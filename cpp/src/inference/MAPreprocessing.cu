#include "fs/inference/MAPreprocessing.hpp"

#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstddef>

namespace fs::ma_preprocessing {
namespace {

__global__ void prepare_rgb_and_rays_kernel(float* input, int width, int pixel_count,
                                          float fx, float fy, float cx, float cy) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t plane = pixel_count;
    if (index >= plane) return;
    input[index] = (input[index] / 255.0F - 0.485F) / 0.229F;
    input[plane + index] = (input[plane + index] / 255.0F - 0.456F) / 0.224F;
    input[2 * plane + index] = (input[2 * plane + index] / 255.0F - 0.406F) / 0.225F;

    // Match get_rays_in_camera_frame(..., normalize_to_unit_sphere=True).
    const float x = (float(index % width) - cx) / fx;
    const float y = (float(index / width) - cy) / fy;
    const float norm = sqrtf(x * x + y * y + 1.0F);
    input[3 * plane + index] = x / norm;
    input[4 * plane + index] = y / norm;
    input[5 * plane + index] = 1.0F / norm;
}

} // namespace

cudaError_t prepare_rgb_and_rays(float* input, int width, int height,
                               float fx, float fy, float cx, float cy, cudaStream_t stream) noexcept {
    if (!input || width <= 0 || height <= 0 || width > INT_MAX / height ||
        !std::isfinite(fx) || fx <= 0.0F || !std::isfinite(fy) || fy <= 0.0F ||
        !std::isfinite(cx) || !std::isfinite(cy))
        return cudaErrorInvalidValue;
    constexpr int threads = 256;
    const int pixel_count = width * height;
    prepare_rgb_and_rays_kernel<<<1 + (pixel_count - 1) / threads, threads, 0, stream>>>(
        input, width, pixel_count, fx, fy, cx, cy);
    return cudaGetLastError();
}

} // namespace fs::ma_preprocessing
