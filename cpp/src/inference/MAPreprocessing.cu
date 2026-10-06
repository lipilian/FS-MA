#include "fs/inference/MAPreprocessing.hpp"

#include <cuda_runtime.h>
#include <climits>
#include <cstddef>

namespace fs::ma_preprocessing {
namespace {

__global__ void normalize_dinov2_rgb_kernel(float* rgb, int pixel_count) {
    const std::size_t index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::size_t plane = pixel_count;
    if (index >= plane) return;
    rgb[index] = (rgb[index] / 255.0F - 0.485F) / 0.229F;
    rgb[plane + index] = (rgb[plane + index] / 255.0F - 0.456F) / 0.224F;
    rgb[2 * plane + index] = (rgb[2 * plane + index] / 255.0F - 0.406F) / 0.225F;
}

} // namespace

cudaError_t normalize_dinov2_rgb(float* rgb, int width, int height, cudaStream_t stream) noexcept {
    if (!rgb || width <= 0 || height <= 0 || width > INT_MAX / height)
        return cudaErrorInvalidValue;
    constexpr int threads = 256;
    const int pixel_count = width * height;
    normalize_dinov2_rgb_kernel<<<1 + (pixel_count - 1) / threads, threads, 0, stream>>>(rgb, pixel_count);
    return cudaGetLastError();
}

} // namespace fs::ma_preprocessing
