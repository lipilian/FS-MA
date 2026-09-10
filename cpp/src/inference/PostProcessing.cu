#include "fs/inference/PostProcessing.hpp"

#include <cuda_runtime.h>
#include <climits>

namespace fs::postprocessing {
namespace {

__global__ void filter_disparity_kernel(
    float* disparity,
    const unsigned char* selection_mask,
    int pixel_count,
    int width,
    bool remove_invisible) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) return;

    const float d = disparity[index];
    const int u = index % width;
    const bool valid = isfinite(d) && d > 0.0F &&
        (selection_mask == nullptr || selection_mask[index] != 0) &&
        (!remove_invisible || u >= d);
    disparity[index] = valid ? d : 0.0F;
}

} // namespace

cudaError_t filter_disparity(
    float* disparity,
    const unsigned char* selection_mask,
    int width,
    int height,
    bool remove_invisible,
    cudaStream_t stream) noexcept {
    if (disparity == nullptr ||
        width <= 0 || height <= 0 || width > INT_MAX / height) {
        return cudaErrorInvalidValue;
    }

    constexpr int threads = 256;
    const int pixel_count = width * height;
    const int blocks = 1 + (pixel_count - 1) / threads;
    filter_disparity_kernel<<<blocks, threads, 0, stream>>>(
        disparity, selection_mask,
        pixel_count, width, remove_invisible);
    return cudaGetLastError();
}

// Add separate depth/XYZ and neighbourhood-denoising kernels here as needed.

} // namespace fs::postprocessing
