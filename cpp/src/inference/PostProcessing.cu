#include "fs/inference/PostProcessing.hpp"

#include <cuda_runtime.h>
#include <climits>
#include <cmath>
#include <cstddef>

namespace fs::postprocessing {
namespace {

__global__ void compute_xyz_map_kernel(
    const float* disparity,
    float* xyz_map,
    const unsigned char* selection_mask,
    int pixel_count, int width,
    float fx, float fy, float cx, float cy, float baseline_meters,
    float min_depth_m, float max_depth_m) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= pixel_count) return;

    const float d = disparity[index];
    const int u = index % width;
    const int v = index / width;
    float x = 0.0F, y = 0.0F, z = 0.0F;
    if (isfinite(d) && d > 0.0F &&
        (selection_mask == nullptr || selection_mask[index] != 0) &&
        u >= d) {
        const float depth = fx * baseline_meters / d;
        if (isfinite(depth) && depth > 0.0F &&
            depth >= min_depth_m && depth <= max_depth_m) {
            const float point_x = (u - cx) * depth / fx;
            const float point_y = (v - cy) * depth / fy;
            if (isfinite(point_x) && isfinite(point_y)) {
                x = point_x;
                y = point_y;
                z = depth;
            }
        }
    }

    const std::size_t offset = static_cast<std::size_t>(index) * 3;
    xyz_map[offset] = x;
    xyz_map[offset + 1] = y;
    xyz_map[offset + 2] = z;
}

} // namespace

cudaError_t compute_xyz_map(
    const float* disparity,
    float* xyz_map,
    const unsigned char* selection_mask,
    int width, int height,
    float fx, float fy, float cx, float cy, float baseline_meters,
    float min_depth_m, float max_depth_m,
    cudaStream_t stream) noexcept {
    if (disparity == nullptr || xyz_map == nullptr ||
        width <= 0 || height <= 0 || width > INT_MAX / height ||
        !std::isfinite(fx) || fx <= 0.0F || !std::isfinite(fy) || fy <= 0.0F ||
        !std::isfinite(cx) || !std::isfinite(cy) ||
        !std::isfinite(baseline_meters) || baseline_meters <= 0.0F ||
        !std::isfinite(min_depth_m) || !std::isfinite(max_depth_m) ||
        min_depth_m < 0.0F || max_depth_m <= min_depth_m) {
        return cudaErrorInvalidValue;
    }

    constexpr int threads = 256;
    const int pixel_count = width * height;
    const int blocks = 1 + (pixel_count - 1) / threads;
    compute_xyz_map_kernel<<<blocks, threads, 0, stream>>>(
        disparity, xyz_map, selection_mask, pixel_count, width,
        fx, fy, cx, cy, baseline_meters, min_depth_m, max_depth_m);
    return cudaGetLastError();
}

// Add neighbourhood-denoising kernels here as needed.

} // namespace fs::postprocessing
