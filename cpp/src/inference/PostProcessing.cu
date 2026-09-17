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

__global__ void denoise_xyz_map_kernel(const float* xyz, float* output,
    const unsigned char* selected, int width, int height,
    float max_distance, int min_neighbors, int edge_min_neighbors) {
    const int index=blockIdx.x*blockDim.x+threadIdx.x;
    if (index>=width*height) return;
    const size_t offset=size_t(index)*3;
    const float x=xyz[offset], y=xyz[offset+1], z=xyz[offset+2];
    // The selection limits denoising, not the full-image XYZ result.
    if (selected && !selected[index]) {
        output[offset]=x; output[offset+1]=y; output[offset+2]=z;
        return;
    }
    bool keep=isfinite(x) && isfinite(y) && isfinite(z) && z>0;
    if (keep) {
        const int u=index%width, v=index/width;
        int selected_neighbors=0, good_neighbors=0;
        for (int dy=-1;dy<=1;++dy) for (int dx=-1;dx<=1;++dx) {
            if (dx==0 && dy==0) continue;
            const int nu=u+dx, nv=v+dy;
            if (nu<0 || nu>=width || nv<0 || nv>=height) continue;
            const int neighbor=nv*width+nu;
            if (selected && !selected[neighbor]) continue;
            ++selected_neighbors;
            const size_t at=size_t(neighbor)*3;
            const float nx=xyz[at], ny=xyz[at+1], nz=xyz[at+2];
            if (!isfinite(nx) || !isfinite(ny) || !isfinite(nz) || nz<=0) continue;
            const float ax=nx-x, ay=ny-y, az=nz-z;
            // Round the FP32 operations separately, as in the Python torch expression.
            const float distance=sqrtf(__fadd_rn(__fadd_rn(__fmul_rn(ax,ax),__fmul_rn(ay,ay)),__fmul_rn(az,az)));
            if (distance<=max_distance) ++good_neighbors;
        }
        const int required=selected_neighbors==8 ? min_neighbors : max(1,min(selected_neighbors,edge_min_neighbors));
        keep=good_neighbors>=required;
    }
    output[offset]=keep ? x : 0; output[offset+1]=keep ? y : 0; output[offset+2]=keep ? z : 0;
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

cudaError_t denoise_xyz_map(const float* xyz, float* output, const unsigned char* selection_mask,
                           int width, int height, float max_neighbor_distance_m,
                           int min_neighbors, int edge_min_neighbors, cudaStream_t stream) noexcept {
    if (!xyz || !output || xyz==output || width<=0 || height<=0 || width>INT_MAX/height ||
        !std::isfinite(max_neighbor_distance_m) || max_neighbor_distance_m<=0 || min_neighbors<=0 || edge_min_neighbors<=0)
        return cudaErrorInvalidValue;
    constexpr int threads=256;
    denoise_xyz_map_kernel<<<1+(width*height-1)/threads,threads,0,stream>>>(
        xyz,output,selection_mask,width,height,max_neighbor_distance_m,min_neighbors,edge_min_neighbors);
    return cudaGetLastError();
}

} // namespace fs::postprocessing
