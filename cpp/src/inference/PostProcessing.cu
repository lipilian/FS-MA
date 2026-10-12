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

// Smallest covariance eigenvector: the least-squares plane normal. Work in
// centre-relative coordinates to avoid subtracting squared camera distances.
__device__ bool plane_normal(float covariance[3][3], float normal[3]) {
    const float trace=covariance[0][0]+covariance[1][1]+covariance[2][2];
    if (!isfinite(trace) || trace<=0) return false;
    for (int r=0;r<3;++r) for (int c=0;c<3;++c) covariance[r][c]/=trace;
    float vectors[3][3]={{1,0,0},{0,1,0},{0,0,1}};
    // Jacobi rotations for a symmetric 3x3 matrix, largest off-diagonal first.
    for (int iteration=0;iteration<12;++iteration) {
        int p=0, q=1;
        if (fabsf(covariance[0][2])>fabsf(covariance[p][q])) { p=0; q=2; }
        if (fabsf(covariance[1][2])>fabsf(covariance[p][q])) { p=1; q=2; }
        if (fabsf(covariance[p][q])<=1e-7F) break;
        const float tau=(covariance[q][q]-covariance[p][p])/(2*covariance[p][q]);
        const float t=copysignf(1.0F/(fabsf(tau)+sqrtf(1+tau*tau)),tau);
        const float c=rsqrtf(1+t*t), s=t*c;
        covariance[p][p]-=t*covariance[p][q];
        covariance[q][q]+=t*covariance[p][q];
        covariance[p][q]=covariance[q][p]=0;
        for (int r=0;r<3;++r) {
            if (r!=p && r!=q) {
                const float rp=covariance[r][p], rq=covariance[r][q];
                covariance[r][p]=covariance[p][r]=c*rp-s*rq;
                covariance[r][q]=covariance[q][r]=s*rp+c*rq;
            }
            const float vp=vectors[r][p], vq=vectors[r][q];
            vectors[r][p]=c*vp-s*vq;
            vectors[r][q]=s*vp+c*vq;
        }
    }
    if (fmaxf(fabsf(covariance[0][1]),fmaxf(fabsf(covariance[0][2]),fabsf(covariance[1][2])))>1e-6F)
        return false;
    int least=0, greatest=0;
    for (int i=1;i<3;++i) {
        if (covariance[i][i]<covariance[least][least]) least=i;
        if (covariance[i][i]>covariance[greatest][greatest]) greatest=i;
    }
    if (least==greatest) return false;
    const float low=covariance[least][least], high=covariance[greatest][greatest];
    const float middle=covariance[3-least-greatest][3-least-greatest];
    // Coincident/collinear points or a repeated smallest eigenvalue do not
    // determine a unique normal. Tolerances are relative to the largest spread.
    if (middle<=1e-6F*high || middle-low<=1e-6F*high) return false;
    for (int i=0;i<3;++i) normal[i]=vectors[i][least];
    return true;
}

__global__ void denoise_xyz_map_kernel(const float* xyz, float* output,
    const unsigned char* selected, int width, int height,
    float max_distance, int min_neighbors, float min_normal_cosine) {
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
        int good_neighbors=0;
        float sum[3]={}, products[3][3]={};
        const float distance_squared=max_distance*max_distance;
        for (int dy=-2;dy<=2;++dy) for (int dx=-2;dx<=2;++dx) {
            if (dx==0 && dy==0) continue;
            const int nu=u+dx, nv=v+dy;
            if (nu<0 || nu>=width || nv<0 || nv>=height) continue;
            const int neighbor=nv*width+nu;
            if (selected && !selected[neighbor]) continue;
            const size_t at=size_t(neighbor)*3;
            const float nx=xyz[at], ny=xyz[at+1], nz=xyz[at+2];
            if (!isfinite(nx) || !isfinite(ny) || !isfinite(nz) || nz<=0) continue;
            const float ax=nx-x, ay=ny-y, az=nz-z;
            if (ax*ax+ay*ay+az*az>=distance_squared) continue;
            ++good_neighbors;
            const float delta[3]={ax,ay,az};
            for (int r=0;r<3;++r) {
                sum[r]+=delta[r];
                for (int c=r;c<3;++c) products[r][c]+=delta[r]*delta[c];
            }
        }
        keep=good_neighbors>=min_neighbors;
        if (keep) {
            float covariance[3][3], normal[3];
            for (int r=0;r<3;++r) for (int c=r;c<3;++c)
                covariance[r][c]=covariance[c][r]=products[r][c]-sum[r]*sum[c]/good_neighbors;
            keep=plane_normal(covariance,normal);
            if (keep) {
                // XYZ is in the left-camera frame: the viewing ray is -P.
                // A PCA normal has arbitrary sign, so orient it toward the camera
                // by taking the absolute dot product. No acos is needed.
                const float normal_squared=normal[0]*normal[0]+normal[1]*normal[1]+normal[2]*normal[2];
                const float cosine=fabsf(normal[0]*x+normal[1]*y+normal[2]*z)*rsqrtf(normal_squared)*rsqrtf(x*x+y*y+z*z);
                keep=isfinite(cosine) && cosine>min_normal_cosine;
            }
        }
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
                           int min_neighbors, float max_normal_angle_degrees, cudaStream_t stream) noexcept {
    if (!xyz || !output || xyz==output || width<=0 || height<=0 || width>INT_MAX/height ||
        !std::isfinite(max_neighbor_distance_m) || max_neighbor_distance_m<=0 || min_neighbors<3 || min_neighbors>24 ||
        !std::isfinite(max_normal_angle_degrees) || max_normal_angle_degrees<=0 || max_normal_angle_degrees>=90)
        return cudaErrorInvalidValue;
    const float min_normal_cosine=static_cast<float>(std::cos(max_normal_angle_degrees*3.14159265358979323846/180.0));
    constexpr int threads=256;
    denoise_xyz_map_kernel<<<1+(width*height-1)/threads,threads,0,stream>>>(
        xyz,output,selection_mask,width,height,max_neighbor_distance_m,min_neighbors,min_normal_cosine);
    return cudaGetLastError();
}

} // namespace fs::postprocessing
