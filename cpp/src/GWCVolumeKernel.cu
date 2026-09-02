#include "GWCVolumeKernel.hpp"

#include <cuda_fp16.h>

namespace fs::trt {
namespace cuda {
// =========================================================================
// 1. GWC Volume
// =========================================================================
template <typename T>
__device__ __forceinline__ float toFloat(T v) {
    return static_cast<float>(v);
}

template <>
__device__ __forceinline__ float toFloat<__half>(__half v) {
    return __half2float(v);
}

template <typename T>
__device__ __forceinline__ T fromFloat(float v) {
    return static_cast<T>(v);
}

template <>
__device__ __forceinline__ __half fromFloat<__half>(float v) {
    return __float2half(v);
}

// Tensor dimensions:
//   left, right: [B, C, H, W]
//   volume:      [B, groups, maxDisparity, H, W]
// Fixed FoundationStereo instance: [1, 224, 200, 240] -> [1, 8, 48, 200, 240].

template <typename InputT, typename OutputT>
__global__ void buildGWCVolumeKernel(
    InputT const* left,
    InputT const* right,
    OutputT* volume,
    int B, int C, int H, int W,
    int maxDisparity,
    int groups) { // TODO: we can remove the groups in the future since it is always 8 in our case.
    const int w = blockIdx.x * blockDim.x + threadIdx.x; // x coordinate of the pixels
    const int h = blockIdx.y * blockDim.y + threadIdx.y; // y coordinate of the pixels 
    const int dgb = blockIdx.z; 

    if (w >= W || h >= H) return;

    const int d = dgb % maxDisparity; // disparity index
    const int g = (dgb / maxDisparity) % groups; // group index
    const int b = dgb / (maxDisparity * groups); // batch index

    if (b >= B) return;

    const int K = C / groups; // number of channels per group. 224 / 8 = 28
    const int w_right = w - d; // corresponding pixel in the right image

    const int out_idx = ((b * groups + g) * maxDisparity + d) * H * W + h * W + w; 

    if (w_right < 0) { 
        volume[out_idx] = fromFloat<OutputT>(0.0f); 
        return; 
    }

    const int spatialStride = H * W;
    const int leftBase = (b * C + g * K) * H * W + h * W + w;
    const int rightBase = (b * C + g * K) * H * W + h * W + w_right;

    // Match F.normalize(..., dim=2): calculate each groups L2 norm in FP32.
    float leftSquaredNorm = 0.0f;
    float rightSquaredNorm = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float leftValue =
            toFloat<InputT>(left[leftBase + k * spatialStride]);
        const float rightValue =
            toFloat<InputT>(right[rightBase + k * spatialStride]);
        leftSquaredNorm += leftValue * leftValue;
        rightSquaredNorm += rightValue * rightValue;
    }

    const float leftInverseNorm =
        1.0f / fmaxf(sqrtf(leftSquaredNorm), 1.0e-12f);
    const float rightInverseNorm =
        1.0f / fmaxf(sqrtf(rightSquaredNorm), 1.0e-12f);

    // Sum the dot product of the normalized group vectors.
    float correlation = 0.0f;
    for (int k = 0; k < K; ++k) {
        const float normalizedLeft =
            toFloat<InputT>(left[leftBase + k * spatialStride]) * leftInverseNorm;
        const float normalizedRight =
            toFloat<InputT>(right[rightBase + k * spatialStride]) * rightInverseNorm;
        correlation += normalizedLeft * normalizedRight;
    }

    volume[out_idx] = fromFloat<OutputT>(correlation);
}

} // namespace cuda

cudaError_t launchGwcVolumeKernel(
    void const* left,
    void const* right,
    void* volume,
    cudaStream_t stream) noexcept {
    // Fixed FoundationStereo ViT-L x4 feature interface for 800 x 960 input.
    constexpr int B = 1;
    constexpr int C = 224;
    constexpr int H = 200;
    constexpr int W = 240;
    constexpr int maxDisparity = 48;
    constexpr int groups = 8;

    dim3 const block(16, 16);
    dim3 const grid(
        (W + block.x - 1) / block.x,
        (H + block.y - 1) / block.y,
        B * groups * maxDisparity);

    cuda::buildGWCVolumeKernel<float, float><<<grid, block, 0, stream>>>(
        static_cast<float const*>(left),
        static_cast<float const*>(right),
        static_cast<float*>(volume),
        B, C, H, W, maxDisparity, groups);

    return cudaGetLastError();
}

} // namespace fs::trt
