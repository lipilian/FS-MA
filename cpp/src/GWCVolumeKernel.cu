#include "GWCVolumeKernel.hpp"

namespace fs::trt {
namespace cuda {




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
    const int dgb = blockIdx.z; // dgb = B * groups * maxDisparity - > 1 * 8 * 48  

    if (w >= W || h >= H) return;

    const int d = dgb % maxDisparity; // disparity index
    const int g = (dgb / maxDisparity) % groups; // group index
    const int b = dgb / (maxDisparity * groups); // batch index

    if (b >= B) return;

    int K = C / groups; // number of channels per group. 224 / 8 = 28
}

} // namespace cuda

} // namespace fs::trt
