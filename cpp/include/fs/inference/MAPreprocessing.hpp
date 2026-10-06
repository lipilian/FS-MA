#pragma once

#include <cuda_runtime_api.h>

namespace fs::ma_preprocessing {

// Prepare one FP32 [7,H,W] view in a single pixel kernel: normalize raw RGB
// [0,255] in channels 0..2 with DINOv2 mean/std and fill channels 3..5 with unit
// camera rays normalize([(u-cx)/fx, (v-cy)/fy, 1]). Intrinsics describe the resized
// image with integer pixel centers (OpenCV convention). Depth channel 6 is left
// untouched. Call once after each raw upload. No allocations or synchronization;
// the caller keeps the buffer alive until stream completion.
cudaError_t prepare_rgb_and_rays(float* input, int width, int height,
                               float fx, float fy, float cx, float cy, cudaStream_t stream) noexcept;

// Convert resized metric Z-depth in channel 6 to distance along the unit rays
// already stored in channels 3..5: distance = Z / ray_z. Invalid/nonpositive
// depths (including Lanczos undershoot) become zero. Only channel 6 is written.
cudaError_t prepare_ray_distance(float* input, int width, int height, cudaStream_t stream) noexcept;

} // namespace fs::ma_preprocessing
