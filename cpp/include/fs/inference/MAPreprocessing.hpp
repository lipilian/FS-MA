#pragma once

#include <cuda_runtime_api.h>

namespace fs::ma_preprocessing {

// Normalize one contiguous FP32 RGB CHW image in place, from [0,255] to
// (RGB / 255 - mean) / std with DINOv2's per-channel constants. Call once after
// each raw upload. Only the three RGB planes are touched. No allocations or
// synchronization; the caller keeps the buffer alive until stream completion.
cudaError_t normalize_dinov2_rgb(float* rgb, int width, int height, cudaStream_t stream) noexcept;

} // namespace fs::ma_preprocessing
