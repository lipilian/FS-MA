#pragma once

#include <cuda_runtime_api.h>

namespace fs::postprocessing {

/**
 * Filter a contiguous width x height FP32 disparity map on the caller's stream.
 *
 * All pointers refer to device memory owned by the caller. The selection mask
 * may be nullptr to select the whole image; otherwise nonzero bytes select
 * pixels. Outputs must use separate storage from the inputs and each other.
 * Valid disparities retain their original values; invalid values become zero.
 * The output validity mask contains 0 or 1 per pixel.
 *
 * The caller owns the stream and keeps every buffer alive until work completes.
 * This launcher will not allocate, synchronize, or destroy caller resources.
 *
 * Scaffold only: currently returns cudaErrorNotSupported without writing outputs.
 */
cudaError_t filter_disparity(
    const float* disparity,
    const unsigned char* selection_mask,
    float* filtered_disparity,
    unsigned char* valid_mask,
    int width,
    int height,
    bool remove_invisible,
    cudaStream_t stream) noexcept;

} // namespace fs::postprocessing
