#pragma once

#include <cuda_runtime_api.h>

namespace fs::postprocessing {

/**
 * Filter a contiguous width x height FP32 disparity map in place on the caller's stream.
 *
 * All pointers refer to device memory owned by the caller. The selection mask
 * may be nullptr to select the whole image; otherwise nonzero bytes select
 * pixels. The selection mask must not overlap the disparity buffer.
 * Valid disparities retain their original values; invalid values become zero.
 * Removed values cannot be recovered by relaxing the filter; rerun inference.
 * Validity can be determined from disparity > 0; no validity mask is stored.
 *
 * The caller owns the stream and keeps every buffer alive until work completes.
 * This launcher will not allocate, synchronize, or destroy caller resources.
 *
 * A pixel is valid when disparity is finite and positive, its selection byte
 * is nonzero (if supplied), and u >= disparity when remove_invisible is enabled.
 * Width and height must be positive and their product must fit in an int;
 * required pointers must be non-null. Each thread processes one pixel.
 *
 * Returns cudaErrorInvalidValue for invalid dimensions or missing required
 * pointers; otherwise returns the CUDA launch status. Execution is asynchronous:
 * the caller must check completion errors when synchronizing its stream.
 */
cudaError_t filter_disparity(
    float* disparity,
    const unsigned char* selection_mask,
    int width,
    int height,
    bool remove_invisible,
    cudaStream_t stream) noexcept;

} // namespace fs::postprocessing
