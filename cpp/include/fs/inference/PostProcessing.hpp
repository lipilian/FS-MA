#pragma once

#include <cuda_runtime_api.h>

namespace fs::postprocessing {

/**
 * Fuse disparity validation, depth clipping and XYZ reconstruction in one kernel.
 * Disparity is read-only, in pixels on the same width x height grid as intrinsics.
 * Output is interleaved FP32 [height][width][3] (X,Y,Z), in metres. Invalid points
 * are (0,0,0); valid points have Z > 0 and min_depth_m <= Z <= max_depth_m.
 * Thresholds must be finite with 0 <= min_depth_m < max_depth_m.
 *
 * All buffers belong to the caller; XYZ must not overlap either input. An optional
 * selection mask uses nonzero bytes for selected pixels; nullptr selects all.
 * Always require u >= disparity. Each thread handles one pixel.
 * Dimensions must be positive and their product must fit in an int. Intrinsics
 * must be finite, with positive fx/fy and baseline_meters.
 *
 * No allocation or synchronization is performed. Returns cudaErrorInvalidValue
 * for invalid parameters, otherwise the launch status; the caller checks stream
 * completion before reading or releasing results. Input buffers stay unchanged.
 */
cudaError_t compute_xyz_map(
    const float* disparity,
    float* xyz_map,
    const unsigned char* selection_mask,
    int width, int height,
    float fx, float fy, float cx, float cy, float baseline_meters,
    float min_depth_m, float max_depth_m,
    cudaStream_t stream) noexcept;

} // namespace fs::postprocessing
