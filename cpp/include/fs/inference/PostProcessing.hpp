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

/** Single 5x5 pass. Input/output must be distinct; the centre is not a neighbour.
 * Require min_neighbors selected, valid neighbours strictly closer than the 3D
 * distance limit, including at image/mask boundaries. Fit a least-squares plane
 * to those neighbours (smallest covariance eigenvector); reject undefined normals.
 * The unsigned angle between its normal and the point-to-left-camera ray must be
 * strictly below max_normal_angle_degrees. XYZ must be in the left-camera frame.
 * Null selection means the whole image; out-of-image neighbours do not count.
 * Valid input points have finite XYZ and positive Z. Invalid/unsupported outputs
 * within the selection are zero; unselected XYZ is copied unchanged.
 * No allocations or synchronization. Distance must be finite/positive, neighbour
 * count in [3,24], angle finite and in (0,90). All input buffers remain unchanged.
 */
cudaError_t denoise_xyz_map(const float* xyz, float* output, const unsigned char* selection_mask,
                           int width, int height, float max_neighbor_distance_m,
                           int min_neighbors, float max_normal_angle_degrees, cudaStream_t stream) noexcept;

} // namespace fs::postprocessing
