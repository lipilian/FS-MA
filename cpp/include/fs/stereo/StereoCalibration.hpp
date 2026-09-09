#pragma once

#include <filesystem>

#include <opencv2/core.hpp>

/**
 * Raw pinhole stereo calibration, using the existing capture JSON convention.
 * Matrices must be single-channel CV_32F or CV_64F and match the input image grid.
 * Translation is in metres. Extrinsics are passed to stereoRectify unchanged,
 * as in the existing file-input pipeline; no inversion or unit conversion occurs.
 */
struct StereoCalibration {
    cv::Mat left_camera_matrix;       // 3 x 3, positive fx/fy, last row [0, 0, 1].
    cv::Mat right_camera_matrix;      // 3 x 3, positive fx/fy, last row [0, 0, 1].
    cv::Mat left_distortion;          // Row/column vector: 4, 5, 8, 12 or 14 values.
    cv::Mat right_distortion;         // Use explicit zeros for no distortion.
    cv::Mat right_to_left_rotation;   // 3 x 3 proper rotation matrix.
    cv::Mat right_to_left_translation; // Row/column vector: 3 values, nonzero baseline.

    /** Read the six required JSON matrices; StereoFrame validates their values. */
    static StereoCalibration from_file(const std::filesystem::path& path);
};
