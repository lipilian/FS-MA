#pragma once

#include "fs/stereo/StereoCalibration.hpp"

#include <filesystem>
#include <optional>

#include <opencv2/core.hpp>

struct StereoCameraParameters {
    double fx{0.0};
    double fy{0.0};
    double cx{0.0};
    double cy{0.0};
    float baseline_meters{0.0F};
};

class StereoFrame {
public:
    /** Load a raw stereo pair as RGB and the project's OpenCV-style JSON calibration. */
    StereoFrame(const std::filesystem::path& left_image_path,
                const std::filesystem::path& right_image_path,
                const std::filesystem::path& calibration_path);

    /**
     * Snapshot raw RGB CV_8UC3 images and full calibration from memory.
     * Inputs are validated and deep-copied, including non-contiguous image ROIs.
     * Calibration is retained as CV_64F; callers may reuse their buffers after
     * construction. No RGB/BGR conversion is performed by this overload.
     * Invalid images or calibration values throw std::invalid_argument.
     */
    StereoFrame(const cv::Mat& left_rgb, const cv::Mat& right_rgb,
                const StereoCalibration& calibration);

    /**
     * Snapshot an already rectified horizontal stereo pair without resampling.
     * Both images must share these intrinsics; baseline is positive, in metres.
     * Camera parameters must describe the supplied image grid after any crop.
     */
    StereoFrame(const cv::Mat& left_rgb, const cv::Mat& right_rgb,
                const StereoCameraParameters& rectified_camera);

    /** Undistort and rectify raw inputs; already rectified inputs are unchanged. */
    void rectify();
    bool input_is_rectified() const { return rectified_input_camera_.has_value(); }

    const cv::Mat& left() const { return left_; }
    const cv::Mat& right() const { return right_; }
    const cv::Mat& rectified_left() const { return rectified_left_; }
    const cv::Mat& rectified_right() const { return rectified_right_; }
    /** Return the rectified left-camera intrinsics and rectified stereo baseline. */
    StereoCameraParameters rectified_camera_parameters() const;

private:
    std::optional<StereoCameraParameters> rectified_input_camera_;
    cv::Mat left_;
    cv::Mat right_;
    cv::Mat k1_;
    cv::Mat d1_;
    cv::Mat k2_;
    cv::Mat d2_;
    cv::Mat right_to_left_rotation_;
    cv::Mat right_to_left_translation_;

    cv::Mat r1_;
    cv::Mat r2_;
    cv::Mat p1_;
    cv::Mat p2_;
    cv::Mat q_;
    cv::Mat rectified_left_;
    cv::Mat rectified_right_;
};
