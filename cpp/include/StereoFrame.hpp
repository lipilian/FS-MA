#pragma once

#include <filesystem>

#include <opencv2/core.hpp>

class StereoFrame {
public:
    /** Load a raw stereo pair as RGB and the project's OpenCV-style JSON calibration. */
    StereoFrame(const std::filesystem::path& left_image_path,
                const std::filesystem::path& right_image_path,
                const std::filesystem::path& calibration_path);

    /** Undistort and rectify both RGB images using cv::CALIB_ZERO_DISPARITY. */
    void rectify();

    const cv::Mat& left() const { return left_; }
    const cv::Mat& right() const { return right_; }
    const cv::Mat& rectified_left() const { return rectified_left_; }
    const cv::Mat& rectified_right() const { return rectified_right_; }
    cv::Mat rectified_left_camera_matrix() const { return p1_(cv::Rect(0, 0, 3, 3)); }
    double baseline_meters() const;

private:
    static cv::Mat read_matrix(const cv::FileStorage& storage, const char* key);
    static void require_same_size(const cv::Mat& left, const cv::Mat& right);

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
