#include "fs/stereo/StereoFrame.hpp"
#include "fs/core/Logger.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>
#include <string>

#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

constexpr int kMapType = CV_32FC1;

std::string as_string(const std::filesystem::path& path) {
    return path.string();
}

cv::Mat read_rgb_image(const std::filesystem::path& path, const char* side) {
    const cv::Mat bgr = cv::imread(as_string(path), cv::IMREAD_COLOR);
    if (bgr.empty()) {
        throw std::runtime_error(std::string("Unable to read ") + side + " image: " + as_string(path));
    }

    cv::Mat rgb;
    cv::cvtColor(bgr, rgb, cv::COLOR_BGR2RGB);
    return rgb;
}

// Always return independent storage, even when the input is already CV_64F.
cv::Mat copy_matrix(const cv::Mat& matrix, const char* name) {
    if (matrix.empty() || matrix.dims != 2 || matrix.channels() != 1 ||
        (matrix.depth() != CV_32F && matrix.depth() != CV_64F)) {
        throw std::invalid_argument(std::string(name) + " must be a non-empty single-channel float matrix");
    }
    cv::Mat copy;
    matrix.convertTo(copy, CV_64F);
    if (!cv::checkRange(copy)) {
        throw std::invalid_argument(std::string(name) + " must contain only finite values");
    }
    return copy;
}

cv::Mat copy_intrinsics(const cv::Mat& matrix, const char* name) {
    cv::Mat copy = copy_matrix(matrix, name);
    if (copy.size() != cv::Size(3, 3) || copy.at<double>(0, 0) <= 0.0 ||
        copy.at<double>(1, 1) <= 0.0 || std::abs(copy.at<double>(2, 0)) > 1e-6 ||
        std::abs(copy.at<double>(2, 1)) > 1e-6 || std::abs(copy.at<double>(2, 2) - 1.0) > 1e-6) {
        throw std::invalid_argument(std::string(name) + " must be 3x3 with positive fx/fy and last row [0, 0, 1]");
    }
    return copy;
}

cv::Mat copy_distortion(const cv::Mat& matrix, const char* name) {
    cv::Mat copy = copy_matrix(matrix, name);
    const auto count = copy.total();
    if ((copy.rows != 1 && copy.cols != 1) ||
        (count != 4 && count != 5 && count != 8 && count != 12 && count != 14)) {
        throw std::invalid_argument(std::string(name) + " must be a vector of 4, 5, 8, 12 or 14 coefficients");
    }
    return copy.reshape(1, static_cast<int>(count));
}

cv::Mat copy_rotation(const cv::Mat& matrix) {
    cv::Mat copy = copy_matrix(matrix, "right_to_left_rotation");
    if (copy.size() != cv::Size(3, 3) ||
        cv::norm(copy.t() * copy - cv::Mat::eye(3, 3, CV_64F), cv::NORM_INF) > 1e-5 ||
        std::abs(cv::determinant(copy) - 1.0) > 1e-5) {
        throw std::invalid_argument("right_to_left_rotation must be a 3x3 proper rotation matrix");
    }
    return copy;
}

cv::Mat copy_translation(const cv::Mat& matrix) {
    cv::Mat copy = copy_matrix(matrix, "right_to_left_translation");
    if ((copy.rows != 1 && copy.cols != 1) || copy.total() != 3 || cv::norm(copy) <= 0.0) {
        throw std::invalid_argument("right_to_left_translation must be a nonzero 3-element vector in metres");
    }
    return copy.reshape(1, 3);
}

void validate_images(const cv::Mat& left, const cv::Mat& right) {
    if (left.empty() || right.empty() || left.dims != 2 || right.dims != 2 ||
        left.type() != CV_8UC3 || right.type() != CV_8UC3) {
        throw std::invalid_argument("Stereo images must be non-empty 2D RGB CV_8UC3 matrices");
    }
    if (left.size() != right.size()) {
        throw std::invalid_argument("Left and right images must have the same dimensions; got " +
                                    std::to_string(left.cols) + "x" + std::to_string(left.rows) + " and " +
                                    std::to_string(right.cols) + "x" + std::to_string(right.rows));
    }
}

}  // namespace

StereoFrame::StereoFrame(const std::filesystem::path& left_image_path,
                         const std::filesystem::path& right_image_path,
                         const std::filesystem::path& calibration_path)
    : StereoFrame(read_rgb_image(left_image_path, "left"),
                  read_rgb_image(right_image_path, "right"),
                  StereoCalibration::from_file(calibration_path)) {}

StereoFrame::StereoFrame(const cv::Mat& left_rgb, const cv::Mat& right_rgb,
                         const StereoCalibration& calibration) {
    validate_images(left_rgb, right_rgb);
    k1_ = copy_intrinsics(calibration.left_camera_matrix, "left_camera_matrix");
    k2_ = copy_intrinsics(calibration.right_camera_matrix, "right_camera_matrix");
    d1_ = copy_distortion(calibration.left_distortion, "left_distortion");
    d2_ = copy_distortion(calibration.right_distortion, "right_distortion");
    right_to_left_rotation_ = copy_rotation(calibration.right_to_left_rotation);
    right_to_left_translation_ = copy_translation(calibration.right_to_left_translation);
    left_ = left_rgb.clone();
    right_ = right_rgb.clone();
}

void StereoFrame::rectify() {
    const cv::Size image_size = left_.size();
    cv::stereoRectify(k1_, d1_, k2_, d2_, image_size,
                      right_to_left_rotation_, right_to_left_translation_,
                      r1_, r2_, p1_, p2_, q_, cv::CALIB_ZERO_DISPARITY);

    cv::Mat left_map_x, left_map_y, right_map_x, right_map_y;
    cv::initUndistortRectifyMap(k1_, d1_, r1_, p1_, image_size, kMapType, left_map_x, left_map_y);
    cv::initUndistortRectifyMap(k2_, d2_, r2_, p2_, image_size, kMapType, right_map_x, right_map_y);
    cv::remap(left_, rectified_left_, left_map_x, left_map_y, cv::INTER_LINEAR);
    cv::remap(right_, rectified_right_, right_map_x, right_map_y, cv::INTER_LINEAR);

    std::ostringstream summary;
    summary << "Rectification completed in memory.\n"
            << "  left:  " << rectified_left_.cols << "x" << rectified_left_.rows << '\n'
            << "  right: " << rectified_right_.cols << "x" << rectified_right_.rows << '\n'
            << "Baseline: " << rectified_camera_parameters().baseline_meters << " m";
    Logger::log(summary.str());
}

StereoCameraParameters StereoFrame::rectified_camera_parameters() const {
    if (p1_.empty() || p2_.empty() || p2_.at<double>(0, 0) == 0.0) {
        throw std::runtime_error("Rectification has not completed or has invalid projection matrices");
    }

    return {
        p1_.at<double>(0, 0),
        p1_.at<double>(1, 1),
        p1_.at<double>(0, 2),
        p1_.at<double>(1, 2),
        static_cast<float>(std::abs(p2_.at<double>(0, 3) / p2_.at<double>(0, 0))),
    };
}
