#include "StereoFrame.hpp"
#include "Logger.hpp"

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

}  // namespace

StereoFrame::StereoFrame(const std::filesystem::path& left_image_path,
                         const std::filesystem::path& right_image_path,
                         const std::filesystem::path& calibration_path) {
    left_ = read_rgb_image(left_image_path, "left");
    right_ = read_rgb_image(right_image_path, "right");
    require_same_size(left_, right_);

    cv::FileStorage calibration(as_string(calibration_path), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!calibration.isOpened()) {
        throw std::runtime_error("Unable to open calibration JSON: " + as_string(calibration_path));
    }

    k1_ = read_matrix(calibration, "left_camera_matrix");
    d1_ = read_matrix(calibration, "left_distortion");
    k2_ = read_matrix(calibration, "right_camera_matrix");
    d2_ = read_matrix(calibration, "right_distortion");
    right_to_left_rotation_ = read_matrix(calibration, "right_to_left_rotation");
    right_to_left_translation_ = read_matrix(calibration, "right_to_left_translation");

    if (k1_.size() != cv::Size(3, 3) || k2_.size() != cv::Size(3, 3) ||
        right_to_left_rotation_.size() != cv::Size(3, 3) ||
        right_to_left_translation_.total() != 3) {
        throw std::runtime_error("Calibration matrices have invalid dimensions");
    }
}

cv::Mat StereoFrame::read_matrix(const cv::FileStorage& storage, const char* key) {
    const cv::FileNode node = storage[key];
    if (node.empty()) {
        throw std::runtime_error(std::string("Calibration is missing '") + key + "'");
    }

    cv::Mat matrix;
    node >> matrix;
    if (matrix.empty()) {
        throw std::runtime_error(std::string("Calibration matrix '") + key + "' is empty or invalid");
    }
    matrix.convertTo(matrix, CV_64F);
    return matrix;
}

void StereoFrame::require_same_size(const cv::Mat& left, const cv::Mat& right) {
    if (left.size() != right.size()) {
        throw std::runtime_error("Left and right images must have the same dimensions; got " +
                                 std::to_string(left.cols) + "x" + std::to_string(left.rows) + " and " +
                                 std::to_string(right.cols) + "x" + std::to_string(right.rows));
    }
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
