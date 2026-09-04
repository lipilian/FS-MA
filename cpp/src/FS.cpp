#include "FS.hpp"

#include <opencv2/imgproc.hpp>

#include <stdexcept>

FS::FS() = default;

void FS::prepare_stereo_images(const cv::Mat& left, const cv::Mat& right) {
    if (left.empty() || right.empty()) {
        throw std::invalid_argument("FS coarse input preparation requires non-empty left and right images");
    }
    if (left.size() != right.size()) {
        throw std::invalid_argument("FS coarse input preparation requires matching left and right dimensions");
    }
    const cv::Size model_size(kTensorRtInputWidth, kTensorRtInputHeight);
    cv::resize(left, model_left_, model_size, 0.0, 0.0, cv::INTER_LINEAR);
    cv::resize(right, model_right_, model_size, 0.0, 0.0, cv::INTER_LINEAR);
}
