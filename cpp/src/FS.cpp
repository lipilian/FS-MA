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
    // Outer pad the input images from 2048x2448 to 2048x2464.
    outer_padder_ = std::make_unique<InputPadder>(left.size());
    outer_left_ = outer_padder_->pad(left);
    outer_right_ = outer_padder_->pad(right);

    // Convert the outer padded image to float 32 for interpolation
    outer_left_.convertTo(outer_left_, CV_32F);
    outer_right_.convertTo(outer_right_, CV_32F);

    cv::resize(outer_left_, small_left_, cv::Size(), kSmallRatio, kSmallRatio, cv::INTER_LINEAR);
    cv::resize(outer_right_, small_right_, cv::Size(), kSmallRatio, kSmallRatio, cv::INTER_LINEAR);

    


}
