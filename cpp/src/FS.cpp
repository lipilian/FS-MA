#include "FS.hpp"

#include <stdexcept>

FS::FS() = default;

void FS::prepare_stereo_images(const cv::Mat& left, const cv::Mat& right) {
    if (left.empty() || right.empty()) {
        throw std::invalid_argument("FS coarse input preparation requires non-empty left and right images");
    }
    if (left.size() != right.size()) {
        throw std::invalid_argument("FS coarse input preparation requires matching left and right dimensions");
    }

    outer_padder_ = std::make_unique<InputPadder>(left.size());
    outer_left_ = outer_padder_->pad(left);
    outer_right_ = outer_padder_->pad(right);
}
