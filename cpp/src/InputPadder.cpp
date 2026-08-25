#include "InputPadder.hpp"
#include "Logger.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>

#include <opencv2/imgproc.hpp>

InputPadder::InputPadder(const cv::Size& left_image_size, int divis_by, bool force_square)
    : left_image_size_(left_image_size), divis_by_(divis_by), force_square_(force_square) {
    if (left_image_size_.width <= 0 || left_image_size_.height <= 0) {
        throw std::invalid_argument("InputPadder requires a non-empty left image size");
    }
    if (divis_by_ <= 0) {
        throw std::invalid_argument("InputPadder divis_by must be positive");
    }

    if (force_square_) {
        const int max_side = std::max(left_image_size_.height, left_image_size_.width);
        pad_ht_ = ((max_side / divis_by_) + 1) * divis_by_ - left_image_size_.height;
        pad_wd_ = ((max_side / divis_by_) + 1) * divis_by_ - left_image_size_.width;
    } else {
        pad_ht_ = (((left_image_size_.height / divis_by_) + 1) * divis_by_ -
                   left_image_size_.height) %
                  divis_by_;
        pad_wd_ = (((left_image_size_.width / divis_by_) + 1) * divis_by_ -
                   left_image_size_.width) %
                  divis_by_;
    }
    Logger::log("InputPadder: width=" + std::to_string(left_image_size_.width) +
                ", height=" + std::to_string(left_image_size_.height) +
                ", pad_ht=" + std::to_string(pad_ht_) +
                ", pad_wd=" + std::to_string(pad_wd_) +
                ", padded_width=" + std::to_string(left_image_size_.width + pad_wd_) +
                ", padded_height=" + std::to_string(left_image_size_.height + pad_ht_));
}

cv::Mat InputPadder::pad(const cv::Mat& image) const {
    if (image.empty()) {
        throw std::invalid_argument("InputPadder cannot pad an empty image");
    }
    if (image.size() != left_image_size_) {
        throw std::invalid_argument("InputPadder image size does not match its construction size");
    }

    const int left = pad_wd_ / 2;
    const int right = pad_wd_ - left;
    const int top = pad_ht_ / 2;
    const int bottom = pad_ht_ - top;

    cv::Mat padded;
    cv::copyMakeBorder(image, padded, top, bottom, left, right, cv::BORDER_REPLICATE);
    return padded;
}
