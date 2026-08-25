#pragma once

#include <opencv2/core.hpp>

class InputPadder {
public:
    InputPadder(const cv::Size& left_image_size, int divis_by = 32, bool force_square = false);

    /** Apply symmetric replicate padding, matching Python's default sintel mode. */
    cv::Mat pad(const cv::Mat& image) const;

private:
    cv::Size left_image_size_;
    int divis_by_;
    bool force_square_;
    int pad_ht_;
    int pad_wd_;
};
