#pragma once

#include <memory>

#include <opencv2/core.hpp>

#include "InputPadder.hpp"

/**
 * Future TensorRT FoundationStereo inference engine.
 *
 * This class will own engine inference and OpenCV image-to-GPU transfer.
 */
class FS {
public:
    FS();

    /** Prepare outer-padded stereo inputs for the FoundationStereo pipeline. */
    void prepare_stereo_images(const cv::Mat& left, const cv::Mat& right);

private:
    static constexpr double kSmallRatio = 0.5;

    std::unique_ptr<InputPadder> outer_padder_ = nullptr; // outer padder to pad 2048x2448 to 2048x2464
    cv::Mat outer_left_; // The outer padded left image, should be 2048x2464
    cv::Mat outer_right_; // The outer padded right image, should be 2048x2464
};
