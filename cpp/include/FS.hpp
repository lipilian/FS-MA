#pragma once

#include <opencv2/core.hpp>

/**
 * Future TensorRT FoundationStereo inference engine.
 *
 * This class will own engine inference and OpenCV image-to-GPU transfer.
 */
class FS {
public:
    static constexpr int kTensorRtInputHeight = 800;
    static constexpr int kTensorRtInputWidth = 960;

    FS();

    /** Resize matching RGB stereo images to the fixed TensorRT input grid. */
    void prepare_stereo_images(const cv::Mat& left, const cv::Mat& right);

private:
    cv::Mat model_left_;
    cv::Mat model_right_;

};
