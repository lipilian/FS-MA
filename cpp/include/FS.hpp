#pragma once

#include "StereoFrame.hpp"

#include <filesystem>
#include <memory>

#include <opencv2/core.hpp>

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}

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
    ~FS();

    FS(const FS&) = delete;
    FS& operator=(const FS&) = delete;

    /** Register the GWC plugin and deserialize a fixed 800 × 960 TensorRT engine. */
    void loadEngine(const std::filesystem::path& engine_path);
    bool isEngineLoaded() const noexcept { return engine_ != nullptr && execution_context_ != nullptr; }

    /** Scale rectified intrinsics to the fixed TensorRT input grid and retain the baseline in metres. */
    void set_model_camera_parameters(const StereoCameraParameters& rectified_camera_parameters,
                                     const cv::Size& rectified_image_size);

    /** Resize matching RGB stereo images to the fixed TensorRT input grid. */
    void prepare_stereo_images(const cv::Mat& left, const cv::Mat& right);

private:
    struct TensorRtRuntimeDeleter {
        void operator()(nvinfer1::IRuntime* object) const noexcept;
    };
    struct TensorRtEngineDeleter {
        void operator()(nvinfer1::ICudaEngine* object) const noexcept;
    };
    struct TensorRtContextDeleter {
        void operator()(nvinfer1::IExecutionContext* object) const noexcept;
    };

    std::unique_ptr<nvinfer1::IRuntime, TensorRtRuntimeDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TensorRtEngineDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TensorRtContextDeleter> execution_context_;
    StereoCameraParameters model_camera_parameters_;
    cv::Mat model_left_;
    cv::Mat model_right_;

};
