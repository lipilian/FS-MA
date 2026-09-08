#pragma once

#include "StereoFrame.hpp"

#include <filesystem>
#include <memory>

#include <cuda_runtime_api.h>
#include <opencv2/core.hpp>

namespace nvinfer1 {
class IRuntime;
class ICudaEngine;
class IExecutionContext;
}

/**
 * Future TensorRT FoundationStereo inference engine.
 *
 * This class owns the TensorRT engine and reusable OpenCV-to-GPU input buffers.
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

    /**
     * Resize matching RGB images, convert them to FP32 NCHW, and queue their
     * transfer into the already-bound TensorRT input buffers on FS's CUDA
     * stream.
     */
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
    struct CudaDeviceBufferDeleter {
        void operator()(float* pointer) const noexcept;
    };
    struct CudaHostBufferDeleter {
        void operator()(float* pointer) const noexcept;
    };

    void allocate_input_buffers();
    static void check_cuda(cudaError_t status, const char* operation);

    std::unique_ptr<nvinfer1::IRuntime, TensorRtRuntimeDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TensorRtEngineDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TensorRtContextDeleter> execution_context_;
    StereoCameraParameters model_camera_parameters_;
    cv::Mat model_left_;
    cv::Mat model_right_;
    std::unique_ptr<float, CudaDeviceBufferDeleter> left_input_device_;
    std::unique_ptr<float, CudaDeviceBufferDeleter> right_input_device_;
    std::unique_ptr<float, CudaDeviceBufferDeleter> disparity_output_device_;
    std::unique_ptr<float, CudaHostBufferDeleter> left_input_host_;
    std::unique_ptr<float, CudaHostBufferDeleter> right_input_host_;
    cudaStream_t stream_{nullptr};

};
