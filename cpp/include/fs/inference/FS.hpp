#pragma once

#include "fs/stereo/StereoFrame.hpp"
#include "fs/geometry/MeshBuilderGPU.hpp"

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
 * TensorRT FoundationStereo inference engine and XYZ reconstruction.
 *
 * This class owns the TensorRT engine, CUDA stream, and reusable input/output buffers.
 * Calls on an instance must be serialized by the caller, in this order:
 * loadEngine -> set_model_camera_parameters -> prepare_stereo_images
 * -> inference -> compute_xyz_map.
 * Input preparation and inference readiness are not tracked internally.
 */
class FS {
public:
    static constexpr int kTensorRtInputHeight = 800;
    static constexpr int kTensorRtInputWidth = 960;

    FS();
    ~FS();

    FS(const FS&) = delete;
    FS& operator=(const FS&) = delete;

    /** Register the GWC plugin, load the fixed 800 × 960 engine/context and allocate GPU I/O/XYZ/mask/denoising scratch, pinned host and CPU resize buffers. */
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

    /**
     * Queue TensorRT inference on FS's CUDA stream. The disparity result stays
     * in FS's preallocated device buffer for subsequent CUDA processing.
     */
    void inference();

    /**
     * Fuse full-image disparity validation and XYZ reconstruction, then synchronize.
     * Uses model_camera_parameters_; call set_model_camera_parameters first.
     * Preserves disparity_output_device_ and reuses xyz_map_device_ (FP32 HWC XYZ,
     * in metres). Invalid or out-of-range points are (0,0,0); valid Z is positive
     * and within [min_depth_m, max_depth_m], inclusive. Defaults to 0 < Z <= 1 m.
     * Always excludes pixels with u < disparity.
     * May be repeated with different thresholds without another inference.
     * Computes the whole image; selection_mask only limits denoise_xyz_map.
     */
    void compute_xyz_map(float min_depth_m = 0.0F, float max_depth_m = 1.0F);

    /** Resize a rectified-left CV_8UC1 mask to 960x800 with nearest-neighbour and
     * upload to reusable GPU storage. Nonzero pixels select the denoising region; empty
     * clears the selection (full image). Call after prepare_stereo_images, which
     * clears the previous mask. Completes the upload before returning.
     */
    void set_selection_mask(const cv::Mat& rectified_mask);

    /** One 3x3 XYZ denoising pass matching Python within the selection on FS's stream, then synchronize.
     * XYZ outside the selection is copied unchanged; only selected neighbours count.
     * Call compute_xyz_map first. Counts valid neighbours within the Euclidean
     * distance limit; mask-interior points require min_neighbors, boundary points
     * require max(1, min(selected_neighbour_count, edge_min_neighbors)).
     * Uses a separate preallocated output so removed points do not affect neighbours.
     */
    void denoise_xyz_map(float max_neighbor_distance_m = 0.01F,
                         int min_neighbors = 3, int edge_min_neighbors = 2);

    /** Download the last computed XYZ map as an owned CV_32FC3 matrix (800 rows × 960 columns, metres).
     * Call compute_xyz_map first. Completes the transfer before returning.
     */
    cv::Mat download_xyz_map();

    /** Upload the current nonempty rectified-left mask on the FS stream and
     * borrow GPU RGB and the latest XYZ (including the denoising buffer swap). Requires
     * completed XYZ for the current inputs. No inference or XYZ transfer occurs.
     * The mask upload completes before return; consume this view immediately in
     * the same serialized action, before any further FS mutation or destruction.
     */
    fs::MeshGPUInputs prepare_gpu_mesh_inputs(const cv::Mat& rectified_mask);

    /** Wait for queued GPU work and report asynchronous execution errors. */
    void synchronize();

    /** Execute ten inferences, print each GPU execution time, and return their mean in milliseconds. */
    float inference_time_measure();

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
        void operator()(unsigned char* pointer) const noexcept;
    };
    struct CudaHostBufferDeleter {
        void operator()(float* pointer) const noexcept;
        void operator()(unsigned char* pointer) const noexcept;
    };

    void allocate_input_buffers();
    static void check_cuda(cudaError_t status, const char* operation);

    std::unique_ptr<nvinfer1::IRuntime, TensorRtRuntimeDeleter> runtime_;
    std::unique_ptr<nvinfer1::ICudaEngine, TensorRtEngineDeleter> engine_;
    std::unique_ptr<nvinfer1::IExecutionContext, TensorRtContextDeleter> execution_context_;
    StereoCameraParameters model_camera_parameters_;
    cv::Mat model_left_;
    cv::Mat model_right_;
    cv::Mat model_mask_;
    bool selection_mask_ready_{false};
    bool xyz_ready_{false};
    std::unique_ptr<float, CudaDeviceBufferDeleter> left_input_device_;
    std::unique_ptr<float, CudaDeviceBufferDeleter> right_input_device_;
    std::unique_ptr<float, CudaDeviceBufferDeleter> disparity_output_device_;
    std::unique_ptr<float, CudaDeviceBufferDeleter> xyz_map_device_;
    std::unique_ptr<float, CudaDeviceBufferDeleter> xyz_scratch_device_;
    std::unique_ptr<unsigned char, CudaDeviceBufferDeleter> selection_mask_device_;
    std::unique_ptr<unsigned char, CudaHostBufferDeleter> selection_mask_host_;
    std::unique_ptr<float, CudaHostBufferDeleter> left_input_host_;
    std::unique_ptr<float, CudaHostBufferDeleter> right_input_host_;
    cudaStream_t stream_{nullptr};

};
