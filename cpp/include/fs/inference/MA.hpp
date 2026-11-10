#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <opencv2/core/matx.hpp>
#include <vector>

namespace cv { class Mat; }
struct StereoCameraParameters;

namespace fs {

// MapAnything FMHA encoder followed by the dynamic RAW pose/scale engine.
// Owns a non-blocking CUDA stream independently of FS. Calls on this instance
// must be serialized on the pipeline thread.
class MA {
public:
    static constexpr int kMinViews = 2;
    static constexpr int kMaxViews = 5;
    static constexpr int kInputHeight = 434;
    static constexpr int kInputWidth = 518;
    static constexpr int kFeatureChannels = 1536;
    static constexpr int kFeatureHeight = 31;
    static constexpr int kFeatureWidth = 37;
    static constexpr std::size_t kInputElementsPerView = std::size_t{7} * kInputHeight * kInputWidth;
    // Every encoder binding needs 256-byte alignment: 12 padding floats/view.
    static constexpr std::size_t kInputStrideElements = (kInputElementsPerView + 63) / 64 * 64;
    static constexpr std::size_t kTokenElementsPerView = std::size_t{kFeatureChannels} * kFeatureHeight * kFeatureWidth;
    static constexpr std::size_t kClsElementsPerView = kFeatureChannels;
    static constexpr std::size_t kInputElements = kMaxViews * kInputStrideElements;
    static constexpr std::size_t kTokenElements = kMaxViews * kTokenElementsPerView;
    static constexpr std::size_t kClsElements = kMaxViews * kClsElementsPerView;
    static constexpr std::size_t kPoseElements = kMaxViews * 7;
    static constexpr std::size_t kScaleElements = 1;
    static constexpr std::size_t kIOBytes =
        (kInputElements + kTokenElements + kClsElements + kPoseElements + kScaleElements) * sizeof(float);

    // Load both engines, create one context per engine, and allocate V=5 FP32
    // I/O and token/CLS banks during construction. Both contexts share one
    // workspace sized to max(encoder, pose/scale) and run serially on stream().
    // No lazy buffer allocation in inference. Destruction drains the stream
    // and destroys both contexts before releasing their shared workspace.
    MA(const std::filesystem::path& encoder_path, const std::filesystem::path& pose_scale_path);
    ~MA();
    MA(const MA&) = delete;
    MA& operator=(const MA&) = delete;

    bool isLoaded() const noexcept;
    cudaStream_t stream() const noexcept; // Borrowed handle; do not destroy.

    // Resize a rectified CV_8UC3 RGB image directly to 518x434 on the CPU with
    // INTER_LANCZOS4 (no crop), pack raw FP32 CHW, then enqueue H2D followed by
    // DINOv2 RGB normalization and unit camera rays in one kernel on stream().
    // camera must describe rectified_rgb (baseline unused). Intrinsics are
    // scaled per axis; principal points use (c+0.5)*scale-0.5 for pixel centers.
    // Writes channels 0..5 of view_index (0..4); other slots and depth channel 6
    // stay unchanged. No inference or CPU wait for the current GPU work.
    // Owns the pinned staging memory until the upload completes.
    void uploadColor(int view_index, const cv::Mat& rectified_rgb, const StereoCameraParameters& camera);

    // Borrowed 518x434 CV_8UC3 RGB from the last successful uploadColor, before
    // normalization. Copy to retain it; the next uploadColor reuses this storage.
    const cv::Mat& resizedColor() const noexcept;

    // After uploadColor for this slot, resize aligned CV_32FC1 metric Z-depth
    // directly to 518x434 with INTER_LANCZOS4 (no crop). Invalid source depths
    // become zero before resize. Enqueue pinned H2D and a pixel kernel on our
    // stream to write metric ray distance into channel 6; RGB/rays and other
    // views are preserved. Source storage may be reused when this returns.
    void uploadDepth(int view_index, const cv::Mat& depth_z);

    // Forget prepared slots, cached features and results for a new sequence.
    void resetInputs() noexcept;

    // Encode changed slots among the first view_count complete slots (2..5),
    // then run pose/scale directly on their contiguous token/CLS banks. The
    // single encoder context is rebound to each view's fixed input/output slots
    // and enqueued serially on stream(); unchanged features are reused.
    // A preceding inference completes before the pose/scale shape changes.
    // RGB/ray/depth uploads invalidate that view's features and current poses.
    // Retaking RGB also invalidates its depth until uploadDepth is called.
    void inference(int view_count);
    void synchronize();

    // Download only the latest V poses and scalar scale (at most 144 bytes),
    // synchronizing our stream, then decode metric camera-to-world matrices.
    // Preserves the model's world frame, including the first camera's pose.
    std::vector<cv::Matx44d> downloadCameraPoses();

    // Borrowed device pointers, valid from construction until destruction.
    // Contents are uninitialized until written. Callers must not free them;
    // queue accesses on stream(). Access from another stream requires explicit
    // synchronization by the caller.
    // After inference + synchronize, the first V token/CLS slots are valid;
    // poses/scale hold RAW heads before MapAnything output adaptors. Tokens
    // use the engine's channel-major F_fused layout, not [V,1147,1536].
    // View i starts at tokensDevice()+i*kTokenElementsPerView and
    // clsDevice()+i*kClsElementsPerView. Write inputs through upload methods
    // so cached features are invalidated correctly.
    // View i: inputDevice()+i*kInputStrideElements; first [7,434,518] values
    // are data, followed by padding. Feature and output banks have no padding.
    float* inputDevice() const noexcept;
    float* tokensDevice() const noexcept; // F_fused [5,1536,31,37]
    float* clsDevice() const noexcept;    // CLS [5,1536,1]
    float* posesDevice() const noexcept;  // [5,7]
    float* scaleDevice() const noexcept;  // [1,1,1]

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fs
