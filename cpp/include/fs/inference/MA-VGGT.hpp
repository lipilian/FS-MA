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

// MapAnything dynamic RAW engine (the MA-VGGT integration entry point).
// Owns a non-blocking CUDA stream independently of FS. Calls on this instance
// must be serialized on the pipeline thread.
class MA_VGGT {
public:
    static constexpr int kMinViews = 2;
    static constexpr int kMaxViews = 5;
    static constexpr int kInputHeight = 434;
    static constexpr int kInputWidth = 518;
    static constexpr std::size_t kInputElements = std::size_t{kMaxViews} * 7 * kInputHeight * kInputWidth;
    static constexpr std::size_t kDepthElements = std::size_t{kMaxViews} * 6 * kInputHeight * kInputWidth;
    static constexpr std::size_t kPoseElements = kMaxViews * 7;
    static constexpr std::size_t kScaleElements = 1;
    static constexpr std::size_t kIOBytes =
        (kInputElements + kDepthElements + kPoseElements + kScaleElements) * sizeof(float);

    // Create the stream and allocate all four FP32 I/O buffers once for V=5.
    MA_VGGT();
    ~MA_VGGT();
    MA_VGGT(const MA_VGGT&) = delete;
    MA_VGGT& operator=(const MA_VGGT&) = delete;

    // Load weights and a user-managed TensorRT context, validate FP32 I/O and
    // the 2..5-view profile and bind the existing I/O buffers. No inference is
    // run or activation workspace allocated. A failed reload preserves the
    // previous model; both successful and failed reloads preserve I/O addresses.
    void loadEngine(const std::filesystem::path& path);
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

    // Forget prepared slots when starting a new capture sequence.
    void resetInputs() noexcept;

    // Enqueue the first view_count complete slots (2..5) on our stream, after
    // their queued RGB/ray/depth preparation. Allocates the engine's maximum
    // activation workspace on first use and reuses it thereafter. A preceding
    // inference is completed before modifying the context's dynamic shape.
    // Retaking RGB invalidates that slot's depth until uploadDepth is called.
    void inference(int view_count);
    void synchronize();

    // Download only the latest V poses and scalar scale (at most 144 bytes),
    // synchronizing our stream, then decode metric camera-to-world matrices.
    // Preserves the model's world frame, including the first camera's pose.
    std::vector<cv::Matx44d> downloadCameraPoses();

    // Borrowed device pointers, valid from construction until destruction.
    // Contents are uninitialized until written. Callers must not free them;
    // queue accesses on stream(), serialized with model loading. Access from
    // another stream requires explicit synchronization by the caller.
    // After inference + synchronize, the first V entries hold RAW model heads
    // (before MapAnything output adaptors); a later inference replaces them.
    float* inputDevice() const noexcept;  // [5,7,434,518]
    float* depthsDevice() const noexcept; // [5,6,434,518]
    float* posesDevice() const noexcept;  // [5,7]
    float* scaleDevice() const noexcept;  // [1,1,1]

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fs
