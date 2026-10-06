#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <filesystem>
#include <memory>

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

    // Borrowed device pointers, valid from construction until destruction.
    // Contents are uninitialized until written. Callers must not free them;
    // queue accesses on stream(), serialized with model loading. Access from
    // another stream requires explicit synchronization by the caller.
    float* inputDevice() const noexcept;  // [5,7,434,518]
    float* depthsDevice() const noexcept; // [5,6,434,518]
    float* posesDevice() const noexcept;  // [5,7]
    float* scaleDevice() const noexcept;  // [1,1,1]

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fs
