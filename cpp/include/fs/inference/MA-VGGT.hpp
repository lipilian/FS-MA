#pragma once

#include <cuda_runtime_api.h>
#include <filesystem>
#include <memory>

class FS;

namespace fs {

// MapAnything dynamic RAW engine (the MA-VGGT integration entry point).
// Calls must be serialized with FS on the pipeline thread. Keeps FS alive and
// borrows its CUDA stream; never creates or destroys a separate stream.
class MA_VGGT {
public:
    static constexpr int kMinViews = 2;
    static constexpr int kMaxViews = 5;
    static constexpr int kInputHeight = 434;
    static constexpr int kInputWidth = 518;

    explicit MA_VGGT(std::shared_ptr<FS> fs);
    ~MA_VGGT();
    MA_VGGT(const MA_VGGT&) = delete;
    MA_VGGT& operator=(const MA_VGGT&) = delete;

    // Load weights and a user-managed TensorRT context, validate FP32 I/O and
    // the 2..5-view profile. No inference, I/O buffers or activation workspace
    // are allocated here. A failed reload preserves the previous loaded model.
    void loadEngine(const std::filesystem::path& path);
    bool isLoaded() const noexcept;
    cudaStream_t stream() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace fs
