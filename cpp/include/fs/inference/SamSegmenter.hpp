#pragma once
#include <filesystem>
#include <memory>
#include <vector>
#include <opencv2/core.hpp>

namespace fs {
struct SamPrompt {
    cv::Point2f point; // Full-resolution RGB image coordinates.
    int label; // 1 foreground, 0 background, 2 box top-left, 3 box bottom-right.
};

// Serialized calls on the owning worker thread. Owns a CUDA stream independent of FS.
class SamSegmenter {
public:
    static constexpr int kImageSize = 1024;
    static constexpr int kMaxPrompts = 64;
    SamSegmenter();
    ~SamSegmenter();
    SamSegmenter(const SamSegmenter&) = delete;
    SamSegmenter& operator=(const SamSegmenter&) = delete;
    // Load both engines and allocate contexts, GPU I/O, cached features and pinned staging.
    void loadEngines(const std::filesystem::path& encoder, const std::filesystem::path& decoder);
    bool isLoaded() const noexcept;
    bool hasImage() const noexcept;
    void clearImage() noexcept;
    // RGB uint8 input; resize and normalize, encode once, retain features on GPU.
    void setImage(const cv::Mat& rgb);
    // Same-image prompts only run the decoder. Returns owned full-image CV_8UC1 (0/255).
    cv::Mat predict(const std::vector<SamPrompt>& prompts);
    float score() const noexcept;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
}
