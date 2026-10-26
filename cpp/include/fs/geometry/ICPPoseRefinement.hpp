#pragma once

#include "fs/geometry/MeshBuilderGPU.hpp"
#include <opencv2/core/matx.hpp>
#include <functional>
#include <string>
#include <vector>

namespace fs {
// Borrowed immutable camera-space FS points. device=-1 denotes host memory
// (also useful for offline verification); otherwise it is a CUDA device index.
// Only finite XYZ samples with Z>0 and a nonzero selection mask are used.
struct ICPView {
    const MeshGPUVertex* vertices{nullptr};
    std::size_t vertex_count{0};
    int device{-1};
    cudaStream_t stream{nullptr};
    cv::Matx44d camera_to_world{cv::Matx44d::eye()};
    // Optional host mask, one byte per vertex slot; empty selects all valid points.
    std::vector<unsigned char> selection_mask;
};

struct ICPViewResult {
    cv::Matx44d camera_to_world{cv::Matx44d::eye()};
    bool reference{false}, computed{false}, accepted{false};
    std::size_t point_count{0}; // Valid selected samples before voxel downsampling.
    double fitness{0}, rmse_m{0}, translation_m{0}, rotation_deg{0};
    std::string message;
};

// tbsa_panorama defaults: reference view 0, accumulated previous views,
// point-to-plane / Tukey, three scales, reject excessive corrections.
// Does not modify input buffers or poses. Cancellation throws via checkpoint;
// callers should publish all returned poses together only after completion.
std::vector<ICPViewResult> refine_camera_poses_icp(
    const std::vector<ICPView>& views,
    const std::function<void()>& checkpoint = {},
    const std::function<void(std::size_t, const ICPViewResult&)>& on_view = {});
} // namespace fs
