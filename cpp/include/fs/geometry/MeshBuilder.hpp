#pragma once

#include <functional>
#include <opencv2/core.hpp>
#include <optional>
#include <vector>

namespace fs {
// XYZ is expressed in the rectified left camera frame: origin at the optical
// centre, +X right, +Y down, +Z forward. Intrinsics use image_size's pixel
// grid.
struct MeshCamera {
    cv::Matx33d intrinsics;
    cv::Size image_size;
};

struct MeshResult {
    std::vector<cv::Vec3f> vertices; // Metres; only retained XYZ samples.
    std::vector<cv::Vec3b> colors;   // RGB bytes, one per vertex.
    std::vector<cv::Vec3i> triangles;
    double area_m2{0};
    int skipped_components{0};
    std::optional<MeshCamera> camera; // Display metadata; not a mesh vertex.
};

// Matches fs_tensorrt800x960_gwc_plugin: every valid selected point, exterior
// contour constraints, centroid-mask / 3D edge / depth-jump rejection. Internal
// holes are not constraints; rejected triangles are not filled afterwards.
// All inputs are CPU-owned, aligned to the same image grid, and left unchanged.
MeshResult build_constrained_mesh(const cv::Mat &xyz, const cv::Mat &selection,
                                  const cv::Mat &rgb, double max_edge_m = .02,
                                  double max_depth_jump_m = .01,
                                  const std::function<void()> &checkpoint = {});
} // namespace fs
