#pragma once
#include "MeshBuilderGPU.hpp"
#include <opencv2/core.hpp>
#include <functional>
#include <vector>

namespace fs {
struct TsdfOptions {
    double voxel_size_mm{1.0};
    double truncation_voxels{8.0};
    int block_resolution{8};
    int block_count{30000};
    double mesh_weight_threshold{0.0};
    double point_weight_threshold{0.0};
    double truncation_mm() const { return voxel_size_mm * truncation_voxels; }
    void validate() const;
};
struct TsdfView {
    cv::Mat rgb;   // RGB uint8, aligned to depth and intrinsics.
    cv::Mat depth; // Filtered float32 camera Z, metres. Never modified by fusion.
    cv::Mat mask;  // Rectified-left uint8; resized with nearest neighbour.
    cv::Matx33d intrinsics;
    cv::Matx44d camera_to_world;
};
struct TsdfGeometry {
    std::vector<MeshGPUVertex> vertices; // Compact world-space XYZ and RGB [0,1].
    MeshGPUStats stats;
};
struct TsdfResult {
    TsdfGeometry mesh, points;
    int integrated_views{0};
    int64_t active_blocks{0};
};
// Open3D float TSDF/weight/color on CUDA:0. Rebuild from immutable captures;
// use masks when drawn; absent/all-zero masks keep depth, matching tbsa_panorama.
// Extraction is raw (no mesh post-processing). Progress is called between GPU stages.
TsdfResult fuse_tsdf(const std::vector<TsdfView>& views, const TsdfOptions& options,
                     const std::function<void(int, const char*)>& progress = {});
} // namespace fs
