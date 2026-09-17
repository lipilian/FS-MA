#include "fs/geometry/MeshBuilderGPU.hpp"
#include <cmath>
#include <stdexcept>

namespace fs {
std::optional<MeshResult> build_mesh_gpu(const MeshGPUInputs& inputs, const cv::Mat& rgb,
                                         double max_edge_m, double max_depth_jump_m) {
    if (!inputs.xyz || !inputs.mask || !inputs.stream || inputs.width <= 0 || inputs.height <= 0 ||
        rgb.type() != CV_8UC3 || rgb.size() != cv::Size(inputs.width, inputs.height) ||
        !std::isfinite(max_edge_m) || max_edge_m <= 0 ||
        !std::isfinite(max_depth_jump_m) || max_depth_jump_m <= 0)
        throw std::invalid_argument("GPU mesh needs aligned XYZ, mask and RGB, the FS stream and positive thresholds");

    // TODO: Implement the agreed GPU meshing algorithm here on inputs.stream.
    // No CPU meshing fallback, XYZ transfer, or fabricated output in this scaffold.
    return std::nullopt;
}
} // namespace fs
