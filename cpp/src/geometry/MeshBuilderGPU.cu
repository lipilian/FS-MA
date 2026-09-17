#include "fs/geometry/MeshBuilderGPU.hpp"
#include <cmath>
#include <stdexcept>

namespace fs {
MeshGPUStatus build_mesh_gpu(const MeshGPUInputs& inputs,
                             double max_edge_m, double max_depth_jump_m) {
    if (!inputs.xyz || !inputs.rgb || !inputs.mask || !inputs.stream || inputs.width <= 0 || inputs.height <= 0 ||
        !std::isfinite(max_edge_m) || max_edge_m <= 0 ||
        !std::isfinite(max_depth_jump_m) || max_depth_jump_m <= 0)
        throw std::invalid_argument("GPU mesh needs aligned XYZ, mask and RGB, the FS stream and positive thresholds");

    // TODO: Implement the agreed GPU meshing algorithm here on inputs.stream.
    // TODO: Share the resulting GPU buffers with OpenGL for rendering.
    return MeshGPUStatus::NotImplemented;
}
} // namespace fs
