#pragma once

#include "fs/geometry/MeshBuilder.hpp"
#include <cuda_runtime_api.h>

namespace fs {
// Borrowed, read-only inputs owned by FS. XYZ is packed FP32 [height][width][3]
// in metres; mask is packed uint8 [height][width], with nonzero selecting pixels.
// Use only during the serialized worker action. Do not retain these pointers:
// reconstruction/denoising can replace XYZ, and mask uploads overwrite the mask.
struct MeshGPUInputs {
    const float* xyz{nullptr};
    const unsigned char* mask{nullptr};
    int width{0}, height{0};
    cudaStream_t stream{nullptr}; // The existing FS stream, never a new stream.
};

// GPU backend entry point. RGB remains a CPU snapshot for future vertex colors.
// Future kernels must use inputs.stream and leave the borrowed inputs unchanged.
// A populated result must own completed CPU mesh data for the existing viewer.
// Currently validates inputs and returns nullopt: no mesh algorithm is chosen.
std::optional<MeshResult> build_mesh_gpu(const MeshGPUInputs& inputs, const cv::Mat& rgb,
                                         double max_edge_m = .02,
                                         double max_depth_jump_m = .01);
} // namespace fs
