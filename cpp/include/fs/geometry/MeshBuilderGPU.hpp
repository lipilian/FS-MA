#pragma once

#include <cuda_runtime_api.h>

namespace fs {
// Borrowed, read-only inputs owned by FS. XYZ is packed FP32 [height][width][3]
// in metres; mask is packed uint8 [height][width], with nonzero selecting pixels.
// RGB is the FS left input: FP32 NCHW [1][3][height][width], RGB order, 0..255.
// Use only during the serialized worker action. Do not retain these pointers:
// reconstruction/denoising can replace XYZ, input preparation overwrites RGB,
// and mask uploads overwrite the mask.
struct MeshGPUInputs {
    const float* xyz{nullptr};
    const float* rgb{nullptr};
    const unsigned char* mask{nullptr};
    int width{0}, height{0};
    cudaStream_t stream{nullptr}; // The existing FS stream, never a new stream.
};

// Temporary status while the GPU algorithm and rendering output are undefined.
enum class MeshGPUStatus { NotImplemented };

// Future kernels must use inputs.stream and leave the borrowed inputs unchanged.
// Intended output: GPU mesh buffers shared with OpenGL, without a CPU MeshResult.
// CUDA/OpenGL buffer ownership and synchronization remain to be implemented.
// Currently only validates inputs and returns NotImplemented.
MeshGPUStatus build_mesh_gpu(const MeshGPUInputs& inputs,
                             double max_edge_m = .02,
                             double max_depth_jump_m = .01);
} // namespace fs
