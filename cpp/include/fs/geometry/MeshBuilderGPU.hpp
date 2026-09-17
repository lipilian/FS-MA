#pragma once

#include <cuda_runtime_api.h>
#include <cstddef>
#include <memory>

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

// Rendering layout: XYZ in camera-space metres and RGB in [0,1].
struct MeshGPUVertex { float x, y, z, r, g, b; };
struct MeshGPUStats {
    double area_m2{0};
    unsigned long long triangle_count{0};
    float low[3]{}, high[3]{}; // Bounds of emitted vertices; zero for an empty mesh.
};

// Reusable, owned device output and reduction scratch. Does not borrow FS data.
// Each cell has six vertex slots (two triangles); unused slots are zeroed,
// forming degenerate triangles. There is no compaction or hole filling.
// Access only after build_mesh_gpu returns; the next build overwrites the data.
// The viewer copies completed output device-to-device into a registered GL VBO.
class MeshGPUBuffer {
public:
    MeshGPUBuffer();
    ~MeshGPUBuffer();
    MeshGPUBuffer(const MeshGPUBuffer&) = delete;
    MeshGPUBuffer& operator=(const MeshGPUBuffer&) = delete;
    const MeshGPUVertex* vertices() const;
    std::size_t vertex_slots() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
    friend MeshGPUStats build_mesh_gpu(const MeshGPUInputs&, MeshGPUBuffer&, double, double);
};

// One 16x16-block thread per pixel cell. Four valid corners produce ABD/ADC;
// three valid corners produce one triangle. Fewer produce none. Point validity:
// selected mask, finite XYZ, Z>0. Reject each triangle independently on all three
// 3D edge lengths, max(Z)-min(Z), or zero area. No links beyond the 2x2 cell.
// All GPU work uses inputs.stream. Completes before returning; only the final
// area/count/bounds are downloaded. Geometry and normalized RGB remain on the GPU.
MeshGPUStats build_mesh_gpu(const MeshGPUInputs& inputs, MeshGPUBuffer& output,
                            double max_edge_m = .02,
                            double max_depth_jump_m = .01);
} // namespace fs
