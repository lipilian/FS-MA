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

// One view of MA's RAW dense output [6,H,W], RGB/rays/input depth [7,H,W],
// and shared raw log scale [1]. All pointers stay on the MA CUDA device.
struct MAPointCloudGPUInputs {
    const float* dense{nullptr};
    const float* input{nullptr};
    const float* scale{nullptr};
    int width{0}, height{0};
    cudaStream_t stream{nullptr};
};

// Rendering layout: XYZ in camera-space metres and RGB in [0,1].
struct MeshGPUVertex { float x, y, z, r, g, b; };
struct MeshGPUStats {
    double area_m2{0};
    unsigned long long triangle_count{0};
    unsigned long long point_count{0}; // Valid point-cloud samples; zero for triangle output.
    float low[3]{}, high[3]{}; // Bounds of emitted vertices; zero for an empty mesh.
};

// Reusable, owned device output and reduction scratch. Does not borrow FS data.
// Meshes use six slots per cell; point clouds use one slot per pixel. Unused
// slots are zeroed and clipped by the viewer. No compaction or hole filling.
// Access only after a build returns; the next build overwrites the data.
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
    friend MeshGPUStats build_mesh_gpu(const MeshGPUInputs&, MeshGPUBuffer&, double);
    friend MeshGPUStats build_point_cloud_gpu(const MeshGPUInputs&, MeshGPUBuffer&);
    friend MeshGPUStats build_ma_point_cloud_gpu(const MAPointCloudGPUInputs&, MeshGPUBuffer&);
};

// One 16x16-block thread per pixel cell. Four valid corners produce ABD/ADC;
// three valid corners produce one triangle. Fewer produce none. Point validity:
// selected mask, finite XYZ, Z>0. Reject each triangle independently on all three
// 3D edge lengths (10 mm by default), or zero/nonfinite area. No links beyond the 2x2 cell.
// All GPU work uses inputs.stream. Completes before returning; only the final
// area/count/bounds are downloaded. Geometry and normalized RGB remain on the GPU.
MeshGPUStats build_mesh_gpu(const MeshGPUInputs& inputs, MeshGPUBuffer& output,
                            double max_edge_m = .01);
// Preserve every selected finite XYZ sample with Z>0, including isolated points.
// XYZ/RGB stay on the device; only point count and bounds are downloaded.
MeshGPUStats build_point_cloud_gpu(const MeshGPUInputs& inputs, MeshGPUBuffer& output);
// Decode predicted unit rays * exp(depth) * exp(scale) into camera-space metres
// and undo DINOv2 RGB normalization on GPU. Keep finite forward-facing points
// with a positive non-ambiguous mask logit and finite positive input depth in
// channel 6. Zero/invalid input depth produces a zero vertex even when MA
// predicts valid depth. Only count/bounds leave the device.
MeshGPUStats build_ma_point_cloud_gpu(const MAPointCloudGPUInputs& inputs, MeshGPUBuffer& output);
} // namespace fs
