#pragma once
#include "fs/geometry/MeshBuilder.hpp"
#include "fs/geometry/MeshBuilderGPU.hpp"
#include <QMetaType>

// Immutable, completed device output. Never includes borrowed FS input pointers.
// Keeps the FS stream owner alive through queued delivery, upload and GL context
// recreation. The worker must not rebuild into a buffer referenced by a frame.
struct GPUMeshFrame {
    std::shared_ptr<const fs::MeshGPUBuffer> buffer;
    fs::MeshGPUStats stats;
    fs::MeshCamera camera;
    bool point_cloud{false}; // One GPU vertex slot per pixel, rendered with GL_POINTS.
    cudaStream_t stream{nullptr};
    int device{0};
    std::shared_ptr<const void> stream_owner;
};
using SharedGPUMesh = std::shared_ptr<const GPUMeshFrame>;
Q_DECLARE_METATYPE(SharedGPUMesh)
