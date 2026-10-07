#pragma once
#include "fs/geometry/MeshBuilder.hpp"
#include "fs/geometry/MeshBuilderGPU.hpp"
#include <QMetaType>
#include <QImage>

// Immutable, completed device output. Never includes borrowed model input pointers.
// Keeps the FS or MA stream owner alive through queued delivery, upload and GL context
// recreation. The worker must not rebuild into a buffer referenced by a frame.
struct GPUMeshFrame {
    std::shared_ptr<const fs::MeshGPUBuffer> buffer;
    fs::MeshGPUStats stats;
    fs::MeshCamera camera;
    bool point_cloud{false}; // One GPU vertex slot per pixel, rendered with GL_POINTS.
    cudaStream_t stream{nullptr};
    int device{0};
    std::shared_ptr<const void> stream_owner;
    quint64 image_id{0}; // Match MA poses to the capture, including retakes.
};
using SharedGPUMesh = std::shared_ptr<const GPUMeshFrame>;
Q_DECLARE_METATYPE(SharedGPUMesh)
using SharedGPUClouds = std::shared_ptr<const std::vector<SharedGPUMesh>>;

// Small, immutable display snapshot in MA's predicted world coordinate frame.
// Intrinsics define each wireframe's shape; camera_to_world defines its pose.
struct PredictedCameraFrame {
    fs::MeshCamera camera;
    cv::Matx44d camera_to_world;
    quint64 image_id{0};
    QImage image; // Owned MA-sized RGB snapshot; shared across pose updates.
};
using SharedPredictedCameras = std::shared_ptr<const std::vector<PredictedCameraFrame>>;
