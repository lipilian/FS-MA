#pragma once
#include "fs/geometry/MeshBuilder.hpp"
#include <QSurfaceFormat>
#include <QImage>
#include <QWidget>
#include <memory>
#include <vector>
#include "GPUMeshFrame.hpp"

class MeshCanvas;
class MeshView : public QWidget {
    Q_OBJECT
  public:
    // Also install as the default before QApplication creates shared contexts.
    static QSurfaceFormat surfaceFormat();
    explicit MeshView(QWidget *parent = nullptr);
    void setMesh(std::shared_ptr<const fs::MeshResult> mesh);
    void setGPUMesh(SharedGPUMesh mesh);
    // Completed captures remain in the scene while the active input changes.
    // A generated CPU/GPU mesh can temporarily replace its capture's points.
    void setCapturedClouds(std::vector<SharedGPUMesh> clouds, SharedGPUMesh hidden = {});
    void setMAClouds(SharedGPUClouds clouds);
    void setShowMAClouds(bool enabled);
    void setPredictedCameras(SharedPredictedCameras cameras);
    void setMode(int mode);
    void resetView();
    void resetCaptureView();
    // Keep the current scene framing while Capture replaces preview geometry.
    void preserveView();
    void setProcessing(bool processing);
    void setRightCameraVisible(bool visible);
    void setCamera(const std::optional<fs::MeshCamera>& camera);
    void setLivePreview(bool enabled);
    void setCameraImage(QImage image);
    void setRightCameraImage(QImage image);

  signals:
    void gpuMeshPresented(SharedGPUMesh mesh);
    void renderFailed(QString message);
    void log(QString message);

  private:
    MeshCanvas *canvas_{};
};
