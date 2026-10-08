#pragma once
#include <QSurfaceFormat>
#include <QImage>
#include <QPolygonF>
#include <QWidget>
#include <memory>
#include <optional>
#include <vector>
#include "GPUMeshFrame.hpp"

class MeshCanvas;
class MeshView : public QWidget {
    Q_OBJECT
  public:
    // Also install as the default before QApplication creates shared contexts.
    static QSurfaceFormat surfaceFormat();
    explicit MeshView(QWidget *parent = nullptr);
    void setGPUMesh(SharedGPUMesh mesh);
    // Completed captures remain in the scene while the active input changes.
    // A generated GPU mesh can temporarily replace its capture's points.
    void setCapturedClouds(std::vector<SharedGPUMesh> clouds, SharedGPUMesh hidden = {});
    void setMAClouds(SharedGPUClouds clouds);
    void setShowMAClouds(bool enabled);
    void setPredictedCameras(SharedPredictedCameras cameras);
    void setMode(int mode);
    void resetView();
    void resetCaptureView();
    // Keep the current scene framing while a new capture is reconstructed.
    void preserveView();
    void setRightCameraVisible(bool visible);
    void setCamera(const std::optional<fs::MeshCamera>& camera);
    void setCameraImage(QImage image);
    void setRightCameraImage(QImage image);
    // Clockwise image corners in logical widget pixels, from the last GL frame.
    // Empty when the active camera has no visible image plane.
    QPolygonF cameraImageQuad() const;

  signals:
    void gpuMeshPresented(SharedGPUMesh mesh);
    void renderFailed(QString message);
    void log(QString message);

  private:
    MeshCanvas *canvas_{};
};
