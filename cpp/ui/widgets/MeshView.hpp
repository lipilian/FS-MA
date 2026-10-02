#pragma once
#include "fs/geometry/MeshBuilder.hpp"
#include <QSurfaceFormat>
#include <QImage>
#include <QWidget>
#include <memory>
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
    void setMode(int mode);
    void resetView();
    void setRightCameraVisible(bool visible);
    void setCamera(const std::optional<fs::MeshCamera>& camera);
    void setCameraImage(QImage image);
    void setRightCameraImage(QImage image);

  signals:
    void gpuMeshPresented(SharedGPUMesh mesh);
    void renderFailed(QString message);
    void log(QString message);

  private:
    MeshCanvas *canvas_{};
};
