#pragma once
#include "fs/geometry/MeshBuilder.hpp"
#include <QWidget>
#include <memory>
#include "GPUMeshFrame.hpp"

class MeshCanvas;
class MeshView : public QWidget {
    Q_OBJECT
  public:
    explicit MeshView(QWidget *parent = nullptr);
    void setMesh(std::shared_ptr<const fs::MeshResult> mesh);
    void setGPUMesh(SharedGPUMesh mesh);
    void setMode(int mode);
    void resetView();
    void setCameraVisible(bool visible);

  signals:
    void gpuMeshPresented(SharedGPUMesh mesh);
    void renderFailed(QString message);
    void log(QString message);

  private:
    MeshCanvas *canvas_{};
};
