#pragma once
#include "fs/geometry/MeshBuilder.hpp"
#include <QWidget>
#include <memory>

class MeshCanvas;
class MeshView : public QWidget {
  public:
    explicit MeshView(QWidget *parent = nullptr);
    void setMesh(std::shared_ptr<const fs::MeshResult> mesh);
    void setMode(int mode);
    void resetView();

  private:
    MeshCanvas *canvas_{};
};
