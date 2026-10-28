#pragma once
#include "fs/geometry/TsdfFusion.hpp"
#include <QDialog>
class QDoubleSpinBox;
class QSpinBox;

class TsdfSettingsDialog : public QDialog {
public:
    explicit TsdfSettingsDialog(const fs::TsdfOptions& options, QWidget* parent=nullptr);
    fs::TsdfOptions options() const;
    // Closing the popup (title bar, Close or Escape) commits the current edits.
    void done(int result) override;
private:
    void setOptions(const fs::TsdfOptions& options);
    void updateDistance();
    QDoubleSpinBox *voxel_, *truncation_, *distance_, *mesh_weight_, *point_weight_;
    QSpinBox *resolution_, *count_;
};
