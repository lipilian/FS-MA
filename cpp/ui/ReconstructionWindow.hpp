#pragma once
#include "PipelineController.hpp"
#include <QMainWindow>
#include <QElapsedTimer>
class QPushButton;
class QLabel;
class QCheckBox;
class QDoubleSpinBox;
class QProgressBar;
class QPlainTextEdit;
class QTabWidget;
class StereoImageView;
class MaskEditor;

class ReconstructionWindow : public QMainWindow {
    Q_OBJECT
public:
    ReconstructionWindow(PipelineController& controller, ConfirmedCalibration calibration, const QString& path);
    void allowClose();
signals:
    void closeRequested();
protected:
    void closeEvent(QCloseEvent* event) override;
private:
    void refresh();
    void showImages();
    void exportMask();
    PipelineController& controller_;
    PipelineState state_;
    bool busy_{false}, closing_{false}, allow_close_{false};
    QElapsedTimer elapsed_;
    QImage raw_left_, raw_right_, rectified_left_, rectified_right_, live_left_, live_right_;
    QPushButton *import_, *camera_, *preview_, *capture_, *run_, *draw_, *finish_, *undo_, *clear_, *export_mask_;
    QCheckBox *capture_calibration_, *rectified_, *epilines_;
    QDoubleSpinBox *minimum_, *maximum_;
    QLabel *input_, *calibration_, *status_, *time_, *selection_, *scene_status_, *depth_status_, *steps_;
    QProgressBar* progress_;
    QPlainTextEdit* log_;
    QTabWidget* tabs_;
    StereoImageView *left_, *right_, *depth_left_, *depth_map_;
    MaskEditor* mask_;
};
