#pragma once
#include "PipelineController.hpp"
#include <QMainWindow>
#include <QElapsedTimer>
class QPushButton;
class QLabel;
class QCheckBox;
class QDoubleSpinBox;
class QSpinBox;
class QProgressBar;
class QPlainTextEdit;
class QTabWidget;
class StereoImageView;
class MaskEditor;
class MeshView;
class QComboBox;
class QLineEdit;

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
    void refreshWorkflow();
    PipelineController& controller_;
    PipelineState state_;
    quint64 mask_request_id_{0};
    bool reconstruction_valid_{false}, mesh_valid_{false};
    SharedMesh mesh_result_;
    SharedGPUMesh gpu_mesh_result_;
    bool gpu_upload_pending_{false};
    bool busy_{false}, closing_{false}, allow_close_{false};
    bool preview_rectified_{true}; // User preference for cameras requiring software rectification.
    QElapsedTimer elapsed_;
    QImage raw_left_, raw_right_, rectified_left_, rectified_right_, live_left_, live_right_;
    QPushButton *import_, *camera_, *preview_, *capture_, *run_, *finish_draw_, *clear_, *build_mesh_cpu_, *build_mesh_gpu_, *next_;
    QPushButton *sam_box_, *sam_foreground_, *sam_background_, *sam_remove_, *sam_undo_, *brush_, *eraser_;
    QSpinBox* brush_size_;
    QCheckBox *capture_calibration_, *rectified_, *epilines_, *denoise_;
    QDoubleSpinBox *minimum_, *maximum_, *neighbor_distance_, *mesh_edge_, *mesh_jump_;
    QLabel *input_, *calibration_, *status_, *time_, *scene_status_, *depth_status_, *steps_;
    QProgressBar* progress_;
    QPlainTextEdit* log_;
    QTabWidget* tabs_;
    QLabel* tab_status_[4]{};
    StereoImageView *left_, *right_, *depth_left_, *depth_map_;
    MaskEditor* mask_;
    QPushButton *save_selected_, *save_all_, *browse_save_;
    QCheckBox *save_images_, *save_calibration_, *save_mask_, *save_mesh_, *save_depth_;
    QLineEdit* save_directory_;
    QLabel* save_calibration_name_;
    MeshView* mesh_view_;
    QComboBox* mesh_mode_;
    QComboBox* camera_mode_;
};
