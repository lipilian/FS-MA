#pragma once
#include "PipelineController.hpp"
#include <QMainWindow>
#include <QElapsedTimer>
#include <array>
class QPushButton;
class QLabel;
class QCheckBox;
class QDoubleSpinBox;
class QSpinBox;
class QProgressBar;
class QPlainTextEdit;
class QTabWidget;
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
    enum WorkspaceTab { SceneTab, RegionTab };
    void refresh();
    void showImages();
    void refreshWorkspace();
    PipelineController& controller_;
    PipelineState state_;
    quint64 mask_request_id_{0};
    bool reconstruction_valid_{false}, mesh_valid_{false};
    SharedMesh mesh_result_;
    SharedGPUMesh gpu_mesh_result_;
    std::vector<SharedGPUMesh> captured_clouds_;
    bool gpu_upload_pending_{false};
    QString render_error_;
    bool busy_{false}, closing_{false}, allow_close_{false};
    QElapsedTimer elapsed_;
    QImage rectified_left_, rectified_right_, live_left_, live_right_, depth_image_;
    QPushButton *import_, *camera_, *retake_, *capture_more_, *capture_, *run_, *finish_draw_, *clear_, *build_mesh_cpu_, *build_mesh_gpu_;
    QPushButton *sam_box_, *sam_foreground_, *sam_background_, *sam_remove_, *sam_undo_, *brush_, *eraser_;
    QSpinBox* brush_size_;
    QCheckBox *capture_calibration_, *denoise_;
    QDoubleSpinBox *minimum_, *maximum_, *neighbor_distance_, *mesh_edge_, *mesh_jump_;
    QLabel *input_, *calibration_, *status_, *time_, *scene_status_, *depth_status_, *steps_;
    QProgressBar* progress_;
    std::array<QWidget*,PipelineState::kMaxCaptures> capture_segments_{};
    QPlainTextEdit* log_;
    QTabWidget* tabs_;
    QWidget *input_panel_, *depth_panel_, *geometry_panel_, *mask_panel_, *save_panel_;
    MaskEditor* mask_;
    QPushButton *save_selected_, *save_all_, *browse_save_;
    QCheckBox *save_images_, *save_calibration_, *save_mask_, *save_mesh_, *save_depth_;
    QLineEdit* save_directory_;
    QLabel* save_calibration_name_;
    MeshView* mesh_view_;
    QComboBox* mesh_mode_;
    QComboBox* camera_image_mode_;
    QComboBox* camera_mode_;
};
