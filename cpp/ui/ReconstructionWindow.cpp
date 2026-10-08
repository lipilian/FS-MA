#include "ReconstructionWindow.hpp"
#include "fs/capture/RealSenseStereoSource.hpp"
#include "widgets/MaskEditor.hpp"
#include "widgets/MeshView.hpp"
#include <QCheckBox>
#include <QButtonGroup>
#include <QCloseEvent>
#include <QComboBox>
#include <QDateTime>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFileInfo>
#include <QDir>
#include <QLineEdit>
#include <QMessageBox>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QToolButton>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QRadioButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QStandardItemModel>
#include <QSplitter>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <algorithm>
#include <stdexcept>

namespace {
QLabel* label(const QString& text) {
    auto* l = new QLabel(text); l->setWordWrap(true); l->setTextFormat(Qt::PlainText); return l;
}
QPushButton* button(const QString& text, const char* name = "") {
    auto* b = new QPushButton(text); b->setObjectName(name); b->setMinimumHeight(34); return b;
}
QGroupBox* group(const QString& title, QVBoxLayout*& layout) {
    auto* g = new QGroupBox(title); layout = new QVBoxLayout(g); return g;
}
QScrollArea* scrollPanel(QWidget* content, int minimum) {
    auto* s = new QScrollArea; s->setFrameShape(QFrame::NoFrame); s->setWidgetResizable(true);
    s->setWidget(content); s->setMinimumWidth(minimum); return s;
}
QDoubleSpinBox* decimal(double value, double minimum, double maximum, const QString& suffix) {
    auto* s = new QDoubleSpinBox; s->setDecimals(3); s->setRange(minimum, maximum); s->setValue(value); s->setSuffix(suffix); return s;
}
}
ReconstructionWindow::ReconstructionWindow(PipelineController& controller, ConfirmedCalibration calibration, const QString& path)
    : controller_(controller), state_(controller.state()), has_confirmed_calibration_(bool(calibration)) {
    if (!state_.engine_ready || !state_.sam_ready || !state_.ma_ready)
        throw std::logic_error("Reconstruction window requires loaded FoundationStereo, SAM and MapAnything sessions.");
    setObjectName("reconstructionWindow"); setWindowTitle("FoundationStereo · Reconstruction"); resize(1500, 950); setMinimumSize(1120, 740);
    if (!calibration) setWindowTitle("FoundationStereo · RealSense Scan");
    setStyleSheet(R"(
        QMainWindow, QWidget#root { background: #f3f6fa; color: #20314a; }
        QWidget { font-family: 'Noto Sans', sans-serif; font-size: 13px; }
        QGroupBox { background: white; border: 1px solid #dce3ed; border-radius: 8px; margin-top: 15px; padding: 16px 12px 12px; font-weight: 600; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
        QPushButton { background: white; color: #243c60; border: 1px solid #cbd6e5; border-radius: 5px; padding: 5px 10px; }
        QPushButton:checked { background: #dbeafe; color: #1d4ed8; border-color: #2464d9; }
        QPushButton:hover { background: #eaf1fc; border-color: #6c97cf; }
        QPushButton:disabled { color: #9aa8b9; background: #edf1f6; border-color: #dfe5ed; }
        QPushButton#run { background: #2464d9; color: white; border: 0; }
        QPushButton#run:disabled { background: #c3d1e6; }
        QDoubleSpinBox, QSpinBox, QComboBox { min-height: 28px; border: 1px solid #ced8e5; border-radius: 4px; background: white; color: #20314a; padding: 2px 5px; }
        QWidget#inputCard { background: white; border: 1px solid #dce3ed; border-radius: 8px; }
        QToolButton#inputDetailsToggle { border: 0; background: transparent; text-align: left; padding: 10px 8px; font-weight: 600; }
        QToolButton#inputDetailsToggle:hover { background: #eaf1fc; }
        QLabel#title { font-size: 27px; font-weight: 700; color: #192c49; }
        QLabel#step { font-size: 11px; font-weight: 700; color: #2464d9; }
        QLabel#status { padding: 10px; background: #e5edfa; border-radius: 5px; color: #234772; }
        QTabWidget::pane { border: 1px solid #dce3ed; background: white; }
        QTabBar::tab { padding: 10px 16px; background: #e6edf7; color: #3e5470; }
        QTabBar::tab:selected { background: white; color: #2464d9; }
        QProgressBar { border: 1px solid #b9d9c5; border-radius: 5px; background: #e8f3ec; color: #173d27; min-height: 24px; text-align: center; }
        QProgressBar::chunk { background: #22c55e; border-radius: 4px; }
        QPlainTextEdit { background: #172337; color: #d8e4f3; border: 0; border-radius: 5px; }
    )");
    auto* root = new QWidget; root->setObjectName("root"); setCentralWidget(root);
    auto* outer = new QVBoxLayout(root); outer->setContentsMargins(22,18,22,18); outer->setSpacing(10);
    auto* step = label(calibration ? "01  /  CALIBRATION  ✓     →     02  /  RECONSTRUCTION"
                                  : "MODELS READY  ✓     →     INTEL REALSENSE SCANNING");
    step->setObjectName("step"); outer->addWidget(step);
    auto* title = label(calibration ? "Reconstruction workspace" : "RealSense scan workspace"); title->setObjectName("title"); outer->addWidget(title);
    auto* toolbar = new QHBoxLayout;
    import_ = button("Import capture…", "importCapture"); camera_ = button("Connect cameras", "connectCameras");
    camera_mode_ = new QComboBox; camera_mode_->setObjectName("cameraMode");
    if (calibration) camera_mode_->addItem("Sentech stereo", int(CameraMode::Sentech));
    camera_mode_->addItem("Intel RealSense D435", int(CameraMode::RealSenseD435));
    camera_mode_->setToolTip(calibration
        ? "Sentech: confirmed calibration. D435: factory-calibrated IR pair, center-cropped from 1280 × 800 to 960 × 800. Disconnect before switching."
        : "D435: factory-calibrated IR pair, center-cropped from 1280 × 800 to 960 × 800.");
    if (!RealSenseStereoSource::available()) {
        auto* item = static_cast<QStandardItemModel*>(camera_mode_->model())->item(camera_mode_->findData(int(CameraMode::RealSenseD435)));
        item->setEnabled(false); item->setToolTip("Install librealsense2-dev and rebuild to enable D435 support.");
    }
    camera_mode_->setCurrentIndex(camera_mode_->findData(int(state_.camera_mode)));
    retake_ = button("Retake", "retake"); capture_more_ = button("Capture more", "captureMore");
    clean_ = button("Clean", "cleanCaptures");
    clean_->setToolTip("Clear all captured views and reconstruction results, then return to live preview for the first capture.");
    retake_->setToolTip("Return to preview. A single view's reconstruction is cleared; with multiple views, completed captures are retained until the replacement succeeds.");
    capture_more_->setToolTip("Keep completed captures and preview the next view. Up to five pairs.");
    capture_ = button("Capture pair", "capturePair");
    capture_->setToolTip("Capture, reconstruct and display all valid points automatically. No mask drawing required.");
    run_ = button("Reconstruct", "run"); finish_draw_ = button("Finish draw", "finishDraw");
    build_mesh_cpu_ = button("CPU mesh", "buildMeshCPU");
    build_mesh_gpu_ = button("GPU mesh", "buildMeshGPU");
    build_mesh_cpu_->setToolTip("Generate a constrained Delaunay mesh on the CPU.");
    build_mesh_gpu_->setToolTip("Build a local grid mesh on the GPU and display it through CUDA/OpenGL interoperability.");
    toolbar->addWidget(import_); toolbar->addWidget(camera_mode_);
    for (auto* b : {camera_, retake_, capture_more_, clean_}) toolbar->addWidget(b);
    toolbar->addStretch(); toolbar->addWidget(capture_); toolbar->addWidget(finish_draw_); toolbar->addWidget(run_); toolbar->addWidget(build_mesh_cpu_); toolbar->addWidget(build_mesh_gpu_); outer->addLayout(toolbar);
    auto* split = new QSplitter;
    auto* settings = new QWidget; auto* settings_layout = new QVBoxLayout(settings); settings_layout->setContentsMargins(0,0,8,0);
    QVBoxLayout* layout;
    auto* input_group = new QWidget; input_group->setObjectName("inputCard");
    input_panel_ = input_group;
    auto* input_layout = new QVBoxLayout(input_group); input_layout->setContentsMargins(4,4,4,4); input_layout->setSpacing(0);
    auto* input_toggle = new QToolButton; input_toggle->setObjectName("inputDetailsToggle");
    input_toggle->setText("Input and calibration"); input_toggle->setCheckable(true); input_toggle->setChecked(false);
    input_toggle->setToolButtonStyle(Qt::ToolButtonTextBesideIcon); input_toggle->setArrowType(Qt::RightArrow);
    input_toggle->setSizePolicy(QSizePolicy::Expanding, QSizePolicy::Fixed); input_layout->addWidget(input_toggle);
    auto* input_details = new QWidget; input_details->setObjectName("inputDetails");
    layout = new QVBoxLayout(input_details); layout->setContentsMargins(8,4,8,10);
    input_layout->addWidget(input_details); input_details->hide();
    connect(input_toggle, &QToolButton::toggled, this, [input_details, input_toggle](bool expanded) {
        input_details->setVisible(expanded); input_toggle->setArrowType(expanded ? Qt::DownArrow : Qt::RightArrow);
    });
    input_ = label("No stereo pair loaded"); input_->setTextInteractionFlags(Qt::TextSelectableByMouse); layout->addWidget(input_);
    capture_calibration_ = new QCheckBox("Use capture calibration"); capture_calibration_->setChecked(true); capture_calibration_->setObjectName("captureCalibration");
    capture_calibration_->setToolTip(calibration
        ? "The folder must contain left.png, right.png and calibration JSON (calibration.json or sentech_stereo_calibration.json; the confirmed calibration filename is also accepted). Uncheck to use the confirmed calibration with matching image dimensions; all three files are still required."
        : "Imported captures must contain left.png, right.png and their own calibration JSON.");
    layout->addWidget(capture_calibration_);
    calibration_ = label(path); calibration_->setTextInteractionFlags(Qt::TextSelectableByMouse); layout->addWidget(calibration_);
    if (calibration) layout->addWidget(label(QString("Confirmed camera calibration\nL %1 · R %2\n%3 × %4 · %5")
        .arg(QString::fromStdString(calibration->left_serial), QString::fromStdString(calibration->right_serial))
        .arg(calibration->image_size.width).arg(calibration->image_size.height)
        .arg(calibration->checked ? "Independent check performed" : "No independent check (optional)")));
    else layout->addWidget(label("Live capture uses the RealSense factory IR calibration. No manual calibration is required."));
    settings_layout->addWidget(input_group);
    auto* depth = group("Depth range", layout); auto* form = new QFormLayout;
    depth_panel_ = depth;
    minimum_ = decimal(0,0,1000," m"); maximum_ = decimal(1,0.001,1000," m"); minimum_->setObjectName("minimumDepth"); maximum_->setObjectName("maximumDepth");
    minimum_->setSingleStep(0.05); maximum_->setSingleStep(0.05);
    form->addRow("Minimum", minimum_); form->addRow("Maximum", maximum_); layout->addLayout(form);
    layout->addWidget(label("Applied to full-image XYZ.")); settings_layout->addWidget(depth);
    auto* filters = group("Geometry", layout);
    geometry_panel_ = filters;
    denoise_ = new QCheckBox("3 × 3 neighbour filtering"); denoise_->setObjectName("xyzDenoise"); denoise_->setChecked(true); layout->addWidget(denoise_);
    neighbor_distance_ = decimal(0.01,0.001,1," m"); neighbor_distance_->setObjectName("neighborDistance"); neighbor_distance_->setSingleStep(0.001);
    auto* neighbor_form=new QFormLayout; neighbor_form->addRow("Neighbour distance",neighbor_distance_); layout->addLayout(neighbor_form);
    mesh_edge_=decimal(.02,.001,1," m"); mesh_jump_=decimal(.01,.001,1," m");
    mesh_edge_->setObjectName("meshMaxEdge"); mesh_jump_->setObjectName("meshMaxDepthJump");
    mesh_edge_->setSingleStep(.001); mesh_jump_->setSingleStep(.001);
    auto* mesh_form=new QFormLayout; mesh_form->addRow("Max mesh edge",mesh_edge_); mesh_form->addRow("Max depth jump",mesh_jump_);
    layout->addLayout(mesh_form); settings_layout->addWidget(filters);
    auto* settings_panel=scrollPanel(settings,260); settings_panel->setObjectName("workspaceTools");
    split->addWidget(settings_panel);

    tabs_ = new QTabWidget; tabs_->setObjectName("workspaceTabs");
    auto* scene = new QWidget; auto* scene_layout = new QVBoxLayout(scene); auto* scene_controls = new QHBoxLayout;
    auto* cloud_controls=new QHBoxLayout;
    fs_cloud_=new QRadioButton("FS point cloud"); fs_cloud_->setObjectName("fsPointCloud");
    ma_cloud_=new QRadioButton("MA point cloud"); ma_cloud_->setObjectName("maPointCloud");
    auto* cloud_source=new QButtonGroup(this); cloud_source->setExclusive(true);
    cloud_source->addButton(fs_cloud_); cloud_source->addButton(ma_cloud_);
    fs_cloud_->setChecked(true);
    cloud_controls->addWidget(fs_cloud_); cloud_controls->addWidget(ma_cloud_); cloud_controls->addStretch();
    scene_layout->addLayout(cloud_controls);
    camera_image_mode_ = new QComboBox; camera_image_mode_->setObjectName("cameraImageMode");
    camera_image_mode_->addItems({"Left image", "Depth map"});
    camera_image_mode_->setToolTip("Display the rectified left image or Jet depth map on the camera image plane.");
    mesh_mode_=new QComboBox; mesh_mode_->addItems({"Point cloud","Mesh","Wireframe"}); mesh_mode_->setCurrentIndex(0); mesh_mode_->setObjectName("meshMode");
    auto* reset=button("Reset view","resetMeshView");
    auto* show_right_camera=new QCheckBox("Show right camera"); show_right_camera->setObjectName("showRightCamera");
    show_right_camera->setToolTip("Show only the rectified right image beside the left camera, with the same image-plane size.");
    scene_controls->addWidget(camera_image_mode_); scene_controls->addWidget(show_right_camera);
    scene_controls->addWidget(mesh_mode_); scene_controls->addWidget(reset); scene_controls->addStretch();
    scene_layout->addLayout(scene_controls);
    mesh_view_=new MeshView; mesh_view_->setObjectName("meshView"); scene_layout->addWidget(mesh_view_,1);
    mesh_view_->setMode(0);
    scene_status_=label("Connect cameras to see the left live image in 3D, or import a capture.");
    scene_status_->setObjectName("meshStatus"); scene_layout->addWidget(scene_status_);
    depth_status_ = label(""); depth_status_->setObjectName("depthRange"); scene_layout->addWidget(depth_status_);
    scene_layout->addWidget(label("Left drag: rotate · Right drag: pan · Wheel: zoom"));
    tabs_->addTab(scene,"3D workspace");
    tabs_->setTabToolTip(SceneTab,"Live images, depth map and mesh in one 3D scene.");
    connect(camera_image_mode_,&QComboBox::currentIndexChanged,this,[this] { showImages(); });
    connect(mesh_mode_,&QComboBox::currentIndexChanged,this,[this](int mode) { mesh_view_->setMode(mode); });
    connect(reset,&QPushButton::clicked,this,[this] { mesh_view_->resetView(); });
    connect(show_right_camera,&QCheckBox::toggled,this,[this](bool visible) { mesh_view_->setRightCameraVisible(visible); });
    connect(ma_cloud_,&QRadioButton::toggled,this,[this] { refresh(); });
    auto* region = new QWidget; auto* region_layout = new QVBoxLayout(region);
    mask_ = new MaskEditor; mask_->setObjectName("maskEditor"); region_layout->addWidget(mask_,1);
    auto* overlay = new QCheckBox("Show selection overlay"); overlay->setChecked(true); region_layout->addWidget(overlay);
    region_layout->addWidget(label("The mask limits neighbourhood filtering; XYZ outside it is preserved."));
    tabs_->addTab(region,"Region measurement");
    split->addWidget(tabs_);

    auto* sam = group("Mask draw", layout);
    mask_panel_ = sam;
    sam_box_=button("Draw box","samBox"); sam_foreground_=button("Foreground point (+)","samForeground");
    sam_background_=button("Background point (−)","samBackground"); sam_remove_=button("Remove prompt","samRemove");
    sam_undo_=button("Undo prompt","samUndo");
    brush_=button("Brush (+)","maskBrush"); eraser_=button("Eraser (−)","maskEraser");
    clear_=button("Clear mask","clearMask");
    brush_size_=new QSpinBox; brush_size_->setObjectName("maskBrushSize");
    brush_size_->setRange(1,100); brush_size_->setValue(mask_->brushSize()); brush_size_->setSuffix(" px");
    brush_size_->setToolTip("Brush width in full-resolution rectified-left pixels. Shift+wheel over the image adjusts the size.");
    auto* sam_tools = new QButtonGroup(this); sam_tools->setExclusive(true);
    for (auto* b : {sam_box_,sam_foreground_,sam_background_,sam_remove_,brush_,eraser_}) { b->setCheckable(true); sam_tools->addButton(b); }
    sam_box_->setChecked(true);
    auto* auto_draw = label("SAM 2.1 auto draw");
    auto_draw->setStyleSheet("font-weight: 600; color: #526680;"); layout->addWidget(auto_draw);
    for (auto* b : {sam_box_,sam_foreground_,sam_background_,sam_remove_,sam_undo_}) layout->addWidget(b);
    layout->addSpacing(8);
    auto* manual_draw = label("Manual draw");
    manual_draw->setStyleSheet("font-weight: 600; color: #526680;"); layout->addWidget(manual_draw);
    auto* brushes=new QHBoxLayout; brushes->addWidget(brush_); brushes->addWidget(eraser_); layout->addLayout(brushes);
    auto* brush_form=new QFormLayout; brush_form->addRow("Brush size",brush_size_); layout->addLayout(brush_form);
    layout->addWidget(clear_);
    layout->addWidget(label("Wheel: zoom at cursor (1–5×)\nShift+wheel: brush size"));
    sam->setObjectName("maskDrawPanel"); depth->setObjectName("depthSettings"); filters->setObjectName("geometrySettings");
    settings_layout->addWidget(sam);
    auto* save_panel=group("Save results",layout); save_panel->setObjectName("saveResultsPanel");
    save_panel_ = save_panel;
    save_images_=new QCheckBox("Raw left / right images"); save_images_->setObjectName("saveRawImages");
    save_calibration_=new QCheckBox("Calibration JSON"); save_calibration_->setObjectName("saveCalibration");
    save_mask_=new QCheckBox("Mask (mask.png)"); save_mask_->setObjectName("saveMask");
    save_mask_->setToolTip("Full-resolution confirmed mask in rectified-left image coordinates.");
    save_depth_=new QCheckBox("Depth (depth.tiff)"); save_depth_->setObjectName("saveDepth");
    save_depth_->setToolTip("Filtered Z depth in metres, single-channel float32 TIFF. Invalid pixels are zero. Aligned with the rectified left image resized to the depth resolution.");
    save_mesh_=new QCheckBox("Mesh (mesh.ply)"); save_mesh_->setObjectName("saveMesh");
    save_mesh_->setToolTip("Generate the current mesh before saving it.");
    for (auto* option:{save_images_,save_calibration_,save_mask_,save_mesh_,save_depth_}) option->setChecked(true);
    save_mask_->setChecked(false);
    layout->addWidget(save_images_); layout->addWidget(label("left.png · right.png"));
    layout->addWidget(save_calibration_); save_calibration_name_=label(state_.calibration_filename); layout->addWidget(save_calibration_name_);
    layout->addWidget(save_mask_); layout->addWidget(save_depth_); layout->addWidget(save_mesh_);
    save_directory_=new QLineEdit; save_directory_->setObjectName("saveDirectory"); save_directory_->setReadOnly(true); save_directory_->setPlaceholderText("Choose output folder");
    layout->addWidget(save_directory_); browse_save_=button("Browse…","browseSaveDirectory"); layout->addWidget(browse_save_);
    save_selected_=button("Save selected","saveSelected"); layout->addWidget(save_selected_);
    save_all_=button("Save all","saveAll"); layout->addWidget(save_all_);
    settings_layout->addWidget(save_panel); settings_layout->addStretch();
    split->setSizes({290,1170}); split->setStretchFactor(1,1); outer->addWidget(split,1);
    steps_ = label(""); outer->addWidget(steps_);
    auto* footer = new QHBoxLayout; status_ = label(""); status_->setObjectName("status"); footer->addWidget(status_,1);
    time_ = label("Idle"); footer->addWidget(time_); auto* show_log = button("Show log"); show_log->setCheckable(true); footer->addWidget(show_log); outer->addLayout(footer);
    log_ = new QPlainTextEdit; log_->setReadOnly(true); log_->setMaximumBlockCount(500); log_->setMaximumHeight(130); log_->hide(); outer->addWidget(log_);
    auto* capture_slots = new QWidget; capture_slots->setObjectName("captureSlots"); capture_slots->setFixedHeight(4);
    auto* slots_layout = new QHBoxLayout(capture_slots); slots_layout->setContentsMargins(0,0,0,0); slots_layout->setSpacing(6);
    for (int i=0; i<PipelineState::kMaxCaptures; ++i) {
        auto* segment = new QWidget; segment->setObjectName(QString("captureSegment%1").arg(i+1));
        segment->setFixedHeight(4); segment->setProperty("captured",false);
        segment->setAccessibleName(QString("Capture %1").arg(i+1));
        segment->setStyleSheet("background: #dce3ed; border-radius: 2px;");
        capture_segments_[i]=segment; slots_layout->addWidget(segment,1);
    }
    outer->addWidget(capture_slots);
    progress_ = new QProgressBar; progress_->setObjectName("calculationProgress");
    progress_->setRange(0,100); progress_->setValue(0); progress_->setTextVisible(true);
    progress_->setToolTip("Progress updates when each calculation stage completes."); outer->addWidget(progress_);
    connect(show_log,&QPushButton::toggled,log_,&QWidget::setVisible);
    connect(show_log,&QPushButton::toggled,this,[show_log](bool shown) { show_log->setText(shown ? "Hide log" : "Show log"); });
    connect(&controller_,&PipelineController::log,this,[this](const QString& message) { log_->appendPlainText(QDateTime::currentDateTime().toString("hh:mm:ss") + "  " + message); });
    connect(&controller_,&PipelineController::stateChanged,this,[this](PipelineState state) {
        const bool just_connected = state.connected && !state_.connected;
        const bool clean_preview = state.live && !state.has_rectified && state.image_id!=state_.image_id;
        if (state.live || state.image_id != state_.image_id) reconstruction_valid_ = false;
        if (state.capture_count==0) captured_clouds_.clear();
        state_ = std::move(state);
        if (clean_preview) {
            capture_processing_=false; mesh_view_->setProcessing(false);
            mesh_valid_=false; gpu_upload_pending_=false;
            mesh_result_.reset(); gpu_mesh_result_.reset(); mesh_view_->setMesh({});
            rectified_left_={}; rectified_right_={}; live_left_={}; live_right_={}; depth_image_={};
            render_error_.clear();
            mask_request_id_=controller_.requestMask(state_.image_id,{});
            const QSignalBlocker blocker(mask_);
            mask_->setImage({});
            tabs_->setCurrentIndex(SceneTab);
        }
        refresh();
        if (just_connected || clean_preview) mesh_view_->resetCaptureView();
    });
    connect(&controller_,&PipelineController::busyChanged,this,[this](bool busy) {
        busy_ = busy;
        if (busy) { render_error_.clear(); elapsed_.start(); }
        else {
            time_->setText(QString("Last task: %1 s").arg(elapsed_.elapsed()/1000.0,0,'f',1));
            if (capture_processing_ && (state_.action_failed || !gpu_upload_pending_)) {
                capture_processing_=false; mesh_view_->setProcessing(false);
            }
        }
        refresh();
    });
    connect(&controller_,&PipelineController::images,this,[this](QImage rl,QImage rr) {
        rectified_left_ = std::move(rl); rectified_right_ = std::move(rr);
        reconstruction_valid_=false; mask_->setImage(rectified_left_); save_mask_->setChecked(false); tabs_->setCurrentIndex(SceneTab);
    });
    connect(&controller_,&PipelineController::maskReady,this,[this](quint64 image_id,quint64 request_id,QImage mask,QString message) {
        if (closing_ || image_id!=state_.image_id || request_id!=mask_request_id_) return;
        mask_->setPrediction(std::move(mask)); status_->setText(message);
    });
    connect(&controller_,&PipelineController::meshReady,this,[this](SharedMesh mesh) {
        if (closing_ || !reconstruction_valid_) return;
        gpu_mesh_result_.reset(); gpu_upload_pending_=false;
        mesh_result_=std::move(mesh); mesh_valid_=true;
        mesh_mode_->setCurrentIndex(mesh_result_->triangles.empty() ? 0 : 1);
        mesh_view_->setMesh(mesh_result_); refresh(); tabs_->setCurrentIndex(SceneTab);
    });
    connect(&controller_,&PipelineController::gpuMeshReady,this,[this](SharedGPUMesh mesh) {
        if (closing_ || !reconstruction_valid_) return;
        mesh_result_.reset(); mesh_valid_=false;
        gpu_mesh_result_=std::move(mesh); gpu_upload_pending_=true;
        if (gpu_mesh_result_->point_cloud && state_.capture_count>0) {
            captured_clouds_.resize(state_.capture_count);
            captured_clouds_.back()=gpu_mesh_result_;
        }
        mesh_mode_->setCurrentIndex(gpu_mesh_result_->point_cloud ? 0 : 1);
        tabs_->setCurrentIndex(SceneTab);
        mesh_view_->setGPUMesh(gpu_mesh_result_); refresh();
    });
    connect(mesh_view_,&MeshView::gpuMeshPresented,this,[this](SharedGPUMesh mesh) {
        if (closing_ || !reconstruction_valid_ || mesh!=gpu_mesh_result_) return;
        capture_processing_=false; mesh_view_->setProcessing(false);
        gpu_upload_pending_=false; mesh_valid_=true; refresh();
    },Qt::QueuedConnection); // Updating the progress bar may repaint; wait until paintGL returns.
    connect(mesh_view_,&MeshView::renderFailed,this,[this](const QString& message) {
        capture_processing_=false; mesh_view_->setProcessing(false);
        render_error_=message; gpu_upload_pending_=false; mesh_valid_=false; refresh();
        scene_status_->setText(message); status_->setText(message);
        log_->appendPlainText(QDateTime::currentDateTime().toString("hh:mm:ss")+"  "+message);
    },Qt::QueuedConnection);
    connect(mesh_view_,&MeshView::log,this,[this](const QString& message) {
        log_->appendPlainText(QDateTime::currentDateTime().toString("hh:mm:ss")+"  "+message);
    });
    connect(&controller_,&PipelineController::depthImage,this,[this](QImage depth,float minimum,float maximum) {
        if (closing_) return;
        reconstruction_valid_=true;
        depth_image_ = std::move(depth);
        depth_status_->setText(QString("Jet: %1 m (blue) → %2 m (red) · Black: invalid, out of range or filtered")
            .arg(minimum,0,'f',3).arg(maximum,0,'f',3));
        refresh();
        tabs_->setCurrentIndex(SceneTab);
    });
    connect(&controller_,&PipelineController::preview,this,[this](QImage l,QImage r) {
        if (!state_.live) return;
        live_left_ = std::move(l); live_right_ = std::move(r);
        showImages();
    });
    connect(import_,&QPushButton::clicked,this,[this] {
        const QString path = QFileDialog::getExistingDirectory(this,"Choose capture directory (left.png, right.png, calibration JSON)");
        if (path.isEmpty()) return;
        const bool own_calibration = capture_calibration_->isChecked(); controller_.submit([=](auto& w) { w.importCapture(path,own_calibration); });
    });
    connect(camera_,&QPushButton::clicked,this,[this] {
        const bool connected = state_.connected;
        const auto mode = static_cast<CameraMode>(camera_mode_->currentData().toInt());
        live_left_ = {}; live_right_ = {};
        controller_.submit([=](auto& w) { if (connected) w.disconnectCameras(); else w.connectCameras(mode); });
    });
    const auto start_capture_preview=[this](bool append) {
        live_left_={}; live_right_={}; tabs_->setCurrentIndex(SceneTab);
        controller_.submit([append](auto& worker) { worker.startCapturePreview(append); });
    };
    connect(retake_,&QPushButton::clicked,this,[start_capture_preview] { start_capture_preview(false); });
    connect(capture_more_,&QPushButton::clicked,this,[start_capture_preview] { start_capture_preview(true); });
    connect(clean_,&QPushButton::clicked,this,[this] {
        mask_request_id_=controller_.requestMask(state_.image_id,{});
        controller_.submit([](auto& worker) { worker.cleanCaptures(); });
    });
    connect(capture_,&QPushButton::clicked,this,[this] {
        mesh_view_->preserveView();
        capture_processing_=true; mesh_view_->setProcessing(true);
        reconstruction_valid_=false; refresh();
        const float low=minimum_->value(), high=maximum_->value(), distance=neighbor_distance_->value();
        const bool denoise=denoise_->isChecked();
        controller_.submit([=](auto& w) { w.captureAndReconstruct(low,high,denoise,distance); });
    });
    connect(run_,&QPushButton::clicked,this,[this] {
        reconstruction_valid_=false; refresh();
        const float low = minimum_->value(), high = maximum_->value();
        const QImage selected=mask_->mask();
        const bool denoise=denoise_->isChecked(); const float distance=neighbor_distance_->value();
        controller_.submit([=](auto& w) { w.reconstruct(low,high,selected,denoise,distance); });
    });
    connect(browse_save_,&QPushButton::clicked,this,[this] {
        const auto directory=QFileDialog::getExistingDirectory(this,"Choose output folder",save_directory_->text());
        if (!directory.isEmpty()) save_directory_->setText(directory);
    });
    connect(save_directory_,&QLineEdit::textChanged,this,[this] { refresh(); });
    for (auto* option:{save_images_,save_calibration_,save_mask_,save_mesh_,save_depth_})
        connect(option,&QCheckBox::toggled,this,[this] { refresh(); });
    const auto save_results=[this](ReconstructionSaveOptions options) {
        const QString directory=save_directory_->text();
        const QDir dir(directory);
        QStringList names;
        if (options.images) names << "left.png" << "right.png";
        if (options.calibration) names << state_.calibration_filename;
        if (options.mask) names << "mask.png";
        if (options.depth) names << "depth.tiff";
        if (options.mesh) names << "mesh.ply";
        QStringList existing;
        for (const auto& name:names) if (QFileInfo::exists(dir.filePath(name)) || QFileInfo(dir.filePath(name)).isSymLink()) existing << name;
        const bool overwrite=!existing.isEmpty();
        if (overwrite && QMessageBox::question(this,"Replace existing files?",
            "These files already exist in the chosen folder:\n"+existing.join("\n")+"\n\nReplace them?",
            QMessageBox::Yes|QMessageBox::No,QMessageBox::No)!=QMessageBox::Yes) return;
        const QImage selection=mask_->mask();
        const SharedMesh mesh=mesh_valid_ ? mesh_result_ : SharedMesh{};
        const SharedGPUMesh gpu_mesh=mesh_valid_ ? gpu_mesh_result_ : SharedGPUMesh{};
        controller_.submit([=](auto& worker) { worker.saveResults(directory,options,selection,mesh,overwrite,gpu_mesh); });
    };
    connect(save_selected_,&QPushButton::clicked,this,[this,save_results] {
        save_results({save_images_->isChecked(),save_calibration_->isChecked(),save_mask_->isChecked(),save_mesh_->isChecked(),save_depth_->isChecked()});
    });
    connect(save_all_,&QPushButton::clicked,this,[this,save_results] {
        save_results({true,true,mask_->hasSelection(),true,true});
    });
    connect(build_mesh_cpu_,&QPushButton::clicked,this,[this] {
        gpu_upload_pending_=false; mesh_valid_=false; refresh();
        const double edge=mesh_edge_->value(), jump=mesh_jump_->value();
        controller_.submit([=](auto& worker) { worker.buildMeshCPU(edge,jump); });
    });
    connect(build_mesh_gpu_,&QPushButton::clicked,this,[this] {
        gpu_upload_pending_=false; mesh_valid_=false; refresh();
        const QImage selected=mask_->mask();
        const double edge=mesh_edge_->value(), jump=mesh_jump_->value();
        controller_.submit([=](auto& worker) { worker.buildMeshGPU(selected,edge,jump); });
    });
    const auto mesh_settings_changed=[this] {
        // Triangle constraints do not change a point-only result.
        const bool point_cloud=(mesh_result_ && mesh_result_->triangles.empty()) ||
                               (gpu_mesh_result_ && gpu_mesh_result_->point_cloud);
        if (!point_cloud) { gpu_upload_pending_=false; mesh_valid_=false; }
        refresh();
    };
    connect(mesh_edge_,&QDoubleSpinBox::valueChanged,this,mesh_settings_changed);
    connect(mesh_jump_,&QDoubleSpinBox::valueChanged,this,mesh_settings_changed);
    const auto depth_settings_changed = [this] {
        // Changing parameters invalidates derived results even if the old values are restored later.
        reconstruction_valid_=false;
        refresh();
    };
    connect(minimum_,&QDoubleSpinBox::valueChanged,this,depth_settings_changed);
    connect(maximum_,&QDoubleSpinBox::valueChanged,this,depth_settings_changed);
    connect(denoise_,&QCheckBox::toggled,this,depth_settings_changed);
    connect(neighbor_distance_,&QDoubleSpinBox::valueChanged,this,depth_settings_changed);
    connect(clear_,&QPushButton::clicked,mask_,&MaskEditor::clearMask);
    connect(overlay,&QCheckBox::toggled,mask_,&MaskEditor::setOverlayVisible);
    connect(brush_size_,&QSpinBox::valueChanged,mask_,&MaskEditor::setBrushSize);
    connect(mask_,&MaskEditor::brushSizeChanged,brush_size_,&QSpinBox::setValue);
    const auto tool = [this,overlay,region](QPushButton* button, MaskEditor::Tool tool) {
        connect(button,&QPushButton::clicked,this,[this,overlay,region,tool] {
            tabs_->setCurrentWidget(region); overlay->setChecked(true); mask_->setTool(tool);
        });
    };
    tool(sam_box_,MaskEditor::Tool::Box); tool(sam_foreground_,MaskEditor::Tool::Foreground);
    tool(sam_background_,MaskEditor::Tool::Background); tool(sam_remove_,MaskEditor::Tool::Remove);
    tool(brush_,MaskEditor::Tool::Brush); tool(eraser_,MaskEditor::Tool::Eraser);
    connect(sam_undo_,&QPushButton::clicked,mask_,&MaskEditor::undoPrompt);
    connect(finish_draw_,&QPushButton::clicked,this,[this] {
        mask_->acceptPrediction();
        if (mask_->hasSelection()) tabs_->setCurrentIndex(SceneTab);
    });
    connect(mask_,&MaskEditor::promptsChanged,this,[this] {
        const auto prompts=mask_->prompts(); mask_request_id_=controller_.requestMask(state_.image_id,prompts);
        if (!prompts.empty()) status_->setText("Updating SAM mask…");
    });
    connect(mask_,&MaskEditor::selectionChanged,this,[this] { reconstruction_valid_=false; refresh(); }); connect(mask_,&MaskEditor::hint,status_,&QLabel::setText);
    auto* timer = new QTimer(this); timer->setInterval(200); connect(timer,&QTimer::timeout,this,[this] { if (busy_) time_->setText(QString("Running: %1 s").arg(elapsed_.elapsed()/1000.0,0,'f',1)); }); timer->start();
    connect(tabs_,&QTabWidget::currentChanged,this,&ReconstructionWindow::refreshWorkspace);
    refresh();
}
void ReconstructionWindow::refresh() {
    if (!reconstruction_valid_ || !state_.depth_ready) { mesh_valid_=false; gpu_upload_pending_=false; }
    // Result validity controls processing/export. Completed scene geometry has
    // its own lifetime and survives preview, new inference and failed retakes.
    // Update it before controls that can cause a synchronous OpenGL repaint.
    if (closing_) captured_clouds_.clear();
    mesh_view_->setPredictedCameras(closing_ ? SharedPredictedCameras{} : state_.predicted_cameras);
    mesh_view_->setMAClouds(closing_ ? SharedGPUClouds{} : state_.ma_clouds);
    const bool ma_available=!closing_ && state_.ma_clouds && !state_.ma_clouds->empty();
    if (!ma_available && ma_cloud_->isChecked()) {
        const QSignalBlocker fs_blocker(fs_cloud_), ma_blocker(ma_cloud_);
        fs_cloud_->setChecked(true);
    }
    const bool show_ma=ma_cloud_->isChecked();
    fs_cloud_->setEnabled(!closing_); ma_cloud_->setEnabled(ma_available);
    ma_cloud_->setToolTip(ma_available ? "Display MapAnything's predicted point clouds."
        : "Available after MapAnything finishes inference on at least two captures.");
    mesh_view_->setShowMAClouds(show_ma);
    const bool replacement=(mesh_valid_ || gpu_upload_pending_) &&
        (mesh_result_ || (gpu_mesh_result_ && !gpu_mesh_result_->point_cloud));
    mesh_view_->setCapturedClouds(captured_clouds_,replacement && !captured_clouds_.empty()
        ? captured_clouds_.back() : SharedGPUMesh{});
    const bool idle = !busy_ && !closing_ && !gpu_upload_pending_, frozen = state_.has_rectified && !state_.live;
    import_->setEnabled(idle); capture_calibration_->setEnabled(idle && has_confirmed_calibration_); camera_->setEnabled(idle);
    camera_->setText(state_.connected ? "Disconnect cameras" : "Connect cameras");
    camera_mode_->setEnabled(idle && !state_.connected);
    if (state_.connected) {
        const QSignalBlocker blocker(camera_mode_);
        camera_mode_->setCurrentIndex(camera_mode_->findData(int(state_.camera_mode)));
    }
    const bool depth_valid = minimum_->value() < maximum_->value();
    retake_->setEnabled(idle && state_.connected && !state_.live);
    clean_->setEnabled(idle && state_.connected && (state_.has_rectified || state_.capture_count>0));
    capture_more_->setEnabled(idle && state_.connected && frozen && reconstruction_valid_ && mesh_valid_ &&
                              state_.capture_count>0 && state_.capture_count<PipelineState::kMaxCaptures);
    capture_->setEnabled(idle && state_.live && state_.engine_ready && depth_valid);
    minimum_->setEnabled(idle); maximum_->setEnabled(idle);
    mesh_edge_->setEnabled(idle); mesh_jump_->setEnabled(idle);
    build_mesh_cpu_->setEnabled(idle && frozen && reconstruction_valid_ && state_.depth_ready);
    build_mesh_gpu_->setEnabled(build_mesh_cpu_->isEnabled() && state_.gpu_ready && state_.engine_ready);
    denoise_->setEnabled(idle); neighbor_distance_->setEnabled(idle && denoise_->isChecked());
    run_->setEnabled(idle && frozen && state_.engine_ready && depth_valid);
    run_->setToolTip(!state_.engine_ready ? "FoundationStereo is not initialized." : !depth_valid ? "Minimum depth must be less than maximum." : !frozen ? "Import or capture a frozen stereo pair first." : "Reconstruct depth and display valid points. Without a confirmed mask, process the full image.");
    mask_->setEditingEnabled(idle && frozen);
    for (auto* b : {brush_,eraser_}) b->setEnabled(idle && frozen);
    brush_size_->setEnabled(idle && frozen);
    for (auto* b : {sam_box_,sam_foreground_,sam_background_,sam_remove_}) b->setEnabled(idle && frozen && state_.sam_ready);
    sam_undo_->setEnabled(idle && frozen && mask_->hasPrompts());
    finish_draw_->setEnabled(idle && frozen && mask_->hasPrediction());
    clear_->setEnabled(idle && (mask_->hasPrompts() || mask_->hasPrediction()));
    input_->setText(state_.input); calibration_->setText(state_.calibration);
    if (!closing_) status_->setText(!render_error_.isEmpty() ? render_error_ : !busy_ && !state_.action_failed && !state_.live && state_.depth_ready && !reconstruction_valid_
        ? "Selection or post-processing settings changed. Reconstruct again to update the point cloud." : state_.status);
    if (!state_.depth_ready || !reconstruction_valid_) {
        depth_image_ = {};
        depth_status_->setText("Jet uses the Minimum and Maximum depth settings for each reconstruction.");
    }
    const bool result_current=reconstruction_valid_ && state_.depth_ready;
    steps_->setText(QString("%1 Rectification   →   %2 FS inference   →   %3 GPU XYZ / filtering   →   %4 Depth map   →   %5 Valid point cloud")
        .arg(state_.has_rectified ? "✓" : "○", result_current ? "✓" : "○", result_current ? "✓" : "○", result_current ? "✓" : "○", mesh_valid_ ? "✓" : "○"));
    refreshWorkspace();
    const int progress=!render_error_.isEmpty() ? std::min(state_.progress,95) : gpu_upload_pending_ ? 97 : state_.progress;
    const QString progress_stage=!render_error_.isEmpty() ? "Display failed · " + render_error_
        : gpu_upload_pending_ ? "Transferring GPU geometry to OpenGL…" : state_.progress_stage;
    progress_->setValue(progress);
    progress_->setFormat(QString::number(progress) + "% · " + progress_stage);
    for (int i=0; i<PipelineState::kMaxCaptures; ++i) {
        const bool captured=i<state_.capture_count;
        auto* segment=capture_segments_[i];
        if (segment->property("captured").toBool()!=captured) {
            segment->setProperty("captured",captured);
            segment->setStyleSheet(captured ? "background: #8b5cf6; border-radius: 2px;"
                                           : "background: #dce3ed; border-radius: 2px;");
        }
    }
    if (!mesh_valid_ && !gpu_upload_pending_ && (mesh_result_ || gpu_mesh_result_)) {
        mesh_result_.reset(); gpu_mesh_result_.reset(); mesh_view_->setMesh({});
    }
    const bool point_cloud=(mesh_valid_ || gpu_upload_pending_) && ((mesh_result_ && mesh_result_->triangles.empty()) ||
                                                                  (gpu_mesh_result_ && gpu_mesh_result_->point_cloud));
    save_mesh_->setText(point_cloud ? "FS point cloud (mesh.ply)" : "FS mesh (mesh.ply)");
    save_mesh_->setToolTip("Save the current FS points or mesh as PLY; the FS / MA selector controls the 3D display.");
    mesh_mode_->setEnabled(mesh_valid_ && !show_ma);
    for (int mode : {1,2}) static_cast<QStandardItemModel*>(mesh_mode_->model())->item(mode)->setEnabled(!point_cloud);
    browse_save_->setEnabled(idle);
    for (auto* option:{save_images_,save_calibration_,save_mask_,save_mesh_,save_depth_}) option->setEnabled(idle);
    if (!mask_->hasSelection()) {
        const QSignalBlocker blocker(save_mask_);
        save_mask_->setChecked(false);
    }
    save_mask_->setEnabled(idle && mask_->hasSelection());
    save_calibration_name_->setText(state_.calibration_filename);
    const bool any_save=save_images_->isChecked() || save_calibration_->isChecked() || save_mask_->isChecked() || save_mesh_->isChecked() || save_depth_->isChecked();
    const bool can_save=any_save && frozen && (!save_mask_->isChecked() || mask_->hasSelection()) &&
        (!save_depth_->isChecked() || result_current) &&
        (!save_mesh_->isChecked() || (mesh_valid_ && (mesh_result_ || gpu_mesh_result_)));
    save_selected_->setEnabled(idle && can_save && !save_directory_->text().isEmpty());
    save_all_->setEnabled(idle && frozen && result_current && mesh_valid_ && (mesh_result_ || gpu_mesh_result_) && !save_directory_->text().isEmpty());
    scene_status_->setText(capture_processing_ ? "Processing · Reconstructing the captured point cloud…"
        : point_cloud ? QString("%1 point cloud · %2 valid points%3")
        .arg(gpu_mesh_result_ ? "GPU" : "RGB")
        .arg(gpu_mesh_result_ ? gpu_mesh_result_->stats.point_count : mesh_result_->vertices.size())
        .arg(gpu_upload_pending_ ? " · preparing OpenGL" : "")
        : gpu_mesh_result_ ? QString("GPU mesh · %1 triangles · %2 cm²%3")
        .arg(gpu_mesh_result_->stats.triangle_count).arg(gpu_mesh_result_->stats.area_m2*1e4,0,'f',2)
        .arg(gpu_upload_pending_ ? " · preparing OpenGL" : "")
        : mesh_valid_ && mesh_result_ ? QString("%1 vertices · %2 triangles · %3 cm²")
        .arg(mesh_result_->vertices.size()).arg(mesh_result_->triangles.size()).arg(mesh_result_->area_m2*1e4,0,'f',2)
        : state_.live ? "LIVE · Pixel point cloud at 40 cm · Capture a pair to reconstruct depth."
        : result_current ? "Depth ready. Switch the camera image or generate a CPU / GPU mesh."
        : frozen ? "Captured left image · Reconstruct to display valid points. Mask drawing is optional."
        : "Connect cameras to see the left live image in 3D, or import a capture.");
    if (show_ma) {
        unsigned long long points=0;
        for (const auto& cloud:*state_.ma_clouds) points+=cloud->stats.point_count;
        scene_status_->setText(QString("MA point cloud · %1 valid points · %2 views")
            .arg(points).arg(state_.ma_clouds->size()));
    } else if (!captured_clouds_.empty())
        scene_status_->setText(scene_status_->text()+QString(" · %1 completed %2 visible")
            .arg(captured_clouds_.size()).arg(captured_clouds_.size()==1 ? "capture" : "captures"));
    showImages();
}
void ReconstructionWindow::refreshWorkspace() {
    const bool frozen=state_.has_rectified && !state_.live;
    const bool depth_ready=frozen && state_.depth_ready && reconstruction_valid_;
    if (!frozen && tabs_->currentIndex()==RegionTab) tabs_->setCurrentIndex(SceneTab);
    tabs_->setTabEnabled(RegionTab,frozen && !busy_);
    tabs_->setTabToolTip(RegionTab,frozen ? "Optional: draw and confirm a mask for region measurement." : "Capture or import a stereo pair first.");

    const bool scene=tabs_->currentIndex()==SceneTab;
    input_panel_->setVisible(scene);
    mask_panel_->setVisible(!scene);
    depth_panel_->setVisible(scene); geometry_panel_->setVisible(scene); save_panel_->setVisible(scene && frozen);
    for (auto* b : {import_,camera_,capture_,clean_}) b->setVisible(scene);
    retake_->setVisible(scene && !state_.live);
    capture_more_->setVisible(scene && frozen);
    camera_mode_->setVisible(scene);
    finish_draw_->setVisible(!scene);
    run_->setVisible(scene && frozen);
    build_mesh_cpu_->setVisible(scene && depth_ready);
    build_mesh_gpu_->setVisible(scene && depth_ready);
}
void ReconstructionWindow::showImages() {
    const bool depth_available = !state_.live && reconstruction_valid_ && state_.depth_ready && !depth_image_.isNull();
    static_cast<QStandardItemModel*>(camera_image_mode_->model())->item(1)->setEnabled(depth_available);
    if (!depth_available && camera_image_mode_->currentIndex()==1) {
        const QSignalBlocker blocker(camera_image_mode_);
        camera_image_mode_->setCurrentIndex(0);
    }
    const bool show_depth = depth_available && camera_image_mode_->currentIndex()==1;
    const auto camera = state_.live || !state_.has_rectified ? state_.live_camera : state_.image_camera;
    mesh_view_->setCamera(camera);
    mesh_view_->setLivePreview(state_.live);
    mesh_view_->setCameraImage(show_depth ? depth_image_ : state_.live ? live_left_ : rectified_left_);
    mesh_view_->setRightCameraImage(state_.live ? live_right_ : rectified_right_);
    depth_status_->setVisible(show_depth);
}
void ReconstructionWindow::closeEvent(QCloseEvent* event) {
    if (allow_close_) { event->accept(); return; }
    event->ignore(); if (closing_) return;
    capture_processing_=false; mesh_view_->setProcessing(false);
    closing_ = true; gpu_upload_pending_=false; mesh_valid_=false; mask_request_id_=controller_.requestMask(state_.image_id,{}); refresh(); status_->setText("Closing · waiting for background work and releasing cameras / GPU…"); emit closeRequested();
}
void ReconstructionWindow::allowClose() { allow_close_ = true; close(); }
