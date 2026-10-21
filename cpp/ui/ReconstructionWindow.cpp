#include "ReconstructionWindow.hpp"
#include "ScanExport.hpp"
#include "fs/capture/RealSenseStereoSource.hpp"
#include "widgets/MaskEditor.hpp"
#include "widgets/MeshView.hpp"
#include "widgets/StereoImageView.hpp"
#include "widgets/CaptureTransition.hpp"
#include <QCheckBox>
#include <QButtonGroup>
#include <QCloseEvent>
#include <QComboBox>
#include <QCoreApplication>
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
#include <QStackedWidget>
#include <QTabWidget>
#include <QTimer>
#include <QVBoxLayout>
#include <algorithm>
#include <stdexcept>

namespace {
QString defaultImportDirectory() {
    const QString relative = "data/scan_data";
    for (const auto& base : {QDir::currentPath(), QCoreApplication::applicationDirPath() + "/../.."}) {
        const QFileInfo candidate(QDir(base).filePath(relative));
        if (candidate.isDir()) return candidate.canonicalFilePath();
    }
    return QDir::current().absoluteFilePath(relative);
}
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
        QPushButton#restartCaptures { background: #dc2626; color: white; border-color: #dc2626; }
        QPushButton#restartCaptures:hover:enabled { background: #b91c1c; border-color: #b91c1c; }
        QPushButton#restartCaptures:pressed:enabled { background: #991b1b; border-color: #991b1b; }
        QPushButton#restartCaptures:disabled { background: #fee2e2; color: #b97979; border-color: #fecaca; }
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
    capture_toolbar_=toolbar;
    import_ = button("Import folder…", "importCapture"); camera_ = button("Connect cameras", "connectCameras");
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
    retake_ = button("Retake", "retake");
    restart_ = button("Restart", "restartCaptures");
    restart_->setToolTip("Clear all captured views and reconstruction results, then return to live preview for the first capture.");
    retake_->setToolTip("Return to preview. A single view's reconstruction is cleared; with multiple views, completed captures are retained until the replacement succeeds.");
    capture_ = button("Capture pair", "capturePair");
    start_draw_ = button("Start draw", "startDraw");
    finish_draw_ = button("Finish draw", "finishDraw");
    toolbar->addWidget(import_); toolbar->addWidget(camera_mode_);
    toolbar->addWidget(camera_);
    if (has_confirmed_calibration_) {
        manual_reconstruct_ = button("Reconstruct", "run");
        manual_reconstruct_->setStyleSheet("QPushButton { background: #2464d9; color: white; border: 0; }"
                                          "QPushButton:disabled { background: #c3d1e6; }");
        toolbar->addWidget(manual_reconstruct_);
        connect(manual_reconstruct_,&QPushButton::clicked,this,&ReconstructionWindow::reconstructCurrent);
    }
    toolbar->addStretch(); toolbar->addWidget(start_draw_); toolbar->addWidget(finish_draw_);
    toolbar->addWidget(restart_);
    toolbar->addWidget(retake_);
    toolbar->addSpacing(48);
    toolbar->addWidget(capture_);
    outer->addLayout(toolbar);
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
        ? "Choose a folder containing matching left/right images and calibration JSON. Numbered PNG, JPG and TIFF pairs are imported in numeric order. Uncheck to use the confirmed calibration with matching image dimensions."
        : "Choose a folder containing matching left/right images and calibration.json, stereo_calibration.json or sentech_stereo_calibration.json. Numbered PNG, JPG and TIFF pairs are imported in numeric order.");
    layout->addWidget(capture_calibration_);
    connect(capture_calibration_,&QCheckBox::toggled,this,[this] { refresh(); });
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
    denoise_ = new QCheckBox("5 × 5 neighbour + normal filtering"); denoise_->setObjectName("xyzDenoise"); denoise_->setChecked(true); layout->addWidget(denoise_);
    denoise_->setToolTip("Keep points with at least 7 neighbours strictly within the distance limit and a fitted normal less than 60° from the viewing ray. The same rules apply at boundaries.");
    neighbor_distance_ = decimal(0.003,0.001,1," m"); neighbor_distance_->setObjectName("neighborDistance"); neighbor_distance_->setSingleStep(0.001);
    auto* neighbor_form=new QFormLayout; neighbor_form->addRow("Neighbour distance",neighbor_distance_); layout->addLayout(neighbor_form);
    mesh_edge_=decimal(.01,.001,1," m");
    mesh_edge_->setObjectName("meshMaxEdge");
    mesh_edge_->setSingleStep(.001);
    auto* mesh_form=new QFormLayout; mesh_form->addRow("Max mesh edge",mesh_edge_);
    layout->addLayout(mesh_form); settings_layout->addWidget(filters);
    auto* settings_panel=scrollPanel(settings,260); settings_panel->setObjectName("workspaceTools");
    split->addWidget(settings_panel);

    tabs_ = new QTabWidget; tabs_->setObjectName("workspaceTabs");
    auto* scene = new QWidget; auto* scene_layout = new QVBoxLayout(scene); scene_controls_ = new QHBoxLayout;
    scene_controls_->setSpacing(4);
    auto* cloud_controls=new QHBoxLayout;
    fs_cloud_=new QRadioButton("FS"); fs_cloud_->setObjectName("fsPointCloud");
    ma_cloud_=new QRadioButton("MA"); ma_cloud_->setObjectName("maPointCloud");
    auto* cloud_source=new QButtonGroup(this); cloud_source->setExclusive(true);
    cloud_source->addButton(fs_cloud_); cloud_source->addButton(ma_cloud_);
    fs_cloud_->setChecked(true);
    cloud_controls->addWidget(fs_cloud_); cloud_controls->addWidget(ma_cloud_); cloud_controls->addStretch();
    scene_layout->addLayout(cloud_controls);
    camera_image_mode_ = new QComboBox; camera_image_mode_->setObjectName("cameraImageMode");
    camera_image_mode_->addItems({"Left image", "Depth map"});
    camera_image_mode_->setToolTip("Display the rectified left image or Jet depth map in the preview and captured camera image plane.");
    mesh_mode_=new QComboBox; mesh_mode_->addItems({"Point cloud","Mesh","Wireframe"}); mesh_mode_->setCurrentIndex(0); mesh_mode_->setObjectName("meshMode");
    auto* reset=button("Reset view","resetMeshView");
    auto* show_right_camera=new QCheckBox("Show right camera"); show_right_camera->setObjectName("showRightCamera");
    show_right_camera->setToolTip("Show the right image below the left preview and beside the captured left camera in 3D.");
    scene_controls_->addWidget(camera_image_mode_); scene_controls_->addWidget(show_right_camera);
    scene_controls_->addWidget(mesh_mode_); scene_controls_->addWidget(reset); scene_controls_->addStretch();
    scene_layout->addLayout(scene_controls_);
    auto* scene_split = new QSplitter(Qt::Horizontal); scene_split->setObjectName("previewSceneSplit");
    scene_split->setChildrenCollapsible(false);
    auto* previews = new QWidget; auto* preview_layout = new QVBoxLayout(previews);
    preview_layout->setContentsMargins(0,0,0,0);
    preview_stack_=new QStackedWidget; preview_stack_->setObjectName("leftPreviewStack");
    preview_ = new StereoImageView("Left preview"); preview_->setObjectName("capturePreview");
    right_preview_ = new StereoImageView("Right preview"); right_preview_->setObjectName("rightCapturePreview");
    preview_stack_->addWidget(preview_);
    preview_layout->addWidget(preview_stack_,1); preview_layout->addWidget(right_preview_,1);
    right_preview_->hide(); scene_split->addWidget(previews);
    mesh_view_=new MeshView; mesh_view_->setObjectName("meshView"); scene_split->addWidget(mesh_view_);
    scene_split->setStretchFactor(0,1); scene_split->setStretchFactor(1,2);
    scene_split->setSizes({360,720}); scene_layout->addWidget(scene_split,1);
    capture_transition_ = new CaptureTransition(scene);
    connect(capture_transition_, &CaptureTransition::finished, this, [this] {
        if (transition_camera_id_) highlighted_camera_id_=transition_camera_id_;
        transition_camera_id_=0;
        preview_->setImageVisible(true);
        refresh(); // Hand the landed image to the actual OpenGL camera texture.
    });
    mesh_view_->setMode(0);
    scene_status_=label("Connect cameras to preview on the left, or choose an import folder.");
    scene_status_->setObjectName("meshStatus"); scene_layout->addWidget(scene_status_);
    depth_status_ = label(""); depth_status_->setObjectName("depthRange"); scene_layout->addWidget(depth_status_);
    scene_layout->addWidget(label("Click camera / view label: show image · Left drag: rotate · Right drag: pan · Wheel: zoom"));
    tabs_->addTab(scene,"3D workspace");
    tabs_->setTabToolTip(SceneTab,"Image preview on the left; captured point clouds and mesh on the right.");
    connect(camera_image_mode_,&QComboBox::currentIndexChanged,this,[this] { showImages(); });
    connect(mesh_mode_,&QComboBox::currentIndexChanged,this,[this](int mode) { mesh_view_->setMode(mode); refresh(); });
    connect(reset,&QPushButton::clicked,this,[this] { mesh_view_->resetView(); });
    connect(show_right_camera,&QCheckBox::toggled,this,[this](bool visible) {
        right_preview_->setVisible(visible); mesh_view_->setRightCameraVisible(visible);
    });
    connect(ma_cloud_,&QRadioButton::toggled,this,[this](bool checked) {
        ma_upload_pending_=checked && displayed_ma_clouds_ && !ma_cloud_valid_;
        refresh();
    });
    auto* region = new QWidget; region_=region; auto* region_layout = new QVBoxLayout(region);
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
    if (!calibration) {
        // Scan uses the existing defaults; the scene gets the full width.
        // Keep the hidden parameter widgets as the shared pipeline's settings.
        settings_panel->hide();
        auto* mask_tools = new QWidget; mask_tools->setObjectName("scanMaskTools");
        auto* mask_row = new QHBoxLayout(mask_tools); mask_row->setContentsMargins(0,0,0,0); mask_row->setSpacing(2);
        const auto add_tool=[&](QPushButton* button, const QString& text) {
            button->setToolTip(button->text()); button->setAccessibleName(button->text()); button->setText(text);
            button->setStyleSheet("QPushButton { padding: 5px 2px; }");
            button->setSizePolicy(QSizePolicy::Maximum,QSizePolicy::Fixed);
            mask_row->addWidget(button);
        };
        add_tool(sam_box_,"Box"); add_tool(sam_foreground_,"FG +"); add_tool(sam_background_,"BG −");
        add_tool(sam_remove_,"Del"); add_tool(sam_undo_,"Undo");
        add_tool(brush_,"Brush"); add_tool(eraser_,"Erase"); add_tool(clear_,"Clear");
        brush_size_->setSizePolicy(QSizePolicy::Maximum,QSizePolicy::Fixed); mask_row->addWidget(brush_size_);
        overlay->setToolTip(overlay->text()); overlay->setAccessibleName(overlay->text()); overlay->setText("Overlay");
        mask_row->addWidget(overlay); mask_row->addStretch();
        region_layout->insertWidget(0,mask_tools);
        mask_panel_ = mask_tools;
        for (auto* b : {start_draw_,finish_draw_}) b->setStyleSheet("QPushButton { padding: 5px 4px; }");

        open_save_ = button("Save results…","openSaveResults");
        connect(open_save_,&QPushButton::clicked,this,&ReconstructionWindow::saveScanResults);
    }
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
    progress_->setToolTip("Progress updates when each calculation stage completes.");
    auto* bottom_row = new QHBoxLayout; bottom_row->addWidget(progress_,1);
    if (open_save_) bottom_row->addWidget(open_save_);
    outer->addLayout(bottom_row);
    connect(show_log,&QPushButton::toggled,log_,&QWidget::setVisible);
    connect(show_log,&QPushButton::toggled,this,[show_log](bool shown) { show_log->setText(shown ? "Hide log" : "Show log"); });
    connect(&controller_,&PipelineController::log,this,[this](const QString& message) { log_->appendPlainText(QDateTime::currentDateTime().toString("hh:mm:ss") + "  " + message); });
    connect(&controller_,&PipelineController::stateChanged,this,[this](PipelineState state) {
        const bool just_connected = state.connected && !state_.connected;
        const bool clean_preview = !state.has_rectified && state.image_id!=state_.image_id;
        if (state.live || state.image_id != state_.image_id) {
            reconstruction_valid_ = false;
            preview_camera_id_=0;
            drawing_mask_=false; mask_pending_=false;
            mask_request_id_=controller_.requestMask(state.image_id,{});
        }
        if (state.import_directory!=state_.import_directory) {
            drawing_mask_=false; mask_pending_=false; preview_camera_id_=0;
            mask_request_id_=controller_.requestMask(state.image_id,{});
            const QSignalBlocker blocker(mask_);
            mask_->setImage(rectified_left_);
        }
        if (state.capture_count==0) { captured_clouds_.clear(); capture_previews_.clear(); }
        state_ = std::move(state);
        if (clean_preview) {
            cancelCapturePresentation();
            mesh_valid_=false; gpu_upload_pending_=false;
            gpu_mesh_result_.reset(); mesh_view_->setGPUMesh({});
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
            if (capture_mesh_requested_) {
                capture_mesh_requested_=false;
                setCaptureProcessing(false);
                // Wait for FS meshing on the first capture, or for the current
                // MA camera poses before presenting multi-view FS points.
                if (!state_.action_failed && state_.capture_count>=2 && state_.ma_clouds &&
                    state_.ma_clouds!=displayed_ma_clouds_) {
                    capture_transition_pending_=true;
                    transition_camera_id_=state_.image_id; highlighted_camera_id_=0;
                    presentMAResult();
                } else if (!state_.action_failed && pending_capture_mesh_) {
                    captured_clouds_={pending_capture_cloud_};
                    capture_transition_pending_=true;
                    transition_camera_id_=state_.image_id; highlighted_camera_id_=0;
                    presentGeometry(std::move(pending_capture_mesh_));
                } else if (pending_capture_cloud_ && state_.capture_count<2) {
                    // Meshing may reject every triangle; retain usable points
                    // and the worker's failure message without a success animation.
                    presentGeometry(pending_capture_cloud_);
                }
                pending_capture_cloud_.reset(); pending_capture_mesh_.reset();
            } else if (capture_processing_ && (state_.action_failed || !gpu_upload_pending_)) {
                setCaptureProcessing(false);
            }
            if (!state_.action_failed && state_.ma_clouds && state_.ma_clouds!=displayed_ma_clouds_) presentMAResult();
        }
        refresh();
    });
    connect(&controller_,&PipelineController::images,this,[this](QImage rl,QImage rr,QImage rectified_mask) {
        rectified_left_ = std::move(rl); rectified_right_ = std::move(rr);
        reconstruction_valid_=false;
        {
            const QSignalBlocker blocker(mask_);
            MaskEditor::Snapshot drawing;
            drawing.base_mask=std::move(rectified_mask);
            drawing.accepted=!drawing.base_mask.isNull();
            mask_->restore(rectified_left_,drawing);
        }
        save_mask_->setChecked(mask_->hasSelection()); tabs_->setCurrentIndex(SceneTab);
        if (capture_processing_) {
            // Replace the last live frame with the exact pair used for inference.
            preview_->setImage(rectified_left_); right_preview_->setImage(rectified_right_);
        }
        showImages();
    });
    connect(&controller_,&PipelineController::maskReady,this,[this](quint64 image_id,quint64 request_id,QImage mask,QString message) {
        if (closing_ || image_id!=(inlineMaskMode() ? preview_camera_id_ : state_.image_id) ||
            request_id!=mask_request_id_ || (inlineMaskMode() && !drawing_mask_)) return;
        mask_pending_=false;
        mask_->setPrediction(std::move(mask)); status_->setText(message);
    });
    connect(&controller_,&PipelineController::gpuMeshReady,this,[this](SharedGPUMesh mesh) {
        if (closing_ || !reconstruction_valid_) return;
        if (mesh->point_cloud && mesh->image_id==state_.image_id)
            capture_previews_.insert(mesh->image_id,{rectified_left_,rectified_right_,depth_image_,depth_description_,mask_->snapshot(),mask_->mask()});
        if (capture_mesh_requested_) {
            if (mesh->point_cloud) pending_capture_cloud_=std::move(mesh);
            else pending_capture_mesh_=std::move(mesh);
            return;
        }
        presentGeometry(std::move(mesh));
    });
    connect(mesh_view_,&MeshView::cameraClicked,this,[this](quint64 image_id) {
        if (busy_ || closing_ || drawing_mask_ || state_.live || capture_processing_ || capture_transition_pending_ ||
            capture_transition_->isRunning() ||
            !capture_previews_.contains(image_id)) return;
        if (std::none_of(captured_clouds_.begin(),captured_clouds_.end(),
            [=](const auto& cloud) { return cloud && cloud->image_id==image_id; })) return;
        preview_camera_id_=image_id;
        if (inlineMaskMode()) {
            mask_request_id_=controller_.requestMask(image_id,{});
            mask_pending_=false;
            const auto& selected=capture_previews_[image_id];
            const QSignalBlocker blocker(mask_);
            mask_->restore(selected.left,selected.drawing);
        }
        const QSignalBlocker blocker(camera_image_mode_);
        camera_image_mode_->setCurrentIndex(0);
        refresh();
    });
    connect(mesh_view_,&MeshView::gpuMeshPresented,this,[this](SharedGPUMesh mesh) {
        if (closing_ || !reconstruction_valid_ || mesh!=gpu_mesh_result_) return;
        if (!busy_) setCaptureProcessing(false);
        gpu_upload_pending_=false; mesh_valid_=true;
        if (capture_transition_pending_) startCaptureTransition();
        refresh();
    },Qt::QueuedConnection); // Updating the progress bar may repaint; wait until paintGL returns.
    connect(mesh_view_,&MeshView::maCloudsPresented,this,[this](SharedGPUClouds clouds) {
        if (closing_ || clouds!=displayed_ma_clouds_) return;
        ma_upload_pending_=false; ma_cloud_valid_=true;
        if (capture_transition_pending_) startCaptureTransition();
        refresh();
    },Qt::QueuedConnection);
    connect(mesh_view_,&MeshView::renderFailed,this,[this](const QString& message) {
        cancelCapturePresentation();
        render_error_=message; gpu_upload_pending_=false; mesh_valid_=false;
        ma_upload_pending_=false; ma_cloud_valid_=false; refresh();
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
        depth_description_=QString("Jet: %1 m (blue) → %2 m (red) · Black: invalid, out of range or filtered")
            .arg(minimum,0,'f',3).arg(maximum,0,'f',3);
        refresh();
        tabs_->setCurrentIndex(SceneTab);
    });
    connect(&controller_,&PipelineController::preview,this,[this](QImage l,QImage r) {
        if (!state_.live || capture_processing_) return;
        live_left_ = std::move(l); live_right_ = std::move(r);
        showImages();
    });
    connect(import_,&QPushButton::clicked,this,[this] {
        const QString initial_directory=state_.import_directory.isEmpty() ? defaultImportDirectory() : state_.import_directory;
        const QString path = QFileDialog::getExistingDirectory(this,"Choose import folder (left/right images and calibration JSON)",initial_directory);
        if (path.isEmpty()) return;
        controller_.submit([=](auto& w) { w.setImportDirectory(path); });
    });
    connect(camera_,&QPushButton::clicked,this,[this] {
        const bool connected = state_.connected;
        const auto mode = static_cast<CameraMode>(camera_mode_->currentData().toInt());
        live_left_ = {}; live_right_ = {};
        controller_.submit([=](auto& w) { if (connected) w.disconnectCameras(); else w.connectCameras(mode); });
    });
    connect(retake_,&QPushButton::clicked,this,[this] {
        live_left_={}; live_right_={}; tabs_->setCurrentIndex(SceneTab);
        controller_.submit([](auto& worker) { worker.startCapturePreview(false); });
    });
    connect(restart_,&QPushButton::clicked,this,[this] {
        mask_request_id_=controller_.requestMask(state_.image_id,{});
        controller_.submit([](auto& worker) { worker.cleanCaptures(); });
    });
    connect(capture_,&QPushButton::clicked,this,[this] {
        const bool importing=!state_.import_directory.isEmpty();
        if (!importing && !state_.live) {
            live_left_={}; live_right_={}; tabs_->setCurrentIndex(SceneTab);
            controller_.submit([](auto& worker) { worker.startCapturePreview(true); });
            return;
        }
        if (!captured_clouds_.empty()) mesh_view_->preserveView();
        const bool single_capture=state_.capture_count==0 || (importing && !state_.import_sequence_started);
        const bool own_calibration=capture_calibration_->isChecked();
        capture_mesh_requested_=true;
        pending_capture_cloud_.reset(); pending_capture_mesh_.reset();
        setCaptureProcessing(true);
        reconstruction_valid_=false; refresh();
        const float low=minimum_->value(), high=maximum_->value(), distance=neighbor_distance_->value();
        const bool denoise=denoise_->isChecked();
        const double edge=mesh_edge_->value();
        controller_.submit([=](auto& w) {
            if (importing) {
                w.importNextPair(own_calibration);
                w.reconstruct(low,high,{},denoise,distance);
            } else w.captureAndReconstruct(low,high,denoise,distance);
            if (single_capture) w.buildMeshGPU({},edge);
        });
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
        const QImage selection=exportMask();
        const SharedGPUMesh gpu_mesh=mesh_valid_ ? gpu_mesh_result_ : SharedGPUMesh{};
        const SharedGPUClouds ma_geometry=ma_cloud_->isChecked() && ma_cloud_valid_
            ? displayed_ma_clouds_ : SharedGPUClouds{};
        controller_.submit([=](auto& worker) { worker.saveResults(directory,options,selection,gpu_mesh,overwrite,ma_geometry); });
    };
    connect(save_selected_,&QPushButton::clicked,this,[this,save_results] {
        save_results({save_images_->isChecked(),save_calibration_->isChecked(),save_mask_->isChecked(),save_mesh_->isChecked(),save_depth_->isChecked()});
    });
    connect(save_all_,&QPushButton::clicked,this,[this,save_results] {
        save_results({true,true,!exportMask().isNull(),true,true});
    });
    const auto mesh_settings_changed=[this] {
        // Triangle constraints do not change a point-only result.
        const bool point_cloud=gpu_mesh_result_ && gpu_mesh_result_->point_cloud;
        if (!point_cloud) { gpu_upload_pending_=false; mesh_valid_=false; }
        refresh();
    };
    connect(mesh_edge_,&QDoubleSpinBox::valueChanged,this,mesh_settings_changed);
    const auto depth_settings_changed = [this] {
        // Changing parameters invalidates derived results even if the old values are restored later.
        reconstruction_valid_=false;
        refresh();
    };
    connect(minimum_,&QDoubleSpinBox::valueChanged,this,depth_settings_changed);
    connect(maximum_,&QDoubleSpinBox::valueChanged,this,depth_settings_changed);
    connect(denoise_,&QCheckBox::toggled,this,depth_settings_changed);
    connect(neighbor_distance_,&QDoubleSpinBox::valueChanged,this,depth_settings_changed);
    connect(clear_,&QPushButton::clicked,this,[this] {
        mask_->clearMask();
        if (!has_confirmed_calibration_ && !inlineMaskMode()) reconstructCurrent();
    });
    connect(overlay,&QCheckBox::toggled,mask_,&MaskEditor::setOverlayVisible);
    connect(brush_size_,&QSpinBox::valueChanged,mask_,&MaskEditor::setBrushSize);
    connect(mask_,&MaskEditor::brushSizeChanged,brush_size_,&QSpinBox::setValue);
    const auto tool = [this,overlay,region](QPushButton* button, MaskEditor::Tool tool) {
        connect(button,&QPushButton::clicked,this,[this,overlay,region,tool] {
            if (!inlineMaskMode()) tabs_->setCurrentWidget(region);
            overlay->setChecked(true); mask_->setTool(tool);
        });
    };
    tool(sam_box_,MaskEditor::Tool::Box); tool(sam_foreground_,MaskEditor::Tool::Foreground);
    tool(sam_background_,MaskEditor::Tool::Background); tool(sam_remove_,MaskEditor::Tool::Remove);
    tool(brush_,MaskEditor::Tool::Brush); tool(eraser_,MaskEditor::Tool::Eraser);
    connect(sam_undo_,&QPushButton::clicked,mask_,&MaskEditor::undoPrompt);
    connect(start_draw_,&QPushButton::clicked,this,[this,overlay] {
        if (!inlineMaskMode() || !capture_previews_.contains(preview_camera_id_)) return;
        drawing_mask_=true;
        camera_image_mode_->setCurrentIndex(0); overlay->setChecked(true);
        refresh(); mask_->setFocus();
        status_->setText("Draw a mask for the selected camera, then click Finish draw.");
    });
    connect(finish_draw_,&QPushButton::clicked,this,[this] {
        if (inlineMaskMode() && mask_pending_) return;
        mask_->acceptPrediction();
        if (inlineMaskMode()) { finishInlineDraw(); return; }
        if (mask_->hasSelection()) tabs_->setCurrentIndex(SceneTab);
    });
    connect(mask_,&MaskEditor::selectionConfirmed,this,&ReconstructionWindow::finishInlineDraw);
    connect(mask_,&MaskEditor::promptsChanged,this,[this] {
        if (inlineMaskMode() && !drawing_mask_) return;
        const auto prompts=mask_->prompts();
        mask_pending_=!prompts.empty();
        mask_request_id_=controller_.requestMask(inlineMaskMode() ? preview_camera_id_ : state_.image_id,prompts,
            inlineMaskMode() ? capture_previews_.value(preview_camera_id_).left : QImage{});
        refresh();
        if (!prompts.empty()) status_->setText("Updating SAM mask…");
    });
    connect(mask_,&MaskEditor::selectionChanged,this,[this] {
        if (inlineMaskMode()) { refresh(); return; }
        reconstruction_valid_=false; refresh();
        if (!has_confirmed_calibration_ && mask_->hasSelection()) {
            // Finish draw (or Enter) commits the selection. Wait for its UI
            // updates to finish before starting work on the frozen input.
            const auto image_id=state_.image_id;
            QTimer::singleShot(0,this,[this,image_id] {
                if (state_.image_id==image_id && mask_->hasSelection()) reconstructCurrent();
            });
        }
    });
    connect(mask_,&MaskEditor::hint,status_,&QLabel::setText);
    auto* timer = new QTimer(this); timer->setInterval(200); connect(timer,&QTimer::timeout,this,[this] { if (busy_) time_->setText(QString("Running: %1 s").arg(elapsed_.elapsed()/1000.0,0,'f',1)); }); timer->start();
    connect(tabs_,&QTabWidget::currentChanged,this,&ReconstructionWindow::refreshWorkspace);
    refresh();
}
bool ReconstructionWindow::inlineMaskMode() const {
    return !has_confirmed_calibration_ && !state_.import_directory.isEmpty();
}
void ReconstructionWindow::finishInlineDraw() {
    if (!inlineMaskMode() || !drawing_mask_ || mask_pending_) return;
    auto selected=capture_previews_.find(preview_camera_id_);
    if (selected==capture_previews_.end()) return;
    selected->drawing=mask_->snapshot(); selected->mask=mask_->mask();
    drawing_mask_=false;
    mask_request_id_=controller_.requestMask(preview_camera_id_,{});
    refresh();
    status_->setText("Mask saved for this view in the current session.");
}
QImage ReconstructionWindow::exportMask() const {
    // The existing export saves the latest pair, even while reviewing an older camera.
    return inlineMaskMode() ? capture_previews_.value(state_.image_id).mask : mask_->mask();
}
void ReconstructionWindow::saveScanResults() {
    const auto directory=QFileDialog::getExistingDirectory(this,"Save scan results",save_directory_->text());
    if (directory.isEmpty()) return;
    save_directory_->setText(directory);
    QStringList existing;
    for (const auto& name:scanResultFileNames(state_.capture_count)) {
        const QFileInfo file(QDir(directory).filePath(name));
        if (file.exists() || file.isSymLink()) existing << name;
    }
    const bool overwrite=!existing.isEmpty();
    if (overwrite && QMessageBox::question(this,"Replace existing files?",
        "These files already exist in the chosen folder:\n"+existing.join("\n")+"\n\nReplace them?",
        QMessageBox::Yes|QMessageBox::No,QMessageBox::No)!=QMessageBox::Yes) return;
    QHash<quint64,QImage> masks;
    for (auto it=capture_previews_.cbegin();it!=capture_previews_.cend();++it)
        if (!it->mask.isNull()) masks.insert(it.key(),it->mask);
    if (!inlineMaskMode() && mask_->hasSelection()) masks.insert(state_.image_id,mask_->mask());
    controller_.submit([=](auto& worker) { worker.saveScanResults(directory,masks,overwrite); });
}
void ReconstructionWindow::buildMeshGPU() {
    if (busy_ || closing_ || gpu_upload_pending_ || capture_transition_pending_ || capture_transition_->isRunning() || state_.live || !reconstruction_valid_ ||
        !state_.depth_ready || !state_.gpu_ready || !state_.engine_ready || state_.capture_count>=2) return;
    mesh_valid_=false; refresh();
    const QImage selected=inlineMaskMode() ? QImage{} : mask_->mask();
    const double edge=mesh_edge_->value();
    controller_.submit([=](auto& worker) { worker.buildMeshGPU(selected,edge); });
}
void ReconstructionWindow::reconstructCurrent() {
    if (inlineMaskMode() || busy_ || closing_ || capture_transition_pending_ || capture_transition_->isRunning() || state_.live || !state_.has_rectified || !state_.engine_ready) return;
    reconstruction_valid_=false; refresh();
    const float low=minimum_->value(), high=maximum_->value(), distance=neighbor_distance_->value();
    const QImage selected=mask_->mask();
    const bool denoise=denoise_->isChecked();
    controller_.submit([=](auto& w) { w.reconstruct(low,high,selected,denoise,distance); });
}
void ReconstructionWindow::refresh() {
    if (!reconstruction_valid_ || !state_.depth_ready) {
        mesh_valid_=false; gpu_upload_pending_=false; ma_cloud_valid_=false; ma_upload_pending_=false;
    }
    // Result validity controls processing/export. Completed scene geometry has
    // its own lifetime and survives preview, new inference and failed retakes.
    // Update it before controls that can cause a synchronous OpenGL repaint.
    if (closing_) captured_clouds_.clear();
    if (closing_ || !state_.ma_clouds) {
        displayed_ma_clouds_.reset(); displayed_cameras_.reset();
        ma_cloud_valid_=false; ma_upload_pending_=false;
    }
    if (!state_.capture_count) highlighted_camera_id_=0;
    mesh_view_->setPredictedCameras(displayed_cameras_);
    mesh_view_->setMAClouds(displayed_ma_clouds_);
    const bool ma_available=displayed_ma_clouds_ && !displayed_ma_clouds_->empty();
    if (!ma_available && ma_cloud_->isChecked()) {
        const QSignalBlocker fs_blocker(fs_cloud_), ma_blocker(ma_cloud_);
        fs_cloud_->setChecked(true);
    }
    const bool show_ma=ma_cloud_->isChecked();
    fs_cloud_->setEnabled(!closing_ && !busy_ && !capture_transition_pending_ && !capture_transition_->isRunning());
    ma_cloud_->setEnabled(fs_cloud_->isEnabled() && ma_available);
    ma_cloud_->setToolTip(ma_available ? "Display MapAnything's GPU point clouds at the predicted camera poses."
        : "Available after MapAnything finishes inference on at least two captures.");
    mesh_view_->setShowMAClouds(show_ma);
    const bool replacement=(mesh_valid_ || gpu_upload_pending_) && gpu_mesh_result_ && !gpu_mesh_result_->point_cloud;
    mesh_view_->setCapturedClouds(captured_clouds_,replacement && !captured_clouds_.empty()
        ? captured_clouds_.back() : SharedGPUMesh{});
    QHash<quint64,QImage> display_masks;
    if (inlineMaskMode() && !closing_)
        for (auto it=capture_previews_.cbegin();it!=capture_previews_.cend();++it)
            if (!it->mask.isNull()) display_masks.insert(it.key(),it->mask);
    mesh_view_->setFSMasks(std::move(display_masks),inlineMaskMode());
    const bool transferring=capture_transition_pending_ || capture_transition_->isRunning();
    const bool ready = !busy_ && !closing_ && !gpu_upload_pending_ && !ma_upload_pending_ && !transferring;
    const bool idle = ready && !drawing_mask_, frozen = state_.has_rectified && !state_.live;
    const bool importing=!state_.import_directory.isEmpty();
    import_->setEnabled(idle); capture_calibration_->setEnabled(idle && has_confirmed_calibration_); camera_->setEnabled(idle);
    import_->setToolTip(importing ? QString("Import folder: %1\n%2 / %3 pairs imported. Choose another folder to start from its first pair.")
        .arg(state_.import_directory).arg(state_.import_index).arg(state_.import_total)
        : "Choose the folder to use for Import pair. Selecting a folder does not load an image.");
    camera_->setText(state_.connected ? "Disconnect cameras" : "Connect cameras");
    camera_mode_->setEnabled(idle && !state_.connected);
    if (state_.connected) {
        const QSignalBlocker blocker(camera_mode_);
        camera_mode_->setCurrentIndex(camera_mode_->findData(int(state_.camera_mode)));
    }
    const bool depth_valid = minimum_->value() < maximum_->value();
    retake_->setEnabled(idle && !importing && state_.connected && frozen);
    restart_->setEnabled(idle && (importing || state_.connected) && (state_.has_rectified || state_.capture_count>0));
    restart_->setToolTip(importing ? "Clear captured views and results, keep the import folder, and start again from its first pair."
                                  : "Clear all captured views and reconstruction results, then return to live preview for the first capture.");
    const bool can_append=frozen && reconstruction_valid_ && (mesh_valid_ || ma_cloud_valid_) &&
                          state_.capture_count>0 && state_.capture_count<PipelineState::kMaxCaptures;
    const bool can_import=state_.import_index<state_.import_total &&
        (!state_.import_sequence_started || state_.capture_count<PipelineState::kMaxCaptures);
    capture_->setEnabled(idle && state_.engine_ready && depth_valid &&
        (importing ? can_import : state_.connected && (state_.live || can_append)));
    capture_->setText(importing ? "Import pair" : state_.capture_count>0 ? "Capture more" : "Capture pair");
    capture_->setToolTip(importing
        ? (can_import ? QString("Import pair %1 of %2 and reconstruct it.\n%3")
                .arg(state_.import_index+1).arg(state_.import_total).arg(state_.import_directory)
            : state_.import_index>=state_.import_total ? "All pairs have been imported. Restart or choose another folder."
                                                       : "Five views are complete. Restart or choose another folder.")
        : state_.live
        ? (state_.capture_count==0 ? "Capture this stereo pair and automatically display the FS GPU mesh."
                                   : "Capture this stereo pair and display FS point clouds at the updated MA camera poses. MA point clouds remain available.")
        : state_.capture_count>=PipelineState::kMaxCaptures
            ? "Five captures are complete. Retake the latest view or restart."
            : "Keep completed captures and preview the next view, then press again to capture it. Up to five pairs.");
    minimum_->setEnabled(idle); maximum_->setEnabled(idle);
    mesh_edge_->setEnabled(idle);
    denoise_->setEnabled(idle); neighbor_distance_->setEnabled(idle && denoise_->isChecked());
    if (manual_reconstruct_) {
        manual_reconstruct_->setEnabled(idle && frozen && state_.engine_ready && depth_valid);
        manual_reconstruct_->setToolTip(depth_valid
            ? "Reconstruct depth and display valid points. Without a confirmed mask, process the full image."
            : "Minimum depth must be less than maximum.");
    }
    const bool editing=ready && frozen && (!inlineMaskMode() || drawing_mask_);
    mask_->setEditingEnabled(editing);
    for (auto* b : {brush_,eraser_}) b->setEnabled(editing);
    brush_size_->setEnabled(editing);
    for (auto* b : {sam_box_,sam_foreground_,sam_background_,sam_remove_}) b->setEnabled(editing && state_.sam_ready);
    sam_undo_->setEnabled(editing && mask_->hasPrompts());
    start_draw_->setEnabled(inlineMaskMode() && idle && frozen && capture_previews_.contains(preview_camera_id_));
    start_draw_->setToolTip("Click a captured camera or its view label, then start drawing on its rectified left image.");
    finish_draw_->setEnabled(editing && (inlineMaskMode() ? !mask_pending_ : mask_->hasPrediction()));
    clear_->setEnabled(editing && (mask_->hasPrompts() || mask_->hasPrediction()));
    input_->setText(importing ? QString("Import folder: %1\n%2 / %3 pairs imported\n%4")
        .arg(state_.import_directory).arg(state_.import_index).arg(state_.import_total).arg(state_.input) : state_.input);
    calibration_->setText(state_.calibration);
    if (!closing_) status_->setText(!render_error_.isEmpty() ? render_error_ : !busy_ && !state_.action_failed && !state_.live && state_.depth_ready && !reconstruction_valid_
        ? has_confirmed_calibration_ ? "Selection or post-processing settings changed. Reconstruct again to update the point cloud."
                                     : "Confirm the region with Finish draw to update the point cloud."
        : state_.status);
    if (!state_.depth_ready || !reconstruction_valid_) {
        depth_image_ = {};
        depth_status_->setText("Jet uses the Minimum and Maximum depth settings for each reconstruction.");
    }
    const bool result_current=reconstruction_valid_ && state_.depth_ready;
    steps_->setText(QString("%1 Rectification   →   %2 FS inference   →   %3 GPU XYZ / filtering   →   %4 Depth map   →   %5 Valid point cloud")
        .arg(state_.has_rectified ? "✓" : "○", result_current ? "✓" : "○", result_current ? "✓" : "○", result_current ? "✓" : "○", mesh_valid_ ? "✓" : "○"));
    refreshWorkspace();
    const int progress=!render_error_.isEmpty() ? std::min(state_.progress,95) : (gpu_upload_pending_ || ma_upload_pending_) ? 97 : state_.progress;
    const QString progress_stage=!render_error_.isEmpty() ? "Display failed · " + render_error_
        : (gpu_upload_pending_ || ma_upload_pending_) ? "Transferring GPU geometry to OpenGL…" : state_.progress_stage;
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
    if (!mesh_valid_ && !gpu_upload_pending_ && gpu_mesh_result_) {
        gpu_mesh_result_.reset(); mesh_view_->setGPUMesh({});
    }
    const bool point_cloud=(mesh_valid_ || gpu_upload_pending_) && gpu_mesh_result_ && gpu_mesh_result_->point_cloud;
    save_mesh_->setText(show_ma ? "MA point cloud (mesh.ply)"
                              : point_cloud ? "FS point cloud (mesh.ply)" : "FS mesh (mesh.ply)");
    save_mesh_->setToolTip(show_ma ? "Save the displayed MA geometry in its predicted world frame." : "Save the displayed FS geometry as PLY.");
    mesh_mode_->setEnabled(!show_ma && mesh_valid_ && !transferring);
    for (int mode : {1,2}) static_cast<QStandardItemModel*>(mesh_mode_->model())->item(mode)->setEnabled(!show_ma && !point_cloud);
    if ((show_ma || point_cloud) && mesh_mode_->currentIndex()!=0) mesh_mode_->setCurrentIndex(0);
    browse_save_->setEnabled(idle);
    if (open_save_) open_save_->setEnabled(idle && frozen && result_current && state_.capture_count>0);
    for (auto* option:{save_images_,save_calibration_,save_mask_,save_mesh_,save_depth_}) option->setEnabled(idle);
    const bool has_export_mask=inlineMaskMode() ? !capture_previews_.value(state_.image_id).mask.isNull() : mask_->hasSelection();
    if (!has_export_mask) {
        const QSignalBlocker blocker(save_mask_);
        save_mask_->setChecked(false);
    }
    save_mask_->setEnabled(idle && has_export_mask);
    save_mask_->setToolTip(inlineMaskMode() ? "Save the latest imported pair's confirmed mask, in rectified-left image coordinates."
                                          : "Full-resolution confirmed mask in rectified-left image coordinates.");
    save_calibration_name_->setText(state_.calibration_filename);
    const bool any_save=save_images_->isChecked() || save_calibration_->isChecked() || save_mask_->isChecked() || save_mesh_->isChecked() || save_depth_->isChecked();
    const bool geometry_valid=show_ma ? ma_cloud_valid_ : mesh_valid_ && bool(gpu_mesh_result_);
    const bool can_save=any_save && frozen && (!save_mask_->isChecked() || has_export_mask) &&
        (!save_depth_->isChecked() || result_current) &&
        (!save_mesh_->isChecked() || geometry_valid);
    save_selected_->setEnabled(idle && can_save && !save_directory_->text().isEmpty());
    save_all_->setEnabled(idle && frozen && result_current && geometry_valid && !save_directory_->text().isEmpty());
    scene_status_->setText(transferring ? "Capture complete · Placing the image in its camera frame…"
        : capture_processing_ ? "Calculating · Reconstructing the captured surface…"
        : point_cloud ? QString("GPU point cloud · %1 valid points%2")
        .arg(gpu_mesh_result_->stats.point_count)
        .arg(gpu_upload_pending_ ? " · preparing OpenGL" : "")
        : gpu_mesh_result_ ? QString("GPU mesh · %1 triangles · %2 cm²%3")
        .arg(gpu_mesh_result_->stats.triangle_count).arg(gpu_mesh_result_->stats.area_m2*1e4,0,'f',2)
        .arg(gpu_upload_pending_ ? " · preparing OpenGL" : "")
        : state_.live ? "LIVE · Preview on the left · Capture a pair to reconstruct depth."
        : result_current ? "Depth ready. Switch the camera image or generate a GPU mesh."
        : frozen ? (has_confirmed_calibration_ ? "Captured left image · Reconstruct to display valid points. Mask drawing is optional."
                                              : "Captured left image · Confirm a region, retake or import a pair to update the point cloud.")
        : importing ? QString("Import folder ready · %1 / %2 pairs imported · Press Import pair.")
            .arg(state_.import_index).arg(state_.import_total)
        : "Connect cameras to preview on the left, or choose an import folder.");
    if (show_ma && !transferring && !capture_processing_) {
        unsigned long long count=0;
        const auto& frames=displayed_ma_clouds_;
        if (frames) for (const auto& frame:*frames) count+=frame->stats.point_count;
        scene_status_->setText(QString("MA point cloud · %1 valid points · %2 views")
            .arg(count).arg(frames ? frames->size() : 0));
    } else if (!captured_clouds_.empty())
        scene_status_->setText(scene_status_->text()+QString(" · %1 completed %2 visible")
            .arg(captured_clouds_.size()).arg(captured_clouds_.size()==1 ? "capture" : "captures"));
    showImages();
}
void ReconstructionWindow::refreshWorkspace() {
    const bool inline_mask=inlineMaskMode();
    if (inline_mask && mask_->parentWidget()!=preview_stack_) {
        preview_stack_->addWidget(mask_);
        scene_controls_->insertWidget(4,start_draw_);
        scene_controls_->insertWidget(5,finish_draw_);
        scene_controls_->insertWidget(6,mask_panel_,1);
    } else if (!inline_mask && mask_->parentWidget()==preview_stack_) {
        preview_stack_->removeWidget(mask_);
        static_cast<QVBoxLayout*>(region_->layout())->insertWidget(0,mask_,1);
        static_cast<QVBoxLayout*>(region_->layout())->insertWidget(0,mask_panel_);
        capture_toolbar_->insertWidget(capture_toolbar_->indexOf(restart_),start_draw_);
        capture_toolbar_->insertWidget(capture_toolbar_->indexOf(restart_),finish_draw_);
        preview_stack_->setCurrentWidget(preview_);
    }
    const bool frozen=state_.has_rectified && !state_.live;
    if ((!frozen || inline_mask) && tabs_->currentIndex()==RegionTab) tabs_->setCurrentIndex(SceneTab);
    tabs_->setTabVisible(RegionTab,!inline_mask);
    tabs_->setTabEnabled(RegionTab,frozen && !busy_ && !capture_transition_pending_ && !capture_transition_->isRunning());
    tabs_->setTabToolTip(RegionTab,frozen ? "Optional: draw and confirm a mask for region measurement." : "Capture or import a stereo pair first.");

    const bool scene=tabs_->currentIndex()==SceneTab;
    input_panel_->setVisible(scene);
    mask_panel_->setVisible(inline_mask ? drawing_mask_ : !scene);
    depth_panel_->setVisible(scene); geometry_panel_->setVisible(scene); save_panel_->setVisible(scene && frozen);
    for (auto* b : {import_,camera_,capture_,restart_}) b->setVisible(scene);
    retake_->setVisible(scene && state_.has_rectified && state_.import_directory.isEmpty());
    camera_mode_->setVisible(scene);
    start_draw_->setVisible(inline_mask);
    finish_draw_->setVisible(inline_mask || !scene);
    if (manual_reconstruct_) manual_reconstruct_->setVisible(scene && frozen);
    if (open_save_) open_save_->setVisible(scene && frozen);
}
void ReconstructionWindow::showImages() {
    const bool transferring=capture_transition_pending_ || capture_transition_->isRunning();
    const auto selected=capture_previews_.constFind(preview_camera_id_);
    const bool reviewing=!state_.live && !capture_processing_ && !transferring && selected!=capture_previews_.cend();
    const auto& left=reviewing ? selected->left : rectified_left_;
    const auto& right=reviewing ? selected->right : rectified_right_;
    const auto& depth=reviewing ? selected->depth : depth_image_;
    mesh_view_->setCameraPresentation(reviewing ? preview_camera_id_ : state_.image_id,
        reviewing ? preview_camera_id_ : highlighted_camera_id_,transferring ? transition_camera_id_ : 0);
    mesh_view_->setCameraInteractionEnabled(!busy_ && !closing_ && !drawing_mask_ && !state_.live && !capture_processing_ &&
        !transferring && !captured_clouds_.empty());
    camera_image_mode_->setEnabled(!drawing_mask_ && !capture_processing_ && !transferring);
    const bool depth_available = !state_.live && !capture_processing_ && !transferring &&
        (reviewing || (reconstruction_valid_ && state_.depth_ready)) && !depth.isNull();
    static_cast<QStandardItemModel*>(camera_image_mode_->model())->item(1)->setEnabled(depth_available);
    if (!depth_available && camera_image_mode_->currentIndex()==1) {
        const QSignalBlocker blocker(camera_image_mode_);
        camera_image_mode_->setCurrentIndex(0);
    }
    const bool show_depth = depth_available && camera_image_mode_->currentIndex()==1;
    if (inlineMaskMode())
        preview_stack_->setCurrentWidget(reviewing && !show_depth ? static_cast<QWidget*>(mask_) : preview_);
    if (!capture_processing_) {
        preview_->setImage(show_depth ? depth : state_.live ? live_left_ : left);
        right_preview_->setImage(state_.live ? live_right_ : right);
    }
    // The transfer lands on this camera's image plane. Keep its wireframe visible
    // with the mesh, and show the actual texture only after the overlay arrives.
    const bool captured = !state_.live && state_.has_rectified && !capture_processing_;
    mesh_view_->setCamera(captured ? state_.image_camera : std::nullopt);
    mesh_view_->setCameraImage(captured && !transferring ? (show_depth ? depth : left) : QImage{});
    mesh_view_->setRightCameraImage(captured ? right : QImage{});
    depth_status_->setText(reviewing ? selected->depth_description : depth_description_);
    depth_status_->setVisible(show_depth);
}
void ReconstructionWindow::setCaptureProcessing(bool processing) {
    capture_processing_=processing;
    if (processing) preview_camera_id_=0;
    preview_->setProcessing(processing);
    right_preview_->setProcessing(processing);
}
void ReconstructionWindow::presentGeometry(SharedGPUMesh mesh) {
    mesh_valid_=false;
    gpu_mesh_result_=std::move(mesh); gpu_upload_pending_=true;
    if (gpu_mesh_result_->point_cloud && state_.capture_count>0) {
        captured_clouds_.resize(state_.capture_count);
        captured_clouds_.back()=gpu_mesh_result_;
    }
    // QImages share their existing owned storage. Keep only completed slots,
    // dropping the previous image of a retaken view.
    for (auto it=capture_previews_.begin();it!=capture_previews_.end();) {
        if (std::none_of(captured_clouds_.begin(),captured_clouds_.end(),
            [&](const auto& cloud) { return cloud && cloud->image_id==it.key(); })) it=capture_previews_.erase(it);
        else ++it;
    }
    mesh_mode_->setCurrentIndex(0);
    tabs_->setCurrentIndex(SceneTab);
    mesh_view_->setGPUMesh(gpu_mesh_result_);
    if (capture_transition_pending_) mesh_view_->resetCaptureView();
    refresh();
}
void ReconstructionWindow::presentMAResult() {
    displayed_ma_clouds_=state_.ma_clouds;
    displayed_cameras_=state_.predicted_cameras;
    ma_cloud_valid_=false; ma_upload_pending_=false;
    {
        const QSignalBlocker fs_blocker(fs_cloud_), ma_blocker(ma_cloud_);
        fs_cloud_->setChecked(true);
    }
    presentGeometry(pending_capture_cloud_ ? pending_capture_cloud_ : gpu_mesh_result_);
    // MA may change the world origin and every pose. Fit the completed snapshot
    // so the active camera and its transfer destination stay in view.
    mesh_view_->resetCaptureView();
}
void ReconstructionWindow::startCaptureTransition() {
    capture_transition_pending_=false;
    const auto source=[this] {
        const QRectF rect=QRectF(preview_->imageRect()).translated(preview_->mapTo(capture_transition_->parentWidget(),QPoint()));
        return QPolygonF{rect.topLeft(),rect.topRight(),rect.bottomRight(),rect.bottomLeft()};
    };
    const auto destination=[this] {
        return mesh_view_->cameraImageQuad().translated(mesh_view_->mapTo(capture_transition_->parentWidget(),QPoint()));
    };
    if (capture_transition_->start(rectified_left_,source,destination)) preview_->setImageVisible(false);
    else transition_camera_id_=0;
}
void ReconstructionWindow::cancelCapturePresentation() {
    capture_mesh_requested_=false; capture_transition_pending_=false; transition_camera_id_=0;
    pending_capture_cloud_.reset(); pending_capture_mesh_.reset();
    setCaptureProcessing(false);
    capture_transition_->cancel();
    preview_->setImageVisible(true);
}
void ReconstructionWindow::closeEvent(QCloseEvent* event) {
    if (allow_close_) { event->accept(); return; }
    event->ignore(); if (closing_) return;
    closing_=true; cancelCapturePresentation();
    gpu_upload_pending_=false; mesh_valid_=false; mask_request_id_=controller_.requestMask(state_.image_id,{}); refresh(); status_->setText("Closing · waiting for background work and releasing cameras / GPU…"); emit closeRequested();
}
void ReconstructionWindow::allowClose() { allow_close_ = true; close(); }
