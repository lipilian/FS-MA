#include "ReconstructionWindow.hpp"
#include "widgets/MaskEditor.hpp"
#include "widgets/StereoImageView.hpp"
#include <QCheckBox>
#include <QButtonGroup>
#include <QCloseEvent>
#include <QComboBox>
#include <QDateTime>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QLabel>
#include <QToolButton>
#include <QPlainTextEdit>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSpinBox>
#include <QSplitter>
#include <QTabWidget>
#include <QTabBar>
#include <QTimer>
#include <QVBoxLayout>
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
    : controller_(controller), state_(controller.state()) {
    if (!state_.engine_ready || !state_.sam_ready) throw std::logic_error("Reconstruction window requires initialized FoundationStereo and SAM sessions.");
    setObjectName("reconstructionWindow"); setWindowTitle("FoundationStereo · Reconstruction"); resize(1500, 950); setMinimumSize(1120, 740);
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
        QLabel#scene { background: #111c2c; color: #b6c7dc; padding: 32px; border-radius: 7px; font-size: 15px; }
        QTabWidget::pane { border: 1px solid #dce3ed; background: white; }
        QTabBar::tab { padding: 10px 16px; background: #e6edf7; color: #3e5470; }
        QTabBar::tab:selected { background: white; color: #2464d9; }
        QProgressBar { border: 0; background: #e5ebf4; min-height: 6px; max-height: 6px; }
        QProgressBar::chunk { background: #2464d9; }
        QPlainTextEdit { background: #172337; color: #d8e4f3; border: 0; border-radius: 5px; }
    )");
    auto* root = new QWidget; root->setObjectName("root"); setCentralWidget(root);
    auto* outer = new QVBoxLayout(root); outer->setContentsMargins(22,18,22,18); outer->setSpacing(10);
    auto* step = label("01  /  CALIBRATION  ✓     →     02  /  RECONSTRUCTION"); step->setObjectName("step"); outer->addWidget(step);
    auto* title = label("Reconstruction workspace"); title->setObjectName("title"); outer->addWidget(title);
    auto* toolbar = new QHBoxLayout;
    import_ = button("Import capture…", "importCapture"); camera_ = button("Connect cameras", "connectCameras");
    preview_ = button("Resume preview", "preview"); capture_ = button("Capture pair", "capturePair");
    run_ = button("Reconstruct", "run"); finish_draw_ = button("Finish draw", "finishDraw");
    for (auto* b : {import_, camera_, preview_}) toolbar->addWidget(b);
    toolbar->addStretch(); toolbar->addWidget(capture_); toolbar->addWidget(finish_draw_); toolbar->addWidget(run_); outer->addLayout(toolbar);
    auto* split = new QSplitter;
    auto* settings = new QWidget; auto* settings_layout = new QVBoxLayout(settings); settings_layout->setContentsMargins(0,0,8,0);
    QVBoxLayout* layout;
    auto* input_group = new QWidget; input_group->setObjectName("inputCard");
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
    capture_calibration_->setToolTip("Import left.png, right.png and calibration.json from one directory. Uncheck to use the confirmed calibration, requiring matching image dimensions.");
    layout->addWidget(capture_calibration_);
    calibration_ = label(path); calibration_->setTextInteractionFlags(Qt::TextSelectableByMouse); layout->addWidget(calibration_);
    layout->addWidget(label(QString("Confirmed camera calibration\nL %1 · R %2\n%3 × %4 · %5")
        .arg(QString::fromStdString(calibration->left_serial), QString::fromStdString(calibration->right_serial))
        .arg(calibration->image_size.width).arg(calibration->image_size.height)
        .arg(calibration->checked ? "Independent check performed" : "No independent check (optional)")));
    settings_layout->addWidget(input_group);
    auto* depth = group("Depth range", layout); auto* form = new QFormLayout;
    minimum_ = decimal(0,0,1000," m"); maximum_ = decimal(1,0.001,1000," m"); minimum_->setObjectName("minimumDepth"); maximum_->setObjectName("maximumDepth");
    minimum_->setSingleStep(0.05); maximum_->setSingleStep(0.05);
    form->addRow("Minimum", minimum_); form->addRow("Maximum", maximum_); layout->addLayout(form);
    layout->addWidget(label("Applied to full-image XYZ.")); settings_layout->addWidget(depth);
    auto* filters = group("Geometry", layout);
    denoise_ = new QCheckBox("3 × 3 neighbour filtering"); denoise_->setObjectName("xyzDenoise"); denoise_->setChecked(true); layout->addWidget(denoise_);
    neighbor_distance_ = decimal(0.01,0.001,1," m"); neighbor_distance_->setObjectName("neighborDistance"); neighbor_distance_->setSingleStep(0.001);
    auto* neighbor_form=new QFormLayout; neighbor_form->addRow("Neighbour distance",neighbor_distance_); layout->addLayout(neighbor_form);
    layout->addWidget(label("Mesh generation is pending.")); settings_layout->addWidget(filters);
    auto* settings_panel=scrollPanel(settings,260); settings_panel->setObjectName("stepTools");
    split->addWidget(settings_panel);

    tabs_ = new QTabWidget; tabs_->setObjectName("workspaceTabs");
    auto* stereo = new QWidget; auto* stereo_layout = new QVBoxLayout(stereo);
    auto* view_controls = new QHBoxLayout; rectified_ = new QCheckBox("Rectified"); rectified_->setObjectName("rectified"); rectified_->setChecked(true);
    epilines_ = new QCheckBox("Epipolar guides"); epilines_->setObjectName("epipolarGuides"); epilines_->setChecked(false);
    view_controls->addWidget(rectified_); view_controls->addWidget(epilines_); view_controls->addStretch(); stereo_layout->addLayout(view_controls);
    auto* views = new QSplitter; left_ = new StereoImageView("LEFT"); right_ = new StereoImageView("RIGHT");
    left_->setEmptyText("Import a capture directory\nor connect cameras and capture a pair"); right_->setEmptyText("The matching right image\nwill appear here");
    views->addWidget(left_); views->addWidget(right_); stereo_layout->addWidget(views,1);
    stereo_layout->addWidget(label("Rectified applies to live preview and captured pairs. Enable Epipolar guides to compare horizontal alignment."));
    tabs_->addTab(stereo,"Stereo inspection");
    auto* region = new QWidget; auto* region_layout = new QVBoxLayout(region);
    mask_ = new MaskEditor; mask_->setObjectName("maskEditor"); region_layout->addWidget(mask_,1);
    auto* overlay = new QCheckBox("Show selection overlay"); overlay->setChecked(true); region_layout->addWidget(overlay);
    region_layout->addWidget(label("The mask limits neighbourhood filtering; XYZ outside it is preserved."));
    tabs_->addTab(region,"Region measurement");
    auto* depth_page = new QWidget; auto* depth_layout = new QVBoxLayout(depth_page);
    auto* depth_views = new QSplitter;
    depth_left_ = new StereoImageView("RECTIFIED LEFT · 960 × 800"); depth_left_->setObjectName("depthLeft");
    depth_map_ = new StereoImageView("DEPTH · JET · 960 × 800"); depth_map_->setObjectName("depthMap");
    for (auto* view : {depth_left_, depth_map_}) {
        view->setEmptyText("Confirm a region, then\nclick Reconstruct"); depth_views->addWidget(view);
    }
    depth_layout->addWidget(depth_views, 1);
    depth_status_ = label("Jet uses the Minimum and Maximum depth settings for each reconstruction.");
    depth_status_->setObjectName("depthRange"); depth_layout->addWidget(depth_status_);
    tabs_->addTab(depth_page,"Depth map");
    auto* scene = new QWidget; auto* scene_layout = new QVBoxLayout(scene); auto* scene_controls = new QHBoxLayout;
    auto* render = new QComboBox; render->addItems({"Point cloud", "Mesh", "Wireframe"}); render->setEnabled(false);
    auto* color = new QComboBox; color->addItems({"RGB colour", "Depth colour"}); color->setEnabled(false);
    auto* point_size = new QSpinBox; point_size->setRange(1,10); point_size->setValue(2); point_size->setSuffix(" px"); point_size->setEnabled(false);
    auto* reset = button("Reset view"); reset->setEnabled(false);
    for (QWidget* w : std::initializer_list<QWidget*>{render, color, point_size, reset}) { w->setToolTip("Requires CPU geometry and VTK integration."); scene_controls->addWidget(w); }
    scene_layout->addLayout(scene_controls);
    scene_status_ = label("3D viewport · pending\n\nReserved for the VTK point cloud, mesh and wireframe viewer.\n\nRotation · zoom · pan · reset view");
    scene_status_->setObjectName("scene"); scene_status_->setAlignment(Qt::AlignCenter); scene_layout->addWidget(scene_status_,1); tabs_->addTab(scene,"3D browser");
    for (int i=0;i<tabs_->count();++i) {
        tab_status_[i]=new QLabel; tab_status_[i]->setFixedSize(20,20); tab_status_[i]->setAlignment(Qt::AlignCenter);
        tab_status_[i]->setObjectName(QString("stepStatus%1").arg(i));
        tabs_->tabBar()->setTabButton(i,QTabBar::RightSide,tab_status_[i]);
    }
    split->addWidget(tabs_);

    auto* sam = group("Mask draw", layout);
    sam_box_=button("Draw box","samBox"); sam_foreground_=button("Foreground point (+)","samForeground");
    sam_background_=button("Background point (−)","samBackground"); sam_remove_=button("Remove prompt","samRemove");
    sam_undo_=button("Undo prompt","samUndo");
    brush_=button("Brush (+)","maskBrush"); eraser_=button("Eraser (−)","maskEraser");
    clear_=button("Clear mask","clearMask");
    brush_size_=new QSpinBox; brush_size_->setObjectName("maskBrushSize");
    brush_size_->setRange(1,100); brush_size_->setValue(mask_->brushSize()); brush_size_->setSuffix(" px");
    brush_size_->setToolTip("Brush width in full-resolution rectified-left pixels.");
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
    sam->setObjectName("maskDrawPanel"); depth->setObjectName("depthSettings"); filters->setObjectName("geometrySettings");
    settings_layout->addWidget(sam); settings_layout->addStretch();
    split->setSizes({290,1170}); split->setStretchFactor(1,1); outer->addWidget(split,1);
    const auto update_step_tools = [this,input_group,sam,depth,filters,settings_panel] {
        const int step=tabs_->currentIndex();
        input_group->setVisible(step==0); sam->setVisible(step==1);
        depth->setVisible(step==2); filters->setVisible(step==2);
        settings_panel->setVisible(step<3);
    };
    connect(tabs_,&QTabWidget::currentChanged,this,update_step_tools);
    update_step_tools();
    steps_ = label(""); outer->addWidget(steps_);
    progress_ = new QProgressBar; progress_->setRange(0,4); progress_->setValue(0); progress_->setTextVisible(false); outer->addWidget(progress_);
    auto* footer = new QHBoxLayout; status_ = label(""); status_->setObjectName("status"); footer->addWidget(status_,1);
    time_ = label("Idle"); footer->addWidget(time_); auto* show_log = button("Show log"); show_log->setCheckable(true); footer->addWidget(show_log); outer->addLayout(footer);
    log_ = new QPlainTextEdit; log_->setReadOnly(true); log_->setMaximumBlockCount(500); log_->setMaximumHeight(130); log_->hide(); outer->addWidget(log_);
    connect(show_log,&QPushButton::toggled,log_,&QWidget::setVisible);
    connect(show_log,&QPushButton::toggled,this,[show_log](bool shown) { show_log->setText(shown ? "Hide log" : "Show log"); });
    connect(&controller_,&PipelineController::log,this,[this](const QString& message) { log_->appendPlainText(QDateTime::currentDateTime().toString("hh:mm:ss") + "  " + message); });
    connect(&controller_,&PipelineController::stateChanged,this,[this](PipelineState state) { state_ = std::move(state); refresh(); showImages(); });
    connect(&controller_,&PipelineController::busyChanged,this,[this](bool busy) { busy_ = busy; if (busy) elapsed_.start(); else time_->setText(QString("Last task: %1 s").arg(elapsed_.elapsed()/1000.0,0,'f',1)); refresh(); });
    connect(&controller_,&PipelineController::images,this,[this](QImage l,QImage r,QImage rl,QImage rr) {
        raw_left_ = std::move(l); raw_right_ = std::move(r); rectified_left_ = std::move(rl); rectified_right_ = std::move(rr);
        reconstruction_valid_=false; mask_->setImage(rectified_left_); tabs_->setCurrentIndex(1); showImages();
    });
    connect(&controller_,&PipelineController::maskReady,this,[this](quint64 image_id,quint64 request_id,QImage mask,QString message) {
        if (closing_ || image_id!=state_.image_id || request_id!=mask_request_id_) return;
        mask_->setPrediction(std::move(mask)); status_->setText(message);
    });
    connect(&controller_,&PipelineController::depthImages,this,[this,depth_page](QImage left,QImage depth,float minimum,float maximum) {
        if (closing_) return;
        reconstruction_valid_=mask_->hasSelection();
        depth_left_->setImage(std::move(left)); depth_map_->setImage(std::move(depth));
        depth_status_->setText(QString("Jet: %1 m (blue) → %2 m (red) · Black: invalid, out of range or filtered")
            .arg(minimum,0,'f',3).arg(maximum,0,'f',3));
        refreshWorkflow();
        if (reconstruction_valid_) tabs_->setCurrentWidget(depth_page);
    });
    connect(&controller_,&PipelineController::preview,this,[this](QImage l,QImage r,bool rectified) {
        // Ignore queued frames from the mode that was selected before a toggle.
        if (rectified != rectified_->isChecked()) return;
        live_left_ = std::move(l); live_right_ = std::move(r);
        if (state_.live) showImages();
    });
    connect(import_,&QPushButton::clicked,this,[this] {
        const QString path = QFileDialog::getExistingDirectory(this,"Choose capture directory (left.png, right.png, calibration.json)");
        if (path.isEmpty()) return;
        const bool own_calibration = capture_calibration_->isChecked(); controller_.submit([=](auto& w) { w.importCapture(path,own_calibration); });
    });
    connect(camera_,&QPushButton::clicked,this,[this] {
        const bool connected = state_.connected;
        controller_.submit([=](auto& w) { if (connected) w.disconnectCameras(); else w.connectCameras(); });
    });
    connect(preview_,&QPushButton::clicked,this,[this] { const bool live = state_.live; if (!live) { live_left_ = {}; live_right_ = {}; tabs_->setCurrentIndex(0); } controller_.submit([=](auto& w) { w.setLive(!live); }); });
    connect(capture_,&QPushButton::clicked,this,[this] { controller_.submit([](auto& w) { w.freeze(); }); });
    connect(run_,&QPushButton::clicked,this,[this] {
        reconstruction_valid_=false; refresh();
        const float low = minimum_->value(), high = maximum_->value();
        const QImage selected=mask_->mask();
        const bool denoise=denoise_->isChecked(); const float distance=neighbor_distance_->value();
        controller_.submit([=](auto& w) { w.reconstruct(low,high,selected,denoise,distance); });
    });
    const auto depth_settings_changed = [this] {
        // Changing parameters invalidates derived results even if the old values are restored later.
        reconstruction_valid_=false;
        refresh();
    };
    connect(minimum_,&QDoubleSpinBox::valueChanged,this,depth_settings_changed);
    connect(maximum_,&QDoubleSpinBox::valueChanged,this,depth_settings_changed);
    connect(denoise_,&QCheckBox::toggled,this,depth_settings_changed);
    connect(neighbor_distance_,&QDoubleSpinBox::valueChanged,this,depth_settings_changed);
    connect(rectified_,&QCheckBox::toggled,this,[this](bool enabled) {
        live_left_ = {}; live_right_ = {};
        controller_.setPreviewRectified(enabled);
        refresh(); showImages();
    }); connect(epilines_,&QCheckBox::toggled,this,[this] { showImages(); });
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
    connect(finish_draw_,&QPushButton::clicked,this,[this,depth_page] {
        mask_->acceptPrediction();
        if (mask_->hasSelection()) tabs_->setCurrentWidget(depth_page);
    });
    connect(mask_,&MaskEditor::promptsChanged,this,[this] {
        const auto prompts=mask_->prompts(); mask_request_id_=controller_.requestMask(state_.image_id,prompts);
        if (!prompts.empty()) status_->setText("Updating SAM mask…");
    });
    connect(mask_,&MaskEditor::selectionChanged,this,[this] { if (!mask_->hasSelection()) reconstruction_valid_=false; refresh(); }); connect(mask_,&MaskEditor::hint,status_,&QLabel::setText);
    auto* timer = new QTimer(this); timer->setInterval(200); connect(timer,&QTimer::timeout,this,[this] { if (busy_) time_->setText(QString("Running: %1 s").arg(elapsed_.elapsed()/1000.0,0,'f',1)); }); timer->start();
    connect(tabs_,&QTabWidget::currentChanged,this,[this] { refreshWorkflow(); });
    refresh();
}
void ReconstructionWindow::refresh() {
    const bool idle = !busy_ && !closing_, frozen = state_.has_rectified && !state_.live;
    import_->setEnabled(idle); capture_calibration_->setEnabled(idle); camera_->setEnabled(idle);
    camera_->setText(state_.connected ? "Disconnect cameras" : "Connect cameras");
    preview_->setEnabled(idle && state_.connected); preview_->setText(state_.live ? "Pause preview" : "Resume preview"); capture_->setEnabled(idle && state_.live);
    minimum_->setEnabled(idle); maximum_->setEnabled(idle);
    denoise_->setEnabled(idle); neighbor_distance_->setEnabled(idle && denoise_->isChecked());
    const bool depth_valid = minimum_->value() < maximum_->value();
    run_->setEnabled(idle && frozen && state_.engine_ready && depth_valid && mask_->hasSelection());
    run_->setToolTip(!state_.engine_ready ? "FoundationStereo is not initialized." : !depth_valid ? "Minimum depth must be less than maximum." : !frozen ? "Import or capture a frozen stereo pair first." : !mask_->hasSelection() ? "Click Finish draw in Region measurement first." : "Runs full-image XYZ and optional neighbourhood filtering inside the mask, then displays the Jet depth map.");
    rectified_->setEnabled(!closing_ && (state_.live || state_.has_pair));
    epilines_->setEnabled(!closing_ && rectified_->isChecked() && (state_.live || state_.has_rectified));
    mask_->setEditingEnabled(idle && frozen);
    for (auto* b : {brush_,eraser_}) b->setEnabled(idle && frozen);
    brush_size_->setEnabled(idle && frozen);
    for (auto* b : {sam_box_,sam_foreground_,sam_background_,sam_remove_}) b->setEnabled(idle && frozen && state_.sam_ready);
    sam_undo_->setEnabled(idle && frozen && mask_->hasPrompts());
    finish_draw_->setEnabled(idle && frozen && mask_->hasPrediction());
    clear_->setEnabled(idle && (mask_->hasPrompts() || mask_->hasPrediction()));
    input_->setText(state_.input); calibration_->setText(state_.calibration);
    if (!closing_) status_->setText(!busy_ && state_.depth_ready && !reconstruction_valid_
        ? "Selection or post-processing settings changed. Confirm the mask and reconstruct again." : state_.status);
    if (!state_.depth_ready || !reconstruction_valid_) {
        depth_left_->setImage({}); depth_map_->setImage({});
        depth_status_->setText("Jet uses the Minimum and Maximum depth settings for each reconstruction.");
    }
    const bool result_current=reconstruction_valid_ && state_.depth_ready && mask_->hasSelection();
    steps_->setText(QString("%1 Rectification   →   %2 FS inference   →   %3 GPU XYZ   →   %4 Depth map   →   Mesh / area · pending")
        .arg(state_.has_rectified ? "✓" : "○", result_current ? "✓" : "○", result_current ? "✓" : "○", result_current ? "✓" : "○"));
    refreshWorkflow();
    progress_->setRange(0,busy_ ? 0 : 4); if (!busy_) progress_->setValue(result_current ? 4 : state_.has_rectified ? 1 : 0);
    scene_status_->setText((result_current ? "GPU XYZ is ready\n\n" : QString()) + "3D viewport · pending\n\nReserved for VTK point cloud, mesh and wireframe display.\n3D geometry and viewer are not connected yet.\n\nRotation · zoom · pan · reset view");
}
void ReconstructionWindow::refreshWorkflow() {
    const bool pair_done=state_.has_rectified && !state_.live;
    const bool region_done=pair_done && mask_->hasSelection();
    const bool depth_done=region_done && state_.depth_ready && reconstruction_valid_;
    const bool completed[]={pair_done,region_done,depth_done,false}; // 3D rendering is still pending.
    const int last_available=depth_done ? 3 : region_done ? 2 : pair_done ? 1 : 0;
    if (tabs_->currentIndex()>last_available) tabs_->setCurrentIndex(last_available);
    tabs_->setTabEnabled(1,pair_done);
    tabs_->setTabEnabled(2,region_done);
    tabs_->setTabEnabled(3,depth_done);
    for (int i=0;i<4;++i) {
        auto* indicator=tab_status_[i];
        indicator->setText(completed[i] ? "✓" : "●");
        indicator->setStyleSheet(QString("background: transparent; color: %1; font-size: 16px; font-weight: 600;")
            .arg(completed[i] ? "#16a34a" : "#eab308"));
        indicator->setToolTip(completed[i] ? "Completed" : "Pending");
        indicator->setAccessibleName(completed[i] ? "Completed" : "Pending");
    }
    tabs_->setTabToolTip(1,pair_done ? "Draw and confirm a mask." : "Capture or import a stereo pair first.");
    tabs_->setTabToolTip(2,depth_done ? "Depth map ready." : region_done ? "Set the depth range and click Reconstruct." : "Click Finish draw in Region measurement first.");
    tabs_->setTabToolTip(3,depth_done ? "3D rendering is pending implementation." : "Complete depth reconstruction first.");
    capture_->setVisible(tabs_->currentIndex()==0);
    finish_draw_->setVisible(tabs_->currentIndex()==1);
    run_->setVisible(tabs_->currentIndex()==2);
    for (auto* b : {import_,camera_,preview_}) b->setVisible(tabs_->currentIndex()==0);
}
void ReconstructionWindow::showImages() {
    const bool corrected = rectified_->isChecked();
    left_->setImage(state_.live ? live_left_ : corrected ? rectified_left_ : raw_left_);
    right_->setImage(state_.live ? live_right_ : corrected ? rectified_right_ : raw_right_);
    left_->setEpilines(corrected && epilines_->isChecked()); right_->setEpilines(corrected && epilines_->isChecked());
}
void ReconstructionWindow::closeEvent(QCloseEvent* event) {
    if (allow_close_) { event->accept(); return; }
    event->ignore(); if (closing_) return;
    closing_ = true; mask_request_id_=controller_.requestMask(state_.image_id,{}); refresh(); status_->setText("Closing · waiting for background work and releasing cameras / GPU…"); emit closeRequested();
}
void ReconstructionWindow::allowClose() { allow_close_ = true; close(); }
