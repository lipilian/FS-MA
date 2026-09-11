#include "CalibrationWindow.hpp"
#include "widgets/StereoImageView.hpp"
#include <QCheckBox>
#include <QCloseEvent>
#include <QComboBox>
#include <QDoubleSpinBox>
#include <QFileDialog>
#include <QFormLayout>
#include <QGroupBox>
#include <QHBoxLayout>
#include <QLabel>
#include <QListWidget>
#include <QMessageBox>
#include <QProgressBar>
#include <QPushButton>
#include <QScrollArea>
#include <QSignalBlocker>
#include <QSpinBox>
#include <QSplitter>
#include <QStandardPaths>
#include <QVBoxLayout>

namespace {
QLabel* label(const QString& text, QWidget* parent = nullptr) {
    auto* result = new QLabel(text, parent); result->setWordWrap(true); return result;
}
QPushButton* button(const QString& text, const char* name) {
    auto* b = new QPushButton(text); b->setObjectName(name); b->setMinimumHeight(34); return b;
}
QString qualityMetrics(const fs::calibration::Quality& quality) {
    const auto value = [](const QString& name, double rms) {
        const auto color = rms <= fs::calibration::kQualityThresholdPx ? "#15803d" : "#dc2626";
        return QString("<span style=\"color:%1\">%2: %3 px</span>")
            .arg(color, name).arg(rms, 0, 'f', 3);
    };
    return value("Left", quality.left) + "<br>" + value("Right", quality.right) + "<br>" + value("Stereo", quality.stereo);
}
QString qualityText(const CalibrationState& state) {
    const QString threshold = QString("<br><br>Stereo RMS gate: %1 px (fixed)")
        .arg(fs::calibration::kQualityThresholdPx, 0, 'f', 2);
    if (!state.solve_quality) return "No calibration yet.<br>Capture varied positions and tilts." + threshold;
    QString text = QString("<b>SOLVE · %1 pairs</b><br>%2<br><br><b>INDEPENDENT CHECK</b><br>%3")
        .arg(state.solve_quality->pairs).arg(qualityMetrics(*state.solve_quality))
        .arg(state.check_quality ? qualityMetrics(*state.check_quality) : "Not checked in this connection.");
    text += threshold;
    if (state.can_reuse_saved) text += "<br>Saved result available for reuse.";
    if (state.check_quality) {
        const bool passed = state.solve_quality->stereo <= fs::calibration::kQualityThresholdPx &&
                            state.check_quality->stereo <= fs::calibration::kQualityThresholdPx;
        text += passed ? (state.saved_path.isEmpty() ? "<br>PASS · save to finish" : "<br>PASS · saved")
                       : "<br>ABOVE THRESHOLD · collect better samples";
    }
    return text;
}
QDoubleSpinBox* decimal(double value, double maximum, const QString& suffix, int decimals = 2) {
    auto* s = new QDoubleSpinBox; s->setDecimals(decimals); s->setRange(0.001, maximum);
    s->setValue(value); s->setSuffix(suffix); return s;
}
}
CalibrationWindow::CalibrationWindow(CalibrationController& controller) : controller_(controller) {
    setWindowTitle("FoundationStereo · Stereo Calibration"); resize(1440, 900); setMinimumSize(1120, 740);
    setStyleSheet(R"(
        QMainWindow, QWidget#root { background: #f3f6fa; color: #20314a; }
        QWidget { font-family: 'Noto Sans', sans-serif; font-size: 13px; }
        QGroupBox { background: white; border: 1px solid #dce3ed; border-radius: 8px; margin-top: 15px; padding: 16px 12px 12px; font-weight: 600; }
        QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 6px; }
        QPushButton { background: white; color: #243c60; border: 1px solid #cbd6e5; border-radius: 5px; padding: 5px 12px; }
        QPushButton:hover { background: #eaf1fc; border-color: #6c97cf; }
        QPushButton:disabled { color: #9aa8b9; background: #edf1f6; border-color: #dfe5ed; }
        QPushButton#capture, QPushButton#finish { background: #2464d9; color: white; border: 0; }
        QPushButton#capture:disabled, QPushButton#finish:disabled { background: #c3d1e6; color: #f3f6fa; }
        QSpinBox, QDoubleSpinBox, QComboBox { min-height: 28px; border: 1px solid #ced8e5; border-radius: 4px; background: white; color: #20314a; padding-left: 5px; }
        QListWidget { background: white; border: 1px solid #dce3ed; border-radius: 5px; color: #20314a; }
        QListWidget::item { padding: 9px 5px; border-bottom: 1px solid #eef2f7; }
        QListWidget::item:selected { background: #e4edff; color: #174b9e; }
        QLabel#title { font-size: 27px; font-weight: 700; color: #192c49; }
        QLabel#step { font-size: 11px; font-weight: 700; color: #2464d9; }
        QLabel#status { padding: 12px; background: #e5edfa; border-radius: 5px; color: #234772; }
        QProgressBar { border: 0; background: #e5ebf4; border-radius: 3px; min-height: 8px; max-height: 8px; }
        QProgressBar::chunk { background: #2464d9; border-radius: 3px; }
    )");
    auto* root = new QWidget; root->setObjectName("root"); setCentralWidget(root);
    auto* outer = new QVBoxLayout(root); outer->setContentsMargins(22, 18, 22, 18); outer->setSpacing(12);
    auto* step = label("01  /  CALIBRATION     →     02  /  RECONSTRUCTION"); step->setObjectName("step"); outer->addWidget(step);
    auto* heading = new QHBoxLayout;
    auto* title = label("Stereo calibration"); title->setObjectName("title"); heading->addWidget(title); heading->addStretch();
    load_ = button("Load calibration…", "load"); save_ = button("Save calibration…", "save");
    heading->addWidget(load_); heading->addWidget(save_); outer->addLayout(heading);
    outer->addWidget(label("Connect → capture varied board poses → compute → check with a new pose → save."));
    auto* resume = new QHBoxLayout;
    reuse_hint_ = label(""); reuse_hint_->setTextFormat(Qt::PlainText);
    reuse_hint_->setStyleSheet("color: #12634b; font-weight: 600;");
    reuse_ = button("Skip calibration · Continue", "reuseCalibration");
    reuse_->setToolTip("Use the loaded calibration without a new capture or check.");
    resume->addWidget(reuse_hint_, 1); outer->addLayout(resume);
    auto* split = new QSplitter;
    auto* settings = new QWidget; auto* settings_layout = new QVBoxLayout(settings); settings_layout->setContentsMargins(0,0,8,0);
    auto* camera = new QGroupBox("Stereo cameras"); auto* camera_layout = new QVBoxLayout(camera);
    camera_layout->addWidget(label("LEFT   ·   21LJ548\nRIGHT ·   21LJ530\nSentech STC-MCS500U3V"));
    auto* exposure_form = new QFormLayout;
    exposure_ = decimal(SentechStereoOptions{}.exposure_us, 1000000, " µs", 0); exposure_->setMinimum(100);
    exposure_form->addRow("Exposure", exposure_); camera_layout->addLayout(exposure_form);
    connect_ = button("Connect cameras", "connectCameras"); camera_layout->addWidget(connect_);
    camera_layout->addWidget(label("Independent streams. Hold the board still during each capture.")); settings_layout->addWidget(camera);
    board_group_ = new QGroupBox("ChArUco board"); auto* board_layout = new QVBoxLayout(board_group_); auto* form = new QFormLayout;
    squares_x_ = new QSpinBox; squares_x_->setRange(3, 30); squares_x_->setValue(10);
    squares_y_ = new QSpinBox; squares_y_->setRange(3, 30); squares_y_->setValue(8);
    square_mm_ = decimal(66.5, 1000, " mm", 3); marker_mm_ = decimal(50.5, 1000, " mm", 3);
    dictionary_ = new QComboBox;
    for (const auto& name : fs::calibration::dictionary_names()) dictionary_->addItem(QString::fromStdString(name));
    dictionary_->setCurrentText("DICT_4X4_250");
    form->addRow("Squares X", squares_x_); form->addRow("Squares Y", squares_y_);
    form->addRow("Square side", square_mm_); form->addRow("Marker side", marker_mm_); form->addRow("Dictionary", dictionary_);
    board_layout->addLayout(form); apply_ = button("Apply board parameters", "applyBoard"); board_layout->addWidget(apply_);
    board_layout->addWidget(label("Match the physical board. Applying a different board clears samples and calibration."));
    settings_layout->addWidget(board_group_);
    settings_layout->addStretch();
    auto* settings_scroll = new QScrollArea; settings_scroll->setWidgetResizable(true); settings_scroll->setFrameShape(QFrame::NoFrame);
    settings_scroll->setWidget(settings); settings_scroll->setMinimumWidth(270); split->addWidget(settings_scroll);
    auto* center = new QWidget; auto* center_layout = new QVBoxLayout(center); center_layout->setContentsMargins(8,0,8,0);
    auto* view_controls = new QHBoxLayout;
    overlay_ = new QCheckBox("Detection overlay"); overlay_->setChecked(true);
    overlay_->setToolTip("Raw view: detected marker outlines and IDs, plus ChArUco corners and IDs.");
    rectified_ = new QCheckBox("Rectified + epilines"); live_ = button("Return to live", "live");
    view_controls->addWidget(overlay_); view_controls->addWidget(rectified_); view_controls->addStretch(); view_controls->addWidget(live_);
    center_layout->addLayout(view_controls);
    auto* images = new QSplitter;
    left_ = new StereoImageView("LEFT  /  21LJ548"); right_ = new StereoImageView("RIGHT  /  21LJ530");
    images->addWidget(left_); images->addWidget(right_); center_layout->addWidget(images, 1);
    caption_ = label("No camera frames yet."); caption_->setMinimumHeight(45); center_layout->addWidget(caption_);
    auto* actions = new QHBoxLayout;
    capture_ = button("Capture sample", "capture"); compute_ = button("Compute", "compute"); check_ = button("Check · new pose", "check");
    actions->addWidget(capture_); actions->addWidget(compute_); actions->addWidget(check_); center_layout->addLayout(actions);
    center_layout->addWidget(label("At least 3 valid pairs are needed to solve. Use more varied poses and cover the image; 3 pairs alone do not ensure a good calibration."));
    split->addWidget(center);
    auto* history = new QWidget; auto* history_layout = new QVBoxLayout(history); history_layout->setContentsMargins(8,0,0,0);
    sample_count_ = label("SAMPLES  ·  0 / 20"); history_layout->addWidget(sample_count_);
    samples_ = new QListWidget; samples_->setObjectName("samples"); history_layout->addWidget(samples_, 1);
    remove_ = button("Delete selected sample", "removeSample"); history_layout->addWidget(remove_);
    auto* quality_group = new QGroupBox("Calibration quality"); auto* ql = new QVBoxLayout(quality_group);
    quality_ = label(""); quality_->setObjectName("calibrationQuality"); quality_->setTextFormat(Qt::RichText); quality_->setTextInteractionFlags(Qt::TextSelectableByMouse); ql->addWidget(quality_);
    history_layout->addWidget(quality_group); history->setMinimumWidth(250); split->addWidget(history);
    split->setSizes({285, 790, 290}); split->setStretchFactor(1, 1); outer->addWidget(split, 1);
    progress_ = new QProgressBar; progress_->setRange(0,5); progress_->setValue(0); progress_->setTextVisible(false); outer->addWidget(progress_);
    status_ = label(""); status_->setObjectName("status"); status_->setMinimumHeight(46); outer->addWidget(status_);
    saved_ = label("Calibration has not been saved."); saved_->setTextInteractionFlags(Qt::TextSelectableByMouse); outer->addWidget(saved_);
    auto* footer = new QHBoxLayout;
    cancel_ = button("Cancel capture", "cancelCapture"); footer->addWidget(cancel_);
    footer->addWidget(label("Complete calibration to open the reconstruction workspace."), 1);
    footer->addWidget(reuse_);
    finish_ = button("Finish calibration", "finish"); footer->addWidget(finish_); outer->addLayout(footer);

    connect(&controller_, &CalibrationController::actionFinished, this, [this] { awaiting_ = false; refreshActions(); });
    connect(&controller_, &CalibrationController::stateChanged, this, &CalibrationWindow::updateState);
    connect(&controller_, &CalibrationController::preview, this, [this](QImage l, QImage r, const QString& text) {
        if (closing_) return;
        left_->setImage(std::move(l)); right_->setImage(std::move(r)); caption_->setText(text);
    });
    connect(&controller_, &CalibrationController::failed, this, [this](const QString& message) {
        if (!closing_) { status_->setText(message); status_->setToolTip(message); }
    });
    connect(connect_, &QPushButton::clicked, this, [this] {
        const double exposure = exposure_->value(); const bool connected = state_.connected;
        dispatch([=](auto& w) { if (connected) w.disconnectCameras(); else w.connectCameras(exposure); });
    });
    connect(apply_, &QPushButton::clicked, this, [this] {
        if ((!state_.samples.empty() || state_.has_result) && QMessageBox::question(this, "Change board?",
            "Applying a different board clears current samples and calibration. Continue?") != QMessageBox::Yes) return;
        const auto board = boardFromForm(); dispatch([board](auto& w) { w.applyBoard(board); });
    });
    connect(capture_, &QPushButton::clicked, this, [this] { dispatch([](auto& w) { w.capture(false); }); });
    connect(check_, &QPushButton::clicked, this, [this] { dispatch([](auto& w) { w.capture(true); }); });
    connect(compute_, &QPushButton::clicked, this, [this] { dispatch([](auto& w) { w.compute(); }); });
    connect(cancel_, &QPushButton::clicked, this, [this] { dispatch([](auto& w) { w.cancelCapture(); }); });
    connect(reuse_, &QPushButton::clicked, this, [this] {
        const double exposure = exposure_->value(); dispatch([exposure](auto& w) { w.reuseSavedCalibration(exposure); });
    });
    connect(finish_, &QPushButton::clicked, this, [this] { dispatch([](auto& w) { w.finish(); }); });
    connect(live_, &QPushButton::clicked, this, [this] { dispatch([](auto& w) { w.selectSample(-1); }); });
    connect(remove_, &QPushButton::clicked, this, [this] {
        const int index = samples_->currentRow(); dispatch([index](auto& w) { w.deleteSample(index); });
    });
    connect(samples_, &QListWidget::currentRowChanged, this, [this](int index) {
        dispatch([index](auto& w) { w.selectSample(index); });
    });
    const auto display = [this] {
        const bool overlay = overlay_->isChecked(), rectified = rectified_->isChecked();
        dispatch([=](auto& w) { w.setDisplay(overlay, rectified); });
    };
    connect(overlay_, &QCheckBox::toggled, this, display); connect(rectified_, &QCheckBox::toggled, this, display);
    connect(save_, &QPushButton::clicked, this, [this] {
        const QString suggested = state_.suggested_save_path.isEmpty() ? "sentech_stereo_calibration.json" : state_.suggested_save_path;
        const QString path = QFileDialog::getSaveFileName(this, "Save calibration", suggested, "Calibration JSON (*.json)");
        if (!path.isEmpty()) dispatch([path](auto& w) { w.save(path); });
    });
    connect(load_, &QPushButton::clicked, this, [this] {
        if ((!state_.samples.empty() || state_.has_result) && QMessageBox::question(this, "Load calibration?",
            "A successful load replaces current samples and calibration. Continue?") != QMessageBox::Yes) return;
        const QString path = QFileDialog::getOpenFileName(this, "Load calibration", {}, "Calibration JSON (*.json)");
        if (!path.isEmpty()) dispatch([path](auto& w) { w.load(path); });
    });
    for (auto* spin : {squares_x_, squares_y_}) connect(spin, &QSpinBox::valueChanged, this, [this] { refreshActions(); });
    for (auto* spin : {square_mm_, marker_mm_}) connect(spin, &QDoubleSpinBox::valueChanged, this, [this] { refreshActions(); });
    connect(dictionary_, &QComboBox::currentTextChanged, this, [this] { refreshActions(); });
    updateState({});
    dispatch([](auto& w) { w.restoreSavedCalibration(); });
}
fs::calibration::BoardConfig CalibrationWindow::boardFromForm() const {
    return {squares_x_->value(), squares_y_->value(), square_mm_->value()/1000.0, marker_mm_->value()/1000.0,
        dictionary_->currentText().toStdString()};
}
void CalibrationWindow::dispatch(std::function<void(CalibrationWorker&)> action) {
    awaiting_ = true; refreshActions(); controller_.submit(std::move(action));
}
void CalibrationWindow::updateState(CalibrationState state) {
    if (closing_) return;
    state_ = std::move(state);
    QStringList current;
    for (int i = 0; i < samples_->count(); ++i) current << samples_->item(i)->text();
    const QSignalBlocker block(samples_);
    if (current != state_.samples) { samples_->clear(); samples_->addItems(state_.samples); }
    samples_->setCurrentRow(state_.selected >= 0 ? state_.selected : -1);
    sample_count_->setText(QString("SAMPLES  ·  %1 / 20").arg(state_.samples.size()));
    status_->setText(state_.status); quality_->setText(qualityText(state_));
    saved_->setText(state_.saved_path.isEmpty() ? "Current calibration has not been saved." : "Saved: " + state_.saved_path);
    reuse_hint_->setText(state_.can_reuse_saved ? "Saved calibration loaded — you can skip this step.\n" + state_.saved_path : "");
    reuse_hint_->setVisible(state_.can_reuse_saved);
    reuse_->setVisible(state_.can_reuse_saved);
    progress_->setValue(state_.busy ? state_.candidates : 0); refreshActions();
}
void CalibrationWindow::refreshActions() {
    // Disabling a focused button can move focus into the history list and make
    // Qt select its first row. UI refreshes must not enqueue a sample preview.
    const int selected_row = samples_->currentRow();
    const QSignalBlocker block(samples_);
    const bool idle = !awaiting_ && !state_.busy && !closing_;
    const bool dirty = !fs::calibration::same_board(boardFromForm(), state_.board);
    connect_->setEnabled(idle); connect_->setText(state_.connected ? "Disconnect cameras" : "Connect cameras");
    exposure_->setEnabled(idle && !state_.connected); board_group_->setEnabled(idle); apply_->setEnabled(idle && dirty);
    load_->setEnabled(idle && !dirty); save_->setEnabled(idle && !dirty && state_.has_result);
    capture_->setEnabled(idle && !dirty && state_.connected && state_.samples.size() < 20);
    compute_->setEnabled(idle && !dirty && state_.samples.size() >= 3);
    check_->setEnabled(idle && !dirty && state_.connected && state_.has_result);
    finish_->setEnabled(idle && !dirty && state_.can_finish);
    reuse_->setEnabled(idle && !dirty && state_.can_reuse_saved);
    cancel_->setEnabled(!closing_ && state_.busy && state_.candidates <= 5 && state_.status.contains("andidat", Qt::CaseInsensitive));
    samples_->setEnabled(idle); samples_->setCurrentRow(selected_row);
    remove_->setEnabled(idle && selected_row >= 0);
    live_->setEnabled(idle && state_.connected); overlay_->setEnabled(idle); rectified_->setEnabled(idle && state_.has_result);
    if (dirty && idle) status_->setText("Board edits are pending. Apply parameters before capture, calibration or saving.");
}
void CalibrationWindow::closeEvent(QCloseEvent* event) {
    if (allow_close_) { event->accept(); return; }
    event->ignore();
    if (closing_) return;
    closing_ = true; refreshActions(); status_->setText("Closing — waiting for background work and releasing cameras…");
    emit closeRequested();
}
void CalibrationWindow::allowClose() { allow_close_ = true; close(); }
