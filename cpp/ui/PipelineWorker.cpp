#include "PipelineWorker.hpp"
#include <QDir>
#include <QFileInfo>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <stdexcept>

namespace {
QImage image(const cv::Mat& rgb) {
    return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
}
QImage thumbnail(const cv::Mat& rgb) {
    cv::Mat small;
    cv::resize(rgb, small, {}, std::min(1.0, 960.0 / rgb.cols), std::min(1.0, 960.0 / rgb.cols));
    return image(small);
}
}
PipelineWorker::PipelineWorker(ConfirmedCalibration calibration, QString path, SharedStereoSource source,
                               std::shared_ptr<std::atomic_bool> cancel)
    : confirmed_(std::move(calibration)), confirmed_path_(std::move(path)), source_(std::move(source)),
      cancel_(std::move(cancel)), timer_(new QTimer(this)) {
    state_.calibration = confirmed_path_;
    timer_->setInterval(100);
    connect(timer_, &QTimer::timeout, this, &PipelineWorker::poll);
}
void PipelineWorker::publish() { emit stateChanged(state_); }
void PipelineWorker::checkpoint() const {
    if (cancel_->load()) throw std::runtime_error("Stopped at a safe stage boundary.");
}
void PipelineWorker::execute(const std::function<void(PipelineWorker&)>& action) {
    try { checkpoint(); action(*this); }
    catch (const std::exception& e) {
        state_.status = QString::fromUtf8(e.what());
        emit log(state_.status);
    }
    publish(); emit actionFinished();
}
void PipelineWorker::initialize() {
    if (source_ && source_->running()) {
        state_.connected = true; state_.live = true; last_pair_.start(); timer_->start();
        state_.status = "Camera session received from calibration. Previewing raw RGB; capture to freeze a pair.";
    }
    emit log("Confirmed calibration: " + confirmed_path_);
}
void PipelineWorker::prepare(std::unique_ptr<StereoFrame> frame, QString input, QString calibration) {
    QElapsedTimer elapsed; elapsed.start();
    frame->rectify(); checkpoint();
    // Commit together only after validation and rectification succeed. Failed imports retain the previous input.
    frame_ = std::move(frame); state_.input = std::move(input); state_.calibration = std::move(calibration);
    state_.has_pair = true; state_.has_rectified = true; state_.gpu_ready = false; state_.stage = 1;
    state_.live = false; latest_.reset();
    emit images(image(frame_->left()), image(frame_->right()), image(frame_->rectified_left()), image(frame_->rectified_right()));
    state_.status = QString("Pair ready · %1 × %2 · rectified in %3 ms. Check epilines, then start reconstruction.")
        .arg(frame_->left().cols).arg(frame_->left().rows).arg(elapsed.elapsed());
    emit log(state_.input + "\nCalibration: " + state_.calibration + "\n" + state_.status);
}
void PipelineWorker::importCapture(const QString& directory, bool use_capture_calibration) {
    const QDir dir(directory);
    state_.status = "Loading and rectifying capture…"; publish();
    const auto calibration_path = dir.filePath("calibration.json");
    std::unique_ptr<StereoFrame> frame;
    QString description;
    if (use_capture_calibration) {
        if (!QFileInfo::exists(calibration_path))
            throw std::runtime_error("The directory has no calibration.json. Disable 'Use capture calibration' to use the confirmed calibration.");
        frame = std::make_unique<StereoFrame>(dir.filePath("left.png").toStdString(), dir.filePath("right.png").toStdString(), calibration_path.toStdString());
        // Legacy captures are supported like the CLI. Enforce grid metadata when it is present.
        cv::FileStorage metadata(calibration_path.toStdString(), cv::FileStorage::READ);
        const auto width = metadata["image_width"], height = metadata["image_height"];
        if ((!width.empty() || !height.empty()) &&
            (width.empty() || height.empty() || int(width) != frame->left().cols || int(height) != frame->left().rows))
            throw std::runtime_error("Capture images do not match calibration image dimensions.");
        description = calibration_path + (width.empty() ? "\nLegacy capture: image grid / camera identity not verified" : "\nCapture calibration (separate from live cameras)");
    } else {
        // Use the handed-off matrices even if the original JSON has since changed or disappeared.
        cv::Mat left = cv::imread(dir.filePath("left.png").toStdString());
        cv::Mat right = cv::imread(dir.filePath("right.png").toStdString());
        if (left.empty() || right.empty()) throw std::runtime_error("Capture must contain readable left.png and right.png.");
        cv::cvtColor(left, left, cv::COLOR_BGR2RGB); cv::cvtColor(right, right, cv::COLOR_BGR2RGB);
        if (left.size() != confirmed_->image_size || right.size() != confirmed_->image_size)
            throw std::runtime_error("Capture size differs from the confirmed calibration. Supply this capture's calibration.json.");
        frame = std::make_unique<StereoFrame>(left, right, confirmed_->calibration);
        description = confirmed_path_ + "\nConfirmed calibration · matching image dimensions";
    }
    checkpoint(); prepare(std::move(frame), dir.absolutePath(), description);
}
void PipelineWorker::connectCameras(double exposure_us) {
    if (state_.connected) { setLive(true); return; }
    if (source_) { source_->stop(); source_.reset(); }
    SentechStereoOptions options;
    options.left = confirmed_->left_serial; options.right = confirmed_->right_serial; options.exposure_us = exposure_us;
    auto next = std::make_shared<SentechStereoSource>(options);
    state_.status = "Connecting stereo cameras…"; publish();
    next->start(); source_ = std::move(next); state_.connected = true; setLive(true); timer_->start();
}
void PipelineWorker::disconnectCameras() {
    timer_->stop();
    if (source_) source_->stop();
    source_.reset(); latest_.reset(); state_.connected = false; state_.live = false;
    state_.status = "Cameras disconnected. Frozen input is retained.";
}
void PipelineWorker::setLive(bool enabled) {
    if (enabled && (!source_ || !source_->running())) throw std::runtime_error("Cameras are not connected.");
    state_.live = enabled; latest_.reset();
    if (enabled) { last_pair_.restart(); state_.status = "Live raw RGB · capture to freeze a stereo pair. Independent streams; hold the subject still."; }
    else state_.status = "Preview paused. Showing the frozen input, if available.";
}
void PipelineWorker::poll() {
    if (!state_.live) return;
    try {
        auto pair = source_->wait_for_pair(std::chrono::milliseconds(0));
        if (!pair) {
            if (!source_->running() || last_pair_.elapsed() > 5000) throw std::runtime_error("No stereo pair for 5 seconds. Check cameras and reconnect.");
            return;
        }
        if (pair->left.rgb.size() != confirmed_->image_size || pair->right.rgb.size() != confirmed_->image_size)
            throw std::runtime_error("Camera image dimensions do not match the confirmed calibration.");
        last_pair_.restart(); latest_ = std::move(pair);
        emit preview(thumbnail(latest_->left.rgb), thumbnail(latest_->right.rgb));
    } catch (const std::exception& e) {
        disconnectCameras(); state_.status = QString::fromUtf8(e.what()); publish(); emit log(state_.status);
    }
}
void PipelineWorker::freeze() {
    if (!state_.live || !latest_ || last_pair_.elapsed() > 1000)
        throw std::runtime_error("No fresh stereo pair. Wait for the live preview and capture again.");
    auto frame = std::make_unique<StereoFrame>(latest_->left.rgb, latest_->right.rgb, confirmed_->calibration);
    const QString input = QString("Camera capture · L #%1 / R #%2 · host gap %3 ms")
        .arg(latest_->left.frame_id).arg(latest_->right.frame_id)
        .arg(fs::calibration::arrival_skew(*latest_) / 1e6, 0, 'f', 1);
    prepare(std::move(frame), input, confirmed_path_);
}
void PipelineWorker::reconstruct(const QString& engine_path, float minimum, float maximum) {
    if (!frame_ || state_.live) throw std::runtime_error("Import or freeze a stereo pair before reconstruction.");
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum < 0 || maximum <= minimum)
        throw std::runtime_error("Depth range must satisfy 0 ≤ minimum < maximum (metres).");
    state_.gpu_ready = false; state_.stage = 1; publish();
    checkpoint();
    if (!fs_ || loaded_engine_ != engine_path) {
        state_.engine = "Loading TensorRT engine…"; state_.status = state_.engine; publish();
        QElapsedTimer elapsed; elapsed.start();
        try {
            auto next = std::make_unique<FS>(); next->loadEngine(engine_path.toStdString());
            fs_ = std::move(next); loaded_engine_ = engine_path;
        } catch (...) { state_.engine = "Engine unavailable · check the file, GPU and TensorRT compatibility"; throw; }
        state_.engine = "Loaded · 960 × 800 · " + QFileInfo(engine_path).fileName();
        emit log(QString("Engine loaded in %1 ms").arg(elapsed.elapsed()));
    }
    state_.engine = "Loaded · 960 × 800 · " + QFileInfo(loaded_engine_).fileName();
    checkpoint(); state_.status = "Running FoundationStereo inference…"; publish();
    QElapsedTimer elapsed; elapsed.start();
    fs_->set_model_camera_parameters(frame_->rectified_camera_parameters(), frame_->rectified_left().size());
    fs_->prepare_stereo_images(frame_->rectified_left(), frame_->rectified_right());
    fs_->inference(); fs_->synchronize();
    state_.stage = 2; emit log(QString("Inference + input preparation: %1 ms").arg(elapsed.elapsed()));
    checkpoint(); state_.status = "Computing XYZ on GPU…"; publish(); elapsed.restart();
    fs_->compute_xyz_map(minimum, maximum); checkpoint(); state_.stage = 3; state_.gpu_ready = true;
    emit log(QString("GPU XYZ: %1 ms · depth %2–%3 m").arg(elapsed.elapsed()).arg(minimum).arg(maximum));
    state_.status = "GPU XYZ complete. CPU download, depth map, VTK, mesh and area are pending; no 3D or area result is available yet.";
    emit log(state_.status);
}
void PipelineWorker::shutdown() {
    disconnectCameras(); fs_.reset(); frame_.reset(); emit stopped();
}
