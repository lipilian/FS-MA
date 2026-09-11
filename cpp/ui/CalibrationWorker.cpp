#include "CalibrationWorker.hpp"
#include <QSaveFile>
#include <QCoreApplication>
#include <QDir>
#include <QFileInfo>
#include <QSettings>
#include <QThread>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <stdexcept>
#include <cmath>
#include <unistd.h>

namespace cal = fs::calibration;
namespace {
QString defaultCalibrationPath() {
    return QDir::current().absoluteFilePath("sentech_stereo_calibration.json");
}
QString calibrationSettingsPath() {
    // Linux-only application: keep temporary path history separate for each user.
    return QString("/tmp/FS_Engine-%1/fs_gui.conf").arg(getuid());
}
QString lastCalibrationPath() {
    QSettings settings(calibrationSettingsPath(), QSettings::IniFormat);
    return settings.value("calibration/lastFile").toString();
}
void rememberCalibrationPath(const QString& path) {
    QSettings settings(calibrationSettingsPath(), QSettings::IniFormat);
    settings.setValue("calibration/lastFile", path);
    settings.sync();
}
QImage displayImage(const cv::Mat& rgb) {
    cv::Mat small;
    const double scale = std::min(1.0, 1000.0 / rgb.cols);
    cv::resize(rgb, small, {}, scale, scale, cv::INTER_AREA);
    return QImage(small.data, small.cols, small.rows, small.step, QImage::Format_RGB888).copy();
}

}
CalibrationWorker::CalibrationWorker(SourceFactory factory, QObject* parent) : QObject(parent), source_factory_(std::move(factory)), timer_(new QTimer(this)), board_(cal::make_board(state_.board)) {
    if (!source_factory_) source_factory_ = [](double exposure_us) {
        SentechStereoOptions options; options.exposure_us = exposure_us;
        return std::make_unique<SentechStereoSource>(options);
    };
    timer_->setInterval(150);
    connect(timer_, &QTimer::timeout, this, &CalibrationWorker::poll);
}
void CalibrationWorker::execute(const std::function<void(CalibrationWorker&)>& action) {
    if (shutting_down_) return;
    try { action(*this); }
    catch (const std::exception& e) {
        state_.busy = false;
        capture_ = Capture::None; candidate_.reset();
        state_.status = QString::fromUtf8(e.what());
        emit failed(state_.status);
    }
    publish();
    emit actionFinished();
}
bool CalibrationWorker::ready() const {
    return state_.connected && !state_.busy && result_ && result_->image_size == image_size_ &&
        result_->checked && result_->solve.stereo <= cal::kQualityThresholdPx && result_->check.stereo <= cal::kQualityThresholdPx &&
        !state_.saved_path.isEmpty();
}
bool CalibrationWorker::reusable() const {
    return reusable_saved_ && result_ && !state_.busy && QFileInfo(state_.saved_path).isFile() &&
        (!state_.connected || (image_size_.area() > 0 && result_->image_size == image_size_));
}
void CalibrationWorker::publish() {
    state_.has_result = result_.has_value(); state_.can_finish = ready();
    state_.can_reuse_saved = reusable();
    state_.samples.clear();
    for (std::size_t i = 0; i < samples_.size(); ++i) {
        const auto& s = samples_[i];
        state_.samples << QString("%1   L %2 / R %3   ·   %4 shared\n     Host arrival gap %5 ms")
            .arg(i + 1, 2, 10, QLatin1Char('0')).arg(s.left.ids.size()).arg(s.right.ids.size())
            .arg(cal::common_corners(s.left, s.right)).arg(cal::arrival_skew(s.images) / 1e6, 0, 'f', 1);
    }
    state_.solve_quality = result_ ? std::make_optional(result_->solve) : std::nullopt;
    state_.check_quality = result_ && result_->checked ? std::make_optional(result_->check) : std::nullopt;
    emit stateChanged(state_);
}
void CalibrationWorker::invalidate() {
    result_.reset(); check_sample_.reset(); state_.saved_path.clear(); reusable_saved_ = false;
    lx_.release(); ly_.release(); rx_.release(); ry_.release();
}
void CalibrationWorker::connectCameras(double exposure_us) {
    if (state_.connected) return;
    state_.busy = true; state_.status = "Opening left 21LJ548 and right 21LJ530…"; publish();
    source_ = source_factory_(exposure_us);
    if (!source_) throw std::runtime_error("Camera source is unavailable");
    try { source_->start(); } catch (...) { source_->stop(); source_.reset(); throw; }
    exposure_us_ = exposure_us;
    image_size_ = {}; latest_.reset();
    if (result_) result_->checked = false;
    state_.connected = true; state_.busy = false; state_.selected = -1;
    last_frame_clock_.start(); timer_->start();
    state_.status = "Connected. Waiting for the first stereo pair…";
}
void CalibrationWorker::disconnectCameras() {
    timer_->stop(); cancelCapture();
    if (source_) source_->stop();
    source_.reset(); state_.connected = false;
    if (result_) result_->checked = false;
    state_.status = reusable_saved_ ? "Cameras disconnected. You can continue with the loaded calibration."
        : "Cameras disconnected. Reconnect and run a fresh check before finishing.";
}
void CalibrationWorker::applyBoard(cal::BoardConfig config) {
    auto next = cal::make_board(config);
    if (!cal::same_board(config, state_.board)) {
        cancelCapture(); samples_.clear(); invalidate(); latest_.reset(); state_.selected = -1;
        emit preview({}, {}, "Board changed — waiting for new frames");
    }
    board_ = std::move(next); state_.board = config;
    state_.status = "Board parameters applied. Lengths are stored in metres.";
}
void CalibrationWorker::capture(bool for_check) {
    if (!state_.connected || !latest_) throw std::runtime_error("Wait for a stereo preview before capture");
    if (capture_ != Capture::None) throw std::runtime_error("Capture already in progress");
    if (for_check && !result_) throw std::runtime_error("Compute or load calibration first");
    if (!for_check && samples_.size() >= 20) throw std::runtime_error("20-pair limit reached. Delete a sample first.");
    if (for_check) { reusable_saved_ = false; result_->checked = false; state_.saved_path.clear(); check_sample_.reset(); }
    capture_ = for_check ? Capture::Check : Capture::Sample;
    candidate_.reset(); state_.candidates = 0; state_.busy = true; state_.selected = -1;
    (void)source_->wait_for_pair(std::chrono::milliseconds(0)); // Discard the already buffered pair.
    capture_clock_.start();
    state_.status = "Capturing 5 fresh candidates. Keep the board still; both cameras must see it.";
}
void CalibrationWorker::cancelCapture() {
    capture_ = Capture::None; candidate_.reset(); state_.busy = false; state_.candidates = 0;
    state_.status = "Capture cancelled.";
}
void CalibrationWorker::compute() {
    state_.busy = true; state_.status = "Solving camera intrinsics and stereo extrinsics…"; publish();
    invalidate();
    auto result = cal::solve(state_.board, samples_);
    result.threshold_px = cal::kQualityThresholdPx;
    result_ = std::move(result); buildMaps(); state_.busy = false;
    state_.status = "Calibration computed. Move the board to a new pose, then run an independent check.";
}
void CalibrationWorker::selectSample(int index) {
    if (index == -2 && check_sample_) { show(*check_sample_, "Independent check · frozen"); return; }
    state_.selected = index >= 0 && index < int(samples_.size()) ? index : -1;
    if (state_.selected >= 0) show(samples_[state_.selected], QString("Sample %1 · frozen").arg(state_.selected + 1));
    else if (latest_) show(*latest_, "Live preview");
}
void CalibrationWorker::deleteSample(int index) {
    if (index < 0 || index >= int(samples_.size())) return;
    samples_.erase(samples_.begin() + index); invalidate(); state_.selected = -1;
    state_.status = "Sample removed. Recompute calibration from the remaining samples.";
    if (latest_) show(*latest_, "Live preview");
}
void CalibrationWorker::setDisplay(bool detection, bool rectified) {
    detection_ = detection; rectified_ = rectified;
    selectSample(state_.selected);
}
void CalibrationWorker::buildMaps() {
    if (!result_) return;
    const auto& c = result_->calibration;
    cv::Mat l, r, pl, pr, q;
    cv::stereoRectify(c.left_camera_matrix, c.left_distortion, c.right_camera_matrix, c.right_distortion,
        result_->image_size, c.right_to_left_rotation, c.right_to_left_translation, l, r, pl, pr, q, cv::CALIB_ZERO_DISPARITY);
    cv::initUndistortRectifyMap(c.left_camera_matrix, c.left_distortion, l, pl, result_->image_size, CV_32FC1, lx_, ly_);
    cv::initUndistortRectifyMap(c.right_camera_matrix, c.right_distortion, r, pr, result_->image_size, CV_32FC1, rx_, ry_);
}
void CalibrationWorker::show(const cal::Sample& sample, const QString& caption) {
    cv::Mat l, r;
    const bool rect = rectified_ && result_ && sample.images.left.rgb.size() == result_->image_size && !lx_.empty();
    if (rect) {
        cv::remap(sample.images.left.rgb, l, lx_, ly_, cv::INTER_LINEAR);
        cv::remap(sample.images.right.rgb, r, rx_, ry_, cv::INTER_LINEAR);
        for (int y = 60; y < l.rows; y += 120) {
            cv::line(l, {0, y}, {l.cols - 1, y}, {60, 220, 170}, 2);
            cv::line(r, {0, y}, {r.cols - 1, y}, {60, 220, 170}, 2);
        }
    } else {
        l = detection_ ? cal::annotated(sample.images.left.rgb, sample.left) : sample.images.left.rgb;
        r = detection_ ? cal::annotated(sample.images.right.rgb, sample.right) : sample.images.right.rgb;
    }
    emit preview(displayImage(l), displayImage(r), QString("%1 · %2 × %3 · %4\nMarkers L %5 / R %6 · Corners L %7 / R %8 · Shared %9 · Host gap %10 ms")
        .arg(caption).arg(l.cols).arg(l.rows).arg(rect ? "Rectified + epilines" : "Raw RGB")
        .arg(sample.left.markers).arg(sample.right.markers).arg(sample.left.ids.size()).arg(sample.right.ids.size())
        .arg(cal::common_corners(sample.left, sample.right)).arg(cal::arrival_skew(sample.images) / 1e6, 0, 'f', 1));
}
void CalibrationWorker::poll() {
    if (shutting_down_ || QThread::currentThread()->isInterruptionRequested()) return;
    try {
        if (capture_ != Capture::None && capture_clock_.elapsed() > 15000)
            throw std::runtime_error("Capture timed out. Make sure both cameras see the board.");
        auto pair = source_->wait_for_pair(std::chrono::milliseconds(0));
        if (!pair) {
            if (last_frame_clock_.elapsed() > 5000) {
                disconnectCameras(); throw std::runtime_error("No stereo frames for 5 seconds. Check cameras and reconnect.");
            }
            return;
        }
        last_frame_clock_.restart();
        if (pair->left.rgb.size() != pair->right.rgb.size()) throw std::runtime_error("Left/right image sizes differ");
        const auto size = pair->left.rgb.size();
        const bool changed = (image_size_.area() > 0 && image_size_ != size) ||
            (!samples_.empty() && samples_.front().images.left.rgb.size() != size);
        if (changed) { cancelCapture(); samples_.clear(); invalidate(); state_.selected = -1; }
        image_size_ = size;
        if (result_ && result_->image_size != size) {
            invalidate(); state_.status = "Loaded calibration grid differs from camera output. Recalibrate.";
        }
        cal::Sample s; s.images = std::move(*pair);
        if (detection_ || capture_ != Capture::None) {
            s.left = cal::detect(s.images.left.rgb, board_); s.right = cal::detect(s.images.right.rgb, board_);
        }
        latest_ = s;
        if (capture_ != Capture::None) {
            ++state_.candidates;
            if (cal::common_corners(s.left, s.right) >= 4 &&
                (!candidate_ || cal::arrival_skew(s.images) < cal::arrival_skew(candidate_->images))) candidate_ = s;
            state_.status = QString("Candidate %1 / 5 — keep the board still").arg(state_.candidates);
            if (state_.candidates == 5) {
                const auto mode = capture_; capture_ = Capture::None; state_.busy = false;
                if (!candidate_) throw std::runtime_error("No valid pair: need at least 4 shared corners. Adjust board visibility.");
                if (mode == Capture::Sample) {
                    samples_.push_back(std::move(*candidate_)); invalidate();
                    state_.status = QString("Saved sample %1 / 20. Change board position and tilt for the next sample.").arg(samples_.size());
                } else {
                    result_->check = cal::check(*result_, *candidate_); result_->checked = true;
                    check_sample_ = std::move(*candidate_);
                    state_.status = "Independent check complete. Inspect the RMS values; save if acceptable.";
                    // Freeze the independent pair until Return to live or another selection.
                    state_.selected = -2;
                }
                candidate_.reset();
            }
        }
        if (state_.selected == -1) show(s, "Live preview");
        else if (state_.selected == -2 && check_sample_) show(*check_sample_, "Independent check · frozen");
        publish();
    } catch (const std::exception& e) {
        cancelCapture();
        state_.status = QString::fromUtf8(e.what()); emit failed(state_.status); publish();
    }
}
void CalibrationWorker::restoreSavedCalibration() {
    QStringList candidates;
    const QString previous = lastCalibrationPath();
    state_.suggested_save_path = previous.isEmpty() ? defaultCalibrationPath() : previous;
    if (!previous.isEmpty()) candidates << previous;
    for (const auto& directory : {QDir::currentPath(), QCoreApplication::applicationDirPath()}) {
        candidates << QDir(directory).absoluteFilePath("sentech_stereo_calibration.json")
                   << QDir(directory).absoluteFilePath("calibration.json");
    }
    candidates.removeDuplicates();
    QStringList errors;
    for (const auto& path : candidates) {
        if (!QFileInfo(path).isFile()) continue;
        try { load(path); return; }
        catch (const std::exception& e) { errors << path + ": " + QString::fromUtf8(e.what()); }
    }
    state_.status = errors.isEmpty() ? "No saved calibration found. Connect cameras to create one."
        : "Saved calibration could not be loaded. Choose another file or recalibrate.\n" + errors.join("\n");
}
void CalibrationWorker::reuseSavedCalibration(std::optional<double> exposure_us) {
    if (!reusable()) throw std::runtime_error("No matching saved calibration is available to reuse");
    if (!state_.connected && exposure_us) {
        if (!std::isfinite(*exposure_us) || *exposure_us <= 0) throw std::runtime_error("Camera exposure must be positive.");
        exposure_us_ = *exposure_us;
    }
    // Explicit reuse is a separate route; it never pretends a fresh check occurred.
    completeSession();
}
void CalibrationWorker::save(const QString& path) {
    if (!result_) throw std::runtime_error("No calibration to save");
    const auto json = cal::serialize(*result_);
    QSaveFile file(path);
    if (!file.open(QIODevice::WriteOnly) || file.write(json.data(), qint64(json.size())) != qint64(json.size()) || !file.commit())
        throw std::runtime_error("Could not save calibration: " + file.errorString().toStdString());
    state_.saved_path = QFileInfo(path).absoluteFilePath();
    state_.suggested_save_path = state_.saved_path;
    rememberCalibrationPath(state_.saved_path);
    state_.status = "Calibration saved. Finishing also requires a passing check in this camera connection.";
}
void CalibrationWorker::load(const QString& path) {
    auto loaded = cal::load(path.toStdString());
    if (loaded.left_serial != "21LJ548" || loaded.right_serial != "21LJ530")
        throw std::runtime_error("Calibration camera identities do not match the configured left/right cameras");
    if (!cal::same_board(loaded.board, state_.board)) throw std::runtime_error("Calibration board differs from applied board parameters");
    if (state_.connected && image_size_.area() > 0 && image_size_ != loaded.image_size)
        throw std::runtime_error("Calibration image size differs from current camera output");
    loaded.threshold_px = cal::kQualityThresholdPx;
    invalidate(); result_ = std::move(loaded); samples_.clear(); state_.selected = -1;
    buildMaps(); state_.saved_path = QFileInfo(path).absoluteFilePath();
    state_.suggested_save_path = state_.saved_path; reusable_saved_ = true;
    rememberCalibrationPath(state_.saved_path);
    state_.status = "Saved calibration loaded. Skip calibration and continue, or connect cameras to check it again.";
}
void CalibrationWorker::finish() {
    if (!ready()) throw std::runtime_error("Finish requires connected matching cameras, a passing check and a saved calibration");
    completeSession();
}
void CalibrationWorker::completeSession() {
    auto result = std::make_shared<const cal::SessionResult>(cal::clone(*result_));
    timer_->stop();
    state_.connected = false;
    emit completed(std::move(result), state_.saved_path, SharedStereoSource(std::move(source_)), exposure_us_);
}
void CalibrationWorker::shutdown() {
    shutting_down_ = true; disconnectCameras(); emit stopped();
}
