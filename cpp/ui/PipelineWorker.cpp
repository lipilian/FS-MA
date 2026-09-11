#include "PipelineWorker.hpp"
#include <QDir>
#include <QCoreApplication>
#include <QFileInfo>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
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
                               std::shared_ptr<std::atomic_bool> cancel, double exposure_us)
    : confirmed_(std::move(calibration)), confirmed_path_(std::move(path)), source_(std::move(source)),
      exposure_us_(exposure_us), cancel_(std::move(cancel)), timer_(new QTimer(this)) {
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
void PipelineWorker::initialize(const QString& engine_path, const QString& sam_encoder, const QString& sam_decoder) {
    // The same worker retains this instance for every subsequent reconstruction.
    timer_->stop(); state_.live = false; state_.engine_ready = false; state_.gpu_ready = false; state_.depth_ready = false;
    state_.engine_path = QFileInfo(engine_path).absoluteFilePath();
    fs_.reset(); sam_.reset(); state_.sam_ready = false;
    QElapsedTimer elapsed; elapsed.start();
    try {
        state_.engine = "Initializing…";
        state_.status = "Creating FS and the CUDA stream…"; publish(); checkpoint();
        if (engine_path.trimmed().isEmpty() || !QFileInfo(state_.engine_path).isFile())
            throw std::runtime_error("Select an existing TensorRT engine file.");
        auto next = std::make_unique<FS>();
        checkpoint();
        state_.status = "Loading FoundationStereo, creating the TensorRT context and allocating inference buffers…"; publish();
        next->loadEngine(state_.engine_path.toStdString());
        checkpoint();
        state_.status = "Checking that GPU initialization has completed…"; publish();
        next->synchronize(); checkpoint();
        const auto samPath = [&](const QString& specified, const QString& name) {
            if (!specified.isEmpty()) return QFileInfo(specified).absoluteFilePath();
            for (const auto& base : {QFileInfo(state_.engine_path).absolutePath(), QDir::currentPath()+"/onnx",
                                    QCoreApplication::applicationDirPath()+"/../../onnx"}) {
                const QFileInfo candidate(QDir(base).filePath(name));
                if (candidate.isFile()) return candidate.absoluteFilePath();
            }
            throw std::runtime_error(("SAM engine not found: " + name + ". Choose both SAM engines in the splash.").toStdString());
        };
        const auto encoder = samPath(sam_encoder,"sam2.1_hiera_large.encoder.engine");
        const auto decoder = samPath(sam_decoder,"sam2.1_hiera_large.decoder.engine");
        state_.status = "Loading SAM 2.1 Hiera Large encoder / decoder and allocating contexts, GPU features and prompt buffers…";
        publish(); checkpoint();
        auto sam = std::make_unique<fs::SamSegmenter>(); sam->loadEngines(encoder.toStdString(),decoder.toStdString());
        checkpoint();
        emit log("SAM 2.1 ready · independent CUDA stream, contexts, GPU I/O / cached features and pinned buffers allocated.\n" + encoder + "\n" + decoder);
        sam_ = std::move(sam); state_.sam_ready = true;
        fs_ = std::move(next);
        state_.engine_ready = true;
        state_.engine = "Ready · 960 × 800 · " + QFileInfo(state_.engine_path).fileName();
        state_.status = "FoundationStereo and SAM 2.1 are ready. Import a capture directory or preview the cameras.";
        if (source_ && source_->running()) {
            state_.connected = true; state_.live = true; last_pair_.start(); timer_->start();
            state_.status = liveStatus();
        }
        emit log(QString("FoundationStereo initialized in %1 ms · engine, context, GPU I/O / XYZ, pinned host and CPU resize buffers ready.").arg(elapsed.elapsed()));
        emit log("Confirmed calibration: " + confirmed_path_);
    } catch (...) {
        state_.engine = "Initialization failed · check the engine, GPU and TensorRT compatibility";
        throw;
    }
}
void PipelineWorker::prepare(std::unique_ptr<StereoFrame> frame, QString input, QString calibration) {
    QElapsedTimer elapsed; elapsed.start();
    frame->rectify(); checkpoint();
    // Commit together only after validation and rectification succeed. Failed imports retain the previous input.
    frame_ = std::move(frame); state_.input = std::move(input); state_.calibration = std::move(calibration);
    state_.has_pair = true; state_.has_rectified = true; state_.gpu_ready = false; state_.depth_ready = false; state_.stage = 1;
    state_.live = false; latest_.reset();
    ++state_.image_id; if (sam_) sam_->clearImage(); publish();
    emit images(image(frame_->left()), image(frame_->right()), image(frame_->rectified_left()), image(frame_->rectified_right()));
    state_.status = QString("Pair ready · %1 × %2 · rectified in %3 ms. Draw a mask and click Finish draw to continue.")
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
void PipelineWorker::connectCameras() {
    if (state_.connected) { setLive(true); return; }
    if (source_) { source_->stop(); source_.reset(); }
    SentechStereoOptions options;
    options.left = confirmed_->left_serial; options.right = confirmed_->right_serial; options.exposure_us = exposure_us_;
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
    if (enabled) { last_pair_.restart(); state_.status = liveStatus(); }
    else state_.status = "Preview paused. Showing the frozen input, if available.";
}
QString PipelineWorker::liveStatus() const {
    return QString("Live %1 · capture to freeze a stereo pair. Independent streams; hold the subject still.")
        .arg(preview_rectified_ ? "rectified RGB" : "raw RGB");
}
void PipelineWorker::setPreviewRectified(bool enabled) {
    preview_rectified_ = enabled;
    if (state_.live) { state_.status = liveStatus(); publish(); }
}
void PipelineWorker::emitPreview() {
    if (!preview_rectified_) {
        emit preview(thumbnail(latest_->left.rgb), thumbnail(latest_->right.rgb), false);
        return;
    }
    if (preview_left_map_x_.empty()) {
        // Confirmed calibration and camera dimensions are fixed for this worker.
        // Match StereoFrame::rectify(), caching maps across frames and reconnects.
        const auto& c = confirmed_->calibration;
        cv::Mat r1, r2, p1, p2, q, lx, ly, rx, ry;
        cv::stereoRectify(c.left_camera_matrix, c.left_distortion,
                          c.right_camera_matrix, c.right_distortion, confirmed_->image_size,
                          c.right_to_left_rotation, c.right_to_left_translation,
                          r1, r2, p1, p2, q, cv::CALIB_ZERO_DISPARITY);
        cv::initUndistortRectifyMap(c.left_camera_matrix, c.left_distortion, r1, p1,
                                  confirmed_->image_size, CV_32FC1, lx, ly);
        cv::initUndistortRectifyMap(c.right_camera_matrix, c.right_distortion, r2, p2,
                                  confirmed_->image_size, CV_32FC1, rx, ry);
        preview_left_map_x_ = std::move(lx); preview_left_map_y_ = std::move(ly);
        preview_right_map_x_ = std::move(rx); preview_right_map_y_ = std::move(ry);
    }
    cv::remap(latest_->left.rgb, preview_left_, preview_left_map_x_, preview_left_map_y_, cv::INTER_LINEAR);
    cv::remap(latest_->right.rgb, preview_right_, preview_right_map_x_, preview_right_map_y_, cv::INTER_LINEAR);
    emit preview(thumbnail(preview_left_), thumbnail(preview_right_), true);
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
        emitPreview();
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
void PipelineWorker::reconstruct(float minimum, float maximum) {
    if (!state_.engine_ready || !fs_ || !fs_->isEngineLoaded())
        throw std::runtime_error("FoundationStereo must finish splash initialization before reconstruction.");
    if (!frame_ || state_.live) throw std::runtime_error("Import or freeze a stereo pair before reconstruction.");
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum < 0 || maximum <= minimum)
        throw std::runtime_error("Depth range must satisfy 0 ≤ minimum < maximum (metres).");
    state_.gpu_ready = false; state_.depth_ready = false; state_.stage = 1; publish();
    checkpoint(); state_.status = "Running FoundationStereo inference…"; publish();
    QElapsedTimer elapsed; elapsed.start();
    fs_->set_model_camera_parameters(frame_->rectified_camera_parameters(), frame_->rectified_left().size());
    fs_->prepare_stereo_images(frame_->rectified_left(), frame_->rectified_right());
    fs_->inference(); fs_->synchronize();
    state_.stage = 2; emit log(QString("Inference + input preparation: %1 ms").arg(elapsed.elapsed()));
    checkpoint(); state_.status = "Computing XYZ on GPU…"; publish(); elapsed.restart();
    fs_->compute_xyz_map(minimum, maximum); checkpoint(); state_.stage = 3; state_.gpu_ready = true;
    emit log(QString("GPU XYZ: %1 ms · depth %2–%3 m").arg(elapsed.elapsed()).arg(minimum).arg(maximum));
    state_.status = "Preparing depth display…"; publish(); elapsed.restart();
    const cv::Mat xyz = fs_->download_xyz_map(); checkpoint();
    cv::Mat depth, indices, depth_rgb, model_left;
    cv::extractChannel(xyz, depth, 2);
    // Fixed metric range from this run, never per-image min/max normalization.
    depth.convertTo(indices, CV_8U, 255.0 / (maximum - minimum), -255.0 * minimum / (maximum - minimum));
    cv::applyColorMap(indices, depth_rgb, cv::COLORMAP_JET);
    const cv::Mat valid = (depth > 0.0F) & (depth >= minimum) & (depth <= maximum);
    depth_rgb.setTo(cv::Scalar::all(0), ~valid);
    cv::cvtColor(depth_rgb, depth_rgb, cv::COLOR_BGR2RGB);
    // Same dimensions and interpolation as FS::prepare_stereo_images().
    cv::resize(frame_->rectified_left(), model_left,
               cv::Size(FS::kTensorRtInputWidth, FS::kTensorRtInputHeight), 0.0, 0.0, cv::INTER_LINEAR);
    checkpoint();
    state_.depth_ready = true; state_.stage = 4;
    state_.status = "Reconstruction complete · rectified left image and Jet depth map are ready.";
    publish(); // Unlock the depth step before delivering its images.
    emit depthImages(image(model_left), image(depth_rgb), minimum, maximum);
    emit log(QString("Depth display: %1 ms · Jet %2–%3 m").arg(elapsed.elapsed()).arg(minimum).arg(maximum));
    state_.status = "Reconstruction complete · rectified left image and Jet depth map are ready.";
    emit log(state_.status);
}
void PipelineWorker::segment(quint64 image_id, quint64 request_id, const std::vector<fs::SamPrompt>& prompts,
                             const std::shared_ptr<std::atomic_uint64_t>& current_request) {
    const auto current = [&] { return !cancel_->load() && current_request->load()==request_id && state_.image_id==image_id; };
    if (!current()) return;
    try {
        if (!sam_ || !state_.sam_ready || !frame_ || state_.live) throw std::runtime_error("Freeze or import an image before drawing SAM prompts.");
        if (prompts.empty()) return;
        QElapsedTimer elapsed; elapsed.start();
        const bool encode = !sam_->hasImage();
        if (encode) sam_->setImage(frame_->rectified_left());
        if (!current()) return;
        const cv::Mat mask = sam_->predict(prompts);
        if (!current()) return;
        if (cv::countNonZero(mask)==0) {
            emit maskReady(image_id,request_id,{},"No region found. Add a foreground point or adjust the box."); return;
        }
        const QString message = QString("SAM mask ready · score %1 · %2 ms · %3. Refine prompts or click Finish draw.")
            .arg(sam_->score(),0,'f',3).arg(elapsed.elapsed()).arg(encode ? "encoder + decoder" : "cached features + decoder");
        emit maskReady(image_id,request_id,QImage(mask.data,mask.cols,mask.rows,mask.step,QImage::Format_Grayscale8).copy(),message);
        emit log(message);
    } catch (const std::exception& e) {
        if (current()) emit maskReady(image_id,request_id,{},QString::fromUtf8(e.what()));
    }
}
void PipelineWorker::shutdown() {
    disconnectCameras(); sam_.reset(); state_.sam_ready = false; fs_.reset(); state_.engine_ready = false; frame_.reset(); emit stopped();
}
