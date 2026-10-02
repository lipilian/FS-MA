#include "PipelineWorker.hpp"
#include "fs/capture/RealSenseStereoSource.hpp"
#include <QDir>
#include <QCoreApplication>
#include <QFileInfo>
#include <QFile>
#include <QSaveFile>
#include <QDataStream>
#include <QSet>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgcodecs.hpp>
#include <cmath>
#include <algorithm>
#include <array>
#include <unordered_map>
#include <limits>
#include <stdexcept>

namespace {
QImage image(const cv::Mat& rgb) {
    return QImage(rgb.data, rgb.cols, rgb.rows, rgb.step, QImage::Format_RGB888).copy();
}
fs::MeshCamera meshCamera(const StereoFrame& frame, cv::Size mesh_size) {
    const auto camera=frame.rectified_camera_parameters();
    const double sx=double(mesh_size.width)/frame.rectified_left().cols;
    const double sy=double(mesh_size.height)/frame.rectified_left().rows;
    return {cv::Matx33d(camera.fx*sx,0,camera.cx*sx,0,camera.fy*sy,camera.cy*sy,0,0,1),mesh_size};
}
fs::MeshCamera previewCamera(const StereoCalibration& calibration, cv::Size size, bool already_rectified = false) {
    cv::Mat intrinsics;
    if (already_rectified) calibration.left_camera_matrix.convertTo(intrinsics, CV_64F);
    else {
        cv::Mat r1, r2, p1, p2, q;
        cv::stereoRectify(calibration.left_camera_matrix, calibration.left_distortion,
                          calibration.right_camera_matrix, calibration.right_distortion, size,
                          calibration.right_to_left_rotation, calibration.right_to_left_translation,
                          r1, r2, p1, p2, q, cv::CALIB_ZERO_DISPARITY);
        intrinsics = p1(cv::Rect(0, 0, 3, 3));
    }
    return {cv::Matx33d(intrinsics), size};
}
// Match the values actually written to PLY. Exact XYZ equality preserves nearby
// distinct samples; including RGB preserves color differences at the same XYZ.
struct PLYVertexKey {
    std::array<float,3> xyz;
    std::array<unsigned char,3> rgb;
    bool operator==(const PLYVertexKey& other) const { return xyz==other.xyz && rgb==other.rgb; }
};
struct PLYVertexHash {
    std::size_t operator()(const PLYVertexKey& key) const {
        std::size_t hash=0;
        const auto combine=[&](std::size_t value) { hash^=value+0x9e3779b9u+(hash<<6)+(hash>>2); };
        // std::hash<float> treats +0 and -0 equally, matching float equality.
        for (float value:key.xyz) combine(std::hash<float>{}(value));
        for (auto value:key.rgb) combine(value);
        return hash;
    }
};
// Called only for an explicit GPU mesh export, on the pipeline thread.
fs::MeshResult downloadMeshForExport(const GPUMeshFrame& source, const std::function<void()>& checkpoint) {
    const auto vertex_count=source.buffer ? source.buffer->vertex_slots() : 0;
    if (!source.buffer || !source.buffer->vertices() || !source.stream || !source.stream_owner ||
        vertex_count%3 || !source.stats.triangle_count || source.stats.triangle_count>vertex_count/3 ||
        vertex_count>std::size_t(std::numeric_limits<int>::max()))
        throw std::runtime_error("Invalid GPU mesh export source");
    const auto checked=[](cudaError_t e) {
        if (e!=cudaSuccess) throw std::runtime_error(std::string("GPU mesh download failed: ")+cudaGetErrorString(e));
    };
    checkpoint();
    checked(cudaSetDevice(source.device));
    std::vector<fs::MeshGPUVertex> vertices(vertex_count);
    try {
        checked(cudaMemcpyAsync(vertices.data(),source.buffer->vertices(),vertex_count*sizeof(vertices[0]),
                                cudaMemcpyDeviceToHost,source.stream));
        checked(cudaStreamSynchronize(source.stream));
    } catch (...) {
        cudaStreamSynchronize(source.stream);
        throw;
    }
    checkpoint();
    fs::MeshResult mesh;
    mesh.vertices.reserve(source.stats.triangle_count);
    mesh.colors.reserve(source.stats.triangle_count);
    std::unordered_map<PLYVertexKey,int,PLYVertexHash> vertex_indices;
    vertex_indices.reserve(source.stats.triangle_count);
    mesh.triangles.reserve(source.stats.triangle_count);
    for (std::size_t i=0;i<vertex_count;i+=3) {
        if ((i/3)%4096==0) checkpoint();
        cv::Vec3d p[3]; bool valid=true;
        for (int k=0;k<3;++k) {
            const auto& v=vertices[i+k];
            p[k]={v.x,v.y,v.z};
            valid &= std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z) && v.z>0;
        }
        if (!valid) continue; // Includes the zeroed, unused triangle slots.
        const double area=.5*cv::norm((p[1]-p[0]).cross(p[2]-p[0]));
        if (!(area>0) || !std::isfinite(area)) continue;
        cv::Vec3i face;
        for (int k=0;k<3;++k) {
            const auto& v=vertices[i+k];
            if (!std::isfinite(v.r) || !std::isfinite(v.g) || !std::isfinite(v.b))
                throw std::runtime_error("GPU mesh contains invalid colors");
            const auto byte=[](float c) { return static_cast<unsigned char>(std::lround(std::clamp(c,0.f,1.f)*255.f)); };
            const PLYVertexKey key{{v.x,v.y,v.z},{byte(v.r),byte(v.g),byte(v.b)}};
            const auto [entry,inserted]=vertex_indices.emplace(key,int(mesh.vertices.size()));
            if (inserted) {
                mesh.vertices.emplace_back(v.x,v.y,v.z);
                mesh.colors.emplace_back(key.rgb[0],key.rgb[1],key.rgb[2]);
            }
            face[k]=entry->second;
        }
        mesh.triangles.push_back(face);
    }
    if (mesh.triangles.size()!=source.stats.triangle_count)
        throw std::runtime_error("GPU mesh export triangle count does not match the displayed result");
    mesh.area_m2=source.stats.area_m2;
    return mesh;
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
    state_.calibration_filename = QFileInfo(confirmed_path_).fileName();
    if (state_.calibration_filename.isEmpty()) state_.calibration_filename="calibration.json";
    camera_calibration_ = confirmed_->calibration;
    camera_size_ = confirmed_->image_size;
    state_.live_camera = previewCamera(camera_calibration_, camera_size_);
    camera_calibration_description_ = confirmed_path_;
    camera_calibration_filename_ = state_.calibration_filename;
    camera_calibration_json_ = QByteArray::fromStdString(fs::calibration::serialize(*confirmed_));
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
    timer_->stop(); state_.live = false; state_.engine_ready = false; state_.gpu_ready = false; state_.depth_ready = false; latest_mesh_.reset(); latest_gpu_mesh_.reset(); mesh_xyz_.release(); mesh_mask_.release(); mesh_rgb_.release();
    state_.engine_path = QFileInfo(engine_path).absoluteFilePath();
    fs_.reset(); sam_.reset(); state_.sam_ready = false;
    QElapsedTimer elapsed; elapsed.start();
    try {
        state_.engine = "Initializing…";
        state_.status = "Creating FS and the CUDA stream…"; publish(); checkpoint();
        if (engine_path.trimmed().isEmpty() || !QFileInfo(state_.engine_path).isFile())
            throw std::runtime_error("Select an existing TensorRT engine file.");
        auto next = std::make_shared<FS>();
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
void PipelineWorker::prepare(std::unique_ptr<StereoFrame> frame, QString input, QString calibration,
                             QString calibration_filename, QByteArray calibration_json) {
    QElapsedTimer elapsed; elapsed.start();
    const bool already_rectified = frame->input_is_rectified();
    if (!already_rectified) frame->rectify();
    checkpoint();
    // Commit together only after validation and rectification succeed. Failed imports retain the previous input.
    frame_ = std::move(frame); state_.input = std::move(input); state_.calibration = std::move(calibration);
    state_.calibration_filename=std::move(calibration_filename); calibration_json_=std::move(calibration_json);
    state_.has_rectified = true; state_.gpu_ready = false; state_.depth_ready = false; latest_mesh_.reset(); latest_gpu_mesh_.reset(); mesh_xyz_.release(); mesh_mask_.release(); mesh_rgb_.release();
    state_.live = false; latest_.reset();
    state_.image_camera = meshCamera(*frame_, frame_->rectified_left().size());
    ++state_.image_id; if (sam_) sam_->clearImage(); publish();
    emit images(image(frame_->rectified_left()), image(frame_->rectified_right()));
    const QString preparation = already_rectified ? "rectified by camera"
        : QString("rectified in %1 ms").arg(elapsed.elapsed());
    state_.status = QString("Pair ready · %1 × %2 · %3. Draw a mask and click Finish draw to continue.")
        .arg(frame_->left().cols).arg(frame_->left().rows).arg(preparation);
    emit log(state_.input + "\nCalibration: " + state_.calibration + "\n" + state_.status);
}
void PipelineWorker::importCapture(const QString& directory, bool use_capture_calibration) {
    const QDir dir(directory);
    if (!dir.exists()) throw std::runtime_error("Capture directory does not exist.");
    QStringList calibration_names{"calibration.json", "sentech_stereo_calibration.json"};
    const auto confirmed_name=QFileInfo(confirmed_path_).fileName();
    if (!confirmed_name.isEmpty() && !calibration_names.contains(confirmed_name)) calibration_names << confirmed_name;
    QString calibration_path;
    for (const auto& name:calibration_names) {
        if (QFileInfo(dir.filePath(name)).isFile()) { calibration_path=dir.filePath(name); break; }
    }
    QStringList missing;
    for (const auto& name:{"left.png", "right.png"})
        if (!QFileInfo(dir.filePath(name)).isFile()) missing << name;
    if (calibration_path.isEmpty()) missing << "calibration JSON ("+calibration_names.join(" or ")+")";
    if (!missing.isEmpty())
        throw std::runtime_error(("Cannot load capture. Missing required files:\n"+missing.join("\n")).toStdString());
    for (const auto& path:QStringList{dir.filePath("left.png"),dir.filePath("right.png"),calibration_path}) {
        QFile file(path);
        if (!file.open(QIODevice::ReadOnly) || file.size()==0)
            throw std::runtime_error(("Cannot load capture. File is unreadable or empty: "+QFileInfo(path).fileName()).toStdString());
    }
    state_.status = "Loading and rectifying capture…"; publish();
    std::unique_ptr<StereoFrame> frame;
    QString description, calibration_filename;
    QByteArray calibration_json;
    if (use_capture_calibration) {
        frame = std::make_unique<StereoFrame>(dir.filePath("left.png").toStdString(), dir.filePath("right.png").toStdString(), calibration_path.toStdString());
        // Legacy captures are supported like the CLI. Enforce grid metadata when it is present.
        cv::FileStorage metadata(calibration_path.toStdString(), cv::FileStorage::READ);
        const auto width = metadata["image_width"], height = metadata["image_height"];
        if ((!width.empty() || !height.empty()) &&
            (width.empty() || height.empty() || int(width) != frame->left().cols || int(height) != frame->left().rows))
            throw std::runtime_error("Capture images do not match calibration image dimensions.");
        QFile json_file(calibration_path);
        if (!json_file.open(QIODevice::ReadOnly)) throw std::runtime_error("Cannot read capture calibration JSON for the frozen result.");
        calibration_json=json_file.readAll();
        if (json_file.error()!=QFileDevice::NoError || calibration_json.isEmpty()) throw std::runtime_error("Cannot read capture calibration JSON.");
        calibration_filename=QFileInfo(calibration_path).fileName();
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
        calibration_filename=QFileInfo(confirmed_path_).fileName();
        calibration_json=QByteArray::fromStdString(fs::calibration::serialize(*confirmed_));
    }
    if (calibration_filename.isEmpty()) calibration_filename="calibration.json";
    checkpoint(); prepare(std::move(frame), dir.absolutePath(), description, calibration_filename, calibration_json);
}
void PipelineWorker::connectCameras(CameraMode mode) {
    if (state_.connected && state_.camera_mode == mode) { setLive(true); return; }
    disconnectCameras();
    state_.status = mode == CameraMode::RealSenseD435 ? "Connecting Intel RealSense D435…" : "Connecting Sentech stereo cameras…";
    publish();
    SharedStereoSource next;
    StereoCalibration calibration;
    cv::Size size;
    QString description, filename;
    QByteArray json;
    if (mode == CameraMode::RealSenseD435) {
        auto realsense = std::make_shared<RealSenseStereoSource>();
        realsense->start();
        calibration = realsense->calibration(); size = realsense->image_size();
        description = QString("D435 %1 · factory IR calibration · 1280 × 800 → center crop 960 × 800")
            .arg(QString::fromStdString(realsense->serial()));
        filename = "calibration.json";
        json = QByteArray::fromStdString(realsense->calibration_json());
        next = std::move(realsense);
    } else {
        SentechStereoOptions options;
        options.left = confirmed_->left_serial; options.right = confirmed_->right_serial; options.exposure_us = exposure_us_;
        next = std::make_shared<SentechStereoSource>(options);
        next->start();
        calibration = confirmed_->calibration; size = confirmed_->image_size;
        description = confirmed_path_; filename = QFileInfo(confirmed_path_).fileName();
        if (filename.isEmpty()) filename = "calibration.json";
        json = QByteArray::fromStdString(fs::calibration::serialize(*confirmed_));
    }
    const auto live_camera = previewCamera(calibration, size, mode == CameraMode::RealSenseD435);
    camera_calibration_ = std::move(calibration); camera_size_ = size;
    state_.live_camera = live_camera;
    camera_calibration_description_ = std::move(description); camera_calibration_filename_ = std::move(filename);
    camera_calibration_json_ = std::move(json);
    // Switching devices or reconnecting can change both the image grid and intrinsics.
    preview_left_map_x_.release(); preview_left_map_y_.release();
    preview_right_map_x_.release(); preview_right_map_y_.release();
    source_ = std::move(next); state_.camera_mode = mode; state_.connected = true;
    emit log(camera_calibration_description_);
    setLive(true); timer_->start();
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
    if (state_.camera_mode == CameraMode::RealSenseD435)
        return "D435 · live IR rectified by camera · 960 × 800 center crop · capture to freeze a stereo pair.";
    return "Live rectified RGB · capture to freeze a stereo pair. Independent streams; hold the subject still.";
}
void PipelineWorker::emitPreview() {
    if (state_.camera_mode == CameraMode::RealSenseD435) {
        emit preview(thumbnail(latest_->left.rgb), thumbnail(latest_->right.rgb));
        return;
    }
    if (preview_left_map_x_.empty()) {
        // Match StereoFrame::rectify(); maps are invalidated on every connection.
        const auto& c = camera_calibration_;
        cv::Mat r1, r2, p1, p2, q, lx, ly, rx, ry;
        cv::stereoRectify(c.left_camera_matrix, c.left_distortion,
                          c.right_camera_matrix, c.right_distortion, camera_size_,
                          c.right_to_left_rotation, c.right_to_left_translation,
                          r1, r2, p1, p2, q, cv::CALIB_ZERO_DISPARITY);
        cv::initUndistortRectifyMap(c.left_camera_matrix, c.left_distortion, r1, p1,
                                  camera_size_, CV_32FC1, lx, ly);
        cv::initUndistortRectifyMap(c.right_camera_matrix, c.right_distortion, r2, p2,
                                  camera_size_, CV_32FC1, rx, ry);
        preview_left_map_x_ = std::move(lx); preview_left_map_y_ = std::move(ly);
        preview_right_map_x_ = std::move(rx); preview_right_map_y_ = std::move(ry);
    }
    cv::remap(latest_->left.rgb, preview_left_, preview_left_map_x_, preview_left_map_y_, cv::INTER_LINEAR);
    cv::remap(latest_->right.rgb, preview_right_, preview_right_map_x_, preview_right_map_y_, cv::INTER_LINEAR);
    emit preview(thumbnail(preview_left_), thumbnail(preview_right_));
}
void PipelineWorker::poll() {
    if (!state_.live) return;
    try {
        auto pair = source_->wait_for_pair(std::chrono::milliseconds(0));
        if (!pair) {
            if (!source_->running() || last_pair_.elapsed() > 5000) throw std::runtime_error("No stereo pair for 5 seconds. Check cameras and reconnect.");
            return;
        }
        if (pair->left.rgb.size() != camera_size_ || pair->right.rgb.size() != camera_size_)
            throw std::runtime_error("Camera image dimensions do not match the active camera calibration.");
        last_pair_.restart(); latest_ = std::move(pair);
        emitPreview();
    } catch (const std::exception& e) {
        disconnectCameras(); state_.status = QString::fromUtf8(e.what()); publish(); emit log(state_.status);
    }
}
void PipelineWorker::freeze() {
    if (!state_.live || !latest_ || last_pair_.elapsed() > 1000)
        throw std::runtime_error("No fresh stereo pair. Wait for the live preview and capture again.");
    std::unique_ptr<StereoFrame> frame;
    if (state_.camera_mode == CameraMode::RealSenseD435) {
        // D435 Y8 streams are already rectified. Keep the cropped SDK intrinsics
        // and horizontal baseline instead of generating new projection matrices.
        const auto& k = camera_calibration_.left_camera_matrix;
        const StereoCameraParameters camera{k.at<double>(0,0), k.at<double>(1,1),
            k.at<double>(0,2), k.at<double>(1,2),
            static_cast<float>(std::abs(camera_calibration_.right_to_left_translation.at<double>(0)))};
        frame = std::make_unique<StereoFrame>(latest_->left.rgb, latest_->right.rgb, camera);
    } else {
        frame = std::make_unique<StereoFrame>(latest_->left.rgb, latest_->right.rgb, camera_calibration_);
    }
    const QString input = (state_.camera_mode == CameraMode::RealSenseD435 ? "D435 IR · " : "Sentech · ") + QString("Camera capture · L #%1 / R #%2 · host gap %3 ms")
        .arg(latest_->left.frame_id).arg(latest_->right.frame_id)
        .arg(fs::calibration::arrival_skew(*latest_) / 1e6, 0, 'f', 1);
    prepare(std::move(frame), input, camera_calibration_description_, camera_calibration_filename_, camera_calibration_json_);
}
void PipelineWorker::reconstruct(float minimum, float maximum, const QImage& selection_mask,
                                 bool denoise, float max_neighbor_distance_m) {
    if (!state_.engine_ready || !fs_ || !fs_->isEngineLoaded())
        throw std::runtime_error("FoundationStereo must finish splash initialization before reconstruction.");
    if (!frame_ || state_.live) throw std::runtime_error("Import or freeze a stereo pair before reconstruction.");
    if (!std::isfinite(minimum) || !std::isfinite(maximum) || minimum < 0 || maximum <= minimum)
        throw std::runtime_error("Depth range must satisfy 0 ≤ minimum < maximum (metres).");
    if (selection_mask.isNull() || selection_mask.width()!=frame_->rectified_left().cols ||
        selection_mask.height()!=frame_->rectified_left().rows)
        throw std::runtime_error("Confirm a mask aligned with the current rectified left image first.");
    if (!std::isfinite(max_neighbor_distance_m) || max_neighbor_distance_m<=0)
        throw std::runtime_error("Neighbour distance must be positive and finite.");
    state_.gpu_ready = false; state_.depth_ready = false; latest_mesh_.reset(); latest_gpu_mesh_.reset(); mesh_xyz_.release(); mesh_mask_.release(); mesh_rgb_.release(); publish();
    checkpoint(); state_.status = "Running FoundationStereo inference…"; publish();
    QElapsedTimer elapsed; elapsed.start();
    fs_->set_model_camera_parameters(frame_->rectified_camera_parameters(), frame_->rectified_left().size());
    fs_->prepare_stereo_images(frame_->rectified_left(), frame_->rectified_right());
    const QImage grayscale=selection_mask.convertToFormat(QImage::Format_Grayscale8);
    const cv::Mat mask(grayscale.height(),grayscale.width(),CV_8UC1,
                       const_cast<uchar*>(grayscale.constBits()),grayscale.bytesPerLine());
    fs_->set_selection_mask(mask); // Also completes the queued input uploads.
    emit log(QString("Input preparation: %1 ms").arg(elapsed.nsecsElapsed()/1e6,0,'f',3));
    // Synchronized wall time: measure this inference only, excluding preparation.
    elapsed.restart();
    fs_->inference(); fs_->synchronize();
    const double inference_ms=elapsed.nsecsElapsed()/1e6;
    emit log(QString("Inference: %1 ms").arg(inference_ms,0,'f',3));
    checkpoint(); state_.status = "Computing XYZ on GPU…"; publish(); elapsed.restart();
    fs_->compute_xyz_map(minimum, maximum); checkpoint();
    if (denoise) {
        state_.status = "Denoising selected XYZ on GPU…"; publish();
        fs_->denoise_xyz_map(max_neighbor_distance_m,3,2); checkpoint();
    }
    // Both XYZ and denoising APIs synchronize before returning.
    const double xyz_ms=elapsed.nsecsElapsed()/1e6;
    state_.gpu_ready = true;
    emit log(denoise ? QString("Full-image XYZ + selected-region 3 × 3 denoising · %1 m · interior 3 / boundary up to 2 neighbours").arg(max_neighbor_distance_m)
                    : "Full-image XYZ · neighbourhood denoising disabled.");
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
    mesh_xyz_ = xyz; // cv::Mat retains the owned download with no additional copy.
    mesh_rgb_ = model_left;
    cv::resize(mask, mesh_mask_, xyz.size(), 0.0, 0.0, cv::INTER_NEAREST);
    const QImage depth_image=image(depth_rgb);
    const double display_ms=elapsed.nsecsElapsed()/1e6;
    emit log(QString("Post processing: %1 ms (XYZ + denoising: %2 ms; XYZ download + depth display: %3 ms)")
        .arg(xyz_ms+display_ms,0,'f',3).arg(xyz_ms,0,'f',3).arg(display_ms,0,'f',3));
    state_.depth_ready = true;
    state_.status = "Reconstruction complete · rectified left image and Jet depth map are ready.";
    publish(); // Publish readiness before delivering the depth texture.
    emit depthImage(depth_image, minimum, maximum);
    emit log(QString("Jet depth range: %1–%2 m").arg(minimum).arg(maximum));
    emit log(state_.status);
}
void PipelineWorker::buildMeshCPU(double max_edge_m, double max_depth_jump_m) {
    if (!state_.depth_ready || state_.live || !frame_ || mesh_xyz_.empty() || mesh_mask_.empty())
        throw std::runtime_error("Reconstruct depth before generating a mesh.");
    latest_mesh_.reset(); latest_gpu_mesh_.reset();
    checkpoint(); state_.status = "Building constrained Delaunay mesh on CPU…"; publish();
    QElapsedTimer elapsed; elapsed.start();
    auto mesh = std::make_shared<fs::MeshResult>(fs::build_constrained_mesh(
        mesh_xyz_, mesh_mask_, mesh_rgb_, max_edge_m, max_depth_jump_m, [this] { checkpoint(); }));
    checkpoint();
    const double mesh_ms=elapsed.nsecsElapsed()/1e6;
    emit log(QString("Mesh build: %1 ms (CPU)").arg(mesh_ms,0,'f',3));
    mesh->camera=meshCamera(*frame_,mesh_xyz_.size());
    state_.status = QString("Mesh ready · %1 vertices · %2 triangles · %3 cm²")
        .arg(mesh->vertices.size()).arg(mesh->triangles.size()).arg(mesh->area_m2 * 1e4, 0, 'f', 2);
    if (mesh->skipped_components) emit log(QString("Skipped %1 mask components with insufficient points or degenerate boundaries.").arg(mesh->skipped_components));
    latest_mesh_=mesh;
    publish(); emit meshReady(mesh); emit log(state_.status);
}
void PipelineWorker::buildMeshGPU(const QImage& selection_mask, double max_edge_m, double max_depth_jump_m) {
    if (!state_.depth_ready || !state_.gpu_ready || state_.live || !frame_ || !fs_ || !fs_->isEngineLoaded())
        throw std::runtime_error("Reconstruct depth before preparing a GPU mesh.");
    if (selection_mask.isNull() || selection_mask.width() != frame_->rectified_left().cols ||
        selection_mask.height() != frame_->rectified_left().rows)
        throw std::invalid_argument("GPU mesh requires the confirmed rectified-left mask.");
    latest_mesh_.reset(); latest_gpu_mesh_.reset();
    checkpoint();
    state_.status = "Preparing GPU mesh…"; publish();
    QElapsedTimer elapsed; elapsed.start();
    const QImage grayscale = selection_mask.convertToFormat(QImage::Format_Grayscale8);
    const cv::Mat mask(grayscale.height(), grayscale.width(), CV_8UC1,
                       const_cast<uchar*>(grayscale.constBits()), grayscale.bytesPerLine());
    const auto inputs = fs_->prepare_gpu_mesh_inputs(mask);
    checkpoint();
    emit log(QString("GPU mesh input preparation: %1 ms · reuse FS GPU XYZ, RGB and stream; upload current mask (%2 × %3).")
        .arg(elapsed.nsecsElapsed()/1e6,0,'f',3).arg(inputs.width).arg(inputs.height));
    elapsed.restart();
    if (!gpu_mesh_ || !gpu_mesh_.unique()) gpu_mesh_=std::make_shared<fs::MeshGPUBuffer>();
    const auto stats = fs::build_mesh_gpu(inputs, *gpu_mesh_, max_edge_m, max_depth_jump_m);
    checkpoint();
    emit log(QString("GPU mesh build + reduction: %1 ms · %2 triangles · %3 cm²")
        .arg(elapsed.nsecsElapsed()/1e6,0,'f',3).arg(stats.triangle_count).arg(stats.area_m2*1e4,0,'f',2));
    if (!stats.triangle_count) throw std::runtime_error("GPU mesh has no triangles after filtering.");
    auto mesh=std::make_shared<GPUMeshFrame>();
    mesh->buffer=gpu_mesh_; mesh->stats=stats; mesh->stream=inputs.stream; mesh->stream_owner=fs_;
    const auto error=cudaGetDevice(&mesh->device);
    if (error!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
    mesh->camera=meshCamera(*frame_,{inputs.width,inputs.height});
    state_.status = QString("GPU mesh ready · %1 triangles · %2 cm²")
        .arg(stats.triangle_count).arg(stats.area_m2*1e4,0,'f',2);
    latest_gpu_mesh_=mesh;
    publish(); emit log(state_.status); emit gpuMeshReady(std::move(mesh));
}
void PipelineWorker::saveResults(const QString& directory, ReconstructionSaveOptions options,
                                 const QImage& selection, SharedMesh mesh, bool overwrite, SharedGPUMesh gpu_mesh) {
    if (!frame_ || state_.live) throw std::runtime_error("Capture or import a frozen pair before saving.");
    if (!options.images && !options.calibration && !options.mask && !options.mesh && !options.depth)
        throw std::runtime_error("Select at least one item to save.");
    const QDir dir(directory);
    if (directory.trimmed().isEmpty() || !dir.exists()) throw std::runtime_error("Choose an existing output directory.");
    if (options.calibration && (calibration_json_.isEmpty() || state_.calibration_filename.isEmpty() ||
        QFileInfo(state_.calibration_filename).fileName()!=state_.calibration_filename))
        throw std::runtime_error("No valid calibration snapshot is available.");
    if (options.mask && (selection.isNull() || selection.size()!=QSize(frame_->rectified_left().cols,frame_->rectified_left().rows)))
        throw std::runtime_error("Confirm a full-resolution mask before saving it.");
    if (options.depth && (!state_.depth_ready || mesh_xyz_.empty() || mesh_xyz_.type()!=CV_32FC3))
        throw std::runtime_error("Reconstruct depth before saving it.");
    if (options.mesh) {
        const bool current_cpu=mesh && mesh==latest_mesh_ && !mesh->triangles.empty();
        const bool current_gpu=gpu_mesh && gpu_mesh==latest_gpu_mesh_.lock() && gpu_mesh->stats.triangle_count;
        if (!state_.depth_ready || (mesh && gpu_mesh) || (!current_cpu && !current_gpu))
            throw std::runtime_error("Generate the current displayed mesh before saving it.");
    }
    QStringList names;
    if (options.images) names << "left.png" << "right.png";
    if (options.calibration) names << state_.calibration_filename;
    if (options.mask) names << "mask.png";
    if (options.depth) names << "depth.tiff";
    if (options.mesh) names << "mesh.ply";
    for (const auto& name:names) {
        const QFileInfo destination(dir.filePath(name));
        if (destination.isDir() || (!overwrite && (destination.exists() || destination.isSymLink())))
            throw std::runtime_error(("Output already exists: " + destination.filePath()).toStdString());
    }
    if (QSet<QString>(names.begin(),names.end()).size()!=names.size())
        throw std::runtime_error("Calibration filename conflicts with another selected output.");
    checkpoint(); state_.status="Saving selected results…"; publish();
    QElapsedTimer elapsed; elapsed.start();
    if (options.mesh && gpu_mesh) {
        QElapsedTimer download; download.start();
        mesh=std::make_shared<fs::MeshResult>(downloadMeshForExport(*gpu_mesh,[this] { checkpoint(); }));
        emit log(QString("GPU mesh export download + filtering: %1 ms · %2 triangles · %3 unique vertices · RGB uint8 0–255")
            .arg(download.nsecsElapsed()/1e6,0,'f',3).arg(mesh->triangles.size()).arg(mesh->vertices.size()));
    }
    // Encode to temporary files first; each destination is replaced atomically.
    std::vector<std::unique_ptr<QSaveFile>> pending;
    const auto queue=[&](const QString& name,const std::function<void(QSaveFile&)>& write) {
        checkpoint();
        auto file=std::make_unique<QSaveFile>(dir.filePath(name));
        if (!file->open(QIODevice::WriteOnly)) throw std::runtime_error(("Cannot save " + name + ": " + file->errorString()).toStdString());
        write(*file); pending.push_back(std::move(file));
    };
    if (options.images) {
        queue("left.png",[&](auto& file) { if (!image(frame_->left()).save(&file,"PNG")) throw std::runtime_error("Cannot encode left.png"); });
        queue("right.png",[&](auto& file) { if (!image(frame_->right()).save(&file,"PNG")) throw std::runtime_error("Cannot encode right.png"); });
    }
    if (options.calibration) queue(state_.calibration_filename,[&](auto& file) {
        if (file.write(calibration_json_)!=calibration_json_.size()) throw std::runtime_error("Cannot write calibration JSON");
    });
    if (options.mask) queue("mask.png",[&](auto& file) {
        if (!selection.convertToFormat(QImage::Format_Grayscale8).save(&file,"PNG")) throw std::runtime_error("Cannot encode mask.png");
    });
    if (options.depth) queue("depth.tiff",[&](auto& file) {
        // Reuse the filtered CPU XYZ snapshot downloaded for the depth display.
        cv::Mat depth;
        cv::extractChannel(mesh_xyz_,depth,2);
        std::vector<unsigned char> encoded;
        if (!cv::imencode(".tiff",depth,encoded,{cv::IMWRITE_TIFF_COMPRESSION,1}))
            throw std::runtime_error("Cannot encode depth.tiff");
        const auto bytes=static_cast<qint64>(encoded.size());
        if (file.write(reinterpret_cast<const char*>(encoded.data()),bytes)!=bytes)
            throw std::runtime_error("Cannot write depth.tiff");
    });
    if (options.mesh) queue("mesh.ply",[&](auto& file) {
        if (mesh->colors.size()!=mesh->vertices.size()) throw std::runtime_error("Mesh colour count does not match vertices");
        const auto header=QString("ply\nformat binary_little_endian 1.0\ncomment XYZ in metres, rectified left camera frame\n"
            "element vertex %1\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\n"
            "element face %2\nproperty list uchar int vertex_indices\nend_header\n")
            .arg(mesh->vertices.size()).arg(mesh->triangles.size()).toUtf8();
        if (file.write(header)!=header.size()) throw std::runtime_error("Cannot write mesh.ply header");
        QDataStream stream(&file); stream.setByteOrder(QDataStream::LittleEndian);
        stream.setFloatingPointPrecision(QDataStream::SinglePrecision);
        for (size_t i=0;i<mesh->vertices.size();++i) {
            if (i%4096==0) checkpoint();
            const auto p=mesh->vertices[i]; const auto color=mesh->colors[i];
            stream << p[0] << p[1] << p[2] << quint8(color[0]) << quint8(color[1]) << quint8(color[2]);
        }
        for (size_t i=0;i<mesh->triangles.size();++i) {
            if (i%4096==0) checkpoint();
            const auto f=mesh->triangles[i];
            for (int k=0;k<3;++k) if (f[k]<0 || size_t(f[k])>=mesh->vertices.size()) throw std::runtime_error("Invalid mesh triangle index");
            stream << quint8(3) << qint32(f[0]) << qint32(f[1]) << qint32(f[2]);
        }
        if (stream.status()!=QDataStream::Ok) throw std::runtime_error("Cannot write mesh.ply");
    });
    checkpoint();
    for (size_t i=0;i<pending.size();++i) {
        if (!pending[i]->commit()) throw std::runtime_error(("Could not finish saving " + names[int(i)] + ": " + pending[i]->errorString() + ". Earlier files, if any, are listed in the log.").toStdString());
        emit log("Saved " + dir.absoluteFilePath(names[int(i)]));
    }
    state_.status=QString("Saved %1 files to %2 · %3 ms").arg(names.size()).arg(dir.absolutePath()).arg(elapsed.elapsed());
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
    disconnectCameras(); latest_gpu_mesh_.reset(); gpu_mesh_.reset(); sam_.reset(); state_.sam_ready = false; fs_.reset(); state_.engine_ready = false; frame_.reset(); emit stopped();
}
