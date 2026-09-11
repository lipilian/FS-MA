#pragma once
#include "CalibrationWorker.hpp"
#include "fs/inference/FS.hpp"
#include "fs/inference/SamSegmenter.hpp"
#include <atomic>

struct PipelineState {
    bool connected{false}, live{false}, has_pair{false}, has_rectified{false}, gpu_ready{false}, depth_ready{false}, engine_ready{false};
    QString input{"No stereo pair loaded"}, calibration, status{"Import a capture directory or preview the cameras."};
    QString engine{"Not initialized"}, engine_path;
    bool sam_ready{false};
    quint64 image_id{0};
    int stage{0};
};
Q_DECLARE_METATYPE(PipelineState)

// Serialized source, CPU and GPU work. Qt widgets only receive owned QImage snapshots.
class PipelineWorker : public QObject {
    Q_OBJECT
public:
    PipelineWorker(ConfirmedCalibration calibration, QString path, SharedStereoSource source,
                   std::shared_ptr<std::atomic_bool> cancel, double exposure_us = SentechStereoOptions{}.exposure_us);
    void execute(const std::function<void(PipelineWorker&)>& action);
    void initialize(const QString& engine_path, const QString& sam_encoder = {}, const QString& sam_decoder = {});
    void importCapture(const QString& directory, bool use_capture_calibration);
    void connectCameras();
    void disconnectCameras();
    void setLive(bool enabled);
    void setPreviewRectified(bool enabled);
    void freeze();
    void reconstruct(float minimum, float maximum);
    void segment(quint64 image_id, quint64 request_id, const std::vector<fs::SamPrompt>& prompts,
                 const std::shared_ptr<std::atomic_uint64_t>& current_request);
    void shutdown();
signals:
    void stateChanged(PipelineState state);
    void images(QImage raw_left, QImage raw_right, QImage rectified_left, QImage rectified_right);
    void preview(QImage left, QImage right, bool rectified);
    void depthImages(QImage rectified_left, QImage depth_rgb, float minimum, float maximum);
    void maskReady(quint64 image_id, quint64 request_id, QImage mask, QString message);
    void log(QString message);
    void actionFinished();
    void stopped();
private:
    void poll();
    void publish();
    QString liveStatus() const;
    void emitPreview();
    void checkpoint() const;
    void prepare(std::unique_ptr<StereoFrame> frame, QString input, QString calibration);
    ConfirmedCalibration confirmed_;
    QString confirmed_path_;
    SharedStereoSource source_;
    const double exposure_us_; // Requested setting handed off by calibration; also used on reconnect.
    std::shared_ptr<std::atomic_bool> cancel_;
    QTimer* timer_;
    QElapsedTimer last_pair_;
    std::optional<StereoCameraPair> latest_;
    bool preview_rectified_{true};
    cv::Mat preview_left_map_x_, preview_left_map_y_, preview_right_map_x_, preview_right_map_y_;
    cv::Mat preview_left_, preview_right_;
    std::unique_ptr<StereoFrame> frame_;
    std::unique_ptr<FS> fs_;
    std::unique_ptr<fs::SamSegmenter> sam_;
    PipelineState state_;
};
