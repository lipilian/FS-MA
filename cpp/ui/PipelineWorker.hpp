#pragma once
#include "CalibrationWorker.hpp"
#include "fs/inference/FS.hpp"
#include <atomic>

struct PipelineState {
    bool connected{false}, live{false}, has_pair{false}, has_rectified{false}, gpu_ready{false}, engine_ready{false};
    QString input{"No stereo pair loaded"}, calibration, status{"Import a capture directory or preview the cameras."};
    QString engine{"Not initialized"}, engine_path;
    int stage{0};
};
Q_DECLARE_METATYPE(PipelineState)

// Serialized source, CPU and GPU work. Qt widgets only receive owned QImage snapshots.
class PipelineWorker : public QObject {
    Q_OBJECT
public:
    PipelineWorker(ConfirmedCalibration calibration, QString path, SharedStereoSource source,
                   std::shared_ptr<std::atomic_bool> cancel);
    void execute(const std::function<void(PipelineWorker&)>& action);
    void initialize(const QString& engine_path);
    void importCapture(const QString& directory, bool use_capture_calibration);
    void connectCameras(double exposure_us);
    void disconnectCameras();
    void setLive(bool enabled);
    void freeze();
    void reconstruct(float minimum, float maximum);
    void shutdown();
signals:
    void stateChanged(PipelineState state);
    void images(QImage raw_left, QImage raw_right, QImage rectified_left, QImage rectified_right);
    void preview(QImage left, QImage right);
    void log(QString message);
    void actionFinished();
    void stopped();
private:
    void poll();
    void publish();
    void checkpoint() const;
    void prepare(std::unique_ptr<StereoFrame> frame, QString input, QString calibration);
    ConfirmedCalibration confirmed_;
    QString confirmed_path_;
    SharedStereoSource source_;
    std::shared_ptr<std::atomic_bool> cancel_;
    QTimer* timer_;
    QElapsedTimer last_pair_;
    std::optional<StereoCameraPair> latest_;
    std::unique_ptr<StereoFrame> frame_;
    std::unique_ptr<FS> fs_;
    PipelineState state_;
};
