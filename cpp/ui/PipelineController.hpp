#pragma once
#include "PipelineWorker.hpp"
#include <QThread>

class PipelineController : public QObject {
    Q_OBJECT
public:
    PipelineController(ConfirmedCalibration calibration, QString path, SharedStereoSource source,
                       double exposure_us = SentechStereoOptions{}.exposure_us, QObject* parent = nullptr);
    ~PipelineController() override;
    void submit(std::function<void(PipelineWorker&)> action);
    void initialize(const QString& engine_path, const QString& sam_encoder = {}, const QString& sam_decoder = {});
    const PipelineState& state() const { return state_; }
    void setPreviewRectified(bool enabled);
    quint64 requestMask(quint64 image_id, std::vector<fs::SamPrompt> prompts);
    void cancel();
    void stop();
signals:
    void stateChanged(PipelineState state);
    void images(QImage raw_left, QImage raw_right, QImage rectified_left, QImage rectified_right);
    void preview(QImage left, QImage right, bool rectified);
    void depthImages(QImage rectified_left, QImage depth_rgb, float minimum, float maximum);
    void meshReady(SharedMesh mesh);
    void gpuMeshReady(SharedGPUMesh mesh);
    void maskReady(quint64 image_id, quint64 request_id, QImage mask, QString message);
    void log(QString message);
    void busyChanged(bool busy);
    void initializationFinished(bool success, QString message);
    void stopped();
private:
    QThread thread_;
    std::shared_ptr<std::atomic_bool> cancel_;
    std::shared_ptr<std::atomic_uint64_t> mask_request_{std::make_shared<std::atomic_uint64_t>(0)};
    PipelineWorker* worker_;
    PipelineState state_;
    bool busy_{false}, stopping_{false}, initializing_{false};
};
