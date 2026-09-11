#pragma once
#include "PipelineWorker.hpp"
#include <QThread>

class PipelineController : public QObject {
    Q_OBJECT
public:
    PipelineController(ConfirmedCalibration calibration, QString path, SharedStereoSource source, QObject* parent = nullptr);
    ~PipelineController() override;
    void submit(std::function<void(PipelineWorker&)> action);
    void initialize(const QString& engine_path);
    const PipelineState& state() const { return state_; }
    void cancel();
    void stop();
signals:
    void stateChanged(PipelineState state);
    void images(QImage raw_left, QImage raw_right, QImage rectified_left, QImage rectified_right);
    void preview(QImage left, QImage right);
    void log(QString message);
    void busyChanged(bool busy);
    void initializationFinished(bool success, QString message);
    void stopped();
private:
    QThread thread_;
    std::shared_ptr<std::atomic_bool> cancel_;
    PipelineWorker* worker_;
    PipelineState state_;
    bool busy_{false}, stopping_{false}, initializing_{false};
};
