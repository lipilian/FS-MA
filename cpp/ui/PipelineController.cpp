#include "PipelineController.hpp"

PipelineController::PipelineController(ConfirmedCalibration calibration, QString path, SharedStereoSource source, double exposure_us, QObject* parent)
    : QObject(parent), cancel_(std::make_shared<std::atomic_bool>(false)),
      worker_(new PipelineWorker(std::move(calibration), std::move(path), std::move(source), cancel_, exposure_us)) {
    qRegisterMetaType<PipelineState>();
    worker_->moveToThread(&thread_);
    connect(worker_, &PipelineWorker::stateChanged, this, [this](PipelineState state) {
        state_ = std::move(state); emit stateChanged(state_);
    });
    connect(worker_, &PipelineWorker::images, this, &PipelineController::images);
    connect(worker_, &PipelineWorker::preview, this, &PipelineController::preview);
    connect(worker_, &PipelineWorker::depthImages, this, &PipelineController::depthImages);
    connect(worker_, &PipelineWorker::log, this, &PipelineController::log);
    connect(worker_, &PipelineWorker::actionFinished, this, [this] {
        const bool initialized = initializing_; initializing_ = false;
        busy_ = false; emit busyChanged(false);
        if (initialized && !stopping_) emit initializationFinished(state_.engine_ready, state_.status);
    });
    connect(worker_, &PipelineWorker::stopped, &thread_, &QThread::quit, Qt::DirectConnection);
    connect(&thread_, &QThread::finished, worker_, &QObject::deleteLater);
    connect(&thread_, &QThread::finished, this, &PipelineController::stopped);
    thread_.start();
}
PipelineController::~PipelineController() { stop(); thread_.wait(); }
void PipelineController::submit(std::function<void(PipelineWorker&)> action) {
    if (busy_ || stopping_) return;
    busy_ = true; cancel_->store(false); emit busyChanged(true);
    auto* worker = worker_;
    QMetaObject::invokeMethod(worker, [worker, action = std::move(action)] { worker->execute(action); }, Qt::QueuedConnection);
}
void PipelineController::initialize(const QString& engine_path) {
    if (busy_ || stopping_) return;
    initializing_ = true;
    submit([engine_path](auto& worker) { worker.initialize(engine_path); });
}
void PipelineController::setPreviewRectified(bool enabled) {
    if (stopping_) return;
    auto* worker = worker_;
    // A display preference must not be dropped by the task busy guard.
    QMetaObject::invokeMethod(worker, [worker, enabled] { worker->setPreviewRectified(enabled); }, Qt::QueuedConnection);
}
void PipelineController::cancel() { cancel_->store(true); }
void PipelineController::stop() {
    if (stopping_) return;
    stopping_ = true; cancel();
    QMetaObject::invokeMethod(worker_, &PipelineWorker::shutdown, Qt::QueuedConnection);
}
