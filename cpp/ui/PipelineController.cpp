#include "PipelineController.hpp"

PipelineController::PipelineController(ConfirmedCalibration calibration, QString path, SharedStereoSource source, QObject* parent)
    : QObject(parent), cancel_(std::make_shared<std::atomic_bool>(false)),
      worker_(new PipelineWorker(std::move(calibration), std::move(path), std::move(source), cancel_)) {
    qRegisterMetaType<PipelineState>();
    worker_->moveToThread(&thread_);
    connect(worker_, &PipelineWorker::stateChanged, this, &PipelineController::stateChanged);
    connect(worker_, &PipelineWorker::images, this, &PipelineController::images);
    connect(worker_, &PipelineWorker::preview, this, &PipelineController::preview);
    connect(worker_, &PipelineWorker::log, this, &PipelineController::log);
    connect(worker_, &PipelineWorker::actionFinished, this, [this] { busy_ = false; emit busyChanged(false); });
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
void PipelineController::cancel() { cancel_->store(true); }
void PipelineController::stop() {
    if (stopping_) return;
    stopping_ = true; cancel();
    QMetaObject::invokeMethod(worker_, &PipelineWorker::shutdown, Qt::QueuedConnection);
}
