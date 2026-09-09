#include "CalibrationController.hpp"
CalibrationController::CalibrationController(QObject* parent, CalibrationWorker::SourceFactory factory)
    : QObject(parent), worker_(new CalibrationWorker(std::move(factory))) {
    qRegisterMetaType<CalibrationState>(); qRegisterMetaType<ConfirmedCalibration>();
    worker_->moveToThread(&thread_);
    connect(worker_, &CalibrationWorker::stateChanged, this, &CalibrationController::stateChanged);
    connect(worker_, &CalibrationWorker::preview, this, &CalibrationController::preview);
    connect(worker_, &CalibrationWorker::actionFinished, this, &CalibrationController::actionFinished);
    connect(worker_, &CalibrationWorker::failed, this, &CalibrationController::failed);
    connect(worker_, &CalibrationWorker::completed, this, &CalibrationController::completed);
    connect(worker_, &CalibrationWorker::stopped, &thread_, &QThread::quit, Qt::DirectConnection);
    connect(&thread_, &QThread::finished, worker_, &QObject::deleteLater);
    connect(&thread_, &QThread::finished, this, &CalibrationController::stopped);
    thread_.start();
}
CalibrationController::~CalibrationController() { stop(); thread_.wait(); }
void CalibrationController::submit(std::function<void(CalibrationWorker&)> action) {
    if (stopping_) return;
    auto* worker = worker_;
    QMetaObject::invokeMethod(worker, [worker, action = std::move(action)] { worker->execute(action); }, Qt::QueuedConnection);
}
void CalibrationController::stop() {
    if (stopping_) return;
    stopping_ = true;
    thread_.requestInterruption();
    QMetaObject::invokeMethod(worker_, &CalibrationWorker::shutdown, Qt::QueuedConnection);
}
