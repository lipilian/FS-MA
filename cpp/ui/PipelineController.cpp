#include "PipelineController.hpp"

PipelineController::PipelineController(ConfirmedCalibration calibration, QString path, SharedStereoSource source, double exposure_us, QObject* parent)
    : QObject(parent), cancel_(std::make_shared<std::atomic_bool>(false)),
      worker_(new PipelineWorker(std::move(calibration), std::move(path), std::move(source), cancel_, exposure_us)) {
    qRegisterMetaType<PipelineState>();
    qRegisterMetaType<SharedGPUMesh>();
    worker_->moveToThread(&thread_);
    connect(worker_, &PipelineWorker::stateChanged, this, [this](PipelineState state) {
        state_ = std::move(state); emit stateChanged(state_);
    });
    connect(worker_, &PipelineWorker::images, this, &PipelineController::images);
    connect(worker_, &PipelineWorker::preview, this, &PipelineController::preview);
    connect(worker_, &PipelineWorker::depthImage, this, &PipelineController::depthImage);
    connect(worker_, &PipelineWorker::gpuMeshReady, this, &PipelineController::gpuMeshReady);
    connect(worker_, &PipelineWorker::maskReady, this, &PipelineController::maskReady);
    connect(worker_, &PipelineWorker::log, this, &PipelineController::log);
    connect(worker_, &PipelineWorker::actionFinished, this, [this] {
        const bool initialized = initializing_; initializing_ = false;
        busy_ = false; emit busyChanged(false);
        if (initialized && !stopping_)
            emit initializationFinished(!state_.action_failed && state_.engine_ready && state_.sam_ready && state_.ma_ready, state_.status);
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
void PipelineController::initialize(const QString& engine_path, const QString& sam_encoder, const QString& sam_decoder,
                                    const QString& ma_engine) {
    if (busy_ || stopping_) return;
    initializing_ = true;
    submit([engine_path,sam_encoder,sam_decoder,ma_engine](auto& worker) { worker.initialize(engine_path,sam_encoder,sam_decoder,ma_engine); });
}
quint64 PipelineController::requestMask(quint64 image_id, std::vector<fs::SamPrompt> prompts) {
    const quint64 revision = mask_request_->fetch_add(1)+1;
    if (stopping_ || prompts.empty()) return revision;
    auto* worker = worker_; auto current = mask_request_;
    QMetaObject::invokeMethod(worker,[worker,current,image_id,revision,prompts=std::move(prompts)] {
        worker->segment(image_id,revision,prompts,current);
    },Qt::QueuedConnection);
    return revision;
}
void PipelineController::cancel() { cancel_->store(true); }
void PipelineController::stop() {
    if (stopping_) return;
    stopping_ = true; cancel();
    QMetaObject::invokeMethod(worker_, &PipelineWorker::shutdown, Qt::QueuedConnection);
}
