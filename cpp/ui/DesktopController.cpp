#include "DesktopController.hpp"
#include <QApplication>
#include <QMessageBox>
DesktopController::DesktopController() : window_(std::make_unique<CalibrationWindow>(calibration_)) {
    connect(window_.get(), &CalibrationWindow::closeRequested, this, [this] {
        cancelled_ = true; calibration_.stop();
    });
    connect(&calibration_, &CalibrationController::completed, this,
        [this](ConfirmedCalibration result, QString path, SharedStereoSource source) {
        if (cancelled_ || confirmed_) { if (source) source->stop(); return; }
        confirmed_ = std::move(result); saved_path_ = std::move(path); source_ = std::move(source);
        window_->setEnabled(false); calibration_.stop();
    });
    connect(&calibration_, &CalibrationController::stopped, this, [this] {
        if (cancelled_ || !confirmed_) {
            if (source_) source_->stop();
            source_.reset(); window_->allowClose(); QApplication::quit(); return;
        }
        try {
            // The old worker has finished. Model initialization and inference now
            // share this one pipeline worker and its FS instance for the session.
            pipeline_ = std::make_unique<PipelineController>(confirmed_, saved_path_, std::move(source_));
            splash_ = std::make_unique<InferenceSplashWindow>();
            connect(splash_.get(), &InferenceSplashWindow::retryRequested, this, &DesktopController::initializeInference);
            connect(splash_.get(), &InferenceSplashWindow::closeRequested, this, [this] {
                cancelled_ = true; pipeline_->stop();
            });
            connect(pipeline_.get(), &PipelineController::stateChanged, this, [this](const PipelineState& state) {
                if (splash_ && !cancelled_) splash_->setStatus(state.status);
            });
            connect(pipeline_.get(), &PipelineController::initializationFinished, this, [this](bool success, const QString& message) {
                if (cancelled_) return;
                if (success) openReconstruction();
                else if (splash_) splash_->showFailure(message);
            });
            connect(pipeline_.get(), &PipelineController::stopped, this, [this] {
                if (reconstruction_) reconstruction_->allowClose();
                if (splash_) splash_->allowClose();
                QApplication::quit();
            });
            splash_->show(); window_->allowClose(); window_.reset();
            initializeInference(splash_->enginePath());
        } catch (const std::exception& e) {
            pipeline_.reset();
            if (source_) source_->stop();
            source_.reset();
            QMessageBox::critical(window_.get(), "Unable to prepare reconstruction", QString::fromUtf8(e.what()));
            if (window_) window_->allowClose();
            if (splash_) splash_->allowClose();
            QApplication::quit();
        }
    });
}
void DesktopController::initializeInference(const QString& engine_path) {
    if (cancelled_ || !splash_) return;
    splash_->startLoading(); pipeline_->initialize(engine_path);
}
void DesktopController::openReconstruction() {
    if (reconstruction_ || cancelled_) return;
    try {
        reconstruction_ = std::make_unique<ReconstructionWindow>(*pipeline_, confirmed_, saved_path_);
        connect(reconstruction_.get(), &ReconstructionWindow::closeRequested, pipeline_.get(), &PipelineController::stop);
        reconstruction_->show(); splash_->allowClose(); splash_.reset();
    } catch (const std::exception& e) {
        if (splash_) splash_->showFailure(QString::fromUtf8(e.what()));
    }
}
void DesktopController::show() { window_->show(); }
