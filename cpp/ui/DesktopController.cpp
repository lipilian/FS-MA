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
            // The old worker has finished; only the pipeline now touches the same source.
            pipeline_ = std::make_unique<PipelineController>(confirmed_, saved_path_, std::move(source_));
            reconstruction_ = std::make_unique<ReconstructionWindow>(*pipeline_, confirmed_, saved_path_);
            connect(reconstruction_.get(), &ReconstructionWindow::closeRequested, pipeline_.get(), &PipelineController::stop);
            connect(pipeline_.get(), &PipelineController::stopped, this, [this] {
                reconstruction_->allowClose(); QApplication::quit();
            });
            reconstruction_->show(); window_->allowClose(); window_.reset();
        } catch (const std::exception& e) {
            pipeline_.reset();
            if (source_) source_->stop();
            source_.reset();
            QMessageBox::critical(window_.get(), "Unable to open reconstruction", QString::fromUtf8(e.what()));
            window_->allowClose(); QApplication::quit();
        }
    });
}
void DesktopController::show() { window_->show(); }
