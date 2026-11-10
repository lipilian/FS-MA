#include "DesktopController.hpp"
#include <QApplication>
#include <QMessageBox>
#include <QEvent>
#include <QTimer>
#include <QWindow>

namespace {
// Request maximization after the native window is mapped. On GNOME/X11 an
// initial showMaximized() can be lost while the first window size is negotiated.
// This is a one-time startup action; later user restores/resizes are untouched.
class MaximizeOnFirstExpose final : public QObject {
public:
    explicit MaximizeOnFirstExpose(QWidget* window)
        : QObject(window), window_(window), handle_(window->windowHandle()) {
        if (!handle_ || handle_->isExposed()) request();
        else handle_->installEventFilter(this);
    }
protected:
    bool eventFilter(QObject* watched, QEvent* event) override {
        if (watched == handle_ && event->type() == QEvent::Expose && handle_->isExposed()) request();
        return false;
    }
private:
    void request() {
        if (handle_) handle_->removeEventFilter(this);
        QTimer::singleShot(0, window_, [window = window_] {
            if (window->isVisible() && !window->isMinimized()) window->showMaximized();
        });
        deleteLater();
    }
    QWidget* window_;
    QWindow* handle_;
};
void showMainWindow(QWidget* window) {
    window->show();
    new MaximizeOnFirstExpose(window);
}
} // namespace

void DesktopController::prepareInference() {
    if (pipeline_ || cancelled_) return;
    try {
        // Model loading and all later inference share one pipeline worker.
        // ZED obtains rectified calibration from the SDK when it connects.
        pipeline_ = std::make_unique<PipelineController>(ConfirmedCalibration{}, QString{}, SharedStereoSource{});
        splash_ = std::make_unique<InferenceSplashWindow>(false);
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
        splash_->show();
        initializeInference(splash_->enginePath(),splash_->samEncoderPath(),splash_->samDecoderPath(),splash_->maEncoderPath(),splash_->maPoseScalePath());
    } catch (const std::exception& e) {
        pipeline_.reset();
        QMessageBox::critical(splash_.get(), "Unable to prepare reconstruction", QString::fromUtf8(e.what()));
        if (splash_) splash_->allowClose();
        QApplication::quit();
    }
}
void DesktopController::initializeInference(const QString& engine_path, const QString& sam_encoder, const QString& sam_decoder,
                                           const QString& ma_encoder, const QString& ma_pose_scale) {
    if (cancelled_ || !splash_) return;
    splash_->startLoading(); pipeline_->initialize(engine_path,sam_encoder,sam_decoder,ma_encoder,ma_pose_scale);
}
void DesktopController::openReconstruction() {
    if (reconstruction_ || cancelled_) return;
    try {
        reconstruction_ = std::make_unique<ReconstructionWindow>(*pipeline_, ConfirmedCalibration{}, QString{});
        connect(reconstruction_.get(), &ReconstructionWindow::closeRequested, pipeline_.get(), &PipelineController::stop);
        showMainWindow(reconstruction_.get()); splash_->allowClose(); splash_.reset();
        pipeline_->submit([](auto& worker) { worker.connectCameras(CameraMode::Zed); });
    } catch (const std::exception& e) {
        if (splash_) splash_->showFailure(QString::fromUtf8(e.what()));
    }
}
void DesktopController::show() {
    prepareInference();
}
