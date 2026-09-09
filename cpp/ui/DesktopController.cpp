#include "DesktopController.hpp"
#include <QApplication>
#include <QMessageBox>
DesktopController::DesktopController() : window_(std::make_unique<CalibrationWindow>(calibration_)) {
    connect(window_.get(), &CalibrationWindow::closeRequested, this, [this] {
        cancelled_ = true; calibration_.stop();
    });
    connect(&calibration_, &CalibrationController::completed, this, [this](ConfirmedCalibration result, QString path) {
        if (cancelled_) return;
        confirmed_ = std::move(result); saved_path_ = std::move(path);
        window_->setEnabled(false); calibration_.stop();
    });
    connect(&calibration_, &CalibrationController::stopped, this, [this] {
        if (confirmed_) QMessageBox::information(window_.get(), "Calibration complete",
            (confirmed_->checked ? "Calibration checked and saved to:\n" : "Continuing with saved calibration:\n") + saved_path_ +
            "\n\nCameras have been released. The reconstruction window will be added in the next stage.");
        window_->allowClose(); QApplication::quit();
    });
}
void DesktopController::show() { window_->show(); }
