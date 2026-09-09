#pragma once
#include "CalibrationWindow.hpp"
#include <memory>

// Owns the first-stage session. ReconstructionWindow will be added at the
// successful-completion boundary; ordinary close never enters that boundary.
class DesktopController : public QObject {
    Q_OBJECT
public:
    DesktopController();
    void show();
private:
    CalibrationController calibration_;
    std::unique_ptr<CalibrationWindow> window_;
    ConfirmedCalibration confirmed_;
    QString saved_path_;
    bool cancelled_{false};
};
