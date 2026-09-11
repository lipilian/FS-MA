#pragma once
#include "CalibrationWindow.hpp"
#include "ReconstructionWindow.hpp"
#include <memory>

// Creates the second window only after successful calibration and worker shutdown.
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
    SharedStereoSource source_;
    std::unique_ptr<PipelineController> pipeline_;
    std::unique_ptr<ReconstructionWindow> reconstruction_;
    bool cancelled_{false};
};
