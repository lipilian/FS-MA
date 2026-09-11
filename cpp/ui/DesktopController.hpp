#pragma once
#include "CalibrationWindow.hpp"
#include "ReconstructionWindow.hpp"
#include "InferenceSplashWindow.hpp"
#include <memory>

// Calibration -> background inference initialization splash -> ready reconstruction.
class DesktopController : public QObject {
    Q_OBJECT
public:
    DesktopController();
    void show();
private:
    void initializeInference(const QString& engine_path, const QString& sam_encoder, const QString& sam_decoder);
    void openReconstruction();
    CalibrationController calibration_;
    std::unique_ptr<CalibrationWindow> window_;
    ConfirmedCalibration confirmed_;
    QString saved_path_;
    SharedStereoSource source_;
    double camera_exposure_us_{SentechStereoOptions{}.exposure_us};
    std::unique_ptr<PipelineController> pipeline_;
    std::unique_ptr<ReconstructionWindow> reconstruction_;
    std::unique_ptr<InferenceSplashWindow> splash_;
    bool cancelled_{false};
};
