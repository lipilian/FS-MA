#pragma once
#include "ReconstructionWindow.hpp"
#include "InferenceSplashWindow.hpp"
#include <memory>

// Background model loading -> ZED scanning and capture imports.
class DesktopController : public QObject {
    Q_OBJECT
public:
    DesktopController() = default;
    void show();
private:
    void prepareInference();
    void initializeInference(const QString& engine_path, const QString& sam_encoder, const QString& sam_decoder,
                             const QString& ma_encoder, const QString& ma_pose_scale);
    void openReconstruction();
    std::unique_ptr<PipelineController> pipeline_;
    std::unique_ptr<ReconstructionWindow> reconstruction_;
    std::unique_ptr<InferenceSplashWindow> splash_;
    bool cancelled_{false};
};
