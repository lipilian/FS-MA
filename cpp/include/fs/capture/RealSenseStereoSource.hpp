#pragma once

#include "fs/capture/IStereoSource.hpp"
#include "fs/stereo/StereoCalibration.hpp"
#include <memory>
#include <string>

// D435 left/right infrared streams, center-cropped from 1280x800 to 960x800.
// Intrinsics are shifted to the cropped grid. Frames are expanded to RGB for the existing
// FoundationStereo pipeline; depth still comes from FoundationStereo.
class RealSenseStereoSource final : public IStereoSource {
public:
    RealSenseStereoSource();
    ~RealSenseStereoSource() override;
    static bool available() noexcept;
    void start() override;
    void stop() noexcept override;
    bool running() const noexcept override;
    std::optional<StereoCameraPair> wait_for_pair(std::chrono::milliseconds timeout) override;
    const StereoCalibration& calibration() const;
    cv::Size image_size() const;
    std::string serial() const;
    std::string calibration_json() const;
private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
