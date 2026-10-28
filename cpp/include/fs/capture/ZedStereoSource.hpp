#pragma once

#include "fs/capture/IStereoSource.hpp"
#include "fs/stereo/StereoCalibration.hpp"
#include <memory>
#include <string>

// SDK-rectified RGB at 1920x1080, center-cropped to 1296x1080.
// FS performs the same INTER_LINEAR resize to 960x800 used for imported images.
// Calibration describes the cropped grid, in metres; no second rectification.
class ZedStereoSource final : public IStereoSource {
public:
    ZedStereoSource();
    ~ZedStereoSource() override;
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
