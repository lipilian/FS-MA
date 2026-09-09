#pragma once

#include "fs/capture/IStereoSource.hpp"
#include <memory>
#include <string>
#include <vector>

struct SentechDeviceInfo {
    std::string display_name;
    std::string user_name;
    std::string serial;
};

struct SentechStereoOptions {
    // Match exact display name, user-defined name, or serial; never discovery order.
    std::string left{"STC-MCS500U3V(21LJ530)"};
    std::string right{"STC-MCS500U3V(21LJ548)"};
    double exposure_us{100000.0};
    std::chrono::milliseconds max_arrival_skew{100};
};

class SentechStereoSource final : public IStereoSource {
public:
    explicit SentechStereoSource(SentechStereoOptions options = {});
    ~SentechStereoSource() override;
    SentechStereoSource(const SentechStereoSource&) = delete;
    SentechStereoSource& operator=(const SentechStereoSource&) = delete;

    static std::vector<SentechDeviceInfo> list_devices();
    void start() override;
    void stop() noexcept override;
    bool running() const noexcept override;
    std::optional<StereoCameraPair> wait_for_pair(std::chrono::milliseconds timeout) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
