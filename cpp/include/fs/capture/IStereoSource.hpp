#pragma once

#include <chrono>
#include <cstdint>
#include <optional>
#include <opencv2/core.hpp>

struct CameraFrame {
    cv::Mat rgb; // Owned CV_8UC3 snapshot, independent of SDK buffers.
    std::uint64_t frame_id{0};
    std::uint64_t device_timestamp_ns{0}; // Camera-local clock; not comparable across devices.
    std::uint64_t host_arrival_ns{0};    // Host steady clock, measured on buffer retrieval.
};

struct StereoCameraPair {
    CameraFrame left;
    CameraFrame right;
    // A software pair from independent continuous streams, NOT hardware synchronized.
};

class IStereoSource {
public:
    virtual ~IStereoSource() = default;
    virtual void start() = 0;
    virtual void stop() noexcept = 0;
    virtual bool running() const noexcept = 0;
    // Call lifecycle and wait methods from one control thread. nullopt means timeout/stopped.
    virtual std::optional<StereoCameraPair> wait_for_pair(std::chrono::milliseconds timeout) = 0;
};
