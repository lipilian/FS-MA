#pragma once

#include "fs/capture/SentechStereoSource.hpp"
#include <array>
#include <condition_variable>
#include <mutex>
#include <stdexcept>

namespace fs::capture_detail {

inline std::array<std::size_t, 2> select_roles(const std::vector<SentechDeviceInfo>& devices,
                                             const SentechStereoOptions& options) {
    std::array<std::size_t, 2> selected{devices.size(), devices.size()};
    const std::array<std::string, 2> names{options.left, options.right};
    for (std::size_t side = 0; side < 2; ++side) {
        for (std::size_t i = 0; i < devices.size(); ++i) {
            if (names[side] == devices[i].display_name || names[side] == devices[i].user_name ||
                names[side] == devices[i].serial) {
                if (selected[side] != devices.size())
                    throw std::runtime_error("Ambiguous camera identity: " + names[side]);
                selected[side] = i;
            }
        }
        if (selected[side] == devices.size())
            throw std::runtime_error("Camera not found: " + names[side] + "; use --list to inspect devices");
    }
    if (selected[0] == selected[1])
        throw std::runtime_error("Left and right identities resolve to the same physical camera");
    return selected;
}

// Latest-only buffer: each delivered pair consumes one fresh frame from both sides.
// Owned cv::Mat storage is transferred out, never mutated by subsequent publications.
class PairBuffer {
public:
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        frames_ = {};
        error_.clear();
        stopped_ = false;
    }
    void stop() {
        std::lock_guard<std::mutex> lock(mutex_);
        stopped_ = true;
        frames_ = {};
        changed_.notify_all();
    }
    void fail(std::string error) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (error_.empty()) error_ = std::move(error);
        changed_.notify_all();
    }
    void publish(std::size_t side, CameraFrame frame) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!stopped_) frames_.at(side) = std::move(frame);
        changed_.notify_all();
    }
    std::optional<StereoCameraPair> wait(std::chrono::milliseconds timeout,
                                        std::chrono::milliseconds max_skew) {
        const auto deadline = std::chrono::steady_clock::now() + timeout;
        std::unique_lock<std::mutex> lock(mutex_);
        while (true) {
            if (!error_.empty()) throw std::runtime_error(error_);
            if (stopped_) return std::nullopt;
            if (frames_[0] && frames_[1]) {
                const auto left = frames_[0]->host_arrival_ns;
                const auto right = frames_[1]->host_arrival_ns;
                const auto delta = left > right ? left - right : right - left;
                if (delta <= static_cast<std::uint64_t>(max_skew.count()) * 1000000ULL) {
                    if (frames_[0]->rgb.size() != frames_[1]->rgb.size())
                        throw std::runtime_error("Left and right camera image dimensions differ");
                    StereoCameraPair pair{std::move(*frames_[0]), std::move(*frames_[1])};
                    frames_ = {};
                    return pair;
                }
                frames_[left < right ? 0 : 1].reset();
            }
            if (changed_.wait_until(lock, deadline) == std::cv_status::timeout) return std::nullopt;
        }
    }
private:
    std::mutex mutex_;
    std::condition_variable changed_;
    std::array<std::optional<CameraFrame>, 2> frames_;
    bool stopped_{true};
    std::string error_;
};

} // namespace fs::capture_detail
