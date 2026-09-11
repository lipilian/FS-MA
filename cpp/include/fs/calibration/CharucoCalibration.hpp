#pragma once

#include "fs/capture/IStereoSource.hpp"
#include "fs/stereo/StereoCalibration.hpp"
#include <opencv2/aruco/charuco.hpp>
#include <string>
#include <vector>

namespace fs::calibration {
inline constexpr double kQualityThresholdPx = 1.0;
struct BoardConfig {
    int squares_x{10}, squares_y{8};
    double square_length_m{0.0665}, marker_length_m{0.0505};
    std::string dictionary_name{"DICT_4X4_250"};
};
struct Detection {
    int markers{0};
    std::vector<int> ids;
    std::vector<cv::Point2f> corners;
    // Preserve marker geometry for preview/history overlays, even without ChArUco corners.
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners;
};
struct Sample {
    StereoCameraPair images; // Owned snapshots; never mutated after publication.
    Detection left, right;
};
struct Quality {
    double left{0}, right{0}, stereo{0};
    int pairs{0};
};
struct SessionResult {
    BoardConfig board;
    StereoCalibration calibration;
    cv::Size image_size;
    std::string left_serial{"21LJ530"}, right_serial{"21LJ548"};
    Quality solve, check;
    bool checked{false};
    double threshold_px{kQualityThresholdPx}; // Desktop uses the fixed 1 px limit; retained in saved metadata.
};
std::vector<std::string> dictionary_names();
cv::Ptr<cv::aruco::CharucoBoard> make_board(const BoardConfig& config);
bool same_board(const BoardConfig& a, const BoardConfig& b);
Detection detect(const cv::Mat& rgb, const cv::Ptr<cv::aruco::CharucoBoard>& board);
cv::Mat annotated(const cv::Mat& rgb, const Detection& detection);
std::size_t common_corners(const Detection& left, const Detection& right);
std::uint64_t arrival_skew(const StereoCameraPair& pair);
SessionResult solve(const BoardConfig& board, const std::vector<Sample>& samples);
Quality check(const SessionResult& result, const Sample& sample);
void validate(const SessionResult& result);
std::string serialize(const SessionResult& result);
SessionResult load(const std::string& path); // Requires camera and grid metadata; never trusts saved check.
SessionResult clone(const SessionResult& result); // Deep ownership for handoff to the next window.
} // namespace fs::calibration
