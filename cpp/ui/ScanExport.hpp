#pragma once
#include <QImage>
#include <QStringList>
#include <opencv2/core.hpp>
#include <functional>
#include <optional>
#include <vector>

// Owned CPU snapshots of the completed FS inputs/results. Export never runs a
// model or rebuilds geometry. View order defines the two-digit file numbering.
struct ScanExportView {
    cv::Matx33d intrinsic;
    cv::Matx44d camera_to_world; // Currently active pose, retained as "extrinsic".
    QImage rgb, mask;
    cv::Mat depth;
    cv::Matx44d camera_to_world_before_icp{cv::Matx44d::eye()};
    std::optional<cv::Matx44d> camera_to_world_after_icp; // Last completed ICP; null if never run.
    bool icp_applied{false}; // False after Undo ICP, even when a result is retained.
};

QStringList scanResultFileNames(int views);
QStringList writeScanResults(const QString& directory, const std::vector<ScanExportView>& views,
                            bool overwrite = false, const std::function<void()>& checkpoint = {});
