#pragma once
#include <QImage>
#include <QStringList>
#include <opencv2/core.hpp>
#include <functional>
#include <vector>

// Owned CPU snapshots of the completed FS inputs/results. Export never runs a
// model or rebuilds geometry. View order defines the two-digit file numbering.
struct ScanExportView {
    cv::Matx33d intrinsic;
    cv::Matx44d camera_to_world; // Original MA pose before ICP, exported as "extrinsic"; identity for a single view.
    QImage rgb, mask;
    cv::Mat depth;
};

QStringList scanResultFileNames(int views);
QStringList writeScanResults(const QString& directory, const std::vector<ScanExportView>& views,
                            bool overwrite = false, const std::function<void()>& checkpoint = {});
