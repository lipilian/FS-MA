#pragma once

#include <opencv2/core.hpp>

namespace fs {
// XYZ is expressed in the rectified left camera frame: origin at the optical
// centre, +X right, +Y down, +Z forward. Intrinsics use image_size's pixel
// grid.
struct MeshCamera {
    cv::Matx33d intrinsics;
    cv::Size image_size;
};

} // namespace fs
