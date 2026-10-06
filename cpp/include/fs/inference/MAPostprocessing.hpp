#pragma once

#include <opencv2/core/matx.hpp>
#include <vector>

namespace fs::ma_postprocessing {

// Decode RAW [V,7] (tx,ty,tz,qx,qy,qz,qw) and the scalar log-scale exactly as
// the MA pose/scale adaptors: normalized XYZW rotation, metric translation.
// Returns camera-to-world matrices in MA's predicted world, including view 0.
std::vector<cv::Matx44d> camera_poses(const float* raw_poses, int view_count, float raw_scale);

} // namespace fs::ma_postprocessing
