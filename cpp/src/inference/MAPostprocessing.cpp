#include "fs/inference/MAPostprocessing.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace fs::ma_postprocessing {

std::vector<cv::Matx44d> camera_poses(const float* raw_poses, int view_count, float raw_scale) {
    if (!raw_poses || view_count < 2 || view_count > 5)
        throw std::runtime_error("MapAnything: camera poses require 2..5 views");
    const float scale = std::max(std::exp(raw_scale), 1e-8F);
    if (!std::isfinite(raw_scale) || !std::isfinite(scale))
        throw std::runtime_error("MapAnything: non-finite predicted camera scale");
    std::vector<cv::Matx44d> result;
    result.reserve(view_count);
    for (int view = 0; view < view_count; ++view) {
        const float* p = raw_poses + view * 7;
        for (int c = 0; c < 7; ++c)
            if (!std::isfinite(p[c])) throw std::runtime_error("MapAnything: non-finite predicted camera pose");
        const double norm = std::sqrt(double(p[3]) * p[3] + double(p[4]) * p[4] +
                                      double(p[5]) * p[5] + double(p[6]) * p[6]);
        if (norm == 0) throw std::runtime_error("MapAnything: zero predicted camera quaternion");
        // The adaptor and quaternion_to_rotation_matrix both normalize the
        // quaternion. Their combined result is q / norm(q) for nonzero q.
        const double x = p[3] / norm, y = p[4] / norm, z = p[5] / norm, w = p[6] / norm;
        const cv::Vec3f t(p[0] * scale, p[1] * scale, p[2] * scale);
        for (float value : t.val)
            if (!std::isfinite(value)) throw std::runtime_error("MapAnything: non-finite metric camera translation");
        result.emplace_back(
            1 - 2 * (y*y + z*z), 2 * (x*y - w*z), 2 * (x*z + w*y), t[0],
            2 * (x*y + w*z), 1 - 2 * (x*x + z*z), 2 * (y*z - w*x), t[1],
            2 * (x*z - w*y), 2 * (y*z + w*x), 1 - 2 * (x*x + y*y), t[2],
            0, 0, 0, 1);
    }
    return result;
}

} // namespace fs::ma_postprocessing
