#include "fs/stereo/StereoCalibration.hpp"

#include <stdexcept>
#include <string>

namespace {

cv::Mat read_matrix(const cv::FileStorage& storage, const char* key) {
    const cv::FileNode node = storage[key];
    if (node.empty()) {
        throw std::runtime_error(std::string("Calibration is missing '") + key + "'");
    }
    cv::Mat matrix;
    node >> matrix;
    if (matrix.empty()) {
        throw std::runtime_error(std::string("Calibration matrix '") + key + "' is empty or invalid");
    }
    return matrix;
}

}  // namespace

StereoCalibration StereoCalibration::from_file(const std::filesystem::path& path) {
    cv::FileStorage storage(path.string(), cv::FileStorage::READ | cv::FileStorage::FORMAT_JSON);
    if (!storage.isOpened()) {
        throw std::runtime_error("Unable to open calibration JSON: " + path.string());
    }
    return {
        read_matrix(storage, "left_camera_matrix"),
        read_matrix(storage, "right_camera_matrix"),
        read_matrix(storage, "left_distortion"),
        read_matrix(storage, "right_distortion"),
        read_matrix(storage, "right_to_left_rotation"),
        read_matrix(storage, "right_to_left_translation"),
    };
}
