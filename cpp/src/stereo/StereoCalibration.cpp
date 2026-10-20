#include "fs/stereo/StereoCalibration.hpp"

#include <stdexcept>
#include <string>

namespace {

cv::Mat read_matrix(const cv::FileStorage& storage, const char* key, const char* short_key, int rows = 0) {
    const cv::FileNode node = storage[key].empty() ? storage[short_key] : storage[key];
    if (node.empty()) {
        throw std::runtime_error(std::string("Calibration is missing '") + key + "'");
    }
    cv::Mat matrix;
    if (node.isSeq()) {
        if (node.size()==0) throw std::runtime_error(std::string("Empty calibration matrix: ")+key);
        const bool nested=(*node.begin()).isSeq();
        const int columns=nested ? int((*node.begin()).size()) : int(node.size());
        if (columns==0) throw std::runtime_error(std::string("Empty calibration matrix: ")+key);
        matrix.create(nested ? int(node.size()) : 1,columns,CV_64F);
        int row=0, column=0;
        const auto number=[&](const cv::FileNode& value) {
            if (!value.isInt() && !value.isReal())
                throw std::runtime_error(std::string("Calibration matrix must contain numbers: ")+key);
            return double(value);
        };
        for (const auto& entry:node) {
            if (nested) {
                if (!entry.isSeq() || int(entry.size())!=columns)
                    throw std::runtime_error(std::string("Ragged calibration matrix: ")+key);
                column=0;
                for (const auto& value:entry) matrix.at<double>(row,column++)=number(value);
                ++row;
            } else matrix.at<double>(0,column++)=number(entry);
        }
        if (!nested && rows && matrix.total()==std::size_t(rows*rows)) matrix=matrix.reshape(1,rows);
    } else node >> matrix;
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
        read_matrix(storage, "left_camera_matrix", "K1", 3),
        read_matrix(storage, "right_camera_matrix", "K2", 3),
        read_matrix(storage, "left_distortion", "D1"),
        read_matrix(storage, "right_distortion", "D2"),
        read_matrix(storage, "right_to_left_rotation", "R", 3),
        read_matrix(storage, "right_to_left_translation", "T"),
    };
}
