#include <exception>
#include <filesystem>
#include <string>

#include "fs/inference/FS.hpp"
#include "fs/core/Logger.hpp"
#include "fs/stereo/StereoFrame.hpp"

namespace {

void print_usage(const char* executable) {
    Logger::error(std::string("Usage: ") + executable +
                  " <capture-directory> [--measure]\n\nThe directory must contain left.png, right.png, and calibration.json.");
}

}  // namespace

int main(int argc, char* argv[]) {
    const bool measure = argc == 3 && std::string(argv[2]) == "--measure";
    if ((argc != 2 && argc != 3) || (argc == 3 && !measure)) {
        print_usage(argv[0]);
        return 2;
    }

    const std::filesystem::path capture_directory = argv[1];
    const auto left_path = capture_directory / "left.png";
    const auto right_path = capture_directory / "right.png";
    const auto calibration_path = capture_directory / "calibration.json";

    try {
        FS fs;
        fs.loadEngine("onnx/foundationstereo_800x960_gwc_plugin.engine");

        Logger::log("Loading stereo capture.");
        StereoFrame frame(left_path, right_path, calibration_path);

        Logger::log("Rectifying RGB stereo pair.");
        frame.rectify();

        fs.set_model_camera_parameters(
            frame.rectified_camera_parameters(), frame.rectified_left().size());
        fs.prepare_stereo_images(frame.rectified_left(), frame.rectified_right());
        if (measure) {
            fs.inference_time_measure();
        } else {
            fs.inference();
        }

        Logger::log("Computing XYZ map (0 < depth <= 1 m).");
        fs.compute_xyz_map();
        Logger::log("XYZ reconstruction completed on GPU.");
        return 0;
    } catch (const std::exception& error) {
        Logger::error(std::string("stereo_rectify: ") + error.what());
        return 1;
    }
}
