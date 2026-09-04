#include <exception>
#include <filesystem>
#include <string>

#include "FS.hpp"
#include "Logger.hpp"
#include "StereoFrame.hpp"

namespace {

void print_usage(const char* executable) {
    Logger::error(std::string("Usage: ") + executable +
                  " <capture-directory>\n\nThe directory must contain left.png, right.png, and calibration.json.");
}

}  // namespace

int main(int argc, char* argv[]) {
    if (argc != 2) {
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

        fs.prepare_stereo_images(frame.rectified_left(), frame.rectified_right());

        // Continue processing frame.rectified_left() and frame.rectified_right() here.
        return 0;
    } catch (const std::exception& error) {
        Logger::error(std::string("stereo_rectify: ") + error.what());
        return 1;
    }
}
