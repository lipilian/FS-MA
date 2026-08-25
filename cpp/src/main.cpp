#include <exception>
#include <filesystem>
#include <iostream>
#include <string>

#include "StereoFrame.hpp"

namespace {

void print_usage(const char* executable) {
    std::cerr << "Usage: " << executable
              << " <capture-directory>\n"
              << "\nThe directory must contain left.png, right.png, and calibration.json.\n";
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
        StereoFrame frame(left_path, right_path, calibration_path);
        frame.rectify();

        const cv::Mat& rectified_left = frame.rectified_left();
        const cv::Mat& rectified_right = frame.rectified_right();

        // Continue processing the RGB rectified_left and rectified_right images here.
        std::cout << "Rectification completed in memory:\n"
                  << "  left:  " << rectified_left.cols << "x" << rectified_left.rows << '\n'
                  << "  right: " << rectified_right.cols << "x" << rectified_right.rows << '\n'
                  << "Baseline: " << frame.baseline_meters() << " m\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "stereo_rectify: " << error.what() << '\n';
        return 1;
    }
}
