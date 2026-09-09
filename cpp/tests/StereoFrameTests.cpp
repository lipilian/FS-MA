#include "fs/stereo/StereoFrame.hpp"

#include <chrono>
#include <cmath>
#include <filesystem>
#include <functional>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void expect_invalid(const std::function<void()>& action, const std::string& message) {
    try {
        action();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error("Expected invalid_argument: " + message);
}

StereoCalibration calibration() {
    return {
        (cv::Mat_<double>(3, 3) << 90, 0, 32, 0, 90, 24, 0, 0, 1),
        (cv::Mat_<double>(3, 3) << 90, 0, 32, 0, 90, 24, 0, 0, 1),
        cv::Mat::zeros(5, 1, CV_64F),
        cv::Mat::zeros(5, 1, CV_64F),
        cv::Mat::eye(3, 3, CV_64F),
        (cv::Mat_<double>(3, 1) << -0.05, 0, 0),
    };
}

cv::Mat patterned_image() {
    cv::Mat image(48, 64, CV_8UC3);
    cv::RNG random(42);
    random.fill(image, cv::RNG::UNIFORM, 0, 256);
    return image;
}

void same_image(const cv::Mat& a, const cv::Mat& b, const std::string& message) {
    require(a.size() == b.size() && a.type() == b.type() &&
            cv::norm(a, b, cv::NORM_INF) == 0.0, message + " (max difference: " +
            std::to_string(cv::norm(a, b, cv::NORM_INF)) + ")");
}

void same_rectification(StereoFrame& a, StereoFrame& b) {
    a.rectify();
    b.rectify();
    same_image(a.rectified_left(), b.rectified_left(), "Rectified left images differ");
    same_image(a.rectified_right(), b.rectified_right(), "Rectified right images differ");
    const auto x = a.rectified_camera_parameters();
    const auto y = b.rectified_camera_parameters();
    require(std::abs(x.fx - y.fx) < 1e-10 && std::abs(x.fy - y.fy) < 1e-10 &&
            std::abs(x.cx - y.cx) < 1e-10 && std::abs(x.cy - y.cy) < 1e-10 &&
            x.baseline_meters == y.baseline_meters, "Rectified camera parameters differ");
}

struct TemporaryDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("fs-stereo-test-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TemporaryDirectory() { require(std::filesystem::create_directory(path), "Could not create fixture directory"); }
    ~TemporaryDirectory() {
        std::error_code error;
        std::filesystem::remove_all(path, error);
    }
};

void write_calibration(const std::filesystem::path& path, const StereoCalibration& c) {
    cv::FileStorage file(path.string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    require(file.isOpened(), "Could not create calibration fixture");
    file << "left_camera_matrix" << c.left_camera_matrix
         << "right_camera_matrix" << c.right_camera_matrix
         << "left_distortion" << c.left_distortion
         << "right_distortion" << c.right_distortion
         << "right_to_left_rotation" << c.right_to_left_rotation
         << "right_to_left_translation" << c.right_to_left_translation;
}

void compare_capture(const std::filesystem::path& directory) {
    const auto left_path = directory / "left.png";
    const auto right_path = directory / "right.png";
    const auto calibration_path = directory / "calibration.json";
    cv::Mat left = cv::imread(left_path.string());
    cv::Mat right = cv::imread(right_path.string());
    require(!left.empty() && !right.empty(), "Capture images are missing");
    cv::cvtColor(left, left, cv::COLOR_BGR2RGB);
    cv::cvtColor(right, right, cv::COLOR_BGR2RGB);
    StereoFrame file_frame(left_path, right_path, calibration_path);
    StereoFrame memory_frame(left, right, StereoCalibration::from_file(calibration_path));
    same_image(file_frame.left(), left, "File input must convert BGR to RGB exactly once");
    same_image(memory_frame.left(), left, "Memory input must retain RGB values");
    same_image(file_frame.right(), right, "Right file input RGB differs");
    same_image(memory_frame.right(), right, "Right memory input RGB differs");
    same_rectification(file_frame, memory_frame);
}

void test_file_equivalence() {
    TemporaryDirectory fixture;
    const auto rgb = patterned_image();
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    require(cv::imwrite((fixture.path / "left.png").string(), bgr), "Could not save left fixture");
    cv::flip(bgr, bgr, 1);
    require(cv::imwrite((fixture.path / "right.png").string(), bgr), "Could not save right fixture");
    auto c = calibration();
    c.left_distortion.at<double>(0) = -0.1;
    c.right_distortion.at<double>(0) = -0.08;
    write_calibration(fixture.path / "calibration.json", c);
    compare_capture(fixture.path);

    c.left_camera_matrix.at<double>(0, 0) = -1;
    write_calibration(fixture.path / "calibration.json", c);
    expect_invalid([&] { StereoFrame bad(fixture.path / "left.png", fixture.path / "right.png",
                                       fixture.path / "calibration.json"); }, "File input shares calibration validation");
    cv::FileStorage incomplete((fixture.path / "missing.json").string(), cv::FileStorage::WRITE);
    incomplete << "left_camera_matrix" << c.left_camera_matrix;
    incomplete.release();
    bool missing_reported = false;
    try {
        StereoCalibration::from_file(fixture.path / "missing.json");
    } catch (const std::runtime_error& error) {
        missing_reported = std::string(error.what()).find("right_camera_matrix") != std::string::npos;
    }
    require(missing_reported, "Missing calibration field should be named in the error");
}

void test_ownership() {
    auto left = patterned_image();
    auto right = left.clone();
    auto c = calibration();
    StereoFrame expected(left, right, c);
    StereoFrame snapshot(left, right, c);
    left.setTo(cv::Scalar(0, 0, 0));
    right.setTo(cv::Scalar(255, 255, 255));
    for (auto* matrix : {&c.left_camera_matrix, &c.right_camera_matrix, &c.left_distortion,
                        &c.right_distortion, &c.right_to_left_rotation, &c.right_to_left_translation}) {
        matrix->setTo(0);
        matrix->release();
    }
    same_image(snapshot.left(), expected.left(), "Left image aliases the caller buffer");
    same_image(snapshot.right(), expected.right(), "Right image aliases the caller buffer");
    same_rectification(snapshot, expected);
    const auto camera = snapshot.rectified_camera_parameters();
    require(std::abs(camera.baseline_meters - 0.05) < 1e-7, "Baseline metre scale changed");
    require(std::abs(camera.fx - 90.0) < 1e-7, "Rectified focal length changed");
}

void test_strided_float_input() {
    const auto image = patterned_image();
    cv::Mat padded(50, 70, CV_8UC3, cv::Scalar(0, 0, 0));
    auto roi = padded(cv::Rect(2, 1, 64, 48));
    image.copyTo(roi);
    require(!roi.isContinuous(), "Image fixture must be non-contiguous");
    auto c = calibration();
    for (auto* matrix : {&c.left_camera_matrix, &c.right_camera_matrix, &c.left_distortion,
                        &c.right_distortion, &c.right_to_left_rotation, &c.right_to_left_translation}) {
        matrix->convertTo(*matrix, CV_32F);
    }
    // Non-contiguous float intrinsics and row-vector distortion/translation.
    cv::Mat intrinsics_backing = cv::Mat::zeros(3, 5, CV_32F);
    auto intrinsics_roi = intrinsics_backing(cv::Rect(1, 0, 3, 3));
    c.left_camera_matrix.copyTo(intrinsics_roi);
    c.left_camera_matrix = intrinsics_roi;
    c.left_distortion = c.left_distortion.t();
    c.right_to_left_translation = c.right_to_left_translation.t();
    StereoFrame expected(image, image, calibration());
    StereoFrame snapshot(roi, roi, c);
    padded.setTo(0);
    intrinsics_backing.setTo(0);
    same_rectification(snapshot, expected);
}

void test_invalid_inputs() {
    const auto rgb = patterned_image();
    const auto c = calibration();
    expect_invalid([&] { StereoFrame f(cv::Mat(), rgb, c); }, "Empty image");
    expect_invalid([&] { StereoFrame f(rgb, cv::Mat(48, 63, CV_8UC3), c); }, "Different image sizes");
    for (const int type : {CV_8UC1, CV_8UC4, CV_32FC3}) {
        expect_invalid([&] { StereoFrame f(cv::Mat(48, 64, type), rgb, c); }, "Unsupported image type");
    }
    const auto invalid_calibration = [&](const std::function<void(StereoCalibration&)>& mutate) {
        auto bad = calibration();
        mutate(bad);
        expect_invalid([&] { StereoFrame f(rgb, rgb, bad); }, "Invalid calibration");
    };
    invalid_calibration([](StereoCalibration& x) { x.left_camera_matrix = cv::Mat::eye(2, 2, CV_64F); });
    invalid_calibration([](StereoCalibration& x) { x.right_camera_matrix.at<double>(0, 0) = 0; });
    invalid_calibration([](StereoCalibration& x) { x.left_camera_matrix.at<double>(2, 2) = 0; });
    invalid_calibration([](StereoCalibration& x) { x.right_camera_matrix.at<double>(0, 2) = std::numeric_limits<double>::quiet_NaN(); });
    invalid_calibration([](StereoCalibration& x) { x.left_distortion = cv::Mat::zeros(6, 1, CV_64F); });
    invalid_calibration([](StereoCalibration& x) { x.left_distortion = cv::Mat::zeros(2, 2, CV_64F); });
    invalid_calibration([](StereoCalibration& x) { x.right_distortion = cv::Mat::zeros(5, 1, CV_64FC2); });
    invalid_calibration([](StereoCalibration& x) { x.right_distortion.release(); });
    invalid_calibration([](StereoCalibration& x) { x.right_distortion.at<double>(0) = std::numeric_limits<double>::infinity(); });
    invalid_calibration([](StereoCalibration& x) { x.right_to_left_rotation.at<double>(0, 0) = -1; });
    invalid_calibration([](StereoCalibration& x) { x.right_to_left_rotation.at<double>(0, 1) = 0.1; });
    invalid_calibration([](StereoCalibration& x) { x.right_to_left_translation.setTo(0); });
    invalid_calibration([](StereoCalibration& x) { x.right_to_left_translation = cv::Mat::ones(2, 1, CV_64F); });
    invalid_calibration([](StereoCalibration& x) { x.left_camera_matrix.convertTo(x.left_camera_matrix, CV_32S); });
    for (int count : {4, 5, 8, 12, 14}) {
        auto valid = calibration();
        valid.left_distortion = cv::Mat::zeros(1, count, CV_32F);
        valid.right_distortion = cv::Mat::zeros(count, 1, CV_64F);
        StereoFrame accepted(rgb, rgb, valid);
        accepted.rectify();
    }
}

}  // namespace

int main(int argc, char* argv[]) {
    try {
        test_file_equivalence();
        std::cout << "PASS file/memory equivalence and JSON errors\n";
        test_ownership();
        std::cout << "PASS image/calibration ownership and metre scale\n";
        test_strided_float_input();
        std::cout << "PASS strided buffers and float/row-vector calibration\n";
        test_invalid_inputs();
        std::cout << "PASS input validation and supported distortion lengths\n";
        if (argc == 2) {
            compare_capture(argv[1]);
            std::cout << "PASS real capture file/memory equivalence\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "FAIL " << error.what() << '\n';
        return 1;
    }
}
