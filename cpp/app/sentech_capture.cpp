#include "fs/capture/SentechStereoSource.hpp"
#include "fs/inference/FS.hpp"
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

namespace {
struct Options {
    SentechStereoOptions camera;
    bool list{false}, preview{false}, infer{false};
    int timeout_ms{5000};
    std::filesystem::path output, calibration;
    std::filesystem::path engine{"onnx/foundationstereo_800x960_gwc_plugin.engine"};
};

void usage() {
    std::cout << "Usage: fs_sentech [--list] [--left ID] [--right ID]\n"
                 "  [--preview] [--capture NEW_DIRECTORY] [--calibration JSON] [--infer]\n"
                 "  [--engine PATH] [--exposure-us 50000] [--timeout-ms 5000]\n"
                 "  [--max-arrival-skew-ms 100]\n\n"
                 "Default: preview the configured left/right cameras. Space freezes one pair; Q/Esc exits.\n"
                 "--capture without --preview saves the first available pair and exits.\n"
                 "--infer requires --calibration for this camera rig and image resolution.\n"
                 "Continuous independent streams: host arrival gating is NOT exposure synchronization.\n";
}

Options parse(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string flag = argv[i];
        const auto value = [&]() -> std::string {
            if (++i == argc) throw std::invalid_argument("Missing value for " + flag);
            return argv[i];
        };
        const auto integer = [&]() {
            const auto input = value();
            std::size_t used = 0;
            const int number = std::stoi(input, &used);
            if (used != input.size() || number <= 0) throw std::invalid_argument("Invalid value for " + flag);
            return number;
        };
        if (flag == "--list") options.list = true;
        else if (flag == "--preview") options.preview = true;
        else if (flag == "--infer") options.infer = true;
        else if (flag == "--left") options.camera.left = value();
        else if (flag == "--right") options.camera.right = value();
        else if (flag == "--capture") options.output = value();
        else if (flag == "--calibration") options.calibration = value();
        else if (flag == "--engine") options.engine = value();
        else if (flag == "--timeout-ms") options.timeout_ms = integer();
        else if (flag == "--max-arrival-skew-ms") options.camera.max_arrival_skew = std::chrono::milliseconds(integer());
        else if (flag == "--exposure-us") {
            const auto input = value();
            std::size_t used = 0;
            options.camera.exposure_us = std::stod(input, &used);
            if (used != input.size()) throw std::invalid_argument("Invalid exposure");
        } else throw std::invalid_argument("Unknown option: " + flag);
    }
    if (options.infer && options.calibration.empty()) throw std::invalid_argument("--infer requires --calibration");
    if (!options.output.empty() && std::filesystem::exists(options.output))
        throw std::invalid_argument("Capture directory already exists; choose a new directory");
    return options;
}

void preview(const StereoCameraPair& pair) {
    cv::Mat left, right, canvas;
    const double scale = std::min(640.0 / pair.left.rgb.cols, 480.0 / pair.left.rgb.rows);
    cv::resize(pair.left.rgb, left, {}, scale, scale);
    cv::resize(pair.right.rgb, right, left.size());
    cv::cvtColor(left, left, cv::COLOR_RGB2BGR);
    cv::cvtColor(right, right, cv::COLOR_RGB2BGR);
    cv::putText(left, "LEFT", {12, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);
    cv::putText(right, "RIGHT", {12, 30}, cv::FONT_HERSHEY_SIMPLEX, 0.8, {0, 255, 0}, 2);
    cv::hconcat(left, right, canvas);
    cv::imshow("Sentech stereo | free-running | Space: capture, Q: exit", canvas);
}

void save(const StereoCameraPair& pair, const Options& options) {
    if (!std::filesystem::create_directories(options.output))
        throw std::runtime_error("Unable to create new capture directory");
    cv::Mat bgr;
    cv::cvtColor(pair.left.rgb, bgr, cv::COLOR_RGB2BGR);
    if (!cv::imwrite((options.output / "left.png").string(), bgr)) throw std::runtime_error("Saving left image failed");
    cv::cvtColor(pair.right.rgb, bgr, cv::COLOR_RGB2BGR);
    if (!cv::imwrite((options.output / "right.png").string(), bgr)) throw std::runtime_error("Saving right image failed");
    if (!options.calibration.empty())
        std::filesystem::copy_file(options.calibration, options.output / "calibration.json");
    cv::FileStorage metadata((options.output / "capture.json").string(), cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);
    if (!metadata.isOpened()) throw std::runtime_error("Saving capture metadata failed");
    metadata << "pairing" << "independent_continuous_streams_host_arrival_gate"
             << "left_selector" << options.camera.left << "right_selector" << options.camera.right
             << "width" << pair.left.rgb.cols << "height" << pair.left.rgb.rows
             << "exposure_requested_us" << options.camera.exposure_us;
    const auto frame_metadata = [&](const char* name, const CameraFrame& frame) {
        metadata << name << "{" << "frame_id" << std::to_string(frame.frame_id)
                 << "device_timestamp_ns" << std::to_string(frame.device_timestamp_ns)
                 << "host_arrival_ns" << std::to_string(frame.host_arrival_ns) << "}";
    };
    frame_metadata("left", pair.left);
    frame_metadata("right", pair.right);
    std::cout << "Saved capture: " << options.output << '\n';
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--help") { usage(); return 0; }
        const auto options = parse(argc, argv);
        if (options.list) {
            const auto devices = SentechStereoSource::list_devices();
            for (const auto& device : devices)
                std::cout << "Display: " << device.display_name << " | User: " << device.user_name
                          << " | Serial: " << device.serial << '\n';
            std::cout << devices.size() << " device(s) found\n";
            return 0;
        }
        std::optional<StereoCalibration> calibration;
        if (!options.calibration.empty()) calibration = StereoCalibration::from_file(options.calibration);
        const bool show_preview = options.preview || options.output.empty();
        SentechStereoSource source(options.camera);
        source.start();
        std::cout << "Independent continuous streams; not hardware synchronized.\n";
        std::optional<StereoCameraPair> selected;
        auto last_pair = std::chrono::steady_clock::now();
        while (true) {
            auto pair = source.wait_for_pair(std::chrono::milliseconds(100));
            if (pair) {
                last_pair = std::chrono::steady_clock::now();
                selected = std::move(pair);
                if (!show_preview) break;
                preview(*selected);
            }
            if (std::chrono::steady_clock::now() - last_pair > std::chrono::milliseconds(options.timeout_ms))
                throw std::runtime_error("Timed out waiting for a fresh stereo pair; check both cameras and frame timing");
            if (show_preview) {
                const int key = cv::waitKey(10) & 0xff;
                if (key == 27 || key == 'q' || key == 'Q') { source.stop(); cv::destroyAllWindows(); return 0; }
                if (key == ' ' && pair) break;
                if (selected && cv::getWindowProperty("Sentech stereo | free-running | Space: capture, Q: exit", cv::WND_PROP_VISIBLE) < 1)
                    return 0;
            }
        }
        source.stop();
        if (show_preview) cv::destroyAllWindows();
        std::cout << "Captured left frame " << selected->left.frame_id << ", right frame " << selected->right.frame_id << '\n';
        std::optional<StereoFrame> frame;
        if (calibration) frame.emplace(selected->left.rgb, selected->right.rgb, *calibration);
        if (!options.output.empty()) save(*selected, options);
        if (options.infer) {
            frame->rectify();
            FS fs;
            fs.loadEngine(options.engine);
            fs.set_model_camera_parameters(frame->rectified_camera_parameters(), frame->rectified_left().size());
            fs.prepare_stereo_images(frame->rectified_left(), frame->rectified_right());
            fs.inference();
            std::cout << "Inference submitted; disparity remains in FS's GPU buffer.\n";
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "Sentech: " << error.what() << '\n';
        return 1;
    }
}
