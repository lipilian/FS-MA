#include "fs/inference/FS.hpp"
#include "fs/inference/MA-VGGT.hpp"

#include <opencv2/imgproc.hpp>
#include <chrono>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}
void checked(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
std::array<float*, 4> devicePointers(const fs::MA_VGGT& model) {
    return {model.inputDevice(), model.depthsDevice(), model.posesDevice(), model.scaleDevice()};
}
bool deviceAllocation(const float* pointer) {
    cudaPointerAttributes attributes{};
    const auto error = cudaPointerGetAttributes(&attributes, pointer);
    if (error == cudaErrorInvalidValue) { cudaGetLastError(); return false; }
    checked(error);
    return attributes.type == cudaMemoryTypeDevice;
}
template<class Action> void rejects(Action action, const char* expected) {
    try { action(); }
    catch (const std::runtime_error& error) {
        require(std::string(error.what()).find(expected) != std::string::npos, "Unexpected error");
        return;
    }
    throw std::runtime_error("Invalid operation unexpectedly succeeded");
}

void checkColorUploads(fs::MA_VGGT& model) {
    constexpr int h = fs::MA_VGGT::kInputHeight, w = fs::MA_VGGT::kInputWidth;
    constexpr std::size_t plane = std::size_t{h} * w;
    constexpr std::uint32_t untouched = 0x3f3f3f3fU;
    constexpr std::array<double, 3> mean{0.485, 0.456, 0.406}, stddev{0.229, 0.224, 0.225};
    std::array<cv::Mat, fs::MA_VGGT::kMaxViews> expected;
    std::array<StereoCameraParameters, fs::MA_VGGT::kMaxViews> cameras{};
    std::array<cv::Size, fs::MA_VGGT::kMaxViews> source_sizes{};
    const auto* input = model.inputDevice();
    checked(cudaMemsetAsync(model.inputDevice(), 0x3f, fs::MA_VGGT::kInputElements * sizeof(float), model.stream()));

    const auto checkContents = [&] {
        require(model.inputDevice() == input, "Color upload replaced the engine's input allocation");
        std::vector<float> actual(fs::MA_VGGT::kInputElements);
        checked(cudaMemcpyAsync(actual.data(), input, actual.size() * sizeof(float), cudaMemcpyDeviceToHost, model.stream()));
        checked(cudaStreamSynchronize(model.stream()));
        for (int v = 0; v < fs::MA_VGGT::kMaxViews; ++v) {
            for (int c = 0; c < 7; ++c) {
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        const auto offset = (v * 7 + c) * plane + y * w + x;
                        if (c < 3 && !expected[v].empty()) {
                            const double reference = (double(expected[v].at<cv::Vec3b>(y, x)[c]) / 255.0 - mean[c]) / stddev[c];
                            require(std::abs(double(actual[offset]) - reference) <= 1e-6,
                                    "GPU RGB differs from Lanczos4 resize + DINOv2 normalization (channel order, slot or repeated normalization mismatch)");
                        } else if (c < 6 && !expected[v].empty()) {
                            // Independently map the target pixel center back into
                            // the source image and unproject with its original K.
                            const double source_x = (x + 0.5) * source_sizes[v].width / w - 0.5;
                            const double source_y = (y + 0.5) * source_sizes[v].height / h - 0.5;
                            const double rx = (source_x - cameras[v].cx) / cameras[v].fx;
                            const double ry = (source_y - cameras[v].cy) / cameras[v].fy;
                            const double norm = std::sqrt(rx * rx + ry * ry + 1.0);
                            const double reference = (c == 3 ? rx : c == 4 ? ry : 1.0) / norm;
                            require(std::abs(double(actual[offset]) - reference) <= 1e-6,
                                    "GPU ray differs from source-camera unprojection (resize, pixel center, calibration or channel mismatch)");
                            if (c == 3) {
                                const double dx = actual[offset], dy = actual[offset + plane], dz = actual[offset + 2 * plane];
                                require(dz > 0 && std::abs(std::sqrt(dx * dx + dy * dy + dz * dz) - 1.0) <= 1e-6,
                                        "Camera ray must be a forward-facing unit vector");
                            }
                        } else {
                            std::uint32_t bits{};
                            std::memcpy(&bits, &actual[offset], sizeof(bits));
                            require(bits == untouched, "View preparation overwrote depth or another view");
                        }
                    }
                }
            }
        }
    };

    // Different aspect ratios exercise a direct resize; non-contiguous ROIs
    // and immediately modified/destroyed source Mats exercise staging ownership.
    for (int v = 0; v < fs::MA_VGGT::kMaxViews; ++v) {
        const int source_w = 640 + v * 83, source_h = 480 - v * 41;
        cv::Mat storage(source_h + 8, source_w + 16, CV_8UC3);
        cv::Mat rgb = storage(cv::Rect(3, 2, source_w, source_h));
        require(!rgb.isContinuous(), "Test image must have a row stride");
        for (int y = 0; y < rgb.rows; ++y) {
            auto* row = rgb.ptr<cv::Vec3b>(y);
            for (int x = 0; x < rgb.cols; ++x)
                row[x] = cv::Vec3b((x + v * 17) % 256, (y * 3 + v * 29) % 256, (x + y * 2 + v * 43) % 256);
        }
        cv::resize(rgb, expected[v], cv::Size(w, h), 0, 0, cv::INTER_LANCZOS4);
        cameras[v] = {460.0 + v * 70, 530.0 + v * 23, source_w * 0.43 + v * 7, source_h * 0.61 - v * 3, 0};
        source_sizes[v] = rgb.size();
        model.uploadColor(v, rgb, cameras[v]);
        rgb.setTo(cv::Scalar::all(0));
        if (v == 0) checkContents(); // Includes all untouched future slots.
    }
    checkContents();

    // Retake with a new image size and calibration. Earlier views and all
    // depth values must survive; new rays must use the replacement camera.
    const cv::Mat replacement(700, 300, CV_8UC3, cv::Scalar(0, 127, 255));
    expected.back() = cv::Mat(h, w, CV_8UC3, cv::Scalar(0, 127, 255));
    cameras.back() = {630, 410, 149.5, 349.5, 0};
    source_sizes.back() = replacement.size();
    model.uploadColor(fs::MA_VGGT::kMaxViews - 1, replacement, cameras.back());
    checkContents();
    model.uploadColor(fs::MA_VGGT::kMaxViews - 1, replacement, cameras.back());
    checkContents(); // Re-upload starts from raw RGB; it must not normalize twice.
    cameras.back().cx += 13;
    cameras.back().fy *= 0.8;
    model.uploadColor(fs::MA_VGGT::kMaxViews - 1, replacement, cameras.back());
    checkContents(); // Same image size must not retain stale rays after K changes.
    rejects([&] { model.uploadColor(-1, replacement, cameras.back()); }, "view index");
    rejects([&] { model.uploadColor(fs::MA_VGGT::kMaxViews, replacement, cameras.back()); }, "view index");
    rejects([&] { model.uploadColor(0, cv::Mat{}, cameras.back()); }, "CV_8UC3");
    rejects([&] { model.uploadColor(0, cv::Mat(h, w, CV_32FC3), cameras.back()); }, "CV_8UC3");
    auto invalid = cameras.back(); invalid.fx = 0;
    rejects([&] { model.uploadColor(0, replacement, invalid); }, "intrinsics");
    invalid = cameras.back(); invalid.fy = -1;
    rejects([&] { model.uploadColor(0, replacement, invalid); }, "intrinsics");
    invalid = cameras.back(); invalid.cx = std::numeric_limits<double>::quiet_NaN();
    rejects([&] { model.uploadColor(0, replacement, invalid); }, "intrinsics");
    invalid = cameras.back(); invalid.cy = std::numeric_limits<double>::infinity();
    rejects([&] { model.uploadColor(0, replacement, invalid); }, "intrinsics");
    checkContents();
}

void checkDepthUploads(fs::MA_VGGT& model) {
    constexpr int h = fs::MA_VGGT::kInputHeight, w = fs::MA_VGGT::kInputWidth;
    constexpr std::size_t plane = std::size_t{h} * w;
    const StereoCameraParameters camera{1800, 1750, 1100.25, 970.75, 0};
    const cv::Mat rgb(2048, 2448, CV_8UC3, cv::Scalar(40, 110, 230));
    for (int v = 0; v < fs::MA_VGGT::kMaxViews; ++v) model.uploadColor(v, rgb, camera);
    std::vector<float> expected(fs::MA_VGGT::kInputElements);
    checked(cudaMemcpyAsync(expected.data(), model.inputDevice(), expected.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, model.stream()));
    checked(cudaStreamSynchronize(model.stream()));
    std::array<bool, fs::MA_VGGT::kMaxViews> filled{};
    const auto checkContents = [&] {
        std::vector<float> actual(expected.size());
        checked(cudaMemcpyAsync(actual.data(), model.inputDevice(), actual.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, model.stream()));
        checked(cudaStreamSynchronize(model.stream()));
        for (int v = 0; v < fs::MA_VGGT::kMaxViews; ++v) {
            for (int c = 0; c < 7; ++c) {
                const auto offset = (v * 7 + c) * plane;
                if (c != 6 || !filled[v]) {
                    require(std::memcmp(actual.data() + offset, expected.data() + offset, plane * sizeof(float)) == 0,
                            "Depth upload changed RGB, rays or another view");
                    continue;
                }
                for (std::size_t i = 0; i < plane; ++i) {
                    const float value = actual[offset + i], reference = expected[offset + i];
                    require(std::isfinite(value) && value >= 0 && std::abs(value - reference) <= 1e-6F,
                            "GPU ray distance differs from Lanczos4 Z-depth and notebook 3D norm");
                    if (reference == 0) require(value == 0, "Invalid depth must remain zero");
                }
            }
        }
    };
    int undershoot_count = 0;
    const auto upload = [&](int view, const cv::Mat& depth) {
        cv::Mat clean = depth.clone();
        for (int y = 0; y < clean.rows; ++y) {
            float* row = clean.ptr<float>(y);
            for (int x = 0; x < clean.cols; ++x)
                if (!std::isfinite(row[x]) || row[x] <= 0) row[x] = 0;
        }
        cv::Mat resized;
        cv::resize(clean, resized, cv::Size(w, h), 0, 0, cv::INTER_LANCZOS4);
        for (int y = 0; y < h; ++y) {
            for (int x = 0; x < w; ++x) {
                const double z = resized.at<float>(y, x);
                if (z < 0) ++undershoot_count;
                // Independent reference: unproject to a 3D point, then take
                // its Euclidean norm as the notebook does (no GPU ray reads).
                const double rx = ((x + 0.5) * rgb.cols / w - 0.5 - camera.cx) / camera.fx;
                const double ry = ((y + 0.5) * rgb.rows / h - 0.5 - camera.cy) / camera.fy;
                expected[(view * 7 + 6) * plane + y * w + x] =
                    std::isfinite(z) && z > 0 ? float(std::sqrt(z * z * (rx * rx + ry * ry + 1.0))) : 0.0F;
            }
        }
        model.uploadDepth(view, depth);
        filled[view] = true;
    };
    for (int v = 0; v < fs::MA_VGGT::kMaxViews; ++v) {
        // Includes FS's 960x800 grid, other aspect ratios, and strided inputs.
        const int sw = 960 + v * 17, sh = 800 - v * 31;
        cv::Mat storage(sh + 8, sw + 12, CV_32FC1);
        cv::Mat depth = storage(cv::Rect(3, 2, sw, sh));
        require(!depth.isContinuous(), "Test depth must have a row stride");
        for (int y = 0; y < sh; ++y) {
            float* row = depth.ptr<float>(y);
            for (int x = 0; x < sw; ++x)
                row[x] = x < sw / 3 ? 0.0F : 0.2F + v * 0.1F + x * 0.0001F + y * 0.0002F;
        }
        depth(cv::Rect(sw / 2, 20, 20, 20)).setTo(std::numeric_limits<float>::quiet_NaN());
        depth(cv::Rect(sw / 2, 50, 20, 20)).setTo(std::numeric_limits<float>::infinity());
        depth(cv::Rect(sw / 2, 80, 20, 20)).setTo(-0.5F);
        upload(v, depth);
        depth.setTo(9.0F); // The async upload must own the source data now.
        if (v == 0) checkContents();
    }
    checkContents();
    require(undershoot_count > 0, "Depth test must exercise negative Lanczos ringing");
    const cv::Mat replacement(700, 300, CV_32FC1, cv::Scalar(0.42));
    upload(fs::MA_VGGT::kMaxViews - 1, replacement);
    checkContents();
    upload(fs::MA_VGGT::kMaxViews - 1, replacement);
    checkContents(); // No double conversion, even when replacing the same slot.
    upload(0, cv::Mat(800, 960, CV_32FC1, cv::Scalar(0)));
    checkContents(); // A fully invalid replacement must clear the old distances.
    rejects([&] { model.uploadDepth(-1, replacement); }, "view index");
    rejects([&] { model.uploadDepth(fs::MA_VGGT::kMaxViews, replacement); }, "view index");
    rejects([&] { model.uploadDepth(0, cv::Mat{}); }, "CV_32FC1");
    rejects([&] { model.uploadDepth(0, cv::Mat(h, w, CV_8UC1)); }, "CV_32FC1");
    checkContents();
}

struct TempDirectory {
    std::filesystem::path path = std::filesystem::temp_directory_path() /
        ("fs_ma_engine_test_" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    TempDirectory() { require(std::filesystem::create_directory(path), "Cannot create test directory"); }
    ~TempDirectory() { std::error_code error; std::filesystem::remove_all(path, error); }
};
} // namespace

int main(int argc, char** argv) {
    try {
        require(argc == 3, "Usage: fs_ma_engine_test MA.engine SAM_encoder.engine");
        TempDirectory temp;
        const auto missing = temp.path / "missing.engine", empty = temp.path / "empty.engine";
        std::ofstream(empty).close();
        // MA can be constructed without FS and owns a non-blocking stream.
        auto model = std::make_unique<fs::MA_VGGT>();
        rejects([&] { model->uploadDepth(0, cv::Mat(800, 960, CV_32FC1, cv::Scalar(0.4))); }, "camera rays");
        const auto ma_stream = model->stream();
        require(!model->isLoaded() && ma_stream, "Incorrect initial model or stream");
        unsigned int flags{};
        checked(cudaStreamGetFlags(ma_stream, &flags));
        require(flags == cudaStreamNonBlocking, "MA stream is not non-blocking");
        auto owner = std::make_shared<FS>();
        const auto fs_stream = owner->stream();
        std::weak_ptr<FS> lifetime = owner;
        require(ma_stream != fs_stream, "MA borrowed FS's stream");
        {
            fs::MA_VGGT another;
            require(another.stream() != ma_stream && another.stream() != fs_stream,
                    "MA instances share a stream");
        }
        const auto pointers = devicePointers(*model);
        constexpr std::array<std::size_t, 4> elements{5*7*434*518, 5*6*434*518, 5*7, 1};
        for (std::size_t i = 0; i < pointers.size(); ++i) {
            require(pointers[i] && deviceAllocation(pointers[i]), "Constructor did not allocate GPU I/O");
            // Exercise the full V=5 extent with a different byte pattern per tensor.
            checked(cudaMemsetAsync(pointers[i], int(i)+1, elements[i]*sizeof(float), ma_stream));
        }
        const auto checkBuffers = [&] {
            require(devicePointers(*model) == pointers, "Engine loading replaced persistent I/O addresses");
            require(model->stream() == ma_stream, "Engine loading replaced MA's stream");
            checked(cudaStreamSynchronize(ma_stream));
            for (std::size_t i = 0; i < pointers.size(); ++i) {
                std::uint32_t first{}, last{};
                checked(cudaMemcpy(&first, pointers[i], sizeof(first), cudaMemcpyDeviceToHost));
                checked(cudaMemcpy(&last, pointers[i]+elements[i]-1, sizeof(last), cudaMemcpyDeviceToHost));
                const auto expected = std::uint32_t(i+1)*0x01010101U;
                require(first == expected && last == expected, "I/O capacity, independence or contents changed");
            }
        };
        checkBuffers();
        rejects([&] { model->loadEngine(missing); }, "cannot open engine");
        rejects([&] { model->loadEngine(empty); }, "empty engine");
        require(!model->isLoaded(), "Failed initial load marked the model ready");
        checkBuffers();

        model->loadEngine(argv[1]); // Includes runtime output-shape checks for every V=2..5.
        require(model->isLoaded() && model->stream() == ma_stream, "MA did not load on its own stream");
        checkBuffers();
        rejects([&] { model->loadEngine(argv[2]); }, "incompatible tensor");
        rejects([&] { model->loadEngine(missing); }, "cannot open engine");
        require(model->isLoaded(), "Failed reload discarded the working model");
        checkBuffers();
        model->loadEngine(argv[1]);
        require(model->isLoaded() && model->stream() == ma_stream, "Successful reload changed the stream");
        checkBuffers();
        checkColorUploads(*model);
        checkDepthUploads(*model);
        model.reset();
        for (const auto* pointer : pointers)
            require(!deviceAllocation(pointer), "Destructor retained a GPU I/O allocation");
        require(cudaStreamGetFlags(fs_stream, &flags) == cudaSuccess, "MA destroyed FS's stream");
        owner->synchronize();

        // Releasing FS first must neither retain FS nor invalidate MA's stream.
        model = std::make_unique<fs::MA_VGGT>();
        const auto final_stream = model->stream();
        require(final_stream != fs_stream, "New MA instance borrowed FS's stream");
        const auto final_pointers = devicePointers(*model);
        owner.reset();
        require(lifetime.expired(), "MA retained FS");
        require(model->stream() == final_stream, "Releasing FS changed MA's stream");
        checked(cudaStreamGetFlags(final_stream, &flags));
        checked(cudaMemsetAsync(model->inputDevice(), 0, elements[0]*sizeof(float), final_stream));
        checked(cudaStreamSynchronize(final_stream));
        // Destruction also drains queued work before releasing buffers/stream.
        checked(cudaMemsetAsync(model->depthsDevice(), 0, elements[1]*sizeof(float), final_stream));
        model->uploadColor(0, cv::Mat(50, 80, CV_8UC3, cv::Scalar(5, 20, 240)), {60, 55, 39.5, 24.5, 0});
        model->uploadDepth(0, cv::Mat(50, 80, CV_32FC1, cv::Scalar(0.4)));
        model.reset();
        for (const auto* pointer : final_pointers)
            require(!deviceAllocation(pointer), "GPU I/O survived MA release");
        std::cout << "PASS: constructor V=5 GPU I/O capacity, stable addresses/data through load and reload, dynamic shapes, load errors, five-view Lanczos4 uploads with fused GPU DINOv2 normalization/unit camera rays and metric ray distance, pixel-center resize geometry, retake updates, invalid depth/ringing cleanup, channel isolation, GPU buffer release, independent non-blocking streams and FS/MA lifetimes; no inference\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
