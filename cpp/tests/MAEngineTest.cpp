#include "fs/inference/FS.hpp"
#include "fs/inference/MA.hpp"
#include "fs/inference/MAPostprocessing.hpp"

#include <opencv2/imgproc.hpp>
#include <algorithm>
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
std::array<float*, 5> devicePointers(const fs::MA& model) {
    return {model.inputDevice(), model.tokensDevice(), model.clsDevice(), model.posesDevice(), model.scaleDevice()};
}
bool deviceAllocation(const float* pointer) {
    cudaPointerAttributes attributes{};
    const auto error = cudaPointerGetAttributes(&attributes, pointer);
    if (error == cudaErrorInvalidValue) { cudaGetLastError(); return false; }
    checked(error);
    return attributes.type == cudaMemoryTypeDevice;
}
template<class Action> void rejects(Action action, const std::string& expected) {
    try { action(); }
    catch (const std::runtime_error& error) {
        require(std::string(error.what()).find(expected) != std::string::npos, "Unexpected error");
        return;
    }
    throw std::runtime_error("Invalid operation unexpectedly succeeded");
}

void checkColorUploads(fs::MA& model) {
    constexpr int h = fs::MA::kInputHeight, w = fs::MA::kInputWidth;
    constexpr std::size_t plane = std::size_t{h} * w;
    constexpr std::uint32_t untouched = 0x3f3f3f3fU;
    constexpr std::array<double, 3> mean{0.485, 0.456, 0.406}, stddev{0.229, 0.224, 0.225};
    std::array<cv::Mat, fs::MA::kMaxViews> expected;
    std::array<StereoCameraParameters, fs::MA::kMaxViews> cameras{};
    std::array<cv::Size, fs::MA::kMaxViews> source_sizes{};
    const auto* input = model.inputDevice();
    checked(cudaMemsetAsync(model.inputDevice(), 0x3f, fs::MA::kInputElements * sizeof(float), model.stream()));

    const auto checkContents = [&] {
        require(model.inputDevice() == input, "Color upload replaced the engine's input allocation");
        std::vector<float> actual(fs::MA::kInputElements);
        checked(cudaMemcpyAsync(actual.data(), input, actual.size() * sizeof(float), cudaMemcpyDeviceToHost, model.stream()));
        checked(cudaStreamSynchronize(model.stream()));
        for (int v = 0; v < fs::MA::kMaxViews; ++v) {
            for (int c = 0; c < 7; ++c) {
                for (int y = 0; y < h; ++y) {
                    for (int x = 0; x < w; ++x) {
                        const auto offset = v * fs::MA::kInputStrideElements + c * plane + y * w + x;
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
    for (int v = 0; v < fs::MA::kMaxViews; ++v) {
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
    model.uploadColor(fs::MA::kMaxViews - 1, replacement, cameras.back());
    checkContents();
    model.uploadColor(fs::MA::kMaxViews - 1, replacement, cameras.back());
    checkContents(); // Re-upload starts from raw RGB; it must not normalize twice.
    cameras.back().cx += 13;
    cameras.back().fy *= 0.8;
    model.uploadColor(fs::MA::kMaxViews - 1, replacement, cameras.back());
    checkContents(); // Same image size must not retain stale rays after K changes.
    rejects([&] { model.uploadColor(-1, replacement, cameras.back()); }, "view index");
    rejects([&] { model.uploadColor(fs::MA::kMaxViews, replacement, cameras.back()); }, "view index");
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

void checkDepthUploads(fs::MA& model) {
    constexpr int h = fs::MA::kInputHeight, w = fs::MA::kInputWidth;
    constexpr std::size_t plane = std::size_t{h} * w;
    const StereoCameraParameters camera{1800, 1750, 1100.25, 970.75, 0};
    const cv::Mat rgb(2048, 2448, CV_8UC3, cv::Scalar(40, 110, 230));
    for (int v = 0; v < fs::MA::kMaxViews; ++v) model.uploadColor(v, rgb, camera);
    std::vector<float> expected(fs::MA::kInputElements);
    checked(cudaMemcpyAsync(expected.data(), model.inputDevice(), expected.size() * sizeof(float),
                            cudaMemcpyDeviceToHost, model.stream()));
    checked(cudaStreamSynchronize(model.stream()));
    std::array<bool, fs::MA::kMaxViews> filled{};
    const auto checkContents = [&] {
        std::vector<float> actual(expected.size());
        checked(cudaMemcpyAsync(actual.data(), model.inputDevice(), actual.size() * sizeof(float),
                                cudaMemcpyDeviceToHost, model.stream()));
        checked(cudaStreamSynchronize(model.stream()));
        for (int v = 0; v < fs::MA::kMaxViews; ++v) {
            for (int c = 0; c < 7; ++c) {
                const auto offset = v * fs::MA::kInputStrideElements + c * plane;
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
                expected[view * fs::MA::kInputStrideElements + 6 * plane + y * w + x] =
                    std::isfinite(z) && z > 0 ? float(std::sqrt(z * z * (rx * rx + ry * ry + 1.0))) : 0.0F;
            }
        }
        model.uploadDepth(view, depth);
        filled[view] = true;
    };
    for (int v = 0; v < fs::MA::kMaxViews; ++v) {
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
    upload(fs::MA::kMaxViews - 1, replacement);
    checkContents();
    upload(fs::MA::kMaxViews - 1, replacement);
    checkContents(); // No double conversion, even when replacing the same slot.
    upload(0, cv::Mat(800, 960, CV_32FC1, cv::Scalar(0)));
    checkContents(); // A fully invalid replacement must clear the old distances.
    rejects([&] { model.uploadDepth(-1, replacement); }, "view index");
    rejects([&] { model.uploadDepth(fs::MA::kMaxViews, replacement); }, "view index");
    rejects([&] { model.uploadDepth(0, cv::Mat{}); }, "CV_32FC1");
    rejects([&] { model.uploadDepth(0, cv::Mat(h, w, CV_8UC1)); }, "CV_32FC1");
    checkContents();
}

void checkCameraPoseDecoding() {
    // A non-identity first camera must stay in the predicted world: 90 degrees
    // about Z and translation (2,4,6) m after applying exp(log(2)).
    const float r = std::sqrt(0.5F);
    std::array<float,14> raw{1,2,3,0,0,3*r,3*r, -.5F,1,.25F,0,0,0,-2};
    const auto poses = fs::ma_postprocessing::camera_poses(raw.data(),2,std::log(2.0F));
    const cv::Matx44d first(0,-1,0,2, 1,0,0,4, 0,0,1,6, 0,0,0,1);
    const cv::Matx44d second(1,0,0,-1, 0,1,0,2, 0,0,1,.5, 0,0,0,1);
    require(cv::norm(poses[0]-first) < 1e-6 && cv::norm(poses[1]-second) < 1e-6,
            "Pose decoding changed MA's world frame, quaternion order or metric scale");
    const auto clamped = fs::ma_postprocessing::camera_poses(raw.data(),2,-100.0F);
    require(std::abs(clamped[0](0,3)-1e-8) < 1e-14, "MA scale minimum differs from Python adaptor");
    rejects([&] { fs::ma_postprocessing::camera_poses(raw.data(),1,0); }, "2..5 views");
    rejects([&] { fs::ma_postprocessing::camera_poses(raw.data(),2,1000); }, "scale");
    raw[0] = std::numeric_limits<float>::quiet_NaN();
    rejects([&] { fs::ma_postprocessing::camera_poses(raw.data(),2,0); }, "pose");
    raw[0] = 1; raw[5] = 0; raw[6] = 0;
    rejects([&] { fs::ma_postprocessing::camera_poses(raw.data(),2,0); }, "quaternion");
}

std::vector<float> download(const float* pointer, std::size_t count) {
    std::vector<float> values(count);
    checked(cudaMemcpy(values.data(), pointer, count * sizeof(float), cudaMemcpyDeviceToHost));
    return values;
}

void checkInference(fs::MA& model) {
    const auto pointers = devicePointers(model);
    const StereoCameraParameters camera{540, 525, 310.5, 235.25, 0};
    model.resetInputs();
    rejects([&] { model.downloadCameraPoses(); }, "inference outputs");
    rejects([&] { model.inference(1); }, "2..5 views");
    rejects([&] { model.inference(6); }, "2..5 views");
    rejects([&] { model.inference(2); }, "view 1");
    rejects([&] { model.uploadDepth(0, cv::Mat(800, 960, CV_32FC1, cv::Scalar(0.4))); }, "camera rays");
    checked(cudaMemsetAsync(model.tokensDevice(), 0xff, fs::MA::kTokenElements * sizeof(float), model.stream()));
    checked(cudaMemsetAsync(model.clsDevice(), 0xff, fs::MA::kClsElements * sizeof(float), model.stream()));

    const auto run = [&](int views) {
        checked(cudaMemsetAsync(model.posesDevice(), 0xff, fs::MA::kPoseElements * sizeof(float), model.stream()));
        checked(cudaMemsetAsync(model.scaleDevice(), 0xff, sizeof(float), model.stream()));
        const auto start = std::chrono::steady_clock::now();
        model.inference(views);
        model.synchronize();
        std::cout << "MA encoder + pose/scale V=" << views << ": "
                  << std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count()
                  << " ms\n";
        require(devicePointers(model) == pointers, "Inference replaced persistent GPU allocations");
        auto output = download(model.posesDevice(), fs::MA::kPoseElements);
        for (std::size_t i = 0; i < output.size(); ++i) {
            if (i < std::size_t(views) * 7) require(std::isfinite(output[i]), "Non-finite active camera pose");
            else {
                std::uint32_t bits{};
                std::memcpy(&bits, &output[i], sizeof(bits));
                require(bits == 0xffffffffU, "Pose output overwrote an inactive view");
            }
        }
        output.resize(views * 7);
        const float raw_scale = download(model.scaleDevice(), 1).front();
        require(std::isfinite(raw_scale), "Non-finite scene scale");
        const float scale = std::max(std::exp(raw_scale), 1e-8F);
        const auto cameras = model.downloadCameraPoses();
        require(int(cameras.size()) == views, "Camera download has the wrong view count");
        for (int v = 0; v < views; ++v) {
            const auto& pose = cameras[v];
            for (int c = 0; c < 3; ++c)
                require(pose(c,3) == output[v*7+c]*scale, "Camera positions were rebased or use the wrong scale");
            const cv::Matx33d rotation(pose(0,0),pose(0,1),pose(0,2),pose(1,0),pose(1,1),pose(1,2),
                                      pose(2,0),pose(2,1),pose(2,2));
            require(cv::norm(rotation.t()*rotation-cv::Matx33d::eye()) < 1e-6 &&
                    std::abs(cv::determinant(rotation)-1) < 1e-6, "Invalid predicted camera rotation");
        }
        output.push_back(raw_scale);
        return output;
    };
    const auto features = [&] {
        auto values = download(model.tokensDevice(), fs::MA::kTokenElements);
        const auto cls = download(model.clsDevice(), fs::MA::kClsElements);
        values.insert(values.end(), cls.begin(), cls.end());
        return values;
    };
    const auto sameSlot = [](const auto& a, const auto& b, int view) {
        const auto offset = std::size_t(view) * fs::MA::kTokenElementsPerView;
        const auto cls_offset = fs::MA::kTokenElements + std::size_t(view) * fs::MA::kClsElementsPerView;
        return std::memcmp(a.data()+offset,b.data()+offset,fs::MA::kTokenElementsPerView*sizeof(float)) == 0 &&
               std::memcmp(a.data()+cls_offset,b.data()+cls_offset,fs::MA::kClsElementsPerView*sizeof(float)) == 0;
    };
    std::array<cv::Mat, fs::MA::kMaxViews> rgbs, depths;
    std::array<std::vector<float>, fs::MA::kMaxViews + 1> results;
    for (int v = 0; v < fs::MA::kMaxViews; ++v) {
        auto& rgb = rgbs[v]; rgb.create(480, 640, CV_8UC3);
        for (int y = 0; y < rgb.rows; ++y) {
            auto* row = rgb.ptr<cv::Vec3b>(y);
            for (int x = 0; x < rgb.cols; ++x)
                row[x] = cv::Vec3b((x + v * 19) % 256, (y * 2 + v * 11) % 256, (x + y + v * 7) % 256);
        }
        model.uploadColor(v, rgb, camera);
        rejects([&] { model.inference(std::max(2, v + 1)); }, "view " + std::to_string(v + 1));
        depths[v] = cv::Mat(800,960,CV_32FC1,cv::Scalar(0.35+v*0.02));
        depths[v](cv::Rect(100+v*20,100,160,160)).setTo(0);
        model.uploadDepth(v,depths[v]);
        if (v == 0) { rejects([&] { model.inference(2); }, "view 2"); continue; }
        model.synchronize();
        const auto previous = features();
        results[v+1] = run(v+1);
        const auto current = features();
        for (int slot = 0; slot < fs::MA::kMaxViews; ++slot) {
            if ((v > 1 && slot < v) || slot > v)
                require(sameSlot(previous,current,slot), "Encoding changed another view's token/CLS slot");
            if (slot <= v) {
                const auto offset = std::size_t(slot) * fs::MA::kTokenElementsPerView;
                const auto cls_offset = fs::MA::kTokenElements + std::size_t(slot) * fs::MA::kClsElementsPerView;
                require(std::all_of(current.begin()+offset,current.begin()+offset+fs::MA::kTokenElementsPerView,
                                    [](float x) { return std::isfinite(x); }) &&
                        std::all_of(current.begin()+cls_offset,current.begin()+cls_offset+fs::MA::kClsElementsPerView,
                                    [](float x) { return std::isfinite(x); }), "Encoder did not fill an entire feature slot");
            }
        }
    }
    const auto all_features = features();
    // Shrink while inference is still pending, then grow using cached slots.
    model.inference(5);
    for (const int views : {2,5,3,4,2})
        require(run(views) == results[views], "Dynamic view switching changed repeatable pose/scale outputs");
    require(features() == all_features, "Pose/scale changed cached encoder features");

    // A depth-only update must invalidate exactly its own encoding.
    depths[1].setTo(0.62);
    model.uploadDepth(1, depths[1]);
    rejects([&] { model.downloadCameraPoses(); }, "inference outputs");
    require(run(2) != results[2], "Depth-only update retained stale features or camera outputs");
    const auto changed_depth = features();
    require(!sameSlot(all_features,changed_depth,1), "Depth change did not re-encode its view");
    for (const int slot : {0,2,3,4})
        require(sameSlot(all_features,changed_depth,slot), "Depth change overwrote another view's features");

    rgbs[1].setTo(cv::Scalar(190,50,70));
    model.uploadColor(1, rgbs[1], camera);
    rejects([&] { model.downloadCameraPoses(); }, "inference outputs");
    rejects([&] { model.inference(2); }, "view 2");
    model.uploadDepth(1, depths[1]);
    const auto retaken = run(2);
    require(retaken != results[2], "Retake input did not affect MA outputs");
    const auto changed_rgb = features();
    require(!sameSlot(changed_depth,changed_rgb,1), "RGB change did not re-encode its view");
    for (const int slot : {0,2,3,4})
        require(sameSlot(changed_depth,changed_rgb,slot), "Retake overwrote another view's features");

    // Compare incremental/cached results with five back-to-back encoder calls
    // through one context. Reverse upload order also checks fixed slot mapping.
    const auto all_retaken = run(5);
    model.resetInputs();
    rejects([&] { model.inference(2); }, "view 1");
    rejects([&] { model.downloadCameraPoses(); }, "inference outputs");
    checked(cudaMemsetAsync(model.tokensDevice(), 0xff, fs::MA::kTokenElements*sizeof(float), model.stream()));
    checked(cudaMemsetAsync(model.clsDevice(), 0xff, fs::MA::kClsElements*sizeof(float), model.stream()));
    for (int v = fs::MA::kMaxViews - 1; v >= 0; --v) {
        model.uploadColor(v, rgbs[v], camera);
        model.uploadDepth(v, depths[v]);
    }
    require(run(5) == all_retaken, "Serial five-view re-encoding changed pose/scale outputs");
    require(features() == changed_rgb, "Serial encoder reuse changed a view's token/CLS slot");
    require(run(2) == retaken, "Fresh encoding differs from cached/retaken features");
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
        require(argc == 3, "Usage: fs_ma_engine_test encoder_fmha.engine pose_scale_fmha.engine");
        checkCameraPoseDecoding();
        TempDirectory temp;
        const auto missing = temp.path / "missing.engine", empty = temp.path / "empty.engine";
        std::ofstream(empty).close();
        rejects([&] { fs::MA bad(missing,argv[2]); }, "cannot open engine");
        rejects([&] { fs::MA bad(empty,argv[2]); }, "empty engine");
        rejects([&] { fs::MA bad(argv[2],argv[1]); }, "single-view encoder");
        // Fail after loading the encoder, before allocating the shared workspace.
        rejects([&] { fs::MA bad(argv[1],missing); }, "cannot open engine");
        rejects([&] { fs::MA bad(argv[1],argv[1]); }, "pose/scale engine");

        auto model = std::make_unique<fs::MA>(argv[1],argv[2]);
        const auto ma_stream = model->stream();
        require(model->isLoaded() && ma_stream, "Constructor did not prepare both engines");
        unsigned int flags{};
        checked(cudaStreamGetFlags(ma_stream, &flags));
        require(flags == cudaStreamNonBlocking, "MA stream is not non-blocking");
        auto owner = std::make_shared<FS>();
        const auto fs_stream = owner->stream();
        std::weak_ptr<FS> lifetime = owner;
        require(ma_stream != fs_stream, "MA borrowed FS's stream");
        const auto pointers = devicePointers(*model);
        constexpr std::array<std::size_t, 5> elements{
            fs::MA::kInputElements,fs::MA::kTokenElements,fs::MA::kClsElements,fs::MA::kPoseElements,fs::MA::kScaleElements};
        for (std::size_t i = 0; i < pointers.size(); ++i) {
            require(pointers[i] && deviceAllocation(pointers[i]), "Constructor did not allocate GPU I/O and feature banks");
            checked(cudaMemsetAsync(pointers[i], int(i)+1, elements[i]*sizeof(float), ma_stream));
        }
        model->synchronize();
        for (std::size_t i = 0; i < pointers.size(); ++i) {
            std::uint32_t first{}, last{};
            checked(cudaMemcpy(&first, pointers[i], sizeof(first), cudaMemcpyDeviceToHost));
            checked(cudaMemcpy(&last, pointers[i]+elements[i]-1, sizeof(last), cudaMemcpyDeviceToHost));
            const auto expected = std::uint32_t(i+1)*0x01010101U;
            require(first == expected && last == expected, "V=5 buffers overlap or have insufficient capacity");
        }
        checkColorUploads(*model);
        checkDepthUploads(*model);
        checkInference(*model);
        model->inference(2); // Destructor must drain queued pose/scale execution.
        model.reset();
        for (const auto* pointer : pointers)
            require(!deviceAllocation(pointer), "Destructor retained a GPU allocation");
        require(cudaStreamGetFlags(fs_stream, &flags) == cudaSuccess, "MA destroyed FS's stream");
        owner->synchronize();

        model = std::make_unique<fs::MA>(argv[1],argv[2]);
        const auto final_stream = model->stream();
        require(final_stream != fs_stream, "New MA instance borrowed FS's stream");
        const auto final_pointers = devicePointers(*model);
        owner.reset();
        require(lifetime.expired(), "MA retained FS");
        checked(cudaStreamGetFlags(final_stream, &flags));
        for (int view = 0; view < 2; ++view) {
            model->uploadColor(view, cv::Mat(50,80,CV_8UC3,cv::Scalar(5,20,240)), {60,55,39.5,24.5,0});
            model->uploadDepth(view, cv::Mat(50,80,CV_32FC1,cv::Scalar(0.4)));
        }
        model->inference(2); // Both encoder and suffix may still be running.
        model.reset();
        for (const auto* pointer : final_pointers)
            require(!deviceAllocation(pointer), "GPU I/O survived MA release");
        std::cout << "PASS: constructor-owned two-engine resources with shared activation workspace, one encoder context reused across five fixed token/CLS slots, RGB/ray/depth preparation, real V=2..5 inference, dynamic switching, cached views, depth/RGB invalidation, reset, pose decoding, output bounds and destruction with pending encoder/pose work\n";
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
}
