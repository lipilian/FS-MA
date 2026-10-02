#include "fs/capture/RealSenseStereoSource.hpp"
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <stdexcept>

namespace {
constexpr int kSensorWidth = 1280, kSensorHeight = 800;
constexpr int kCropWidth = 960, kCropX = (kSensorWidth-kCropWidth)/2;
}
#ifdef FS_HAS_REALSENSE
#include <librealsense2/rs.hpp>
#endif

struct RealSenseStereoSource::Impl {
    bool running{false};
    StereoCalibration calibration;
    cv::Size size;
    std::string serial;
#ifdef FS_HAS_REALSENSE
    rs2::pipeline pipeline;
#endif
};

RealSenseStereoSource::RealSenseStereoSource() : impl_(std::make_unique<Impl>()) {}
RealSenseStereoSource::~RealSenseStereoSource() { stop(); }
bool RealSenseStereoSource::available() noexcept {
#ifdef FS_HAS_REALSENSE
    return true;
#else
    return false;
#endif
}

#ifdef FS_HAS_REALSENSE
namespace {
cv::Mat camera_matrix(const rs2_intrinsics& i) {
    return (cv::Mat_<double>(3,3) << i.fx,0,i.ppx-kCropX,0,i.fy,i.ppy,0,0,1);
}
cv::Mat distortion(const rs2_intrinsics& i) {
    cv::Mat d = cv::Mat::zeros(1,5,CV_64F);
    // Rectified IR profiles can advertise inverse Brown-Conrady with zero
    // coefficients. Nonzero inverse/modified models are not OpenCV's model.
    const bool zero = std::all_of(std::begin(i.coeffs), std::end(i.coeffs), [](float v) { return v == 0; });
    if (i.model == RS2_DISTORTION_NONE || zero) return d;
    if (i.model != RS2_DISTORTION_BROWN_CONRADY)
        throw std::runtime_error("D435 IR distortion model is not supported by OpenCV rectification.");
    for (int k=0; k<5; ++k) d.at<double>(0,k) = i.coeffs[k];
    return d;
}
CameraFrame snapshot(const rs2::video_frame& frame, std::uint64_t arrival) {
    const cv::Mat gray(frame.get_height(), frame.get_width(), CV_8UC1,
                       const_cast<void*>(frame.get_data()), frame.get_stride_in_bytes());
    CameraFrame result;
    if (gray.size() != cv::Size(kSensorWidth,kSensorHeight))
        throw std::runtime_error("D435 infrared frame dimensions changed unexpectedly.");
    cv::cvtColor(gray(cv::Rect(kCropX,0,kCropWidth,kSensorHeight)), result.rgb, cv::COLOR_GRAY2RGB);
    result.frame_id = frame.get_frame_number();
    result.device_timestamp_ns = static_cast<std::uint64_t>(frame.get_timestamp()*1e6);
    result.host_arrival_ns = arrival;
    return result;
}
}
#endif

void RealSenseStereoSource::start() {
    if (impl_->running) return;
#ifdef FS_HAS_REALSENSE
    rs2::context context;
    std::string serial;
    for (const auto& device : context.query_devices()) {
        if (!device.supports(RS2_CAMERA_INFO_NAME) || !device.supports(RS2_CAMERA_INFO_SERIAL_NUMBER)) continue;
        if (std::string(device.get_info(RS2_CAMERA_INFO_NAME)).find("D435") == std::string::npos) continue;
        if (!serial.empty()) throw std::runtime_error("Multiple D435 cameras found. Connect only the D435 you want to use.");
        serial = device.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER);
    }
    if (serial.empty()) throw std::runtime_error("No Intel RealSense D435 found. Check its USB connection and reconnect.");
    rs2::config config;
    config.enable_device(serial);
    config.enable_stream(RS2_STREAM_INFRARED, 1, kSensorWidth, kSensorHeight, RS2_FORMAT_Y8, 30);
    config.enable_stream(RS2_STREAM_INFRARED, 2, kSensorWidth, kSensorHeight, RS2_FORMAT_Y8, 30);
    const auto profile = impl_->pipeline.start(config);
    impl_->running = true;
    try {
        const auto left = profile.get_stream(RS2_STREAM_INFRARED,1).as<rs2::video_stream_profile>();
        const auto right = profile.get_stream(RS2_STREAM_INFRARED,2).as<rs2::video_stream_profile>();
        const auto li = left.get_intrinsics(), ri = right.get_intrinsics();
        if (li.width != kSensorWidth || li.height != kSensorHeight || li.width != ri.width || li.height != ri.height)
            throw std::runtime_error("D435 infrared stream dimensions do not match.");
        // The legacy field names say right_to_left, but StereoFrame passes them
        // straight to stereoRectify, which requires the LEFT-to-RIGHT transform.
        const auto extrinsics = left.get_extrinsics_to(right);
        StereoCalibration c;
        c.left_camera_matrix = camera_matrix(li); c.right_camera_matrix = camera_matrix(ri);
        c.left_distortion = distortion(li); c.right_distortion = distortion(ri);
        c.right_to_left_rotation = cv::Mat(3,3,CV_64F);
        c.right_to_left_translation = cv::Mat(3,1,CV_64F);
        for (int row=0; row<3; ++row) {
            c.right_to_left_translation.at<double>(row) = extrinsics.translation[row];
            for (int col=0; col<3; ++col)
                c.right_to_left_rotation.at<double>(row,col) = extrinsics.rotation[col*3+row];
        }
        if (!(cv::norm(c.right_to_left_translation) > 0))
            throw std::runtime_error("D435 factory calibration has an invalid stereo baseline.");
        impl_->calibration = std::move(c); impl_->size = {kCropWidth,kSensorHeight}; impl_->serial = serial;
    } catch (...) { stop(); throw; }
#else
    throw std::runtime_error("RealSense support is unavailable. Install librealsense2-dev and rebuild the desktop application.");
#endif
}
void RealSenseStereoSource::stop() noexcept {
#ifdef FS_HAS_REALSENSE
    if (impl_->running) { try { impl_->pipeline.stop(); } catch (...) {} }
#endif
    impl_->running = false;
}
bool RealSenseStereoSource::running() const noexcept { return impl_->running; }
std::optional<StereoCameraPair> RealSenseStereoSource::wait_for_pair(std::chrono::milliseconds timeout) {
#ifdef FS_HAS_REALSENSE
    if (!impl_->running) return std::nullopt;
    rs2::frameset frames;
    const bool ready = timeout.count() > 0
        ? impl_->pipeline.try_wait_for_frames(&frames, static_cast<unsigned int>(timeout.count()))
        : impl_->pipeline.poll_for_frames(&frames);
    if (!ready) return std::nullopt;
    const auto left = frames.get_infrared_frame(1), right = frames.get_infrared_frame(2);
    // A frameset can be partial; don't pair a new frame with a stale sibling.
    if (!left || !right || left.get_frame_number() != right.get_frame_number()) return std::nullopt;
    const auto arrival = std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
    return StereoCameraPair{snapshot(left,arrival), snapshot(right,arrival)};
#else
    (void)timeout;
    return std::nullopt;
#endif
}
const StereoCalibration& RealSenseStereoSource::calibration() const { return impl_->calibration; }
cv::Size RealSenseStereoSource::image_size() const { return impl_->size; }
std::string RealSenseStereoSource::serial() const { return impl_->serial; }
std::string RealSenseStereoSource::calibration_json() const {
    if (!impl_->running) throw std::runtime_error("Connect the D435 before reading factory calibration.");
    cv::FileStorage f(".json", cv::FileStorage::WRITE | cv::FileStorage::MEMORY | cv::FileStorage::FORMAT_JSON);
    f << "calibration_source" << "RealSense D435 factory IR calibration"
      << "device_serial" << impl_->serial << "image_width" << impl_->size.width << "image_height" << impl_->size.height
      << "sensor_width" << kSensorWidth << "sensor_height" << kSensorHeight << "crop_x" << kCropX << "crop_y" << 0;
    const auto& c = impl_->calibration;
    f << "left_camera_matrix" << c.left_camera_matrix << "right_camera_matrix" << c.right_camera_matrix
      << "left_distortion" << c.left_distortion << "right_distortion" << c.right_distortion
      << "right_to_left_rotation" << c.right_to_left_rotation
      << "right_to_left_translation" << c.right_to_left_translation;
    return f.releaseAndGetString();
}
