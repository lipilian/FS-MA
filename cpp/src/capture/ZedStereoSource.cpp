#include "fs/capture/ZedStereoSource.hpp"
#include <opencv2/imgproc.hpp>
#include <cmath>
#include <stdexcept>
#include <thread>

#ifdef FS_HAS_ZED
#include <sl/Camera.hpp>
#endif

namespace {
constexpr int kSensorWidth = 1920, kSensorHeight = 1080;
constexpr int kCropWidth = 1296, kCropX = (kSensorWidth-kCropWidth)/2;
}

struct ZedStereoSource::Impl {
    bool running{false};
    StereoCalibration calibration;
    cv::Size size;
    std::string serial;
    std::uint64_t frame_id{0};
#ifdef FS_HAS_ZED
    sl::Camera camera;
    sl::Mat left, right;
#endif
};

ZedStereoSource::ZedStereoSource() : impl_(std::make_unique<Impl>()) {}
ZedStereoSource::~ZedStereoSource() { stop(); }
bool ZedStereoSource::available() noexcept {
#ifdef FS_HAS_ZED
    return true;
#else
    return false;
#endif
}

#ifdef FS_HAS_ZED
namespace {
void check(sl::ERROR_CODE result, const char* operation) {
    if (result != sl::ERROR_CODE::SUCCESS)
        throw std::runtime_error(std::string("ZED ")+operation+": "+sl::toString(result).c_str());
}
cv::Mat croppedIntrinsics(const sl::CameraParameters& camera) {
    return (cv::Mat_<double>(3,3) << camera.fx,0,camera.cx-kCropX,0,camera.fy,camera.cy,0,0,1);
}
CameraFrame snapshot(sl::Mat& image, std::uint64_t frame_id,
                     std::uint64_t timestamp, std::uint64_t arrival) {
    if (image.getWidth()!=kSensorWidth || image.getHeight()!=kSensorHeight ||
        image.getDataType()!=sl::MAT_TYPE::U8_C4)
        throw std::runtime_error("ZED rectified frame is not 1920 x 1080 BGRA.");
    const cv::Mat bgra(kSensorHeight,kSensorWidth,CV_8UC4,
                       image.getPtr<sl::uchar1>(sl::MEM::CPU),image.getStepBytes());
    CameraFrame frame;
    cv::cvtColor(bgra(cv::Rect(kCropX,0,kCropWidth,kSensorHeight)),frame.rgb,cv::COLOR_BGRA2RGB);
    frame.frame_id=frame_id; frame.device_timestamp_ns=timestamp; frame.host_arrival_ns=arrival;
    return frame;
}
}
#endif

void ZedStereoSource::start() {
    if (impl_->running) return;
#ifdef FS_HAS_ZED
    const auto devices=sl::Camera::getDeviceList();
    if (devices.empty()) throw std::runtime_error("No Stereolabs ZED camera found. Check its USB connection and reconnect.");
    if (devices.size()!=1) throw std::runtime_error("Multiple ZED cameras found. Connect only the camera you want to use.");
    sl::InitParameters init;
    init.input.setFromSerialNumber(devices.front().serial_number);
    init.camera_resolution=sl::RESOLUTION::HD1080;
    init.camera_fps=30;
    init.depth_mode=sl::DEPTH_MODE::NONE; // FoundationStereo owns depth estimation.
    init.coordinate_units=sl::UNIT::METER;
    init.coordinate_system=sl::COORDINATE_SYSTEM::IMAGE;
    init.async_grab_camera_recovery=true;
    try {
        check(impl_->camera.open(init),"open");
        const auto info=impl_->camera.getCameraInformation();
        const auto& config=info.camera_configuration;
        if (config.resolution.width!=kSensorWidth || config.resolution.height!=kSensorHeight)
            throw std::runtime_error("ZED must support HD1080 (1920 x 1080 per eye).");
        const auto& rectified=config.calibration_parameters;
        StereoCalibration c;
        c.left_camera_matrix=croppedIntrinsics(rectified.left_cam);
        c.right_camera_matrix=croppedIntrinsics(rectified.right_cam);
        if (cv::norm(c.left_camera_matrix,c.right_camera_matrix,cv::NORM_INF)>1e-6)
            throw std::runtime_error("ZED rectified left/right intrinsics do not match.");
        const double baseline=rectified.getCameraBaseline();
        if (!std::isfinite(baseline) || baseline<=0)
            throw std::runtime_error("ZED calibration has an invalid stereo baseline.");
        c.left_distortion=cv::Mat::zeros(1,5,CV_64F);
        c.right_distortion=cv::Mat::zeros(1,5,CV_64F);
        c.right_to_left_rotation=cv::Mat::eye(3,3,CV_64F);
        // Legacy field names are consumed as OpenCV LEFT-to-RIGHT extrinsics.
        // The right optical center is +B in the left frame, so X_right=X_left-B.
        c.right_to_left_translation=(cv::Mat_<double>(3,1) << -baseline,0,0);
        impl_->calibration=std::move(c);
        impl_->size={kCropWidth,kSensorHeight};
        impl_->serial=std::to_string(info.serial_number);
        impl_->frame_id=0;
        impl_->running=true;
    } catch (...) { stop(); throw; }
#else
    throw std::runtime_error("ZED support is unavailable. Install the Stereolabs ZED SDK and rebuild scan_gui.");
#endif
}

void ZedStereoSource::stop() noexcept {
#ifdef FS_HAS_ZED
    impl_->left.free(); impl_->right.free();
    impl_->camera.close();
#endif
    impl_->running=false;
}
bool ZedStereoSource::running() const noexcept { return impl_->running; }
std::optional<StereoCameraPair> ZedStereoSource::wait_for_pair(std::chrono::milliseconds timeout) {
#ifdef FS_HAS_ZED
    if (!impl_->running) return std::nullopt;
    const auto deadline=std::chrono::steady_clock::now()+timeout;
    do {
        // grab waits at most for the next sensor frame under normal operation.
        // Async recovery prevents unplugging from blocking the worker indefinitely.
        const auto status=impl_->camera.grab();
        if (status==sl::ERROR_CODE::SUCCESS) {
            const auto timestamp=impl_->camera.getTimestamp(sl::TIME_REFERENCE::IMAGE).getNanoseconds();
            check(impl_->camera.retrieveImage(impl_->left,sl::VIEW::LEFT,sl::MEM::CPU),"retrieve left");
            check(impl_->camera.retrieveImage(impl_->right,sl::VIEW::RIGHT,sl::MEM::CPU),"retrieve right");
            const auto arrival=std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
            const auto id=++impl_->frame_id;
            return StereoCameraPair{snapshot(impl_->left,id,timestamp,arrival),
                                    snapshot(impl_->right,id,timestamp,arrival)};
        }
        if (status!=sl::ERROR_CODE::CAMERA_REBOOTING) check(status,"grab");
        if (std::chrono::steady_clock::now()>=deadline) break;
        std::this_thread::sleep_for(std::chrono::milliseconds(5));
    } while (impl_->running);
#else
    (void)timeout;
#endif
    return std::nullopt;
}

const StereoCalibration& ZedStereoSource::calibration() const { return impl_->calibration; }
cv::Size ZedStereoSource::image_size() const { return impl_->size; }
std::string ZedStereoSource::serial() const { return impl_->serial; }
std::string ZedStereoSource::calibration_json() const {
    if (!impl_->running) throw std::runtime_error("Connect the ZED before reading calibration.");
    cv::FileStorage f(".json",cv::FileStorage::WRITE|cv::FileStorage::MEMORY|cv::FileStorage::FORMAT_JSON);
    f << "calibration_source" << "Stereolabs ZED SDK rectified calibration"
      << "device_serial" << impl_->serial << "length_unit" << "meter" << "images_rectified" << 1
      << "image_width" << impl_->size.width << "image_height" << impl_->size.height
      << "sensor_width" << kSensorWidth << "sensor_height" << kSensorHeight
      << "crop_x" << kCropX << "crop_y" << 0 << "crop_width" << kCropWidth << "crop_height" << kSensorHeight
      << "fs_input_width" << 960 << "fs_input_height" << 800 << "fs_resize" << "INTER_LINEAR";
    const auto& c=impl_->calibration;
    f << "left_camera_matrix" << c.left_camera_matrix << "right_camera_matrix" << c.right_camera_matrix
      << "left_distortion" << c.left_distortion << "right_distortion" << c.right_distortion
      << "right_to_left_rotation" << c.right_to_left_rotation
      << "right_to_left_translation" << c.right_to_left_translation;
    return f.releaseAndGetString();
}
