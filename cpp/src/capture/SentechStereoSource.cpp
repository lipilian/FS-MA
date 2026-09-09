#include "fs/capture/SentechStereoSource.hpp"
#include "StereoCaptureSupport.hpp"

#include <StApi_IP.h>
#include <StApi_TL.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <thread>

namespace {

void configure_transport() {
    if (!std::getenv("GENICAM_GENTL64_PATH") &&
        setenv("GENICAM_GENTL64_PATH", FS_SENTECH_GENTL_DIRECTORY, 0) != 0)
        throw std::runtime_error("Unable to configure GENICAM_GENTL64_PATH");
}

struct DiscoveredDevice {
    SentechDeviceInfo info;
    StApi::IStInterface* interface;
    std::size_t index;
};

std::vector<DiscoveredDevice> discover(StApi::IStSystem& system) {
    std::vector<DiscoveredDevice> result;
    system.UpdateInterfaceList();
    for (std::size_t i = 0; i < system.GetInterfaceCount(); ++i) {
        auto* interface = system.GetIStInterface(i);
        interface->UpdateDeviceList();
        for (std::size_t j = 0; j < interface->GetDeviceCount(); ++j) {
            const auto* info = interface->GetIStDeviceInfo(j);
            result.push_back({{info->GetDisplayName().c_str(), info->GetUserDefinedName().c_str(),
                               info->GetSerialNumber().c_str()}, interface, j});
        }
    }
    return result;
}

bool set_enum(GenApi::INodeMap& map, const char* name, const char* value, bool required = true) {
    GenApi::CEnumerationPtr node(map.GetNode(name));
    if (GenApi::IsWritable(node)) {
        GenApi::CEnumEntryPtr entry(node->GetEntryByName(value));
        if (GenApi::IsAvailable(entry)) {
            node->SetIntValue(entry->GetValue());
            return true;
        }
    }
    if (required) throw std::runtime_error(std::string("Cannot configure ") + name + "=" + value);
    return false;
}

void configure_camera(StApi::IStDevice& device, double exposure_us) {
    auto* map = device.GetRemoteIStPort()->GetINodeMap();
    set_enum(*map, "AcquisitionMode", "Continuous");
    // Explicitly disable supported trigger modes for the reference free-running workflow.
    if (set_enum(*map, "TriggerSelector", "AcquisitionStart", false))
        set_enum(*map, "TriggerMode", "Off");
    if (set_enum(*map, "TriggerSelector", "FrameStart", false))
        set_enum(*map, "TriggerMode", "Off");
    set_enum(*map, "ExposureMode", "Timed");
    set_enum(*map, "ExposureAuto", "Off", false);
    GenApi::CFloatPtr exposure(map->GetNode("ExposureTime"));
    if (!GenApi::IsWritable(exposure)) throw std::runtime_error("ExposureTime is not writable");
    double value = std::clamp(exposure_us, exposure->GetMin(), exposure->GetMax());
    if (exposure->HasInc()) {
        if (exposure->GetIncMode() == GenApi::fixedIncrement && exposure->GetInc() > 0) {
            value = exposure->GetMin() + std::round((value - exposure->GetMin()) / exposure->GetInc()) * exposure->GetInc();
            value = std::clamp(value, exposure->GetMin(), exposure->GetMax());
        } else {
            const auto values = exposure->GetListOfValidValues();
            if (values.size()) {
                double nearest = values[0];
                for (std::size_t i = 1; i < values.size(); ++i)
                    if (std::abs(values[i] - value) < std::abs(nearest - value)) nearest = values[i];
                value = nearest;
            }
        }
    }
    exposure->SetValue(value);
    if (!set_enum(*map, "BalanceWhiteAuto", "Continuous", false))
        std::cerr << "Continuous white balance unavailable; retaining current balance\n";
    std::cout << device.GetIStDeviceInfo()->GetDisplayName() << ": exposure " << exposure->GetValue() << " us\n";
}

} // namespace

struct SentechStereoSource::Impl {
    explicit Impl(SentechStereoOptions value) : options(std::move(value)) {}
    SentechStereoOptions options;
    // SDK initialization must outlive all system/device/stream objects.
    std::optional<StApi::CStApiAutoInit> sdk;
    StApi::CIStSystemPtr system;
    std::array<StApi::CIStDevicePtr, 2> devices;
    std::array<StApi::CIStDataStreamPtr, 2> streams;
    std::array<StApi::CIStPixelFormatConverterPtr, 2> converters;
    std::array<StApi::CIStImageBufferPtr, 2> converted;
    std::array<std::thread, 2> workers;
    std::atomic_bool active{false};
    fs::capture_detail::PairBuffer buffer;

    void acquire(std::size_t side) noexcept {
        try {
            while (active.load()) {
                StApi::CIStStreamBufferPtr input(streams[side]->RetrieveBuffer(50, StApi::StTimeoutHandling_Return));
                if (!input.IsValid()) {
                    if (devices[side]->IsDeviceLost()) throw std::runtime_error("Camera disconnected");
                    continue;
                }
                const auto arrival = std::chrono::steady_clock::now().time_since_epoch();
                const auto* info = input->GetIStStreamBufferInfo();
                if (!info->IsImagePresent()) continue;
                converters[side]->Convert(input->GetIStImage(), converted[side]);
                const auto* image = converted[side]->GetIStImage();
                const auto width = image->GetImageWidth();
                const auto height = image->GetImageHeight();
                if (!width || !height || width > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                    height > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
                    image->GetImageLinePitch() < width * 3 || !image->GetImageBuffer())
                    throw std::runtime_error("Invalid converted RGB image layout");
                cv::Mat view(static_cast<int>(height), static_cast<int>(width), CV_8UC3,
                             image->GetImageBuffer(), image->GetImageLinePitch());
                buffer.publish(side, {view.clone(), info->GetFrameID(), info->GetTimestampNS(),
                    static_cast<std::uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(arrival).count())});
            }
        } catch (const GenICam::GenericException& error) {
            buffer.fail(std::string(side == 0 ? "Left: " : "Right: ") + error.GetDescription());
            active.store(false);
        } catch (const std::exception& error) {
            buffer.fail(std::string(side == 0 ? "Left: " : "Right: ") + error.what());
            active.store(false);
        } catch (...) {
            buffer.fail("Unknown camera acquisition error");
            active.store(false);
        }
    }

    void stop() noexcept {
        active.store(false);
        buffer.stop();
        // Retrieval is bounded to 50 ms; join before touching SDK objects used by workers.
        for (auto& worker : workers) if (worker.joinable()) worker.join();
        for (auto& device : devices) if (device.IsValid()) {
            try { device->AcquisitionStop(); } catch (...) {}
        }
        for (auto& stream : streams) if (stream.IsValid()) {
            try { stream->StopAcquisition(); } catch (...) {}
        }
        for (auto& stream : streams) stream.Reset();
        for (auto& converter : converters) converter.Reset();
        for (auto& image : converted) image.Reset();
        for (auto& device : devices) device.Reset();
        system.Reset();
    }
    ~Impl() { stop(); }
};

SentechStereoSource::SentechStereoSource(SentechStereoOptions options) {
    if (options.left.empty() || options.right.empty() || options.left == options.right ||
        !std::isfinite(options.exposure_us) || options.exposure_us <= 0 || options.max_arrival_skew.count() <= 0)
        throw std::invalid_argument("Distinct left/right identities and positive exposure/skew are required");
    impl_ = std::make_unique<Impl>(std::move(options));
}
SentechStereoSource::~SentechStereoSource() = default;

std::vector<SentechDeviceInfo> SentechStereoSource::list_devices() {
    try {
        configure_transport();
        StApi::CStApiAutoInit sdk;
        StApi::CIStSystemPtr system(StApi::CreateIStSystem());
        std::vector<SentechDeviceInfo> result;
        for (const auto& device : discover(*system)) result.push_back(device.info);
        return result;
    } catch (const GenICam::GenericException& error) { throw std::runtime_error(error.GetDescription()); }
}

void SentechStereoSource::start() {
    if (running()) return;
    impl_->stop();
    impl_->buffer.reset();
    try {
        configure_transport();
        if (!impl_->sdk) impl_->sdk.emplace();
        impl_->system.Reset(StApi::CreateIStSystem());
        const auto found = discover(*impl_->system);
        std::vector<SentechDeviceInfo> identities;
        for (const auto& device : found) identities.push_back(device.info);
        const auto roles = fs::capture_detail::select_roles(identities, impl_->options);
        for (std::size_t side = 0; side < 2; ++side) {
            const auto& selected = found[roles[side]];
            impl_->devices[side].Reset(selected.interface->CreateIStDevice(selected.index));
            std::cout << (side == 0 ? "Left: " : "Right: ") << selected.info.display_name << '\n';
            configure_camera(*impl_->devices[side], impl_->options.exposure_us);
            impl_->streams[side].Reset(impl_->devices[side]->CreateIStDataStream(0));
            impl_->converters[side].Reset(StApi::CreateIStConverter(StApi::StConverterType_PixelFormat));
            impl_->converters[side]->SetDestinationPixelFormat(StApi::StPFNC_RGB8);
            impl_->converted[side].Reset(StApi::CreateIStImageBuffer());
        }
        for (std::size_t side = 0; side < 2; ++side) {
            impl_->streams[side]->StartAcquisition();
            impl_->devices[side]->AcquisitionStart();
        }
        impl_->active.store(true);
        for (std::size_t side = 0; side < 2; ++side)
            impl_->workers[side] = std::thread(&Impl::acquire, impl_.get(), side);
    } catch (const GenICam::GenericException& error) {
        impl_->stop();
        throw std::runtime_error(error.GetDescription());
    } catch (...) { impl_->stop(); throw; }
}

void SentechStereoSource::stop() noexcept { impl_->stop(); }
bool SentechStereoSource::running() const noexcept { return impl_->active.load(); }
std::optional<StereoCameraPair> SentechStereoSource::wait_for_pair(std::chrono::milliseconds timeout) {
    if (timeout.count() < 0) throw std::invalid_argument("Timeout must not be negative");
    try { return impl_->buffer.wait(timeout, impl_->options.max_arrival_skew); }
    catch (...) { impl_->stop(); throw; }
}
