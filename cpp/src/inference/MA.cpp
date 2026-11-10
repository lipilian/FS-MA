#include "fs/inference/MA.hpp"
#include "fs/inference/MAPreprocessing.hpp"
#include "fs/inference/MAPostprocessing.hpp"
#include "fs/stereo/StereoFrame.hpp"

#include <NvInfer.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs {
namespace {
constexpr std::size_t kColorPlane = std::size_t{MA::kInputHeight} * MA::kInputWidth;
constexpr std::size_t kColorElements = 3 * kColorPlane;

class Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[MA TensorRT] " << message << '\n';
    }
} logger;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("MapAnything: " + message);
}
void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("MapAnything: ") + operation + ": " + cudaGetErrorString(error));
}
nvinfer1::Dims dims(std::initializer_list<int64_t> values) {
    nvinfer1::Dims result{};
    result.nbDims = int(values.size());
    std::copy(values.begin(), values.end(), result.d);
    return result;
}
bool equal(nvinfer1::Dims a, nvinfer1::Dims b) {
    return a.nbDims == b.nbDims && a.nbDims >= 0 && std::equal(a.d, a.d + a.nbDims, b.d);
}

struct CudaStream {
    cudaStream_t handle{};

    CudaStream() {
        const auto error = cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking);
        if (error != cudaSuccess)
            throw std::runtime_error(std::string("MapAnything: cannot create CUDA stream: ") + cudaGetErrorString(error));
    }
    ~CudaStream() { cudaStreamDestroy(handle); }
    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    void synchronize() const {
        const auto error = cudaStreamSynchronize(handle);
        if (error != cudaSuccess)
            throw std::runtime_error(std::string("MapAnything: cannot synchronize CUDA stream: ") + cudaGetErrorString(error));
    }
};

struct DeviceFree {
    void operator()(float* pointer) const noexcept { if (pointer) cudaFree(pointer); }
};
using DeviceBuffer = std::unique_ptr<float, DeviceFree>;
DeviceBuffer allocate(std::size_t elements, const char* name) {
    void* pointer = nullptr;
    const auto error = cudaMalloc(&pointer, elements * sizeof(float));
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("MapAnything: cannot allocate GPU ") + name + ": " + cudaGetErrorString(error));
    return DeviceBuffer(static_cast<float*>(pointer));
}

struct HostFree {
    void operator()(float* pointer) const noexcept { if (pointer) cudaFreeHost(pointer); }
};
struct InputUpload {
    std::unique_ptr<float, HostFree> host;
    cv::Mat resized;
    cudaEvent_t done{};
    bool pending{false};

    InputUpload(int type, std::size_t elements) : resized(MA::kInputHeight, MA::kInputWidth, type) {
        void* pointer = nullptr;
        checkCuda(cudaMallocHost(&pointer, elements * sizeof(float)), "cannot allocate pinned input staging buffer");
        host.reset(static_cast<float*>(pointer));
        checkCuda(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "cannot create input upload event");
    }
    ~InputUpload() { cudaEventDestroy(done); }
    InputUpload(const InputUpload&) = delete;
    InputUpload& operator=(const InputUpload&) = delete;

    void wait() const {
        if (pending) checkCuda(cudaEventSynchronize(done), "cannot wait for previous input upload");
    }
    void enqueue(float* destination, std::size_t elements, cudaStream_t stream) {
        checkCuda(cudaMemcpyAsync(destination, host.get(), elements * sizeof(float), cudaMemcpyHostToDevice, stream),
                  "cannot upload input");
        const auto error = cudaEventRecord(done, stream);
        // Drain the H2D read if recording fails before staging can be reused.
        if (error != cudaSuccess) {
            cudaStreamSynchronize(stream);
            checkCuda(error, "cannot record input upload completion");
        }
        pending = true;
    }
};

struct Model {
    cudaStream_t stream;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    int64_t workspace_bytes{};

    Model(cudaStream_t ma_stream, const std::filesystem::path& path) : stream(ma_stream) {
        auto* active_logger = getLogger();
        runtime.reset(nvinfer1::createInferRuntime(active_logger ? *active_logger : logger));
        require(bool(runtime), "cannot create TensorRT runtime");
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(bool(file), "cannot open engine: " + path.string());
        const auto size = file.tellg();
        require(size > 0, "empty engine: " + path.string());
        std::vector<char> bytes(static_cast<std::size_t>(size));
        file.seekg(0);
        require(bool(file.read(bytes.data(), std::streamsize(bytes.size()))), "cannot read engine: " + path.string());
        engine.reset(runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        require(bool(engine), "cannot deserialize engine: " + path.string());
        workspace_bytes = engine->getDeviceMemorySizeV2();
        require(workspace_bytes >= 0, "cannot query TensorRT activation memory");
    }
    // Also drain profile work if construction of the containing MA fails.
    ~Model() { cudaStreamSynchronize(stream); }

    // Borrow MA's shared allocation; it outlives both models and their contexts.
    void createContext(void* workspace, int64_t capacity) {
        require(capacity >= workspace_bytes && (workspace_bytes == 0 || workspace),
                "shared activation workspace is too small");
        context.reset(engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
        require(bool(context), "cannot create TensorRT context");
        context->setDeviceMemoryV2(workspace, capacity);
        require(context->setOptimizationProfileAsync(0, stream), "cannot select the MA engine profile");
    }

    void tensor(const char* name, nvinfer1::Dims shape, nvinfer1::TensorIOMode mode) const {
        require(engine->getTensorIOMode(name) == mode &&
                engine->getTensorDataType(name) == nvinfer1::DataType::kFLOAT &&
                engine->getTensorLocation(name) == nvinfer1::TensorLocation::kDEVICE &&
                engine->getTensorFormat(name) == nvinfer1::TensorFormat::kLINEAR &&
                equal(engine->getTensorShape(name), shape), std::string("incompatible tensor: ") + name);
    }
};
} // namespace

struct MA::Impl {
    // Reverse destruction releases both models/contexts before their shared
    // workspace and I/O buffers, then our stream.
    // RAII also releases the stream if a buffer allocation fails in construction.
    CudaStream stream;
    DeviceBuffer inputs, tokens, cls, poses, scale;
    DeviceBuffer workspace;
    int64_t workspace_bytes{};
    InputUpload color_upload{CV_8UC3, kColorElements};
    InputUpload depth_upload{CV_32FC1, kColorPlane};
    cv::Mat valid_depth;
    std::array<bool, kMaxViews> rays_ready{};
    std::array<bool, kMaxViews> depth_ready{};
    std::array<bool, kMaxViews> encoded{};
    bool inference_pending{false};
    int output_views{0};
    std::unique_ptr<Model> encoder, pose_scale;

    Impl(const std::filesystem::path& encoder_path, const std::filesystem::path& pose_scale_path) {
        inputs = allocate(kInputElements, "inputs");
        tokens = allocate(kTokenElements, "five-view F_fused tokens");
        cls = allocate(kClsElements, "five-view CLS");
        poses = allocate(kPoseElements, "poses");
        scale = allocate(kScaleElements, "scale");

        using Mode = nvinfer1::TensorIOMode;
        using Profile = nvinfer1::OptProfileSelector;
        encoder = std::make_unique<Model>(stream.handle, encoder_path);
        require(encoder->engine->getNbIOTensors() == 3 && encoder->engine->getNbOptimizationProfiles() == 1,
                "expected the single-view encoder with three I/O tensors and one profile");
        encoder->tensor("inputs", dims({1, 7, kInputHeight, kInputWidth}), Mode::kINPUT);
        encoder->tensor("F_fused", dims({1, kFeatureChannels, kFeatureHeight, kFeatureWidth}), Mode::kOUTPUT);
        encoder->tensor("CLS", dims({1, kFeatureChannels, 1}), Mode::kOUTPUT);
        pose_scale = std::make_unique<Model>(stream.handle, pose_scale_path);
        require(pose_scale->engine->getNbIOTensors() == 4 && pose_scale->engine->getNbOptimizationProfiles() == 1,
                "expected the pose/scale engine with four I/O tensors and one profile");
        pose_scale->tensor("F_fused", dims({-1, kFeatureChannels, kFeatureHeight, kFeatureWidth}), Mode::kINPUT);
        pose_scale->tensor("CLS", dims({-1, kFeatureChannels, 1}), Mode::kINPUT);
        pose_scale->tensor("poses", dims({-1, 7}), Mode::kOUTPUT);
        pose_scale->tensor("scale", dims({1, 1, 1}), Mode::kOUTPUT);
        for (const auto profile : {Profile::kMIN, Profile::kOPT, Profile::kMAX}) {
            const int views = profile == Profile::kMIN ? kMinViews : profile == Profile::kOPT ? 3 : kMaxViews;
            require(equal(pose_scale->engine->getProfileShape("F_fused", 0, profile),
                          dims({views, kFeatureChannels, kFeatureHeight, kFeatureWidth})) &&
                    equal(pose_scale->engine->getProfileShape("CLS", 0, profile), dims({views, kFeatureChannels, 1})),
                    "pose/scale profile must support min/opt/max views 2/3/5");
        }
        // Query maximum execution memory over each engine's profiles once at load.
        // Both contexts execute serially on stream, so they borrow the same region.
        workspace_bytes = std::max(encoder->workspace_bytes, pose_scale->workspace_bytes);
        if (workspace_bytes > 0)
            workspace = allocate((static_cast<std::size_t>(workspace_bytes) + sizeof(float) - 1) / sizeof(float),
                                 "shared encoder/pose-scale activation workspace");
        encoder->createContext(workspace.get(), workspace_bytes);
        pose_scale->createContext(workspace.get(), workspace_bytes);
        bindEncoderView(0);
        auto& context = pose_scale->context;
        require(context->setTensorAddress("F_fused", tokens.get()) && context->setTensorAddress("CLS", cls.get()) &&
                context->setTensorAddress("poses", poses.get()) && context->setTensorAddress("scale", scale.get()),
                "cannot bind pose/scale to the preallocated feature banks");
        for (int views = kMinViews; views <= kMaxViews; ++views) setPoseShape(views);
        stream.synchronize();

        constexpr double mib = 1024.0 * 1024.0;
        std::cout << "[MA] FMHA encoder: " << encoder_path << '\n'
                  << "[MA] FMHA pose/scale: " << pose_scale_path << '\n'
                  << "[MA] Constructor allocations: I/O " << double(kIOBytes) / mib
                  << " MiB; shared activation workspace " << double(workspace_bytes) / mib
                  << " MiB (encoder requires " << double(encoder->workspace_bytes) / mib
                  << " MiB; pose/scale requires " << double(pose_scale->workspace_bytes) / mib << " MiB)\n"
                  << "[MA] One encoder context reused serially across five fixed F_fused[1536,31,37] / CLS[1536,1] slots; one pose/scale context\n"
                  << std::flush;
    }
    ~Impl() { cudaStreamSynchronize(stream.handle); }

    void bindEncoderView(int view) {
        auto& context = encoder->context;
        require(context->setTensorAddress("inputs", inputs.get() + view * kInputStrideElements) &&
                context->setTensorAddress("F_fused", tokens.get() + view * kTokenElementsPerView) &&
                context->setTensorAddress("CLS", cls.get() + view * kClsElementsPerView),
                "cannot bind encoder to its fixed view slot");
    }

    void setPoseShape(int views) {
        auto& context = pose_scale->context;
        require(context->setInputShape("F_fused", dims({views, kFeatureChannels, kFeatureHeight, kFeatureWidth})) &&
                context->setInputShape("CLS", dims({views, kFeatureChannels, 1})), "cannot set pose/scale view count");
        require(context->inferShapes(0, nullptr) == 0 &&
                equal(context->getTensorShape("poses"), dims({views, 7})) &&
                equal(context->getTensorShape("scale"), dims({1, 1, 1})), "unexpected pose/scale output shapes");
    }
};

MA::MA(const std::filesystem::path& encoder_path, const std::filesystem::path& pose_scale_path)
    : impl_(std::make_unique<Impl>(encoder_path, pose_scale_path)) {}
MA::~MA() = default;
bool MA::isLoaded() const noexcept { return impl_->encoder && impl_->pose_scale; }
cudaStream_t MA::stream() const noexcept { return impl_->stream.handle; }
float* MA::inputDevice() const noexcept { return impl_->inputs.get(); }
float* MA::tokensDevice() const noexcept { return impl_->tokens.get(); }
float* MA::clsDevice() const noexcept { return impl_->cls.get(); }
float* MA::posesDevice() const noexcept { return impl_->poses.get(); }
float* MA::scaleDevice() const noexcept { return impl_->scale.get(); }
const cv::Mat& MA::resizedColor() const noexcept { return impl_->color_upload.resized; }

void MA::uploadColor(int view_index, const cv::Mat& rectified_rgb, const StereoCameraParameters& camera) {
    require(view_index >= 0 && view_index < kMaxViews, "RGB view index must be in 0..4");
    require(!rectified_rgb.empty() && rectified_rgb.type() == CV_8UC3,
            "RGB upload requires a nonempty CV_8UC3 RGB image");
    const double scale_x = double(kInputWidth) / rectified_rgb.cols;
    const double scale_y = double(kInputHeight) / rectified_rgb.rows;
    const float fx = float(camera.fx * scale_x), fy = float(camera.fy * scale_y);
    // The notebook's OpenCV -> COLMAP -> OpenCV pixel-center adjustment,
    // adapted to independent X/Y resize scales with no crop or crop offset.
    const float cx = float((camera.cx + 0.5) * scale_x - 0.5);
    const float cy = float((camera.cy + 0.5) * scale_y - 0.5);
    require(std::isfinite(fx) && fx > 0.0F && std::isfinite(fy) && fy > 0.0F &&
            std::isfinite(cx) && std::isfinite(cy), "RGB upload requires valid resized camera intrinsics");
    auto& upload = impl_->color_upload;
    cv::resize(rectified_rgb, upload.resized, cv::Size(kInputWidth, kInputHeight),
               0.0, 0.0, cv::INTER_LANCZOS4);
    // Wait only for the preceding H2D read of the reused pinned buffer, not
    // for later MA kernels or anything on FS's independent stream.
    upload.wait();
    float* destination = upload.host.get();
    for (int y = 0; y < kInputHeight; ++y) {
        const auto* row = upload.resized.ptr<cv::Vec3b>(y);
        for (int x = 0; x < kInputWidth; ++x) {
            const auto offset = static_cast<std::size_t>(y) * kInputWidth + x;
            destination[offset] = float(row[x][0]);
            destination[kColorPlane + offset] = float(row[x][1]);
            destination[2 * kColorPlane + offset] = float(row[x][2]);
        }
    }
    float* view_input = inputDevice() + static_cast<std::size_t>(view_index) * kInputStrideElements;
    impl_->rays_ready[view_index] = false;
    impl_->depth_ready[view_index] = false;
    impl_->encoded[view_index] = false;
    impl_->output_views = 0;
    upload.enqueue(view_input, kColorElements, stream());
    // Same-stream ordering starts this kernel after H2D, without a CPU wait.
    // The event above protects only host staging; the kernel uses device input.
    checkCuda(ma_preprocessing::prepare_rgb_and_rays(view_input, kInputWidth, kInputHeight,
                                                    fx, fy, cx, cy, stream()),
              "cannot launch RGB normalization and camera rays");
    impl_->rays_ready[view_index] = true;
}

void MA::uploadDepth(int view_index, const cv::Mat& depth_z) {
    require(view_index >= 0 && view_index < kMaxViews, "Depth view index must be in 0..4");
    require(!depth_z.empty() && depth_z.dims == 2 && depth_z.type() == CV_32FC1,
            "Depth upload requires a nonempty CV_32FC1 metric Z-depth image");
    require(impl_->rays_ready[view_index], "Depth upload requires camera rays from uploadColor for this view");
    auto& upload = impl_->depth_upload;
    auto& valid = impl_->valid_depth;
    valid.create(depth_z.size(), CV_32FC1);
    // Match the notebook's source-depth cleanup before interpolation, so a
    // non-finite source pixel cannot contaminate neighbouring Lanczos samples.
    for (int y = 0; y < depth_z.rows; ++y) {
        const float* source = depth_z.ptr<float>(y);
        float* target = valid.ptr<float>(y);
        for (int x = 0; x < depth_z.cols; ++x)
            target[x] = std::isfinite(source[x]) && source[x] > 0.0F ? source[x] : 0.0F;
    }
    cv::resize(valid, upload.resized, cv::Size(kInputWidth, kInputHeight), 0.0, 0.0, cv::INTER_LANCZOS4);
    upload.wait();
    std::memcpy(upload.host.get(), upload.resized.ptr<float>(), kColorPlane * sizeof(float));
    float* view_input = inputDevice() + static_cast<std::size_t>(view_index) * kInputStrideElements;
    impl_->depth_ready[view_index] = false;
    impl_->encoded[view_index] = false;
    impl_->output_views = 0;
    upload.enqueue(view_input + 6 * kColorPlane, kColorPlane, stream());
    // Same-stream ordering waits for both this H2D and the earlier ray kernel.
    checkCuda(ma_preprocessing::prepare_ray_distance(view_input, kInputWidth, kInputHeight, stream()),
              "cannot launch Z-depth to ray-distance conversion");
    impl_->depth_ready[view_index] = true;
}

void MA::resetInputs() noexcept {
    impl_->rays_ready.fill(false);
    impl_->depth_ready.fill(false);
    impl_->encoded.fill(false);
    impl_->output_views = 0;
}

void MA::inference(int view_count) {
    require(view_count >= kMinViews && view_count <= kMaxViews, "Inference requires 2..5 views");
    for (int view = 0; view < view_count; ++view)
        require(impl_->rays_ready[view] && impl_->depth_ready[view],
                "Inference requires RGB, rays and depth for view " + std::to_string(view + 1));
    // The suffix's dynamic shape cannot change while its previous run is in flight.
    if (impl_->inference_pending) synchronize();
    impl_->output_views = 0;
    impl_->setPoseShape(view_count);
    impl_->inference_pending = true; // A failed enqueue may still submit work.
    for (int view = 0; view < view_count; ++view) {
        if (impl_->encoded[view]) continue;
        // Rebind the same context for each enqueue. All view buffers stay alive,
        // and same-stream ordering serializes reuse of its activation workspace.
        impl_->bindEncoderView(view);
        require(impl_->encoder->context->enqueueV3(stream()), "encoder enqueueV3 failed");
        impl_->encoded[view] = true;
    }
    // Same-stream ordering lets pose/scale reuse the encoder's workspace after
    // its last kernel, including TensorRT's joined auxiliary streams. Token/CLS
    // banks are separate persistent buffers and are read directly by pose/scale.
    require(impl_->pose_scale->context->enqueueV3(stream()), "pose/scale enqueueV3 failed");
    impl_->output_views = view_count;
}

void MA::synchronize() {
    try {
        impl_->stream.synchronize();
    } catch (...) {
        impl_->encoded.fill(false);
        impl_->output_views = 0;
        throw;
    }
    impl_->inference_pending = false;
}

std::vector<cv::Matx44d> MA::downloadCameraPoses() {
    require(impl_->output_views >= kMinViews, "Camera poses require current inference outputs");
    const int views = impl_->output_views;
    std::array<float, kPoseElements> poses{};
    float scale{};
    try {
        checkCuda(cudaMemcpyAsync(poses.data(), posesDevice(), std::size_t(views) * 7 * sizeof(float),
                                  cudaMemcpyDeviceToHost, stream()), "cannot download predicted camera poses");
        checkCuda(cudaMemcpyAsync(&scale, scaleDevice(), sizeof(scale), cudaMemcpyDeviceToHost, stream()),
                  "cannot download predicted camera scale");
        synchronize();
    } catch (...) {
        cudaStreamSynchronize(stream()); // Keep host destinations alive until queued copies finish.
        throw;
    }
    return ma_postprocessing::camera_poses(poses.data(), views, scale);
}

} // namespace fs
