#include "fs/inference/MA-VGGT.hpp"
#include "fs/inference/MAPreprocessing.hpp"

#include <NvInfer.h>
#include <opencv2/imgproc.hpp>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs {
namespace {
constexpr std::size_t kColorPlane = std::size_t{MA_VGGT::kInputHeight} * MA_VGGT::kInputWidth;
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

std::string tensorShapeString(const nvinfer1::Dims& shape) {
    std::ostringstream output;
    output << '[';
    for (int i = 0; i < shape.nbDims; ++i) {
        if (i) output << ", ";
        output << shape.d[i];
    }
    return output.str() + ']';
}

std::size_t freeDeviceMemory() {
    std::size_t free{}, total{};
    const auto error = cudaMemGetInfo(&free, &total);
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("MapAnything: cannot query free device memory: ") + cudaGetErrorString(error));
    return free;
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
struct ColorUpload {
    std::unique_ptr<float, HostFree> host;
    cv::Mat resized;
    cudaEvent_t done{};
    bool pending{false};

    ColorUpload() : resized(MA_VGGT::kInputHeight, MA_VGGT::kInputWidth, CV_8UC3) {
        void* pointer = nullptr;
        checkCuda(cudaMallocHost(&pointer, kColorElements * sizeof(float)), "cannot allocate pinned RGB staging buffer");
        host.reset(static_cast<float*>(pointer));
        checkCuda(cudaEventCreateWithFlags(&done, cudaEventDisableTiming), "cannot create RGB upload event");
    }
    ~ColorUpload() { cudaEventDestroy(done); }
    ColorUpload(const ColorUpload&) = delete;
    ColorUpload& operator=(const ColorUpload&) = delete;
};

struct Model {
    cudaStream_t stream;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    explicit Model(cudaStream_t ma_stream) : stream(ma_stream) {}
    // Also complete profile work before destroying an unsuccessful load.
    ~Model() { cudaStreamSynchronize(stream); }

    void tensor(const char* name, nvinfer1::Dims shape, nvinfer1::TensorIOMode mode) const {
        require(engine->getTensorIOMode(name) == mode &&
                engine->getTensorDataType(name) == nvinfer1::DataType::kFLOAT &&
                engine->getTensorLocation(name) == nvinfer1::TensorLocation::kDEVICE &&
                engine->getTensorFormat(name) == nvinfer1::TensorFormat::kLINEAR &&
                equal(engine->getTensorShape(name), shape), std::string("incompatible tensor: ") + name);
    }
};
} // namespace

struct MA_VGGT::Impl {
    // Reverse destruction releases context/model, I/O buffers, then our stream.
    // RAII also releases the stream if a buffer allocation fails in construction.
    CudaStream stream;
    DeviceBuffer inputs, depths, poses, scale;
    ColorUpload color_upload;
    std::unique_ptr<Model> model;

    Impl() {
        inputs = allocate(kInputElements, "inputs");
        depths = allocate(kDepthElements, "depths");
        poses = allocate(kPoseElements, "poses");
        scale = allocate(kScaleElements, "scale");
    }
    ~Impl() { cudaStreamSynchronize(stream.handle); }
};

MA_VGGT::MA_VGGT() : impl_(std::make_unique<Impl>()) {}
MA_VGGT::~MA_VGGT() = default;
bool MA_VGGT::isLoaded() const noexcept { return bool(impl_->model); }
cudaStream_t MA_VGGT::stream() const noexcept { return impl_->stream.handle; }
float* MA_VGGT::inputDevice() const noexcept { return impl_->inputs.get(); }
float* MA_VGGT::depthsDevice() const noexcept { return impl_->depths.get(); }
float* MA_VGGT::posesDevice() const noexcept { return impl_->poses.get(); }
float* MA_VGGT::scaleDevice() const noexcept { return impl_->scale.get(); }

void MA_VGGT::uploadColor(int view_index, const cv::Mat& rectified_rgb) {
    require(view_index >= 0 && view_index < kMaxViews, "RGB view index must be in 0..4");
    require(!rectified_rgb.empty() && rectified_rgb.type() == CV_8UC3,
            "RGB upload requires a nonempty CV_8UC3 RGB image");
    auto& upload = impl_->color_upload;
    cv::resize(rectified_rgb, upload.resized, cv::Size(kInputWidth, kInputHeight),
               0.0, 0.0, cv::INTER_LANCZOS4);
    // Wait only for the preceding H2D read of the reused pinned buffer, not
    // for later MA kernels or anything on FS's independent stream.
    if (upload.pending) checkCuda(cudaEventSynchronize(upload.done), "cannot wait for previous RGB upload");
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
    float* rgb = inputDevice() + static_cast<std::size_t>(view_index) * 7 * kColorPlane;
    checkCuda(cudaMemcpyAsync(rgb, destination, kColorElements * sizeof(float), cudaMemcpyHostToDevice, stream()),
              "cannot upload RGB input");
    const auto error = cudaEventRecord(upload.done, stream());
    // If recording fails, finish the copy before the staging buffer can be reused.
    if (error != cudaSuccess) {
        cudaStreamSynchronize(stream());
        checkCuda(error, "cannot record RGB upload completion");
    }
    upload.pending = true;
    // Same-stream ordering starts this kernel after H2D, without a CPU wait.
    // The event above protects only host staging; the kernel uses device RGB.
    checkCuda(ma_preprocessing::normalize_dinov2_rgb(rgb, kInputWidth, kInputHeight, stream()),
              "cannot launch DINOv2 RGB normalization");
}

void MA_VGGT::loadEngine(const std::filesystem::path& path) {
    impl_->stream.synchronize();
    auto next = std::make_unique<Model>(stream());
    auto& s = *next;
    auto* active_logger = getLogger();
    s.runtime.reset(nvinfer1::createInferRuntime(active_logger ? *active_logger : logger));
    require(bool(s.runtime), "cannot create TensorRT runtime");
    std::size_t engine_memory_bytes{};
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(bool(file), "cannot open engine: " + path.string());
        const auto size = file.tellg();
        require(size > 0, "empty engine: " + path.string());
        std::vector<char> bytes(static_cast<std::size_t>(size));
        file.seekg(0);
        require(bool(file.read(bytes.data(), std::streamsize(bytes.size()))), "cannot read engine: " + path.string());
        const auto free_before = freeDeviceMemory();
        s.engine.reset(s.runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        require(bool(s.engine), "cannot deserialize engine: " + path.string());
        const auto free_after = freeDeviceMemory();
        engine_memory_bytes = free_before >= free_after ? free_before - free_after : 0;
    }
    using Mode = nvinfer1::TensorIOMode;
    using Profile = nvinfer1::OptProfileSelector;
    require(s.engine->getNbIOTensors() == 4 && s.engine->getNbOptimizationProfiles() == 1,
            "expected the dynamic RAW engine with four I/O tensors and one profile");
    s.tensor("inputs", dims({-1, 7, kInputHeight, kInputWidth}), Mode::kINPUT);
    s.tensor("depths", dims({-1, 6, kInputHeight, kInputWidth}), Mode::kOUTPUT);
    s.tensor("poses", dims({-1, 7}), Mode::kOUTPUT);
    s.tensor("scale", dims({1, 1, 1}), Mode::kOUTPUT);
    require(equal(s.engine->getProfileShape("inputs", 0, Profile::kMIN), dims({kMinViews, 7, kInputHeight, kInputWidth})) &&
            equal(s.engine->getProfileShape("inputs", 0, Profile::kMAX), dims({kMaxViews, 7, kInputHeight, kInputWidth})),
            "input profile must support exactly 2..5 views at 434x518");

    // Future inference will provide activation workspace and enqueue on the
    // MA-owned stream. I/O storage already exists independently of the engine.
    s.context.reset(s.engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    require(bool(s.context), "cannot create TensorRT context");
    require(s.context->setOptimizationProfileAsync(0, stream()), "cannot select the profile on the MA stream");
    require(s.context->setTensorAddress("inputs", inputDevice()) &&
            s.context->setTensorAddress("depths", depthsDevice()) &&
            s.context->setTensorAddress("poses", posesDevice()) &&
            s.context->setTensorAddress("scale", scaleDevice()),
            "cannot bind preallocated GPU I/O buffers");
    for (int views = kMinViews; views <= kMaxViews; ++views) {
        require(s.context->setInputShape("inputs", dims({views, 7, kInputHeight, kInputWidth})), "cannot set view count");
        require(s.context->inferShapes(0, nullptr) == 0 &&
                equal(s.context->getTensorShape("depths"), dims({views, 6, kInputHeight, kInputWidth})) &&
                equal(s.context->getTensorShape("poses"), dims({views, 7})) &&
                equal(s.context->getTensorShape("scale"), dims({1, 1, 1})),
                "unexpected dynamic output shapes for " + std::to_string(views) + " views");
    }
    const auto context_memory_bytes = s.engine->getDeviceMemorySizeV2();
    require(context_memory_bytes >= 0, "cannot query TensorRT context device-memory requirement");
    impl_->stream.synchronize();
    impl_->model = std::move(next);

    constexpr double kBytesPerMiB = 1024.0 * 1024.0;
    std::cout << "[MA-VGGT] TensorRT engine loaded: " << path << '\n'
              << "[MA-VGGT] TensorRT engine deserialization device-memory delta: "
              << engine_memory_bytes << " bytes (" << double(engine_memory_bytes) / kBytesPerMiB << " MiB)\n"
              << "[MA-VGGT] Tensor I/O:\n";
    for (int i = 0; i < s.engine->getNbIOTensors(); ++i) {
        const auto* name = s.engine->getIOTensorName(i);
        // All four tensors have already been validated as FP32 above.
        std::cout << "  " << (s.engine->getTensorIOMode(name) == Mode::kINPUT ? "input" : "output")
                  << "  name=" << name
                  << "  shape=" << tensorShapeString(s.engine->getTensorShape(name))
                  << "  type=FP32\n";
    }
    std::cout << "[MA-VGGT] Optimization profile 0: inputs"
              << "  min=" << tensorShapeString(s.engine->getProfileShape("inputs", 0, Profile::kMIN))
              << "  opt=" << tensorShapeString(s.engine->getProfileShape("inputs", 0, Profile::kOPT))
              << "  max=" << tensorShapeString(s.engine->getProfileShape("inputs", 0, Profile::kMAX)) << '\n'
              << "[MA-VGGT] TensorRT context device-memory requirement: " << context_memory_bytes
              << " bytes (" << double(context_memory_bytes) / kBytesPerMiB
              << " MiB); user-managed activation workspace not allocated\n"
              << "[MA-VGGT] Preallocated GPU I/O (V=" << kMaxViews << "): " << kIOBytes
              << " bytes (" << double(kIOBytes) / kBytesPerMiB << " MiB)\n"
              << "[MA-VGGT] CUDA stream: independent, non-blocking\n" << std::flush;
}

} // namespace fs
