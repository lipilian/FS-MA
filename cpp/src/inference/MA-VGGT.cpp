#include "fs/inference/MA-VGGT.hpp"
#include "fs/inference/FS.hpp"

#include <NvInfer.h>
#include <algorithm>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs {
namespace {
class Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char* message) noexcept override {
        if (severity <= Severity::kWARNING) std::cerr << "[MA TensorRT] " << message << '\n';
    }
} logger;

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error("MapAnything: " + message);
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
} // namespace

struct MA_VGGT::Impl {
    // Reverse destruction releases TensorRT objects before the stream owner.
    std::shared_ptr<FS> fs;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;

    explicit Impl(std::shared_ptr<FS> owner) : fs(std::move(owner)) {
        require(fs && fs->stream(), "an FS-owned CUDA stream is required");
    }
    ~Impl() { cudaStreamSynchronize(fs->stream()); }

    void tensor(const char* name, nvinfer1::Dims shape, nvinfer1::TensorIOMode mode) const {
        require(engine->getTensorIOMode(name) == mode &&
                engine->getTensorDataType(name) == nvinfer1::DataType::kFLOAT &&
                engine->getTensorLocation(name) == nvinfer1::TensorLocation::kDEVICE &&
                engine->getTensorFormat(name) == nvinfer1::TensorFormat::kLINEAR &&
                equal(engine->getTensorShape(name), shape), std::string("incompatible tensor: ") + name);
    }
};

MA_VGGT::MA_VGGT(std::shared_ptr<FS> fs) : impl_(std::make_unique<Impl>(std::move(fs))) {}
MA_VGGT::~MA_VGGT() = default;
bool MA_VGGT::isLoaded() const noexcept { return impl_->engine && impl_->context; }
cudaStream_t MA_VGGT::stream() const noexcept { return impl_->fs->stream(); }

void MA_VGGT::loadEngine(const std::filesystem::path& path) {
    impl_->fs->synchronize();
    auto next = std::make_unique<Impl>(impl_->fs);
    auto& s = *next;
    auto* active_logger = getLogger();
    s.runtime.reset(nvinfer1::createInferRuntime(active_logger ? *active_logger : logger));
    require(bool(s.runtime), "cannot create TensorRT runtime");
    {
        std::ifstream file(path, std::ios::binary | std::ios::ate);
        require(bool(file), "cannot open engine: " + path.string());
        const auto size = file.tellg();
        require(size > 0, "empty engine: " + path.string());
        std::vector<char> bytes(static_cast<std::size_t>(size));
        file.seekg(0);
        require(bool(file.read(bytes.data(), std::streamsize(bytes.size()))), "cannot read engine: " + path.string());
        s.engine.reset(s.runtime->deserializeCudaEngine(bytes.data(), bytes.size()));
        require(bool(s.engine), "cannot deserialize engine: " + path.string());
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

    // We only load the model at this stage. Future inference will provide the
    // activation workspace and I/O addresses, then enqueue on this same stream.
    s.context.reset(s.engine->createExecutionContext(nvinfer1::ExecutionContextAllocationStrategy::kUSER_MANAGED));
    require(bool(s.context), "cannot create TensorRT context");
    require(s.context->setOptimizationProfileAsync(0, stream()), "cannot select the profile on the FS stream");
    for (int views = kMinViews; views <= kMaxViews; ++views) {
        require(s.context->setInputShape("inputs", dims({views, 7, kInputHeight, kInputWidth})), "cannot set view count");
        require(s.context->inferShapes(0, nullptr) == 0 &&
                equal(s.context->getTensorShape("depths"), dims({views, 6, kInputHeight, kInputWidth})) &&
                equal(s.context->getTensorShape("poses"), dims({views, 7})) &&
                equal(s.context->getTensorShape("scale"), dims({1, 1, 1})),
                "unexpected dynamic output shapes for " + std::to_string(views) + " views");
    }
    s.fs->synchronize();
    impl_ = std::move(next);
}

} // namespace fs
