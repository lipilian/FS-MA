#include "FS.hpp"

#include "GWCVolumePlugin.hpp"

#include <opencv2/imgproc.hpp>

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

class TensorRtLogger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, char const* message) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

TensorRtLogger kTensorRtLogger;

char const* tensorIoModeName(nvinfer1::TensorIOMode mode) noexcept {
    switch (mode) {
    case nvinfer1::TensorIOMode::kNONE: return "none";
    case nvinfer1::TensorIOMode::kINPUT: return "input";
    case nvinfer1::TensorIOMode::kOUTPUT: return "output";
    }
    return "unknown";
}

char const* tensorDataTypeName(nvinfer1::DataType type) noexcept {
    switch (type) {
    case nvinfer1::DataType::kFLOAT: return "FP32";
    case nvinfer1::DataType::kHALF: return "FP16";
    case nvinfer1::DataType::kINT8: return "INT8";
    case nvinfer1::DataType::kINT32: return "INT32";
    case nvinfer1::DataType::kBOOL: return "BOOL";
    default: return "other";
    }
}

std::string tensorShapeString(nvinfer1::Dims const& dimensions) {
    std::ostringstream output;
    output << "[";
    for (int index = 0; index < dimensions.nbDims; ++index) {
        if (index != 0) output << ", ";
        output << dimensions.d[index];
    }
    return output.str() + "]";
}

}  // namespace

FS::FS() = default;
FS::~FS() = default;

void FS::TensorRtRuntimeDeleter::operator()(nvinfer1::IRuntime* object) const noexcept {
    delete object;
}

void FS::TensorRtEngineDeleter::operator()(nvinfer1::ICudaEngine* object) const noexcept {
    delete object;
}

void FS::TensorRtContextDeleter::operator()(nvinfer1::IExecutionContext* object) const noexcept {
    delete object;
}

void FS::loadEngine(const std::filesystem::path& engine_path) {
    if (!std::filesystem::is_regular_file(engine_path)) {
        throw std::runtime_error("TensorRT engine does not exist: " + engine_path.string());
    }
    if (!fs::trt::registerGwcVolumePlugin()) {
        throw std::runtime_error("failed to register the GWCVolume TensorRT plugin");
    }

    std::ifstream engine_file(engine_path, std::ios::binary | std::ios::ate);
    if (!engine_file) {
        throw std::runtime_error("unable to open TensorRT engine: " + engine_path.string());
    }
    const std::streamsize byte_count = engine_file.tellg();
    if (byte_count <= 0) {
        throw std::runtime_error("TensorRT engine is empty: " + engine_path.string());
    }

    std::vector<char> serialized_engine(static_cast<std::size_t>(byte_count));
    engine_file.seekg(0, std::ios::beg);
    if (!engine_file.read(serialized_engine.data(), byte_count)) {
        throw std::runtime_error("unable to read TensorRT engine: " + engine_path.string());
    }

    std::unique_ptr<nvinfer1::IRuntime, TensorRtRuntimeDeleter> runtime(
        nvinfer1::createInferRuntime(kTensorRtLogger));
    if (!runtime) {
        throw std::runtime_error("failed to create TensorRT runtime");
    }

    std::unique_ptr<nvinfer1::ICudaEngine, TensorRtEngineDeleter> engine(
        runtime->deserializeCudaEngine(serialized_engine.data(), serialized_engine.size()));
    if (!engine) {
        throw std::runtime_error("failed to deserialize TensorRT engine: " + engine_path.string());
    }

    std::cout << "TensorRT engine loaded: " << engine_path << std::endl;
    std::cout << "Tensor I/O:" << std::endl;
    for (int index = 0; index < engine->getNbIOTensors(); ++index) {
        char const* tensor_name = engine->getIOTensorName(index);
        if (!tensor_name) continue;

        const auto mode = engine->getTensorIOMode(tensor_name);
        const auto shape = engine->getTensorShape(tensor_name);
        const auto data_type = engine->getTensorDataType(tensor_name);
        std::cout << "  " << tensorIoModeName(mode)
                  << "  name=" << tensor_name
                  << "  shape=" << tensorShapeString(shape)
                  << "  type=" << tensorDataTypeName(data_type)
                  << std::endl;
    }

    std::unique_ptr<nvinfer1::IExecutionContext, TensorRtContextDeleter> execution_context(
        engine->createExecutionContext());
    if (!execution_context) {
        throw std::runtime_error("failed to create TensorRT execution context");
    }

    execution_context_ = std::move(execution_context);
    engine_ = std::move(engine);
    runtime_ = std::move(runtime);
}

void FS::set_model_camera_parameters(const StereoCameraParameters& rectified_camera_parameters,
                                     const cv::Size& rectified_image_size) {
    const double scale_x = static_cast<double>(rectified_image_size.width) / kTensorRtInputWidth;
    const double scale_y = static_cast<double>(rectified_image_size.height) / kTensorRtInputHeight;
    model_camera_parameters_ = rectified_camera_parameters;
    model_camera_parameters_.fx /= scale_x;
    model_camera_parameters_.cx /= scale_x;
    model_camera_parameters_.fy /= scale_y;
    model_camera_parameters_.cy /= scale_y;
}

void FS::prepare_stereo_images(const cv::Mat& left, const cv::Mat& right) {
    if (left.empty() || right.empty()) {
        throw std::invalid_argument("FS input preparation requires non-empty left and right images");
    }
    if (left.size() != right.size()) {
        throw std::invalid_argument("FS input preparation requires matching left and right dimensions");
    }
    const cv::Size model_size(kTensorRtInputWidth, kTensorRtInputHeight);
    cv::resize(left, model_left_, model_size, 0.0, 0.0, cv::INTER_LINEAR);
    cv::resize(right, model_right_, model_size, 0.0, 0.0, cv::INTER_LINEAR);
}
