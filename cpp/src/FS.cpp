#include "FS.hpp"

#include "GWCVolumePlugin.hpp"

#include <cuda_runtime_api.h>
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

constexpr char kLeftInputName[] = "left";
constexpr char kRightInputName[] = "right";
constexpr char kDisparityOutputName[] = "disp";
constexpr int kTensorRtInputChannels = 3;
constexpr int kTensorRtOutputChannels = 1;

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

FS::FS() {
    check_cuda(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking),
               "failed to create CUDA stream");
}
FS::~FS() {
    // The stream waits for outstanding work before its buffers are released.
    if (stream_ != nullptr) {
        cudaStreamDestroy(stream_);
    }
}

void FS::TensorRtRuntimeDeleter::operator()(nvinfer1::IRuntime* object) const noexcept {
    delete object;
}

void FS::TensorRtEngineDeleter::operator()(nvinfer1::ICudaEngine* object) const noexcept {
    delete object;
}

void FS::TensorRtContextDeleter::operator()(nvinfer1::IExecutionContext* object) const noexcept {
    delete object;
}

void FS::CudaDeviceBufferDeleter::operator()(float* pointer) const noexcept {
    if (pointer != nullptr) {
        cudaFree(pointer);
    }
}

void FS::CudaHostBufferDeleter::operator()(float* pointer) const noexcept {
    if (pointer != nullptr) {
        cudaFreeHost(pointer);
    }
}

void FS::check_cuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
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

    std::size_t free_memory_before_engine{};
    std::size_t total_device_memory{};
    check_cuda(cudaMemGetInfo(&free_memory_before_engine, &total_device_memory),
               "failed to query free device memory before TensorRT engine deserialization");

    std::unique_ptr<nvinfer1::ICudaEngine, TensorRtEngineDeleter> engine(
        runtime->deserializeCudaEngine(serialized_engine.data(), serialized_engine.size()));
    if (!engine) {
        throw std::runtime_error("failed to deserialize TensorRT engine: " + engine_path.string());
    }

    std::size_t free_memory_after_engine{};
    check_cuda(cudaMemGetInfo(&free_memory_after_engine, &total_device_memory),
               "failed to query free device memory after TensorRT engine deserialization");
    const std::size_t engine_deserialization_memory =
        free_memory_before_engine >= free_memory_after_engine
            ? free_memory_before_engine - free_memory_after_engine : 0;
    std::cout << "TensorRT engine deserialization device-memory delta: "
              << engine_deserialization_memory << " bytes ("
              << static_cast<double>(engine_deserialization_memory) / (1024.0 * 1024.0) << " MiB)"
              << std::endl;


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

    const auto context_memory_bytes = engine->getDeviceMemorySizeV2();
    if (context_memory_bytes < 0) {
        throw std::runtime_error("failed to query TensorRT context device-memory requirement");
    }
    constexpr double kBytesPerMiB = 1024.0 * 1024.0;
    std::cout << "TensorRT context internal device memory: " << context_memory_bytes
              << " bytes (" << static_cast<double>(context_memory_bytes) / kBytesPerMiB
              << " MiB)"
              << std::endl;

    execution_context_ = std::move(execution_context);
    engine_ = std::move(engine);
    runtime_ = std::move(runtime);
    allocate_input_buffers();
}

void FS::allocate_input_buffers() {
    const nvinfer1::Dims input_shape{4, {1, kTensorRtInputChannels,
                                         kTensorRtInputHeight, kTensorRtInputWidth}};
    for (const char* input_name : {kLeftInputName, kRightInputName}) {
        if (engine_->getTensorIOMode(input_name) != nvinfer1::TensorIOMode::kINPUT ||
            engine_->getTensorDataType(input_name) != nvinfer1::DataType::kFLOAT) {
            throw std::runtime_error(std::string("expected FP32 input tensor named '") + input_name + "'");
        }
        if (!execution_context_->setInputShape(input_name, input_shape)) {
            throw std::runtime_error(std::string("failed to set TensorRT input shape for '") + input_name + "'");
        }
    }

    const nvinfer1::Dims output_shape = execution_context_->getTensorShape(kDisparityOutputName);
    if (engine_->getTensorIOMode(kDisparityOutputName) != nvinfer1::TensorIOMode::kOUTPUT ||
        engine_->getTensorDataType(kDisparityOutputName) != nvinfer1::DataType::kFLOAT ||
        output_shape.nbDims != 4 || output_shape.d[0] != 1 ||
        output_shape.d[1] != kTensorRtOutputChannels ||
        output_shape.d[2] != kTensorRtInputHeight || output_shape.d[3] != kTensorRtInputWidth) {
        throw std::runtime_error("expected FP32 output tensor 'disp' with shape [1, 1, 800, 960]");
    }

    const std::size_t input_element_count = static_cast<std::size_t>(kTensorRtInputChannels) *
                                            kTensorRtInputHeight * kTensorRtInputWidth;
    const std::size_t input_byte_count = input_element_count * sizeof(float);
    const std::size_t output_element_count = static_cast<std::size_t>(kTensorRtOutputChannels) *
                                             kTensorRtInputHeight * kTensorRtInputWidth;
    const std::size_t output_byte_count = output_element_count * sizeof(float);

    void* left_device = nullptr;
    check_cuda(cudaMalloc(&left_device, input_byte_count), "failed to allocate left TensorRT input buffer");
    left_input_device_.reset(static_cast<float*>(left_device));

    void* right_device = nullptr;
    check_cuda(cudaMalloc(&right_device, input_byte_count), "failed to allocate right TensorRT input buffer");
    right_input_device_.reset(static_cast<float*>(right_device));

    void* disparity_device = nullptr;
    check_cuda(cudaMalloc(&disparity_device, output_byte_count),
               "failed to allocate TensorRT disparity output buffer");
    disparity_output_device_.reset(static_cast<float*>(disparity_device));

    void* left_host = nullptr;
    check_cuda(cudaMallocHost(&left_host, input_byte_count), "failed to allocate pinned left input buffer");
    left_input_host_.reset(static_cast<float*>(left_host));

    void* right_host = nullptr;
    check_cuda(cudaMallocHost(&right_host, input_byte_count), "failed to allocate pinned right input buffer");
    right_input_host_.reset(static_cast<float*>(right_host));

    if (!execution_context_->setTensorAddress(kLeftInputName, left_input_device_.get()) ||
        !execution_context_->setTensorAddress(kRightInputName, right_input_device_.get()) ||
        !execution_context_->setTensorAddress(kDisparityOutputName, disparity_output_device_.get())) {
        throw std::runtime_error("failed to bind preallocated TensorRT I/O buffers");
    }
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
    if (!isEngineLoaded() || stream_ == nullptr || !left_input_device_ || !right_input_device_) {
        throw std::logic_error("loadEngine must complete before preparing TensorRT inputs");
    }
    if (left.empty() || right.empty()) {
        throw std::invalid_argument("FS input preparation requires non-empty left and right images");
    }
    if (left.size() != right.size()) {
        throw std::invalid_argument("FS input preparation requires matching left and right dimensions");
    }
    const cv::Size model_size(kTensorRtInputWidth, kTensorRtInputHeight);
    cv::resize(left, model_left_, model_size, 0.0, 0.0, cv::INTER_LINEAR);
    cv::resize(right, model_right_, model_size, 0.0, 0.0, cv::INTER_LINEAR);

    if (model_left_.type() != CV_8UC3 || model_right_.type() != CV_8UC3) {
        throw std::invalid_argument("FS input preparation requires 8-bit, three-channel RGB images");
    }

    // The staging buffers are reused, so wait before overwriting data needed by
    // a previous upload or inference enqueued on this same stream.
    check_cuda(cudaStreamSynchronize(stream_), "failed to synchronize previous TensorRT input work");

    const std::size_t plane_size = static_cast<std::size_t>(kTensorRtInputHeight) * kTensorRtInputWidth;
    const auto pack_nchw = [plane_size](const cv::Mat& rgb, float* destination) {
        for (int row = 0; row < rgb.rows; ++row) {
            const cv::Vec3b* source = rgb.ptr<cv::Vec3b>(row);
            const std::size_t offset = static_cast<std::size_t>(row) * rgb.cols;
            for (int column = 0; column < rgb.cols; ++column) {
                const cv::Vec3b& pixel = source[column];
                destination[offset + column] = static_cast<float>(pixel[0]);
                destination[plane_size + offset + column] = static_cast<float>(pixel[1]);
                destination[2 * plane_size + offset + column] = static_cast<float>(pixel[2]);
            }
        }
    };

    pack_nchw(model_left_, left_input_host_.get());
    pack_nchw(model_right_, right_input_host_.get());

    const std::size_t byte_count = kTensorRtInputChannels * plane_size * sizeof(float);
    check_cuda(cudaMemcpyAsync(left_input_device_.get(), left_input_host_.get(), byte_count,
                               cudaMemcpyHostToDevice, stream_),
               "failed to upload left TensorRT input");
    check_cuda(cudaMemcpyAsync(right_input_device_.get(), right_input_host_.get(), byte_count,
                               cudaMemcpyHostToDevice, stream_),
               "failed to upload right TensorRT input");
}

void FS::inference() {
    if (!isEngineLoaded() || stream_ == nullptr || !disparity_output_device_) {
        throw std::logic_error("loadEngine must complete before TensorRT inference");
    }
    if (!execution_context_->enqueueV3(stream_)) {
        throw std::runtime_error("failed to enqueue TensorRT inference");
    }
}

float FS::inference_time_measure() {
    if (!isEngineLoaded() || stream_ == nullptr || !disparity_output_device_) {
        throw std::logic_error("loadEngine must complete before TensorRT inference");
    }

    constexpr int kInferenceIterations = 10;
    float total_milliseconds{0.0F};
    cudaEvent_t start{nullptr};
    cudaEvent_t stop{nullptr};
    try {
        check_cuda(cudaEventCreate(&start), "failed to create TensorRT inference start event");
        check_cuda(cudaEventCreate(&stop), "failed to create TensorRT inference stop event");

        for (int iteration = 0; iteration < kInferenceIterations; ++iteration) {
            check_cuda(cudaEventRecord(start, stream_), "failed to record TensorRT inference start event");

            if (!execution_context_->enqueueV3(stream_)) {
                throw std::runtime_error("failed to enqueue TensorRT inference");
            }

            check_cuda(cudaEventRecord(stop, stream_), "failed to record TensorRT inference stop event");
            check_cuda(cudaEventSynchronize(stop), "failed to synchronize TensorRT inference stop event");

            float elapsed_milliseconds{0.0F};
            check_cuda(cudaEventElapsedTime(&elapsed_milliseconds, start, stop),
                       "failed to measure TensorRT inference time");
            std::cout << elapsed_milliseconds
                      << (iteration + 1 == kInferenceIterations ? '\n' : ' ');
            total_milliseconds += elapsed_milliseconds;
        }

        check_cuda(cudaEventDestroy(stop), "failed to destroy TensorRT inference stop event");
        stop = nullptr;
        check_cuda(cudaEventDestroy(start), "failed to destroy TensorRT inference start event");
        start = nullptr;
        return total_milliseconds / kInferenceIterations;
    } catch (...) {
        if (stop != nullptr) cudaEventDestroy(stop);
        if (start != nullptr) cudaEventDestroy(start);
        throw;
    }
}
