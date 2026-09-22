#include "fs/inference/plugins/GWCVolumePlugin.hpp"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>

namespace {

class Logger final : public nvinfer1::ILogger {
public:
    void log(Severity severity, char const* message) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

struct TrtDeleter {
    template <typename T>
    void operator()(T* object) const noexcept {
        delete object;
    }
};

template <typename T>
using TrtPtr = std::unique_ptr<T, TrtDeleter>;

void printUsage(char const* program) {
    std::cerr
        << "Usage: " << program
        << " <plugin_onnx> <output_engine> [--workspace-mb N]"
        << " [--optimization-level N] [--no-compilation-cache]\n\n"
        << "Build one TensorRT engine from an ONNX graph containing the\n"
        << "foundation_stereo::GWCVolume Plugin V3 node.\n\n"
        << "Strong typing preserves the tensor types encoded in ONNX; default workspace is 4096 MiB.\n";
}

size_t parsePositiveSize(char const* value, char const* option) {
    try {
        const auto parsed = std::stoull(value);
        if (parsed == 0) {
            throw std::invalid_argument("zero");
        }
        return static_cast<size_t>(parsed);
    } catch (std::exception const&) {
        throw std::runtime_error(std::string(option) + " must be a positive integer");
    }
}

int parseOptimizationLevel(char const* value) {
    try {
        const int level = std::stoi(value);
        if (level < 0 || level > 5) {
            throw std::out_of_range("outside TensorRT range");
        }
        return level;
    } catch (std::exception const&) {
        throw std::runtime_error("--optimization-level must be an integer from 0 to 5");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }

    try {
        const std::filesystem::path onnxPath = argv[1];
        const std::filesystem::path enginePath = argv[2];
        size_t workspaceMiB = 4096;
        int optimizationLevel = -1;
        bool disableCompilationCache = false;

        for (int index = 3; index < argc; ++index) {
            const std::string option = argv[index];
            if (option == "--workspace-mb" && index + 1 < argc) {
                workspaceMiB = parsePositiveSize(argv[++index], "--workspace-mb");
            } else if (option == "--optimization-level" && index + 1 < argc) {
                optimizationLevel = parseOptimizationLevel(argv[++index]);
            } else if (option == "--no-compilation-cache") {
                disableCompilationCache = true;
            } else {
                throw std::runtime_error("unknown or incomplete option: " + option);
            }
        }

        if (!std::filesystem::exists(onnxPath)) {
            throw std::runtime_error("ONNX file does not exist: " + onnxPath.string());
        }
        if (!enginePath.parent_path().empty()) {
            std::filesystem::create_directories(enginePath.parent_path());
        }

        // The ONNX parser resolves foundation_stereo::GWCVolume through this
        // creator. Register it before creating/parsing the network.
        if (!fs::trt::registerGwcVolumePlugin()) {
            throw std::runtime_error("failed to register GWCVolume Plugin V3 creator");
        }

        Logger logger;
        TrtPtr<nvinfer1::IBuilder> builder(nvinfer1::createInferBuilder(logger));
        if (!builder) {
            throw std::runtime_error("createInferBuilder failed");
        }

        // TensorRT 10 requires an explicit flag to preserve ONNX tensor types.
        // TensorRT 11 uses strong typing by default.
        uint32_t networkFlags = 0U;
#if NV_TENSORRT_MAJOR < 11
        networkFlags = 1U << static_cast<uint32_t>(
            nvinfer1::NetworkDefinitionCreationFlag::kSTRONGLY_TYPED);
#endif
        TrtPtr<nvinfer1::INetworkDefinition> network(builder->createNetworkV2(networkFlags));
        if (!network) {
            throw std::runtime_error("createNetworkV2 failed");
        }

        TrtPtr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
        if (!parser) {
            throw std::runtime_error("createParser failed");
        }
        if (!parser->parseFromFile(
                onnxPath.string().c_str(),
                static_cast<int32_t>(nvinfer1::ILogger::Severity::kWARNING))) {
            for (int32_t index = 0; index < parser->getNbErrors(); ++index) {
                std::cerr << "[ONNX parser] " << parser->getError(index)->desc() << '\n';
            }
            throw std::runtime_error("failed to parse ONNX: " + onnxPath.string());
        }

        TrtPtr<nvinfer1::IBuilderConfig> config(builder->createBuilderConfig());
        if (!config) {
            throw std::runtime_error("createBuilderConfig failed");
        }
        config->setMemoryPoolLimit(
            nvinfer1::MemoryPoolType::kWORKSPACE, workspaceMiB * 1024ULL * 1024ULL);
        if (optimizationLevel >= 0) {
            config->setBuilderOptimizationLevel(optimizationLevel);
        }
        if (disableCompilationCache) {
            config->setFlag(nvinfer1::BuilderFlag::kDISABLE_COMPILATION_CACHE);
        }

        TrtPtr<nvinfer1::IHostMemory> serialized(
            builder->buildSerializedNetwork(*network, *config));
        if (!serialized) {
            throw std::runtime_error("buildSerializedNetwork failed");
        }

        std::ofstream output(enginePath, std::ios::binary);
        if (!output) {
            throw std::runtime_error("cannot write engine: " + enginePath.string());
        }
        output.write(static_cast<char const*>(serialized->data()), serialized->size());
        if (!output) {
            throw std::runtime_error("failed while writing engine: " + enginePath.string());
        }

        std::cout << "Built engine: " << enginePath << '\n';
        std::cout << "Precision: determined by the ONNX tensor types (strong typing)\n";
        std::cout << "Plugin: foundation_stereo::GWCVolume (Plugin V3)\n";
    } catch (std::exception const& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }

    return 0;
}
