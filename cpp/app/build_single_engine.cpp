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

#if NV_TENSORRT_MAJOR == 10
int rewriteCorrelationAveragePools(nvinfer1::INetworkDefinition& network) {
    // TRT 10.14 can corrupt coarse-volume samples when AveragePool stays inside
    // the fused recurrent subgraph. The equivalent pair average avoids that path
    // without extra engine outputs or a change to the model's precision.
    static constexpr float half = 0.5F; // Weights must live through engine construction.
    const int32_t originalLayerCount = network.getNbLayers();
    int rewritten = 0;
    const auto matches2D = [](nvinfer1::Dims dims, int64_t height, int64_t width) {
        return dims.nbDims == 2 && dims.d[0] == height && dims.d[1] == width;
    };
    for (int32_t index = 0; index < originalLayerCount; ++index) {
        auto* layer = network.getLayer(index);
        if (layer->getType() != nvinfer1::LayerType::kPOOLING) continue;
        auto* pool = static_cast<nvinfer1::IPoolingLayer*>(layer);
        auto* input = pool->getInput(0);
        const auto shape = input->getDimensions();
        if (pool->getPoolingType() != nvinfer1::PoolingType::kAVERAGE ||
            !matches2D(pool->getWindowSizeNd(), 1, 2) ||
            !matches2D(pool->getStrideNd(), 1, 2) ||
            !matches2D(pool->getPrePadding(), 0, 0) ||
            !matches2D(pool->getPostPadding(), 0, 0) ||
            pool->getPaddingMode() != nvinfer1::PaddingMode::kEXPLICIT_ROUND_DOWN ||
            shape.nbDims != 4 || shape.d[0] <= 0 || shape.d[1] <= 0 ||
            shape.d[2] != 1 || shape.d[3] <= 0 || shape.d[3] % 2 != 0 ||
            input->getType() != nvinfer1::DataType::kFLOAT) {
            continue;
        }
        auto pooledShape = shape;
        pooledShape.d[3] /= 2;
        auto* even = network.addSlice(*input, nvinfer1::Dims4{0, 0, 0, 0},
                                     pooledShape, nvinfer1::Dims4{1, 1, 1, 2});
        auto* odd = network.addSlice(*input, nvinfer1::Dims4{0, 0, 0, 1},
                                    pooledShape, nvinfer1::Dims4{1, 1, 1, 2});
        auto* scale = network.addConstant(nvinfer1::Dims4{1, 1, 1, 1},
            nvinfer1::Weights{nvinfer1::DataType::kFLOAT, &half, 1});
        if (!even || !odd || !scale) throw std::runtime_error("Failed to rewrite correlation pooling");
        auto* sum = network.addElementWise(*even->getOutput(0), *odd->getOutput(0),
                                          nvinfer1::ElementWiseOperation::kSUM);
        if (!sum) throw std::runtime_error("Failed to create correlation pair sum");
        auto* average = network.addElementWise(*sum->getOutput(0), *scale->getOutput(0),
                                              nvinfer1::ElementWiseOperation::kPROD);
        if (!average) throw std::runtime_error("Failed to create correlation pair average");
        const std::string name = pool->getName();
        even->setName((name + "/pair_even").c_str());
        odd->setName((name + "/pair_odd").c_str());
        sum->setName((name + "/pair_sum").c_str());
        average->setName((name + "/pair_average").c_str());
        auto* oldOutput = pool->getOutput(0);
        auto* newOutput = average->getOutput(0);
        const std::string outputName = oldOutput->getName();
        oldOutput->setName((outputName + "/replaced").c_str());
        newOutput->setName(outputName.c_str());
        for (int32_t consumerIndex = 0; consumerIndex < originalLayerCount; ++consumerIndex) {
            auto* consumer = network.getLayer(consumerIndex);
            for (int32_t slot = 0; slot < consumer->getNbInputs(); ++slot) {
                if (consumer->getInput(slot) == oldOutput) consumer->setInput(slot, *newOutput);
            }
        }
        if (oldOutput->isNetworkOutput()) {
            network.unmarkOutput(*oldOutput);
            network.markOutput(*newOutput);
        }
        ++rewritten;
    }
    return rewritten;
}
#endif

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

#if NV_TENSORRT_MAJOR == 10
        const int poolingRewrites = rewriteCorrelationAveragePools(*network);
        std::cout << "TensorRT 10 correlation pooling compatibility rewrites: "
                  << poolingRewrites << std::endl;
#endif

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
