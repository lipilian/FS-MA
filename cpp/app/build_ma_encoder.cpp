#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <charconv>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <initializer_list>
#include <iostream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#if NV_TENSORRT_MAJOR < 11
#error "ma_build_encoder requires TensorRT 11 or newer"
#endif

namespace {
namespace fs = std::filesystem;
using namespace nvinfer1;

constexpr int kAttentionCount = 24;
constexpr float kAttentionCount = 0.125F; // 1 / sqrt(1536 / 24)

class Logger final : public ILogger {
public:
    void log(Severity severity, char const* message) noexcept override {
        if (severity <= Severity::kINFO) {
            std::cerr << "[TensorRT] " << message << '\n';
        }
    }
};

void require(bool condition, std::string const& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

void cudaCheck(cudaError_t result) {
    require(result == cudaSuccess, std::string("CUDA: ") + cudaGetErrorString(result));
}

bool shapeIs(Dims const& dims, std::initializer_list<int64_t> expected) {
    if (dims.nbDims != static_cast<int>(expected.size())) {
        return false;
    }
    int i = 0;
    for (auto value : expected) {
        if (dims.d[i++] != value) {
            return false;
        }
    }
    return true;
}

bool sameShape(Dims const& left, Dims const& right) {
    if (left.nbDims != right.nbDims || left.nbDims < 0) {
        return false;
    }
    for (int i = 0; i < left.nbDims; ++i) {
        if (left.d[i] != right.d[i]) {
            return false;
        }
    }
    return true;
}

void tensorIs(ITensor* tensor, DataType type, std::initializer_list<int64_t> shape,
              std::string const& role) {
    require(tensor && tensor->getType() == type && shapeIs(tensor->getDimensions(), shape),
            "Unexpected tensor type/shape for " + role);
}

bool identityPermutation(Permutation const& permutation, int rank) {
    for (int i = 0; i < rank; ++i) {
        if (permutation.order[i] != i) {
            return false;
        }
    }
    return true;
}

void validateNetwork(INetworkDefinition& network) {
    require(network.getNbInputs() == 1 && network.getNbOutputs() == 2,
            "Expected the extracted encoder with one input and two outputs");
    auto* input = network.getInput(0);
    require(std::string(input->getName()) == "inputs", "Expected input named inputs");
    tensorIs(input, DataType::kFLOAT, {1, 7, 434, 518}, "inputs (fixed single view)");
    bool fused = false;
    bool cls = false;
    for (int i = 0; i < network.getNbOutputs(); ++i) {
        auto* output = network.getOutput(i);
        const std::string name = output->getName();
        if (name == "F_fused") {
            tensorIs(output, DataType::kFLOAT, {1, 1536, 31, 37}, name);
            fused = true;
        } else if (name == "CLS") {
            tensorIs(output, DataType::kFLOAT, {1, 1536, 1}, name);
            cls = true;
        } else {
            throw std::runtime_error("Unexpected encoder output: " + name);
        }
    }
    require(fused && cls, "Expected F_fused and CLS outputs");
}

// Inspect the parsed network, without a separate ONNX/protobuf dependency.
// Validate the complete SDPA pattern before replacing any of its consumers.
class Graph {
public:
    explicit Graph(INetworkDefinition& network) {
        for (int i = 0; i < network.getNbLayers(); ++i) {
            auto* layer = network.getLayer(i);
            layers.push_back(layer);
            for (int j = 0; j < layer->getNbOutputs(); ++j) {
                if (auto* tensor = layer->getOutput(j)) {
                    require(producers.emplace(tensor, layer).second, "Duplicate tensor producer");
                }
            }
        }
    }

    ILayer* producer(ITensor* tensor, LayerType type, std::string const& role) const {
        auto it = producers.find(tensor);
        require(it != producers.end() && it->second->getType() == type,
                "Unsupported attention graph at " + role);
        return it->second;
    }

    // The ONNX parser inserts an identity reshape after Softmax.
    ITensor* skipIdentityShuffles(ITensor* tensor) const {
        for (;;) {
            auto it = producers.find(tensor);
            if (it == producers.end() || it->second->getType() != LayerType::kSHUFFLE) {
                return tensor;
            }
            auto* shuffle = static_cast<IShuffleLayer*>(it->second);
            const auto dims = tensor->getDimensions();
            if (shuffle->getNbInputs() != 1 ||
                !sameShape(shuffle->getInput(0)->getDimensions(), dims) ||
                !identityPermutation(shuffle->getFirstTranspose(), dims.nbDims) ||
                !identityPermutation(shuffle->getSecondTranspose(), dims.nbDims)) {
                return tensor;
            }
            tensor = shuffle->getInput(0);
        }
    }

    void validateScale(ITensor* tensor) const {
        // A scalar Constant is broadcast to [1,1,1,1] by a parser Shuffle.
        for (;;) {
            const auto dims = tensor->getDimensions();
            require(dims.nbDims >= 0 && tensor->getType() == DataType::kFLOAT,
                    "Expected FP32 scalar attention scale");
            for (int i = 0; i < dims.nbDims; ++i) {
                require(dims.d[i] == 1, "Attention scale must contain exactly one value");
            }
            auto it = producers.find(tensor);
            require(it != producers.end(), "Attention scale must be constant");
            if (it->second->getType() != LayerType::kSHUFFLE) {
                break;
            }
            auto* shuffle = static_cast<IShuffleLayer*>(it->second);
            require(shuffle->getNbInputs() == 1, "Dynamic attention scale is unsupported");
            tensor = shuffle->getInput(0);
        }
        auto* constant = static_cast<IConstantLayer*>(producer(tensor, LayerType::kCONSTANT, "scale"));
        const auto weights = constant->getWeights();
        require(weights.type == DataType::kFLOAT && weights.count == 1 && weights.values,
                "Expected a scalar FP32 constant for attention scale");
        require(*static_cast<float const*>(weights.values) == kAttentionCount,
                "Attention scale is not 0.125; refusing to change model semantics");
    }

    std::vector<ILayer*> layers;
private:
    std::unordered_map<ITensor*, ILayer*> producers;
};

struct Attention {
    ITensor* q;
    ITensor* k;
    ITensor* v;
    ITensor* output;
    std::vector<std::pair<ILayer*, int>> consumers;
};

IMatrixMultiplyLayer* matmul(Graph const& graph, ITensor* output, std::string const& role) {
    auto* layer = static_cast<IMatrixMultiplyLayer*>(
        graph.producer(output, LayerType::kMATRIX_MULTIPLY, role));
    require(layer->getOperation(0) == MatrixOperation::kNONE &&
            layer->getOperation(1) == MatrixOperation::kNONE,
            "Unexpected matrix transpose at " + role);
    return layer;
}

ITensor* bf16BeforeCast(Graph const& graph, ITensor* tensor, std::string const& role) {
    tensorIs(tensor, DataType::kFLOAT, {1, 24, 1148, 64}, role + " FP32");
    auto* cast = graph.producer(tensor, LayerType::kCAST, role + " Cast");
    auto* input = cast->getInput(0);
    tensorIs(input, DataType::kBF16, {1, 24, 1148, 64}, role + " BF16");
    return input;
}

std::vector<Attention> findAttention(Graph const& graph) {
    std::vector<Attention> result;
    for (auto* layer : graph.layers) {
        const std::string name = layer->getName();
        if (name.rfind("node_scaled_dot_product_attention", 0) != 0) {
            continue;
        }
        require(layer->getType() == LayerType::kCAST, "Expected SDPA output Cast: " + name);
        auto* output = layer->getOutput(0);
        tensorIs(output, DataType::kBF16, {1, 24, 1148, 64}, name);
        tensorIs(layer->getInput(0), DataType::kFLOAT, {1, 24, 1148, 64}, name + " AV");
        auto* av = matmul(graph, layer->getInput(0), name + " AV");
        auto* probabilities = graph.skipIdentityShuffles(av->getInput(0));
        auto* softmax = static_cast<ISoftMaxLayer*>(
            graph.producer(probabilities, LayerType::kSOFTMAX, name + " Softmax"));
        tensorIs(probabilities, DataType::kFLOAT, {1, 24, 1148, 1148}, name + " probabilities");
        require(softmax->getAxes() == (1U << 3), "Expected Softmax over the last axis: " + name);
        auto* mul = static_cast<IElementWiseLayer*>(
            graph.producer(softmax->getInput(0), LayerType::kELEMENTWISE, name + " scale Mul"));
        require(mul->getOperation() == ElementWiseOperation::kPROD,
                "Expected multiplication by attention scale: " + name);
        graph.validateScale(mul->getInput(1));
        auto* qk = matmul(graph, mul->getInput(0), name + " QK");
        tensorIs(qk->getOutput(0), DataType::kFLOAT, {1, 24, 1148, 1148}, name + " scores");
        auto* transpose = static_cast<IShuffleLayer*>(
            graph.producer(qk->getInput(1), LayerType::kSHUFFLE, name + " K transpose"));
        const auto first = transpose->getFirstTranspose();
        require(transpose->getNbInputs() == 1 && first.order[0] == 0 && first.order[1] == 1 &&
                first.order[2] == 3 && first.order[3] == 2 &&
                identityPermutation(transpose->getSecondTranspose(), 4),
                "Expected K transpose [0,1,3,2]: " + name);
        tensorIs(transpose->getOutput(0), DataType::kFLOAT, {1, 24, 64, 1148}, name + " transposed K");
        Attention attention{bf16BeforeCast(graph, qk->getInput(0), name + " Q"),
                            bf16BeforeCast(graph, transpose->getInput(0), name + " K"),
                            bf16BeforeCast(graph, av->getInput(1), name + " V"), output, {}};
        require(!output->isNetworkOutput(), "SDPA intermediate must not be a model output");
        for (auto* consumer : graph.layers) {
            for (int i = 0; i < consumer->getNbInputs(); ++i) {
                if (consumer->getInput(i) == output) {
                    attention.consumers.emplace_back(consumer, i);
                }
            }
        }
        require(!attention.consumers.empty(), "SDPA output has no consumers: " + name);
        result.push_back(std::move(attention));
    }
    require(result.size() == kAttentionCount,
            "Expected exactly 24 encoder attention blocks, found " + std::to_string(result.size()));
    return result;
}

void replaceAttention(INetworkDefinition& network, std::vector<Attention> const& blocks,
                      float const& scaleValue) {
    Dims broadcast{};
    broadcast.nbDims = 4;
    for (int i = 0; i < 4; ++i) {
        broadcast.d[i] = 1;
    }
    // scaleValue must remain alive until buildSerializedNetwork finishes.
    auto* constant = network.addConstant(broadcast, Weights{DataType::kFLOAT, &scaleValue, 1});
    require(constant != nullptr, "Could not create attention scale");
    constant->setName("fmha_query_scale_fp32");
    auto* cast = network.addCast(*constant->getOutput(0), DataType::kBF16);
    require(cast != nullptr, "Could not cast attention scale to BF16");
    cast->setName("fmha_query_scale_bf16");
    for (size_t i = 0; i < blocks.size(); ++i) {
        auto const& block = blocks[i];
        const std::string prefix = "dino_" + std::string(i < 10 ? "0" : "") + std::to_string(i);
        auto* scaled = network.addElementWise(*block.q, *cast->getOutput(0), ElementWiseOperation::kPROD);
        require(scaled != nullptr, "Could not scale Q");
        scaled->setName((prefix + "_query_scale").c_str());
        auto* attention = network.addAttentionV2(*scaled->getOutput(0), *block.k, *block.v,
                                                AttentionNormalizationOp::kSOFTMAX, CausalMaskKind::kNONE);
        require(attention != nullptr, "Could not create native attention");
        require(attention->setName((prefix + "_native_attention").c_str()) &&
                attention->setDecomposable(false), "Could not configure native attention");
        auto* replacement = attention->getOutput(0);
        tensorIs(replacement, DataType::kBF16, {1, 24, 1148, 64}, prefix + " replacement");
        replacement->setName((std::string(block.output->getName()) + "_fmha").c_str());
        for (auto const& consumer : block.consumers) {
            consumer.first->setInput(consumer.second, *replacement);
        }
    }
    // The builder removes the disconnected FP32 MatMul/Softmax/MatMul paths.
    std::cout << "Replaced " << blocks.size() << " attention blocks (BF16, scale=0.125, non-decomposable)\n";
}

uint64_t parseNumber(char const* text, char const* option, uint64_t minimum, uint64_t maximum) {
    const std::string value(text);
    uint64_t parsed{};
    auto result = std::from_chars(value.data(), value.data() + value.size(), parsed);
    require(result.ec == std::errc{} && result.ptr == value.data() + value.size() &&
            parsed >= minimum && parsed <= maximum,
            std::string(option) + " must be an integer in [" + std::to_string(minimum) +
            ", " + std::to_string(maximum) + "]");
    return parsed;
}

void printUsage(char const* program) {
    std::cout << "Usage: " << program << " <encoder.onnx> <output.engine>\n"
              << "       [--workspace-mb N] [--optimization-level N] [--device N]\n\n"
              << "Build the fixed single-view MA encoder with 24 native BF16 attention blocks.\n"
              << "Input: inputs FP32 [1,7,434,518]. Outputs: F_fused and CLS (FP32).\n"
              << "Defaults: workspace 4096 MiB, optimization level 3, device 0, TF32 off.\n"
              << "External weights are resolved relative to the ONNX file.\n"
              << "Writes output.engine and output.engine.layers.json; existing files are never replaced.\n";
}

void ensureAbsent(fs::path const& path) {
    require(!fs::exists(path) && !fs::is_symlink(path), "Refusing to overwrite: " + path.string());
}

void writeFile(fs::path const& path, char const* bytes, size_t size) {
    std::ofstream stream(path, std::ios::binary);
    require(static_cast<bool>(stream), "Cannot open: " + path.string());
    stream.write(bytes, static_cast<std::streamsize>(size));
    stream.close();
    require(static_cast<bool>(stream), "Failed writing: " + path.string());
}

void saveEngine(fs::path const& path, IHostMemory const& serialized, std::string const& layers) {
    const fs::path layerPath = path.string() + ".layers.json";
    const fs::path staging = path.string() + ".building";
    if (!path.parent_path().empty()) {
        fs::create_directories(path.parent_path());
    }
    require(fs::create_directory(staging), "Build staging directory already exists: " + staging.string());
    struct Cleanup {
        fs::path path;
        ~Cleanup() { std::error_code ignored; fs::remove_all(path, ignored); }
    } cleanup{staging};
    writeFile(staging / "engine", static_cast<char const*>(serialized.data()), serialized.size());
    writeFile(staging / "layers.json", layers.data(), layers.size());
    // Same-filesystem hard links publish complete files and fail if a destination
    // exists. No partially written engine is exposed, even during concurrent builds.
    fs::create_hard_link(staging / "layers.json", layerPath);
    try {
        fs::create_hard_link(staging / "engine", path);
    } catch (...) {
        std::error_code ignored;
        fs::remove(layerPath, ignored);
        throw;
    }
}

void validateEngine(ICudaEngine const& engine) {
    require(engine.getNbIOTensors() == 3, "Built engine has unexpected I/O count");
    for (auto const* name : {"inputs", "F_fused", "CLS"}) {
        const bool input = std::string(name) == "inputs";
        require(engine.getTensorIOMode(name) == (input ? TensorIOMode::kINPUT : TensorIOMode::kOUTPUT) &&
                engine.getTensorDataType(name) == DataType::kFLOAT, "Built engine has invalid I/O type");
    }
    require(shapeIs(engine.getTensorShape("inputs"), {1, 7, 434, 518}) &&
            shapeIs(engine.getTensorShape("F_fused"), {1, 1536, 31, 37}) &&
            shapeIs(engine.getTensorShape("CLS"), {1, 1536, 1}), "Built engine has invalid I/O shape");
}
} // namespace

int main(int argc, char** argv) {
    if (argc == 2 && std::string(argv[1]) == "--help") {
        printUsage(argv[0]);
        return 0;
    }
    if (argc < 3) {
        printUsage(argv[0]);
        return 1;
    }
    try {
        const fs::path onnxPath = fs::absolute(argv[1]);
        const fs::path enginePath = fs::absolute(argv[2]);
        uint64_t workspaceMiB = 4096;
        int optimizationLevel = 3;
        int device = 0;
        for (int i = 3; i < argc; ++i) {
            const std::string option = argv[i];
            require(i + 1 < argc, "Missing value for " + option);
            if (option == "--workspace-mb") {
                workspaceMiB = parseNumber(argv[++i], option.c_str(), 1,
                                           std::numeric_limits<size_t>::max() / (1024ULL * 1024ULL));
            } else if (option == "--optimization-level") {
                optimizationLevel = static_cast<int>(parseNumber(argv[++i], option.c_str(), 0, 5));
            } else if (option == "--device") {
                device = static_cast<int>(parseNumber(argv[++i], option.c_str(), 0,
                                                      std::numeric_limits<int>::max()));
            } else {
                throw std::runtime_error("Unknown option: " + option);
            }
        }
        require(fs::is_regular_file(onnxPath), "ONNX file not found: " + onnxPath.string());
        ensureAbsent(enginePath);
        ensureAbsent(enginePath.string() + ".layers.json");
        cudaCheck(cudaSetDevice(device));
        cudaDeviceProp properties{};
        cudaCheck(cudaGetDeviceProperties(&properties, device));
        std::cout << "GPU: " << properties.name << " (SM " << properties.major << properties.minor << ")\n"
                  << "TensorRT: " << NV_TENSORRT_MAJOR << '.' << NV_TENSORRT_MINOR << '.'
                  << NV_TENSORRT_PATCH << '.' << NV_TENSORRT_BUILD << '\n'
                  << "ONNX: " << onnxPath << '\n'
                  << "Workspace: " << workspaceMiB << " MiB; optimization: " << optimizationLevel
                  << "; TF32: off\n" << std::flush;

        Logger logger;
        std::unique_ptr<IBuilder> builder(createInferBuilder(logger));
        require(builder != nullptr, "createInferBuilder failed");
        std::unique_ptr<INetworkDefinition> network(builder->createNetworkV2(0U));
        require(network != nullptr, "createNetworkV2 failed");
        std::unique_ptr<nvonnxparser::IParser> parser(nvonnxparser::createParser(*network, logger));
        require(parser != nullptr, "createParser failed");
        if (!parser->parseFromFile(onnxPath.string().c_str(), static_cast<int>(ILogger::Severity::kWARNING))) {
            for (int i = 0; i < parser->getNbErrors(); ++i) {
                std::cerr << "[ONNX parser] " << parser->getError(i)->desc() << '\n';
            }
            throw std::runtime_error("ONNX parsing failed; check the model and its external weight file");
        }
        validateNetwork(*network);
        Graph graph(*network);
        auto blocks = findAttention(graph);
        const float scale = kAttentionCount;
        replaceAttention(*network, blocks, scale);
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
        require(config != nullptr, "createBuilderConfig failed");
        config->clearFlag(BuilderFlag::kTF32);
        config->setBuilderOptimizationLevel(optimizationLevel);
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, workspaceMiB * 1024ULL * 1024ULL);
        config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
        const auto start = std::chrono::steady_clock::now();
        std::unique_ptr<IHostMemory> serialized(builder->buildSerializedNetwork(*network, *config));
        require(serialized != nullptr, "Engine build failed (native BF16 attention support is required)");
        const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();

        std::unique_ptr<IRuntime> runtime(createInferRuntime(logger));
        require(runtime != nullptr, "createInferRuntime failed");
        std::unique_ptr<ICudaEngine> engine(runtime->deserializeCudaEngine(serialized->data(), serialized->size()));
        require(engine != nullptr, "Built engine could not be deserialized");
        validateEngine(*engine);
        std::unique_ptr<IEngineInspector> inspector(engine->createEngineInspector());
        require(inspector != nullptr, "createEngineInspector failed");
        auto const* information = inspector->getEngineInformation(LayerInformationFormat::kJSON);
        require(information != nullptr, "Engine inspection failed");
        saveEngine(enginePath, *serialized, information);
        std::cout << "Built encoder: " << enginePath << '\n'
                  << "Layer report: " << enginePath.string() << ".layers.json\n"
                  << "Engine bytes: " << serialized->size() << "; build seconds: " << seconds << '\n';
    } catch (std::exception const& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
