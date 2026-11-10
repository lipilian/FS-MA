#include <NvInfer.h>
#include <NvOnnxParser.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
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
#error "ma_build_pose_scale requires TensorRT 11 or newer"
#endif

namespace {
namespace fs = std::filesystem;
using namespace nvinfer1;

constexpr int kGlobalAttentionCount = 8;
constexpr int kPerViewAttentionCount = 8;
constexpr int kAttentionBlockCount = kGlobalAttentionCount + kPerViewAttentionCount;
constexpr int kFirstExportedAttention = 24; // Encoder occupies SDPA 0..23 in the source ONNX.
constexpr int kMinViews = 2;
constexpr int kOptViews = 3;
constexpr int kMaxViews = 5;
constexpr float kAttentionScale = 0.125F; // 1 / sqrt(1536 / 24)

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
    require(network.getNbInputs() == 2 && network.getNbOutputs() == 2,
            "Expected the extracted pose/scale network with two inputs and two outputs");
    bool fused = false, cls = false, poses = false, scale = false;
    for (int i = 0; i < network.getNbInputs(); ++i) {
        auto* input = network.getInput(i);
        const std::string name = input->getName();
        if (name == "F_fused") {
            tensorIs(input, DataType::kFLOAT, {-1, 1536, 31, 37}, name);
            fused = true;
        } else if (name == "CLS") {
            tensorIs(input, DataType::kFLOAT, {-1, 1536, 1}, name);
            cls = true;
        } else {
            throw std::runtime_error("Unexpected pose/scale input: " + name);
        }
        // Shared dimension names tell TensorRT that both inputs have the same V.
        input->setDimensionName(0, "num_views");
    }
    for (int i = 0; i < network.getNbOutputs(); ++i) {
        auto* output = network.getOutput(i);
        const std::string name = output->getName();
        if (name == "poses") {
            tensorIs(output, DataType::kFLOAT, {-1, 7}, name);
            poses = true;
        } else if (name == "scale") {
            require(output->getType() == DataType::kFLOAT &&
                    (shapeIs(output->getDimensions(), {1, 1, -1}) ||
                     shapeIs(output->getDimensions(), {1, 1, 1})), "Unexpected scale shape/type");
            scale = true;
        } else {
            throw std::runtime_error("Unexpected pose/scale output: " + name);
        }
    }
    require(fused && cls && poses && scale, "Missing expected pose/scale I/O");
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

    // The dynamic parser restores Softmax's shape using Shape(softmaxInput).
    // Matching [-1,...,-1] alone cannot prove that a dynamic reshape is identity.
    ITensor* softmaxOutput(ITensor* tensor) const {
        auto it = producers.find(tensor);
        require(it != producers.end(), "Missing Softmax producer");
        if (it->second->getType() == LayerType::kSOFTMAX) {
            return tensor;
        }
        auto* shuffle = static_cast<IShuffleLayer*>(producer(tensor, LayerType::kSHUFFLE,
                                                            "Softmax shape restoration"));
        auto* input = shuffle->getInput(0);
        auto* softmax = producer(input, LayerType::kSOFTMAX, "Softmax before reshape");
        const auto dims = tensor->getDimensions();
        require(sameShape(input->getDimensions(), dims) &&
                identityPermutation(shuffle->getFirstTranspose(), dims.nbDims) &&
                identityPermutation(shuffle->getSecondTranspose(), dims.nbDims),
                "Softmax reshape must not change dimensions or transpose");
        if (shuffle->getNbInputs() == 2) {
            auto* shape = shuffle->getInput(1);
            auto shapeIt = producers.find(shape);
            while (shapeIt != producers.end() && shapeIt->second->getType() == LayerType::kCAST) {
                auto* before = shapeIt->second->getInput(0);
                require((shape->getType() == DataType::kINT64 || shape->getType() == DataType::kINT32) &&
                        (before->getType() == DataType::kINT64 || before->getType() == DataType::kINT32),
                        "Expected integer shape cast");
                shape = before;
                shapeIt = producers.find(shape);
            }
            auto* shapeLayer = producer(shape, LayerType::kSHAPE, "Softmax restore shape");
            require(shapeLayer->getInput(0) == softmax->getInput(0) ||
                    shapeLayer->getInput(0) == input,
                    "Dynamic reshape must restore the exact Softmax shape");
        } else {
            // A one-input reshape is safe only when every dimension is static.
            for (int i = 0; i < dims.nbDims; ++i) {
                require(dims.d[i] > 0, "Unproven static reshape of a dynamic attention tensor");
            }
        }
        return input;
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
        require(*static_cast<float const*>(weights.values) == kAttentionScale,
                "Attention scale is not 0.125; refusing to change model semantics");
    }

    std::vector<ILayer*> layers;
private:
    std::unordered_map<ITensor*, ILayer*> producers;
};

struct Attention {
    int blockIndex;
    bool global;
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

void attentionTensorIs(ITensor* tensor, DataType type, bool global, std::string const& role) {
    if (global) {
        tensorIs(tensor, type, {1, 24, -1, 64}, role);
    } else {
        tensorIs(tensor, type, {-1, 24, 1148, 64}, role);
    }
}

void scoresTensorIs(ITensor* tensor, bool global, std::string const& role) {
    if (global) {
        tensorIs(tensor, DataType::kFLOAT, {1, 24, -1, -1}, role);
    } else {
        tensorIs(tensor, DataType::kFLOAT, {-1, 24, 1148, 1148}, role);
    }
}

ITensor* bf16BeforeCast(Graph const& graph, ITensor* tensor, bool global, std::string const& role) {
    attentionTensorIs(tensor, DataType::kFLOAT, global, role + " FP32");
    auto* cast = graph.producer(tensor, LayerType::kCAST, role + " Cast");
    auto* input = cast->getInput(0);
    attentionTensorIs(input, DataType::kBF16, global, role + " BF16");
    return input;
}

std::vector<Attention> findAttention(Graph const& graph) {
    std::vector<Attention> result;
    for (auto* layer : graph.layers) {
        const std::string name = layer->getName();
        if (name.rfind("node_scaled_dot_product_attention", 0) != 0) {
            continue;
        }
        const std::string prefix = "node_scaled_dot_product_attention_";
        require(name.rfind(prefix, 0) == 0, "Unexpected SDPA name: " + name);
        int exportedIndex = -1;
        auto parsed = std::from_chars(name.data() + prefix.size(), name.data() + name.size(), exportedIndex);
        require(parsed.ec == std::errc{} && parsed.ptr == name.data() + name.size() &&
                exportedIndex >= kFirstExportedAttention &&
                exportedIndex < kFirstExportedAttention + kAttentionBlockCount,
                "Expected exported SDPA indices 24..39: " + name);
        const int blockIndex = exportedIndex - kFirstExportedAttention;
        const bool global = blockIndex % 2 == 0;
        require(layer->getType() == LayerType::kCAST, "Expected SDPA output Cast: " + name);
        auto* output = layer->getOutput(0);
        attentionTensorIs(output, DataType::kBF16, global, name);
        attentionTensorIs(layer->getInput(0), DataType::kFLOAT, global, name + " AV");
        auto* av = matmul(graph, layer->getInput(0), name + " AV");
        auto* probabilities = graph.softmaxOutput(av->getInput(0));
        auto* softmax = static_cast<ISoftMaxLayer*>(
            graph.producer(probabilities, LayerType::kSOFTMAX, name + " Softmax"));
        scoresTensorIs(probabilities, global, name + " probabilities");
        require(softmax->getAxes() == (1U << 3), "Expected Softmax over the last axis: " + name);
        auto* mul = static_cast<IElementWiseLayer*>(
            graph.producer(softmax->getInput(0), LayerType::kELEMENTWISE, name + " scale Mul"));
        require(mul->getOperation() == ElementWiseOperation::kPROD,
                "Expected multiplication by attention scale: " + name);
        graph.validateScale(mul->getInput(1));
        auto* qk = matmul(graph, mul->getInput(0), name + " QK");
        scoresTensorIs(qk->getOutput(0), global, name + " scores");
        auto* transpose = static_cast<IShuffleLayer*>(
            graph.producer(qk->getInput(1), LayerType::kSHUFFLE, name + " K transpose"));
        const auto first = transpose->getFirstTranspose();
        require(transpose->getNbInputs() == 1 && first.order[0] == 0 && first.order[1] == 1 &&
                first.order[2] == 3 && first.order[3] == 2 &&
                identityPermutation(transpose->getSecondTranspose(), 4),
                "Expected K transpose [0,1,3,2]: " + name);
        if (global) {
            tensorIs(transpose->getOutput(0), DataType::kFLOAT, {1, 24, 64, -1}, name + " transposed K");
        } else {
            tensorIs(transpose->getOutput(0), DataType::kFLOAT, {-1, 24, 64, 1148}, name + " transposed K");
        }
        Attention attention{blockIndex, global,
                            bf16BeforeCast(graph, qk->getInput(0), global, name + " Q"),
                            bf16BeforeCast(graph, transpose->getInput(0), global, name + " K"),
                            bf16BeforeCast(graph, av->getInput(1), global, name + " V"), output, {}};
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
    require(result.size() == kAttentionBlockCount,
            "Expected exactly 16 pose/scale attention blocks, found " + std::to_string(result.size()));
    std::sort(result.begin(), result.end(), [](Attention const& a, Attention const& b) {
        return a.blockIndex < b.blockIndex;
    });
    for (int i = 0; i < kAttentionBlockCount; ++i) {
        require(result[i].blockIndex == i, "Missing or duplicate exported attention block");
    }
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
        const std::string prefix = std::string(block.global ? "global_" : "per_view_") +
                                   std::string(i < 10 ? "0" : "") + std::to_string(i);
        auto* scaled = network.addElementWise(*block.q, *cast->getOutput(0), ElementWiseOperation::kPROD);
        require(scaled != nullptr, "Could not scale Q");
        scaled->setName((prefix + "_query_scale").c_str());
        auto* attention = network.addAttentionV2(*scaled->getOutput(0), *block.k, *block.v,
                                                AttentionNormalizationOp::kSOFTMAX, CausalMaskKind::kNONE);
        require(attention != nullptr, "Could not create native attention");
        require(attention->setName((prefix + "_native_attention").c_str()) &&
                attention->setDecomposable(false), "Could not configure native attention");
        auto* replacement = attention->getOutput(0);
        attentionTensorIs(replacement, DataType::kBF16, block.global, prefix + " replacement");
        replacement->setName((std::string(block.output->getName()) + "_fmha").c_str());
        for (auto const& consumer : block.consumers) {
            consumer.first->setInput(consumer.second, *replacement);
        }
    }
    // The builder removes the disconnected FP32 MatMul/Softmax/MatMul paths.
    std::cout << "Replaced " << blocks.size() << " attention blocks: 8 global + 8 per-view (BF16, scale=0.125, non-decomposable)\n";
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
    std::cout << "Usage: " << program << " <pose_scale.onnx> <output.engine>\n"
              << "       [--workspace-mb N] [--optimization-level N] [--device N]\n\n"
              << "Build dynamic MA pose/scale with 16 native BF16 attention blocks; views 2..5.\n"
              << "Inputs: F_fused [V,1536,31,37], CLS [V,1536,1]; outputs: poses [V,7], scale [1,1,1]. All FP32.\n"
              << "Profile: min/opt/max views 2/3/5. Defaults: workspace 8192 MiB, optimization level 3, device 0, TF32 off.\n"
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

void addViewProfile(IBuilder& builder, IBuilderConfig& config) {
    // The builder owns profiles returned by createOptimizationProfile().
    auto* profile = builder.createOptimizationProfile();
    require(profile != nullptr, "createOptimizationProfile failed");
    const std::array<std::pair<OptProfileSelector, int>, 3> shapes{{
        {OptProfileSelector::kMIN, kMinViews},
        {OptProfileSelector::kOPT, kOptViews},
        {OptProfileSelector::kMAX, kMaxViews}}};
    for (auto const& item : shapes) {
        require(profile->setDimensions("F_fused", item.first, Dims4{item.second, 1536, 31, 37}) &&
                profile->setDimensions("CLS", item.first, Dims3{item.second, 1536, 1}),
                "Could not set the two-input view profile");
    }
    require(profile->isValid() && config.addOptimizationProfile(profile) == 0,
            "Could not add dynamic view profile");
}

void validateEngine(ICudaEngine& engine) {
    require(engine.getNbIOTensors() == 4 && engine.getNbOptimizationProfiles() == 1,
            "Built engine has unexpected I/O or profile count");
    for (auto const* name : {"F_fused", "CLS", "poses", "scale"}) {
        const bool input = std::string(name) == "F_fused" || std::string(name) == "CLS";
        require(engine.getTensorIOMode(name) == (input ? TensorIOMode::kINPUT : TensorIOMode::kOUTPUT) &&
                engine.getTensorDataType(name) == DataType::kFLOAT, "Built engine has invalid I/O type");
    }
    require(shapeIs(engine.getTensorShape("F_fused"), {-1, 1536, 31, 37}) &&
            shapeIs(engine.getTensorShape("CLS"), {-1, 1536, 1}) &&
            shapeIs(engine.getTensorShape("poses"), {-1, 7}), "Built engine has invalid I/O shape");
    const std::array<std::pair<OptProfileSelector, int>, 3> shapes{{
        {OptProfileSelector::kMIN, kMinViews},
        {OptProfileSelector::kOPT, kOptViews},
        {OptProfileSelector::kMAX, kMaxViews}}};
    for (auto const& item : shapes) {
        require(shapeIs(engine.getProfileShape("F_fused", 0, item.first), {item.second, 1536, 31, 37}) &&
                shapeIs(engine.getProfileShape("CLS", 0, item.first), {item.second, 1536, 1}),
                "Built engine has invalid profile shapes");
    }
    std::unique_ptr<IExecutionContext> context(engine.createExecutionContext());
    require(context != nullptr, "Could not create context to validate dynamic output shapes");
    for (int views = kMinViews; views <= kMaxViews; ++views) {
        require(context->setInputShape("F_fused", Dims4{views, 1536, 31, 37}) &&
                context->setInputShape("CLS", Dims3{views, 1536, 1}) &&
                context->inferShapes(0, nullptr) == 0, "Dynamic shape inference failed");
        require(shapeIs(context->getTensorShape("poses"), {views, 7}) &&
                shapeIs(context->getTensorShape("scale"), {1, 1, 1}),
                "Unexpected resolved output shapes for V=" + std::to_string(views));
    }
    std::cout << "Validated profile 2/3/5 and output shapes for every view count 2..5\n";
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
        uint64_t workspaceMiB = 8192;
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
        const float scale = kAttentionScale;
        replaceAttention(*network, blocks, scale);
        std::unique_ptr<IBuilderConfig> config(builder->createBuilderConfig());
        require(config != nullptr, "createBuilderConfig failed");
        config->clearFlag(BuilderFlag::kTF32);
        config->setBuilderOptimizationLevel(optimizationLevel);
        config->setMemoryPoolLimit(MemoryPoolType::kWORKSPACE, workspaceMiB * 1024ULL * 1024ULL);
        config->setProfilingVerbosity(ProfilingVerbosity::kDETAILED);
        addViewProfile(*builder, *config);
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
        std::cout << "Built dynamic pose/scale: " << enginePath << '\n'
                  << "Layer report: " << enginePath.string() << ".layers.json\n"
                  << "Engine bytes: " << serialized->size() << "; build seconds: " << seconds << '\n';
    } catch (std::exception const& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
    return 0;
}
