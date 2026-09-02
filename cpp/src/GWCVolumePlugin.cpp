#include "GWCVolumePlugin.hpp"

#include "GWCVolumeKernel.hpp"

#include <NvInfer.h>
#include <cuda_runtime_api.h>

#include <mutex>

namespace fs::trt {
namespace {

constexpr int32_t kBatch = 1;
constexpr int32_t kChannels = 224;
constexpr int32_t kHeight = 200;
constexpr int32_t kWidth = 240;
constexpr int32_t kGroups = 8;
constexpr int32_t kMaxDisparity = 48;

bool hasFeatureShape(nvinfer1::Dims const& dims) noexcept {
    constexpr int32_t kExpected[] = {kBatch, kChannels, kHeight, kWidth};
    if (dims.nbDims != 4) return false;
    for (int32_t i = 0; i < 4; ++i) {
        if (dims.d[i] != kExpected[i]) return false;
    }
    return true;
}

bool hasVolumeShape(nvinfer1::Dims const& dims) noexcept {
    constexpr int32_t kExpected[] = {kBatch, kGroups, kMaxDisparity, kHeight, kWidth};
    if (dims.nbDims != 5) return false;
    for (int32_t i = 0; i < 5; ++i) {
        if (dims.d[i] != kExpected[i]) return false;
    }
    return true;
}

bool isFp32Linear(nvinfer1::PluginTensorDesc const& desc) noexcept {
    return desc.type == nvinfer1::DataType::kFLOAT &&
           desc.format == nvinfer1::TensorFormat::kLINEAR;
}

bool configureDescIsFixedFeature(nvinfer1::DynamicPluginTensorDesc const& desc) noexcept {
    return isFp32Linear(desc.desc) &&
           hasFeatureShape(desc.min) &&
           hasFeatureShape(desc.opt) &&
           hasFeatureShape(desc.max);
}

bool configureDescIsFixedVolume(nvinfer1::DynamicPluginTensorDesc const& desc) noexcept {
    return isFp32Linear(desc.desc) &&
           hasVolumeShape(desc.min) &&
           hasVolumeShape(desc.opt) &&
           hasVolumeShape(desc.max);
}

} // namespace

class GWCVolumePlugin final : public nvinfer1::IPluginV3,
                              public nvinfer1::IPluginV3OneCore,
                              public nvinfer1::IPluginV3OneBuild,
                              public nvinfer1::IPluginV3OneRuntime {
public:
    explicit GWCVolumePlugin(nvinfer1::TensorRTPhase phase) noexcept : phase_(phase) {}

    nvinfer1::IPluginCapability* getCapabilityInterface(
        nvinfer1::PluginCapabilityType type) noexcept override {
        switch (type) {
        case nvinfer1::PluginCapabilityType::kCORE:
            return static_cast<nvinfer1::IPluginV3OneCore*>(this);
        case nvinfer1::PluginCapabilityType::kBUILD:
            return phase_ == nvinfer1::TensorRTPhase::kBUILD
                ? static_cast<nvinfer1::IPluginV3OneBuild*>(this)
                : nullptr;
        case nvinfer1::PluginCapabilityType::kRUNTIME:
            return static_cast<nvinfer1::IPluginV3OneRuntime*>(this);
        }
        return nullptr;
    }

    nvinfer1::IPluginV3* clone() noexcept override {
        return new GWCVolumePlugin(phase_);
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return kGwcVolumePluginName;
    }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return kGwcVolumePluginVersion;
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return kGwcVolumePluginNamespace;
    }

    int32_t configurePlugin(
        nvinfer1::DynamicPluginTensorDesc const* in,
        int32_t nbInputs,
        nvinfer1::DynamicPluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override {
        if (!in || !out || nbInputs != 2 || nbOutputs != 1) return 1;
        return configureDescIsFixedFeature(in[0]) &&
               configureDescIsFixedFeature(in[1]) &&
               configureDescIsFixedVolume(out[0])
            ? 0
            : 1;
    }

    int32_t getOutputDataTypes(
        nvinfer1::DataType* outputTypes,
        int32_t nbOutputs,
        nvinfer1::DataType const* inputTypes,
        int32_t nbInputs) const noexcept override {
        if (!outputTypes || !inputTypes || nbInputs != 2 || nbOutputs != 1 ||
            inputTypes[0] != nvinfer1::DataType::kFLOAT ||
            inputTypes[1] != nvinfer1::DataType::kFLOAT) {
            return 1;
        }
        outputTypes[0] = nvinfer1::DataType::kFLOAT;
        return 0;
    }

    int32_t getOutputShapes(
        nvinfer1::DimsExprs const* inputs,
        int32_t nbInputs,
        nvinfer1::DimsExprs const*,
        int32_t nbShapeInputs,
        nvinfer1::DimsExprs* outputs,
        int32_t nbOutputs,
        nvinfer1::IExprBuilder& exprBuilder) noexcept override {
        if (!inputs || !outputs || nbInputs != 2 || nbShapeInputs != 0 ||
            nbOutputs != 1 || inputs[0].nbDims != 4 || inputs[1].nbDims != 4) {
            return 1;
        }

        outputs[0].nbDims = 5;
        outputs[0].d[0] = exprBuilder.constant(kBatch);
        outputs[0].d[1] = exprBuilder.constant(kGroups);
        outputs[0].d[2] = exprBuilder.constant(kMaxDisparity);
        outputs[0].d[3] = exprBuilder.constant(kHeight);
        outputs[0].d[4] = exprBuilder.constant(kWidth);
        return 0;
    }

    bool supportsFormatCombination(
        int32_t pos,
        nvinfer1::DynamicPluginTensorDesc const* inOut,
        int32_t nbInputs,
        int32_t nbOutputs) noexcept override {
        if (!inOut || nbInputs != 2 || nbOutputs != 1 || pos < 0 || pos >= 3) {
            return false;
        }
        return isFp32Linear(inOut[pos].desc);
    }

    int32_t getNbOutputs() const noexcept override {
        return 1;
    }

    int32_t onShapeChange(
        nvinfer1::PluginTensorDesc const* in,
        int32_t nbInputs,
        nvinfer1::PluginTensorDesc const* out,
        int32_t nbOutputs) noexcept override {
        if (!in || !out || nbInputs != 2 || nbOutputs != 1) return 1;
        return isFp32Linear(in[0]) &&
               isFp32Linear(in[1]) &&
               isFp32Linear(out[0]) &&
               hasFeatureShape(in[0].dims) &&
               hasFeatureShape(in[1].dims) &&
               hasVolumeShape(out[0].dims)
            ? 0
            : 1;
    }

    int32_t enqueue(
        nvinfer1::PluginTensorDesc const* inputDesc,
        nvinfer1::PluginTensorDesc const* outputDesc,
        void const* const* inputs,
        void* const* outputs,
        void*,
        cudaStream_t stream) noexcept override {
        if (!inputDesc || !outputDesc || !inputs || !outputs ||
            !inputs[0] || !inputs[1] || !outputs[0] ||
            onShapeChange(inputDesc, 2, outputDesc, 1) != 0) {
            return 1;
        }

        return launchGwcVolumeKernel(
                   static_cast<float const*>(inputs[0]),
                   static_cast<float const*>(inputs[1]),
                   static_cast<float*>(outputs[0]),
                   stream) == cudaSuccess
            ? 0
            : 1;
    }

    nvinfer1::IPluginV3* attachToContext(nvinfer1::IPluginResourceContext*) noexcept override {
        return clone();
    }

    nvinfer1::PluginFieldCollection const* getFieldsToSerialize() noexcept override {
        return &serializedFields_;
    }

private:
    nvinfer1::TensorRTPhase phase_;
    nvinfer1::PluginFieldCollection serializedFields_{};
};

class GWCVolumePluginCreator final : public nvinfer1::IPluginCreatorV3One {
public:
    nvinfer1::IPluginV3* createPlugin(
        nvinfer1::AsciiChar const*,
        nvinfer1::PluginFieldCollection const* fields,
        nvinfer1::TensorRTPhase phase) noexcept override {
        if (fields && fields->nbFields != 0) return nullptr;
        return new GWCVolumePlugin(phase);
    }

    nvinfer1::PluginFieldCollection const* getFieldNames() noexcept override {
        return &fieldNames_;
    }

    nvinfer1::AsciiChar const* getPluginName() const noexcept override {
        return kGwcVolumePluginName;
    }

    nvinfer1::AsciiChar const* getPluginVersion() const noexcept override {
        return kGwcVolumePluginVersion;
    }

    nvinfer1::AsciiChar const* getPluginNamespace() const noexcept override {
        return kGwcVolumePluginNamespace;
    }

private:
    nvinfer1::PluginFieldCollection fieldNames_{};
};

namespace {
GWCVolumePluginCreator gCreator;
}

bool registerGwcVolumePlugin() noexcept {
    static std::once_flag once;
    static bool registered = false;
    std::call_once(once, [] {
        auto registerIn = [](nvinfer1::IPluginRegistry* registry) noexcept {
            if (!registry) return false;
            if (registry->getCreator(
                    kGwcVolumePluginName,
                    kGwcVolumePluginVersion,
                    kGwcVolumePluginNamespace)) {
                return true;
            }
            return registry->registerCreator(gCreator, kGwcVolumePluginNamespace);
        };

        const bool runtimeRegistered = registerIn(getPluginRegistry());
        const bool builderRegistered =
            registerIn(nvinfer1::getBuilderPluginRegistry(nvinfer1::EngineCapability::kSTANDARD));
        registered = runtimeRegistered || builderRegistered;
    });
    return registered;
}

} // namespace fs::trt

extern "C" bool fs_register_gwc_volume_plugin() {
    return fs::trt::registerGwcVolumePlugin();
}
