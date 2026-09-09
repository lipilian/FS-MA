#pragma once

#include <NvInferRuntime.h>

namespace fs::trt {

inline constexpr char kGwcVolumePluginName[] = "GWCVolume";
inline constexpr char kGwcVolumePluginVersion[] = "1";
inline constexpr char kGwcVolumePluginNamespace[] = "foundation_stereo";

// Fixed Plugin V3 interface:
//   left, right: [1, 224, 200, 240] FP32
//   volume:      [1, 8, 48, 200, 240] FP32
class GWCVolumePlugin;
class GWCVolumePluginCreator;

// Registers the fixed-shape FP32 Plugin V3 creator in TensorRT's registry.
// Calling this more than once is safe.
bool registerGwcVolumePlugin() noexcept;

} // namespace fs::trt

// C ABI entry point for applications which load the plugin library dynamically.
extern "C" bool fs_register_gwc_volume_plugin();
