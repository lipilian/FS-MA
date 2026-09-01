#pragma once

#include <NvInferRuntime.h>

namespace fs::trt {

inline constexpr char kGwcVolumePluginName[] = "GWCVolume";
inline constexpr char kGwcVolumePluginVersion[] = "1";
inline constexpr char kGwcVolumePluginNamespace[] = "foundation_stereo";

// The Plugin V3 implementation and creator are defined in GWCVolumePlugin.cpp.
class GWCVolumePlugin;
class GWCVolumePluginCreator;

} // namespace fs::trt
