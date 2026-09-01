#pragma once

#include <cuda_runtime_api.h>

namespace fs::trt {

// Declares the CUDA launcher implemented in GWCVolumeKernel.cu.
cudaError_t launchGwcVolumeKernel(
    void const* left,
    void const* right,
    void* volume,
    cudaStream_t stream) noexcept;

} // namespace fs::trt
