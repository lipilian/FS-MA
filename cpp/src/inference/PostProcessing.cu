#include "fs/inference/PostProcessing.hpp"

namespace fs::postprocessing {

cudaError_t filter_disparity(
    const float* /*disparity*/,
    const unsigned char* /*selection_mask*/,
    float* /*filtered_disparity*/,
    unsigned char* /*valid_mask*/,
    int /*width*/,
    int /*height*/,
    bool /*remove_invisible*/,
    cudaStream_t /*stream*/) noexcept {
    // TODO: launch the validity-filter kernel on the caller's CUDA stream.
    return cudaErrorNotSupported;
}

// Add separate depth/XYZ and neighbourhood-denoising kernels here as needed.

} // namespace fs::postprocessing
