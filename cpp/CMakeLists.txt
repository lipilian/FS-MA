cmake_minimum_required(VERSION 3.18)

project(TSFS LANGUAGES C CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs imgproc calib3d)
find_package(CUDAToolkit REQUIRED)

# TensorRT is required by the custom GWC Volume plugin and the ONNX engine
# builder below.
find_path(TENSORRT_INCLUDE_DIR
    NAMES NvInfer.h
    HINTS /usr /usr/local
    PATH_SUFFIXES include include/x86_64-linux-gnu
)
find_library(TENSORRT_NVINFER_LIBRARY
    NAMES nvinfer nvinfer.so.11
    HINTS /usr /usr/local
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

find_library(TENSORRT_NVONNXPARSER_LIBRARY
    NAMES nvonnxparser nvonnxparser.so.11
    HINTS /usr /usr/local
    PATH_SUFFIXES lib lib64 lib/x86_64-linux-gnu
)

if(NOT TENSORRT_INCLUDE_DIR OR NOT TENSORRT_NVINFER_LIBRARY OR
   NOT TENSORRT_NVONNXPARSER_LIBRARY)
    message(FATAL_ERROR "TensorRT headers, nvinfer, or nvonnxparser library not found")
endif()

# Native Triangle 1.6; preserve exact-predicate floating-point semantics.
add_library(fs_triangle STATIC third_party/triangle/triangle.c)
target_compile_definitions(fs_triangle PRIVATE TRILIBRARY ANSI_DECLARATORS)
target_include_directories(fs_triangle PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/third_party/triangle)
target_compile_options(fs_triangle PRIVATE $<$<C_COMPILER_ID:GNU,Clang>:-fno-fast-math -ffp-contract=off>)
target_link_libraries(fs_triangle PRIVATE m)

# Shared processing code for the CLI and future desktop application.
add_library(fs_core STATIC
    src/geometry/MeshBuilder.cpp
    src/geometry/MeshBuilderGPU.cu
    src/inference/FS.cpp
    src/inference/SamSegmenter.cpp
    src/inference/PostProcessing.cu
    src/core/Logger.cpp
    src/stereo/StereoFrame.cpp
    src/stereo/StereoCalibration.cpp
)

target_include_directories(fs_core
    PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE ${TENSORRT_INCLUDE_DIR}
)
target_link_libraries(fs_core
    PUBLIC
        ${OpenCV_LIBS}
        CUDA::cudart
    PRIVATE
        fs_triangle
        GWCVolumePlugin
        ${TENSORRT_NVINFER_LIBRARY}
)

target_compile_options(fs_core PRIVATE
    $<$<COMPILE_LANG_AND_ID:CXX,GNU,Clang>:-Wall -Wextra -Wpedantic>
)

add_executable(TSFS
    app/TSFS.cpp
)

target_link_libraries(TSFS PRIVATE fs_core)

target_compile_options(TSFS PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

# Standalone TensorRT Plugin V3 shared library. It is loaded by the engine
# builder/runtime and is intentionally independent from the TSFS executable.
add_library(GWCVolumePlugin SHARED
    src/inference/plugins/GWCVolumePlugin.cpp
    src/inference/plugins/GWCVolumeKernel.cu
)

set_target_properties(GWCVolumePlugin PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    OUTPUT_NAME "GWCVolumePlugin"
)

target_include_directories(GWCVolumePlugin PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(GWCVolumePlugin PRIVATE
    ${TENSORRT_NVINFER_LIBRARY}
    CUDA::cudart
)

target_compile_options(GWCVolumePlugin PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

# Build one TensorRT engine from an ONNX graph containing the
# foundation_stereo::GWCVolume Plugin V3 node. Linking the plugin library
# makes its explicit registration entry point available before ONNX parsing.
add_executable(fs_build_single_engine
    app/build_single_engine.cpp
)

target_include_directories(fs_build_single_engine PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${TENSORRT_INCLUDE_DIR}
)

target_link_libraries(fs_build_single_engine PRIVATE
    GWCVolumePlugin
    ${TENSORRT_NVINFER_LIBRARY}
    ${TENSORRT_NVONNXPARSER_LIBRARY}
    CUDA::cudart
)

target_compile_options(fs_build_single_engine PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

# The Qt desktop owns the camera SDK dependency; file-only builds can disable it.
option(FS_BUILD_DESKTOP "Build the Qt 6 calibration and reconstruction desktop" ON)
if(FS_BUILD_DESKTOP)
    include(cmake/Sentech.cmake)
    include(cmake/Desktop.cmake)
endif()
