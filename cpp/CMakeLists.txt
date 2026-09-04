cmake_minimum_required(VERSION 3.18)

project(TSFS LANGUAGES CXX CUDA)

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

add_executable(TSFS
    app/TSFS.cpp
    src/FS.cpp
    src/Logger.cpp
    src/StereoFrame.cpp
)

target_include_directories(TSFS PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(TSFS PRIVATE ${OpenCV_LIBS})

target_compile_options(TSFS PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

# Standalone TensorRT Plugin V3 shared library. It is loaded by the engine
# builder/runtime and is intentionally independent from the TSFS executable.
add_library(GWCVolumePlugin SHARED
    src/GWCVolumePlugin.cpp
    src/GWCVolumeKernel.cu
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
