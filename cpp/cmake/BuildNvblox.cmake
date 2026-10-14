# Run from any working directory: cmake -P <repo>/cpp/cmake/BuildNvblox.cmake
# Override CMAKE_CUDA_COMPILER, CMAKE_CUDA_ARCHITECTURES, FS_NVBLOX_ROOT,
# FS_NVBLOX_BUILD_DIR or FS_NVBLOX_JOBS with -D before -P when needed.
cmake_minimum_required(VERSION 3.22.1)
get_filename_component(FS_CPP_ROOT "${CMAKE_CURRENT_LIST_DIR}/.." ABSOLUTE)
set(FS_NVBLOX_SOURCE "${FS_CPP_ROOT}/third_party/nvblox")
if(NOT EXISTS "${FS_NVBLOX_SOURCE}/CMakeLists.txt")
    message(FATAL_ERROR "Initialize the source first: git submodule update --init --recursive")
endif()
if(NOT DEFINED FS_NVBLOX_ROOT)
    set(FS_NVBLOX_ROOT "${FS_CPP_ROOT}/build/deps/nvblox/install")
endif()
if(NOT DEFINED FS_NVBLOX_BUILD_DIR)
    set(FS_NVBLOX_BUILD_DIR "${FS_CPP_ROOT}/build/deps/nvblox/build")
endif()
if(NOT DEFINED CMAKE_CUDA_COMPILER)
    if(DEFINED ENV{CUDACXX} AND NOT "$ENV{CUDACXX}" STREQUAL "")
        set(CMAKE_CUDA_COMPILER "$ENV{CUDACXX}")
    else()
        find_program(CMAKE_CUDA_COMPILER nvcc HINTS /usr/local/cuda/bin REQUIRED)
    endif()
endif()
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()
if(NOT DEFINED FS_NVBLOX_JOBS)
    set(FS_NVBLOX_JOBS 4)
endif()

execute_process(COMMAND "${CMAKE_COMMAND}"
    -S "${FS_NVBLOX_SOURCE}" -B "${FS_NVBLOX_BUILD_DIR}"
    -DCMAKE_BUILD_TYPE=Release
    "-DCMAKE_INSTALL_PREFIX=${FS_NVBLOX_ROOT}"
    "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
    "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
    -DBUILD_PYTORCH_WRAPPER=OFF -DBUILD_RENDERER=OFF
    -DBUILD_TESTING=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_EXPERIMENTS=OFF
    COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --build "${FS_NVBLOX_BUILD_DIR}"
    --parallel "${FS_NVBLOX_JOBS}" COMMAND_ERROR_IS_FATAL ANY)
execute_process(COMMAND "${CMAKE_COMMAND}" --install "${FS_NVBLOX_BUILD_DIR}"
    COMMAND_ERROR_IS_FATAL ANY)
message(STATUS "nvblox installed locally: ${FS_NVBLOX_ROOT}")
