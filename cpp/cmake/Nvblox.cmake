# Bootstrap the local dependency on first configure, including after cleaning
# cpp/build. The same build script can still be run manually when needed.
# Keep the source submodule clean and all generated files under cpp/build.
set(FS_NVBLOX_ROOT "${CMAKE_CURRENT_LIST_DIR}/../build/deps/nvblox/install"
    CACHE PATH "Local nvblox installation prefix")
set(FS_NVBLOX_BUILD_DIR "${CMAKE_CURRENT_LIST_DIR}/../build/deps/nvblox/build"
    CACHE PATH "Local nvblox build directory")
set(FS_NVBLOX_JOBS 4 CACHE STRING "Parallel jobs when building nvblox")
list(PREPEND CMAKE_PREFIX_PATH "${FS_NVBLOX_ROOT}")

if(NOT EXISTS "${FS_NVBLOX_ROOT}/share/nvblox/cmake/nvbloxConfig.cmake")
    message(STATUS "nvblox is not installed locally; building it now (this may take several minutes)")
    execute_process(
        COMMAND "${CMAKE_COMMAND}"
            "-DFS_NVBLOX_ROOT=${FS_NVBLOX_ROOT}"
            "-DFS_NVBLOX_BUILD_DIR=${FS_NVBLOX_BUILD_DIR}"
            "-DFS_NVBLOX_JOBS=${FS_NVBLOX_JOBS}"
            "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
            "-DCMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}"
            -P "${CMAKE_CURRENT_LIST_DIR}/BuildNvblox.cmake"
        RESULT_VARIABLE _nvblox_build_result)
    if(NOT _nvblox_build_result STREQUAL "0")
        message(FATAL_ERROR "Automatic nvblox build failed. See the build output above.")
    endif()
endif()

# nvblox 0.0.10's package config imports its targets but does not find the
# public logging dependencies. Import those from the same local prefix first.
set(GFLAGS_USE_TARGET_NAMESPACE FALSE)
find_package(gflags CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(nvblox 0.0.10 EXACT CONFIG REQUIRED)
message(STATUS "nvblox ${nvblox_VERSION} enabled: ${FS_NVBLOX_ROOT}")
