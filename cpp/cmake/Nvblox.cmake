# Build/install with: cmake -P cpp/cmake/BuildNvblox.cmake
# Keep the source submodule clean and all generated files under cpp/build.
set(FS_NVBLOX_ROOT "${CMAKE_CURRENT_LIST_DIR}/../build/deps/nvblox/install"
    CACHE PATH "Local nvblox installation prefix")
list(PREPEND CMAKE_PREFIX_PATH "${FS_NVBLOX_ROOT}")

if(NOT EXISTS "${FS_NVBLOX_ROOT}/share/nvblox/cmake/nvbloxConfig.cmake")
    message(FATAL_ERROR
        "nvblox is not installed at ${FS_NVBLOX_ROOT}. "
        "Run git submodule update --init --recursive, then "
        "cmake -P cpp/cmake/BuildNvblox.cmake from the repository root, "
        "or set FS_NVBLOX_ROOT to an existing installation.")
endif()

# nvblox 0.0.10's package config imports its targets but does not find the
# public logging dependencies. Import those from the same local prefix first.
set(GFLAGS_USE_TARGET_NAMESPACE FALSE)
find_package(gflags CONFIG REQUIRED)
find_package(glog CONFIG REQUIRED)
find_package(nvblox 0.0.10 EXACT CONFIG REQUIRED)
message(STATUS "nvblox ${nvblox_VERSION} enabled: ${FS_NVBLOX_ROOT}")
