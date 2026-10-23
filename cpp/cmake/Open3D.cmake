# Reuse tbsa_panorama's staged, CUDA-enabled Open3D 0.19 dependency. Its Stage
# does not ship Open3DConfig.cmake; import the same shared library directly.
set(FS_OPEN3D_ROOT "$ENV{HOME}/Documents/Stage/cuda_${CUDAToolkit_VERSION_MAJOR}.${CUDAToolkit_VERSION_MINOR}/tbsa2_deps"
    CACHE PATH "tbsa_panorama dependency Stage containing include/open3d and lib/libOpen3D.so")
if(NOT EXISTS "${FS_OPEN3D_ROOT}/include/open3d/Open3DConfig.h" OR
   NOT EXISTS "${FS_OPEN3D_ROOT}/lib/libOpen3D.so" OR
   NOT EXISTS "${FS_OPEN3D_ROOT}/lib/libfmt.a")
    message(FATAL_ERROR "tbsa_panorama Open3D dependency missing. Set FS_OPEN3D_ROOT to its tbsa2_deps directory.")
endif()
add_library(fs_tbsa_open3d SHARED IMPORTED)
set_target_properties(fs_tbsa_open3d PROPERTIES
    IMPORTED_LOCATION "${FS_OPEN3D_ROOT}/lib/libOpen3D.so"
    INTERFACE_LINK_LIBRARIES "${FS_OPEN3D_ROOT}/lib/libfmt.a"
    INTERFACE_INCLUDE_DIRECTORIES "${FS_OPEN3D_ROOT}/include;${FS_OPEN3D_ROOT}/include/open3d/3rdparty")
add_library(fs_icp STATIC src/geometry/ICPPoseRefinement.cpp)
target_include_directories(fs_icp PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(fs_icp PUBLIC opencv_core CUDA::cudart PRIVATE fs_tbsa_open3d)
target_compile_options(fs_icp PRIVATE -Wall -Wextra -Wpedantic)
message(STATUS "Manual ICP uses tbsa_panorama Open3D: ${FS_OPEN3D_ROOT}/lib/libOpen3D.so")
