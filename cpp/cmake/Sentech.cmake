set(FS_SENTECH_ROOT "/opt/sentech" CACHE PATH "Sentech StApi SDK installation prefix")
find_path(FS_SENTECH_INCLUDE StApi_TL.h PATHS "${FS_SENTECH_ROOT}/include/StApi" NO_DEFAULT_PATH)
find_path(FS_SENTECH_GENICAM_INCLUDE GenApi/GenApi.h PATHS "${FS_SENTECH_ROOT}/include/GenICam" NO_DEFAULT_PATH)
foreach(component StApi_TL StApi_IP turbojpeg GCBase GenApi)
    find_library(FS_SENTECH_${component} NAMES ${component}
        PATHS "${FS_SENTECH_ROOT}/lib" "${FS_SENTECH_ROOT}/lib/GenICam" NO_DEFAULT_PATH)
    if(NOT FS_SENTECH_${component})
        message(FATAL_ERROR "Missing Sentech ${component}. Set FS_SENTECH_ROOT or disable FS_BUILD_SENTECH.")
    endif()
endforeach()
if(NOT FS_SENTECH_INCLUDE OR NOT FS_SENTECH_GENICAM_INCLUDE)
    message(FATAL_ERROR "Sentech headers missing. Set FS_SENTECH_ROOT or disable FS_BUILD_SENTECH.")
endif()
find_package(Threads REQUIRED)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs imgproc calib3d highgui)

add_library(fs_sentech_source STATIC src/capture/SentechStereoSource.cpp)
target_include_directories(fs_sentech_source PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_include_directories(fs_sentech_source SYSTEM PRIVATE "${FS_SENTECH_INCLUDE}" "${FS_SENTECH_GENICAM_INCLUDE}")
target_compile_definitions(fs_sentech_source PRIVATE FS_SENTECH_GENTL_DIRECTORY="${FS_SENTECH_ROOT}/lib")
target_link_libraries(fs_sentech_source PUBLIC opencv_core PRIVATE
    Threads::Threads ${FS_SENTECH_StApi_TL} ${FS_SENTECH_StApi_IP}
    ${FS_SENTECH_GCBase} ${FS_SENTECH_GenApi} ${FS_SENTECH_turbojpeg})
target_compile_options(fs_sentech_source PRIVATE -Wall -Wextra -Wpedantic)
# Match the reference project's SDK transitive-library lookup (including turbojpeg).
target_link_options(fs_sentech_source INTERFACE "-Wl,--disable-new-dtags" "-Wl,--no-as-needed")
set_property(TARGET fs_sentech_source PROPERTY INTERFACE_LINK_DIRECTORIES
    "${FS_SENTECH_ROOT}/lib;${FS_SENTECH_ROOT}/lib/GenICam")

add_executable(fs_sentech app/sentech_capture.cpp)
target_link_libraries(fs_sentech PRIVATE fs_sentech_source fs_core opencv_highgui)
target_compile_options(fs_sentech PRIVATE -Wall -Wextra -Wpedantic)
set_target_properties(fs_sentech PROPERTIES BUILD_RPATH "${FS_SENTECH_ROOT}/lib;${FS_SENTECH_ROOT}/lib/GenICam")
