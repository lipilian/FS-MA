find_package(Qt6 6.2 REQUIRED COMPONENTS Widgets OpenGL OpenGLWidgets)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc calib3d aruco)
include(${CMAKE_CURRENT_LIST_DIR}/Open3D.cmake)

# ZED is optional for file imports / Sentech; scan_gui reports missing SDK support.
set(FS_ZED_ROOT "/usr/local/zed" CACHE PATH "Stereolabs ZED SDK installation prefix")
find_path(FS_ZED_INCLUDE sl/Camera.hpp HINTS "${FS_ZED_ROOT}/include")
find_library(FS_ZED_LIBRARY NAMES sl_zed HINTS "${FS_ZED_ROOT}/lib")
add_library(fs_zed_source STATIC src/capture/ZedStereoSource.cpp)
target_include_directories(fs_zed_source PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(fs_zed_source PUBLIC opencv_core PRIVATE opencv_imgproc)
target_compile_options(fs_zed_source PRIVATE -Wall -Wextra -Wpedantic)
if(FS_ZED_INCLUDE AND FS_ZED_LIBRARY)
    target_compile_definitions(fs_zed_source PRIVATE FS_HAS_ZED)
    target_include_directories(fs_zed_source SYSTEM PRIVATE "${FS_ZED_INCLUDE}")
    target_link_libraries(fs_zed_source PRIVATE "${FS_ZED_LIBRARY}" CUDA::cudart)
    message(STATUS "ZED camera support enabled (${FS_ZED_ROOT})")
else()
    message(STATUS "ZED SDK not found; live ZED capture unavailable")
endif()

add_library(fs_calibration STATIC src/calibration/CharucoCalibration.cpp)
target_include_directories(fs_calibration PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(fs_calibration PUBLIC fs_core opencv_aruco)
target_compile_options(fs_calibration PRIVATE -Wall -Wextra -Wpedantic)

add_library(fs_calibration_ui STATIC
    ui/CalibrationWorker.cpp ui/CalibrationWorker.hpp
    ui/CalibrationController.cpp ui/CalibrationController.hpp
    ui/CalibrationWindow.cpp ui/CalibrationWindow.hpp
    ui/DesktopController.cpp ui/DesktopController.hpp
    ui/InferenceSplashWindow.cpp ui/InferenceSplashWindow.hpp
    ui/ReconstructionWindow.cpp ui/ReconstructionWindow.hpp
    ui/PipelineController.cpp ui/PipelineController.hpp
    ui/PipelineWorker.cpp ui/PipelineWorker.hpp
    ui/ScanExport.cpp ui/ScanExport.hpp
    ui/TsdfSettingsDialog.cpp ui/TsdfSettingsDialog.hpp
    ui/widgets/MeshView.cpp ui/widgets/MeshView.hpp
    ui/widgets/CaptureTransition.cpp ui/widgets/CaptureTransition.hpp
    ui/widgets/MaskEditor.cpp ui/widgets/MaskEditor.hpp
    ui/widgets/StereoImageView.cpp ui/widgets/StereoImageView.hpp)
set_target_properties(fs_calibration_ui PROPERTIES AUTOMOC ON)
target_include_directories(fs_calibration_ui PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ui")
target_link_libraries(fs_calibration_ui PUBLIC Qt6::Widgets Qt6::OpenGL Qt6::OpenGLWidgets fs_calibration fs_sentech_source fs_zed_source)
target_link_libraries(fs_calibration_ui PRIVATE fs_icp fs_tsdf)
target_compile_options(fs_calibration_ui PRIVATE -Wall -Wextra -Wpedantic)

add_executable(fs_gui app/desktop_main.cpp)
target_link_libraries(fs_gui PRIVATE fs_calibration_ui)
set_target_properties(fs_gui PROPERTIES BUILD_RPATH "${FS_SENTECH_ROOT}/lib;${FS_SENTECH_ROOT}/lib/GenICam")

add_executable(scan_gui app/scan_main.cpp)
target_link_libraries(scan_gui PRIVATE fs_calibration_ui)
set_target_properties(scan_gui PROPERTIES BUILD_RPATH "${FS_SENTECH_ROOT}/lib;${FS_SENTECH_ROOT}/lib/GenICam")
