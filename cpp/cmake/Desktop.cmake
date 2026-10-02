find_package(Qt6 6.2 REQUIRED COMPONENTS Widgets OpenGLWidgets)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc calib3d aruco)

# Keep the existing desktop usable on systems without the optional D435 SDK.
find_package(realsense2 QUIET CONFIG)
add_library(fs_realsense_source STATIC src/capture/RealSenseStereoSource.cpp)
target_include_directories(fs_realsense_source PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(fs_realsense_source PUBLIC opencv_core PRIVATE opencv_imgproc)
target_compile_options(fs_realsense_source PRIVATE -Wall -Wextra -Wpedantic)
if(realsense2_FOUND)
    target_compile_definitions(fs_realsense_source PRIVATE FS_HAS_REALSENSE)
    target_link_libraries(fs_realsense_source PRIVATE realsense2::realsense2)
    message(STATUS "D435 camera support enabled (librealsense ${realsense2_VERSION})")
else()
    message(STATUS "librealsense2 not found; D435 camera support unavailable")
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
    ui/widgets/MeshView.cpp ui/widgets/MeshView.hpp
    ui/widgets/MaskEditor.cpp ui/widgets/MaskEditor.hpp
    ui/widgets/StereoImageView.cpp ui/widgets/StereoImageView.hpp)
set_target_properties(fs_calibration_ui PROPERTIES AUTOMOC ON)
target_include_directories(fs_calibration_ui PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/ui")
target_link_libraries(fs_calibration_ui PUBLIC Qt6::Widgets Qt6::OpenGLWidgets fs_calibration fs_sentech_source fs_realsense_source)
target_compile_options(fs_calibration_ui PRIVATE -Wall -Wextra -Wpedantic)

add_executable(fs_gui app/desktop_main.cpp)
target_link_libraries(fs_gui PRIVATE fs_calibration_ui)
set_target_properties(fs_gui PROPERTIES BUILD_RPATH "${FS_SENTECH_ROOT}/lib;${FS_SENTECH_ROOT}/lib/GenICam")
