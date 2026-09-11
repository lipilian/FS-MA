find_package(Qt6 6.2 REQUIRED COMPONENTS Widgets OpenGLWidgets)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc calib3d aruco)

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
target_link_libraries(fs_calibration_ui PUBLIC Qt6::Widgets Qt6::OpenGLWidgets fs_calibration fs_sentech_source)
target_compile_options(fs_calibration_ui PRIVATE -Wall -Wextra -Wpedantic)

add_executable(fs_gui app/desktop_main.cpp)
target_link_libraries(fs_gui PRIVATE fs_calibration_ui)
set_target_properties(fs_gui PROPERTIES BUILD_RPATH "${FS_SENTECH_ROOT}/lib;${FS_SENTECH_ROOT}/lib/GenICam")
