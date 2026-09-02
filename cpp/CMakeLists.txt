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

add_executable(TSFS
    src/main.cpp
    src/FS.cpp
    src/InputPadder.cpp
    src/Logger.cpp
    src/StereoFrame.cpp
)

target_include_directories(TSFS PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(TSFS PRIVATE ${OpenCV_LIBS})

target_compile_options(TSFS PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

