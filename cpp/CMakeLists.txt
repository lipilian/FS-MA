cmake_minimum_required(VERSION 3.16)

project(TSFS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(OpenCV 4 REQUIRED COMPONENTS core imgcodecs imgproc calib3d)

add_executable(TSFS
    src/main.cpp
    src/StereoFrame.cpp
)

target_include_directories(TSFS PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(TSFS PRIVATE ${OpenCV_LIBS})

target_compile_options(TSFS PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)
