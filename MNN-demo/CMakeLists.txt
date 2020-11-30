cmake_minimum_required(VERSION 3.0)

project(yolo)

add_compile_options(-std=c++11)

find_package(OpenCV REQUIRED)

include_directories(
        ${OpenCV_INCLUDE_DIRS}
        include
        /usr/bin/include
        /home/user/work/mnn/include
)
link_directories(/home/user/work/mnn/build)
add_executable(
        yolo
        yolo.cpp
)
target_link_libraries(
        yolo
        ${OpenCV_LIBS}
        MNN
)
