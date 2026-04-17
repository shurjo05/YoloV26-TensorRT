cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 87 89 90)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(rtdetr_tensorrt LANGUAGES CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# --- Dependencies ---
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

find_package(CUDA REQUIRED)
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)

find_package(OpenCV REQUIRED)

# TensorRT — same host-mounted paths as yolo26_tensorrt.
# See yolo26_tensorrt/CMakeLists.txt and CLAUDE.md Issue 21 for context.
set(TensorRT_INCLUDE_DIRS /host-trt-include /usr/include /usr/include/aarch64-linux-gnu)
set(TensorRT_LIBRARIES /host-trt-lib /usr/lib/aarch64-linux-gnu)
set(CMAKE_INSTALL_RPATH /host-trt-lib)
set(CMAKE_BUILD_RPATH /host-trt-lib)

list(APPEND INCLUDE_DIRS
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
    include
)

list(APPEND ALL_LIBS
    ${CUDA_LIBRARIES}
    ${CUDA_LIB_DIR}
    ${OpenCV_LIBRARIES}
    ${TensorRT_LIBRARIES}
)

include_directories(${INCLUDE_DIRS})

# --- Target: ROS2 detector node ---
add_executable(rtdetr_detector_node
    src/rtdetr_detector_node.cpp
    src/ros_detection_bridge.cpp
    src/rtdetr.cpp
)
target_link_directories(rtdetr_detector_node PUBLIC ${ALL_LIBS})
target_link_libraries(rtdetr_detector_node nvinfer nvinfer_plugin cudart ${OpenCV_LIBS})
ament_target_dependencies(rtdetr_detector_node
    rclcpp
    sensor_msgs
    vision_msgs
    cv_bridge
    image_transport
)

# --- Install ---
install(TARGETS
    rtdetr_detector_node
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
