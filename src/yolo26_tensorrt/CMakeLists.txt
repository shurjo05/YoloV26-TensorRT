cmake_minimum_required(VERSION 3.8)

set(CMAKE_CUDA_ARCHITECTURES 60 61 62 70 72 75 86 87 89 90)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(yolo26_tensorrt LANGUAGES CXX CUDA)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -O3")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE Release)
option(CUDA_USE_STATIC_CUDA_RUNTIME OFF)

# --- Dependencies ---

# ROS2
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

# CUDA
find_package(CUDA REQUIRED)
message(STATUS "CUDA Libs: \n${CUDA_LIBRARIES}\n")
get_filename_component(CUDA_LIB_DIR ${CUDA_LIBRARIES} DIRECTORY)
message(STATUS "CUDA Headers: \n${CUDA_INCLUDE_DIRS}\n")

# OpenCV
find_package(OpenCV REQUIRED)

# TensorRT
# /host-trt-include and /host-trt-lib are the host JetPack TRT 10 paths mounted
# into the container via docker/run.sh. This is needed because the dustynv base
# image ships TRT 8.6 while the Jetson host has TRT 10.x from JetPack 6.
# Outside Docker (native build on host), these paths also exist and contain TRT 10.
set(TensorRT_INCLUDE_DIRS /host-trt-include /usr/include /usr/include/aarch64-linux-gnu)
set(TensorRT_LIBRARIES /host-trt-lib /usr/lib/aarch64-linux-gnu)
# Bake the TRT 10 lib path into the binary RPATH so it finds libnvinfer.so.10
# at runtime inside the container without needing LD_LIBRARY_PATH.
set(CMAKE_INSTALL_RPATH /host-trt-lib)
set(CMAKE_BUILD_RPATH /host-trt-lib)

message(STATUS "TensorRT Libs: \n${TensorRT_LIBRARIES}\n")
message(STATUS "TensorRT Headers: \n${TensorRT_INCLUDE_DIRS}\n")

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

if(${OpenCV_VERSION} VERSION_GREATER_EQUAL 4.7.0)
    message(STATUS "Build with -DBATCHED_NMS")
    add_definitions(-DBATCHED_NMS)
endif()

# --- Targets ---

# ROS2 node executable
add_executable(yolo_detector_node
    src/yolo_detector_node.cpp
    src/ros_detection_bridge.cpp
    src/yolo26.cpp
)
target_link_directories(yolo_detector_node PUBLIC ${ALL_LIBS})
target_link_libraries(yolo_detector_node nvinfer nvinfer_plugin cudart ${OpenCV_LIBS})
ament_target_dependencies(yolo_detector_node
    rclcpp
    sensor_msgs
    vision_msgs
    cv_bridge
    image_transport
)

# Standalone executable (headless testing without ROS2)
add_executable(yolo_standalone
    src/main_standalone.cpp
    src/yolo26.cpp
)
target_link_directories(yolo_standalone PUBLIC ${ALL_LIBS})
target_link_libraries(yolo_standalone PRIVATE nvinfer nvinfer_plugin cudart ${OpenCV_LIBS})

# --- Install ---

install(TARGETS
    yolo_detector_node
    yolo_standalone
    DESTINATION lib/${PROJECT_NAME}
)

install(PROGRAMS
    scripts/arducam_publisher.py
    DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch/
    DESTINATION share/${PROJECT_NAME}/launch
)

install(DIRECTORY config/
    DESTINATION share/${PROJECT_NAME}/config
)

ament_package()
