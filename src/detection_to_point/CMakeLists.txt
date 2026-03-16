cmake_minimum_required(VERSION 3.8)
project(detection_to_point)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- Dependencies ---

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rclcpp_components REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(image_geometry REQUIRED)
find_package(vision_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(visualization_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(tf2_geometry_msgs REQUIRED)

# --- Library (composable component) ---

add_library(detection_to_point_component SHARED
  src/detection_to_point_component.cpp
)

target_include_directories(detection_to_point_component PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
)

target_compile_features(detection_to_point_component PUBLIC cxx_std_17)

set(COMPONENT_DEPS
  rclcpp
  rclcpp_components
  geometry_msgs
  image_geometry
  vision_msgs
  cv_bridge
  visualization_msgs
  sensor_msgs
  tf2
  tf2_ros
  tf2_geometry_msgs
)

ament_target_dependencies(detection_to_point_component
  ${COMPONENT_DEPS}
)

# Register as a composable node (also creates a standalone executable)
rclcpp_components_register_node(
  detection_to_point_component
  PLUGIN "DetectionToPointNode"
  EXECUTABLE detection_to_point_node
)

# --- Install ---

install(TARGETS detection_to_point_component
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(DIRECTORY include/
  DESTINATION include
)

# --- Testing ---

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  set(ament_cmake_copyright_FOUND TRUE)
  set(ament_cmake_cpplint_FOUND TRUE)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
