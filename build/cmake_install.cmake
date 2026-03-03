# Install script for directory: /home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "Release")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/aarch64-linux-gnu-objdump")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt/yolo_detector_node" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt/yolo_detector_node")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt/yolo_detector_node"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt" TYPE EXECUTABLE FILES "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/build/yolo_detector_node")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt/yolo_detector_node" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt/yolo_detector_node")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt/yolo_detector_node"
         OLD_RPATH "/usr/local/cuda-12.6/lib64/libcudart.so:/usr/local/cuda-12.6/lib64:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_calib3d:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_core:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_dnn:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_features2d:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_flann:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_gapi:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_highgui:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_imgcodecs:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_imgproc:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_ml:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_objdetect:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_photo:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_stitching:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_video:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_videoio:/opt/ros/humble/lib/aarch64-linux-gnu:/opt/ros/humble/lib:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/aarch64-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt/yolo_detector_node")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt/yolo_standalone" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt/yolo_standalone")
    file(RPATH_CHECK
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt/yolo_standalone"
         RPATH "")
  endif()
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt" TYPE EXECUTABLE FILES "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/build/yolo_standalone")
  if(EXISTS "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt/yolo_standalone" AND
     NOT IS_SYMLINK "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt/yolo_standalone")
    file(RPATH_CHANGE
         FILE "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt/yolo_standalone"
         OLD_RPATH "/usr/local/cuda-12.6/lib64/libcudart.so:/usr/local/cuda-12.6/lib64:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_calib3d:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_core:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_dnn:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_features2d:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_flann:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_gapi:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_highgui:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_imgcodecs:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_imgproc:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_ml:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_objdetect:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_photo:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_stitching:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_video:/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/opencv_videoio:"
         NEW_RPATH "")
    if(CMAKE_INSTALL_DO_STRIP)
      execute_process(COMMAND "/usr/bin/aarch64-linux-gnu-strip" "$ENV{DESTDIR}${CMAKE_INSTALL_PREFIX}/lib/yolo26_tensorrt/yolo_standalone")
    endif()
  endif()
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/yolo26_tensorrt/launch" TYPE DIRECTORY FILES "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/launch/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/yolo26_tensorrt/config" TYPE DIRECTORY FILES "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/config/")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/package_run_dependencies" TYPE FILE FILES "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/build/ament_cmake_index/share/ament_index/resource_index/package_run_dependencies/yolo26_tensorrt")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/parent_prefix_path" TYPE FILE FILES "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/build/ament_cmake_index/share/ament_index/resource_index/parent_prefix_path/yolo26_tensorrt")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/yolo26_tensorrt/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/ament_prefix_path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/yolo26_tensorrt/environment" TYPE FILE FILES "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/build/ament_cmake_environment_hooks/ament_prefix_path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/yolo26_tensorrt/environment" TYPE FILE FILES "/opt/ros/humble/share/ament_cmake_core/cmake/environment_hooks/environment/path.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/yolo26_tensorrt/environment" TYPE FILE FILES "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/build/ament_cmake_environment_hooks/path.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/yolo26_tensorrt" TYPE FILE FILES "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/build/ament_cmake_environment_hooks/local_setup.bash")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/yolo26_tensorrt" TYPE FILE FILES "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/build/ament_cmake_environment_hooks/local_setup.sh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/yolo26_tensorrt" TYPE FILE FILES "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/build/ament_cmake_environment_hooks/local_setup.zsh")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/yolo26_tensorrt" TYPE FILE FILES "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/build/ament_cmake_environment_hooks/local_setup.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/yolo26_tensorrt" TYPE FILE FILES "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/build/ament_cmake_environment_hooks/package.dsv")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/ament_index/resource_index/packages" TYPE FILE FILES "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/build/ament_cmake_index/share/ament_index/resource_index/packages/yolo26_tensorrt")
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/yolo26_tensorrt/cmake" TYPE FILE FILES
    "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/build/ament_cmake_core/yolo26_tensorrtConfig.cmake"
    "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/build/ament_cmake_core/yolo26_tensorrtConfig-version.cmake"
    )
endif()

if("x${CMAKE_INSTALL_COMPONENT}x" STREQUAL "xUnspecifiedx" OR NOT CMAKE_INSTALL_COMPONENT)
  file(INSTALL DESTINATION "${CMAKE_INSTALL_PREFIX}/share/yolo26_tensorrt" TYPE FILE FILES "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/package.xml")
endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/farms/orin_ssd/aphid_object_detection/YoloV8-TensorRT-Jetson_Nano/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
