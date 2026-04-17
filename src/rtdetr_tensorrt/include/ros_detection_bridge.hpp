//
// ROS ↔ model type conversion bridge for RT-DETR.
//
// Mirrors yolo26_tensorrt/ros_detection_bridge.hpp so both packages share the same
// public interface. Kept as a separate copy (not a shared package) to keep each
// detector package self-contained.
//

#ifndef RTDETR_ROS_DETECTION_BRIDGE_HPP
#define RTDETR_ROS_DETECTION_BRIDGE_HPP

#include <string>
#include <vector>

#include "rclcpp/rclcpp.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "std_msgs/msg/header.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

#include "common.hpp"  // det::Object

/**
 * @brief Static utility class for converting between ROS messages and inference types.
 *
 * All methods are stateless — no instance needed.
 */
class RosDetectionBridge {
public:
    /**
     * @brief Convert a ROS Image message to an OpenCV BGR Mat.
     */
    static cv::Mat imageMsgToMat(const sensor_msgs::msg::Image::ConstSharedPtr& msg);

    /**
     * @brief Convert an OpenCV BGR Mat to a ROS Image message.
     */
    static sensor_msgs::msg::Image::SharedPtr matToImageMsg(
        const cv::Mat& mat,
        const std_msgs::msg::Header& header);

    /**
     * @brief Convert inference Objects to a Detection2DArray message.
     */
    static vision_msgs::msg::Detection2DArray objectsToDetections(
        const std::vector<det::Object>& objs,
        const std::vector<std::string>& class_names,
        const std_msgs::msg::Header& header);
};

#endif  // RTDETR_ROS_DETECTION_BRIDGE_HPP
