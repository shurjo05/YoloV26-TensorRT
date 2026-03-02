//
// ROS ↔ model type conversion bridge.
//
// Isolates all ROS message conversions from the inference engine and node logic.
// Extend this file when adding classification outputs, extra metadata, or
// dashboard-specific fields.
//

#ifndef ROS_DETECTION_BRIDGE_HPP
#define ROS_DETECTION_BRIDGE_HPP

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
     *
     * @param msg Incoming ROS image message.
     * @return cv::Mat BGR image ready for inference.
     */
    static cv::Mat imageMsgToMat(const sensor_msgs::msg::Image::SharedPtr& msg);

    /**
     * @brief Convert an OpenCV BGR Mat to a ROS Image message.
     *
     * @param mat Annotated BGR image.
     * @param header Header to stamp on the output message (preserves original timestamp).
     * @return sensor_msgs::msg::Image::SharedPtr ROS image message.
     */
    static sensor_msgs::msg::Image::SharedPtr matToImageMsg(
        const cv::Mat& mat,
        const std_msgs::msg::Header& header);

    /**
     * @brief Convert inference Objects to a Detection2DArray message.
     *
     * Each Object becomes one Detection2D with bounding box, class ID, and confidence.
     * Future extension point: add classification results, extra metadata here.
     *
     * @param objs Detection results from PostProcess().
     * @param class_names Class label lookup table (indexed by Object::label).
     * @param header Header to stamp on the output message.
     * @return vision_msgs::msg::Detection2DArray Structured detection results.
     */
    static vision_msgs::msg::Detection2DArray objectsToDetections(
        const std::vector<det::Object>& objs,
        const std::vector<std::string>& class_names,
        const std_msgs::msg::Header& header);
};

#endif  // ROS_DETECTION_BRIDGE_HPP
