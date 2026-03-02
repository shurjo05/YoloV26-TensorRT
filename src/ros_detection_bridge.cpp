//
// ROS ↔ model type conversion bridge implementation.
//
// Created for YOLO26 aphid detection ROS2 node.
// Extend this file when adding classification outputs, extra metadata, or
// dashboard-specific fields.
//

#include "ros_detection_bridge.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"

//----------------------------------------------------------------------------------------
/**
 * @brief Convert a ROS Image message to an OpenCV BGR Mat.
 *
 * Uses cv_bridge to decode the ROS image encoding into a standard BGR cv::Mat
 * suitable for YOLO26 inference input.
 *
 * @param msg Incoming ROS image message (any encoding cv_bridge can handle).
 * @return cv::Mat BGR image ready for inference preprocessing.
 */
cv::Mat RosDetectionBridge::imageMsgToMat(const sensor_msgs::msg::Image::SharedPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    return cv_ptr->image;
}
//----------------------------------------------------------------------------------------
/**
 * @brief Convert an OpenCV BGR Mat to a ROS Image message.
 *
 * Wraps the cv::Mat in a sensor_msgs/Image with the provided header so the
 * output timestamp matches the original input frame.
 *
 * @param mat Annotated BGR image (e.g. with drawn bounding boxes).
 * @param header Header to stamp on the output (preserves original frame timestamp).
 * @return sensor_msgs::msg::Image::SharedPtr Ready-to-publish ROS image message.
 */
sensor_msgs::msg::Image::SharedPtr RosDetectionBridge::matToImageMsg(
    const cv::Mat& mat,
    const std_msgs::msg::Header& header)
{
    return cv_bridge::CvImage(header, "bgr8", mat).toImageMsg();
}
//----------------------------------------------------------------------------------------
/**
 * @brief Convert inference Objects to a Detection2DArray message.
 *
 * Each det::Object becomes one Detection2D containing:
 *   - Bounding box (center x/y, size w/h)
 *   - Class hypothesis (class name string + confidence score)
 *
 * This is the primary extension point for future model outputs.
 * When adding classification or multi-task inference, add results here.
 *
 * @param objs Detection results from YOLO26::PostProcess().
 * @param class_names Class label lookup table (indexed by Object::label).
 * @param header Header to stamp on the output (preserves original frame timestamp).
 * @return vision_msgs::msg::Detection2DArray Structured detection results.
 */
vision_msgs::msg::Detection2DArray RosDetectionBridge::objectsToDetections(
    const std::vector<det::Object>& objs,
    const std::vector<std::string>& class_names,
    const std_msgs::msg::Header& header)
{
    vision_msgs::msg::Detection2DArray det_array;
    det_array.header = header;

    for (const auto& obj : objs) {
        vision_msgs::msg::Detection2D det;

        // Bounding box: center + size (vision_msgs convention)
        det.bbox.center.position.x = obj.rect.x + obj.rect.width / 2.0;
        det.bbox.center.position.y = obj.rect.y + obj.rect.height / 2.0;
        det.bbox.size_x = obj.rect.width;
        det.bbox.size_y = obj.rect.height;

        // Class hypothesis: label name + confidence
        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        hyp.hypothesis.class_id =
            (obj.label >= 0 && obj.label < static_cast<int>(class_names.size()))
                ? class_names[obj.label]
                : ("class_" + std::to_string(obj.label));
        hyp.hypothesis.score = obj.prob;
        det.results.push_back(hyp);

        // Future extension point:
        // - Add classification results as additional hypotheses in det.results
        // - Add pose information via hyp.pose for 3D localization
        // - Add custom metadata fields for dashboard consumption

        det_array.detections.push_back(det);
    }

    return det_array;
}
//----------------------------------------------------------------------------------------
