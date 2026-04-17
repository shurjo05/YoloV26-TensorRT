//
// ROS ↔ RT-DETR type conversion bridge implementation.
//
// Mirrors yolo26_tensorrt/src/ros_detection_bridge.cpp.
//

#include "ros_detection_bridge.hpp"
#include "vision_msgs/msg/detection2_d.hpp"
#include "vision_msgs/msg/object_hypothesis_with_pose.hpp"

//----------------------------------------------------------------------------------------
cv::Mat RosDetectionBridge::imageMsgToMat(const sensor_msgs::msg::Image::ConstSharedPtr& msg)
{
    cv_bridge::CvImagePtr cv_ptr = cv_bridge::toCvCopy(msg, "bgr8");
    return cv_ptr->image;
}
//----------------------------------------------------------------------------------------
sensor_msgs::msg::Image::SharedPtr RosDetectionBridge::matToImageMsg(
    const cv::Mat& mat,
    const std_msgs::msg::Header& header)
{
    return cv_bridge::CvImage(header, "bgr8", mat).toImageMsg();
}
//----------------------------------------------------------------------------------------
vision_msgs::msg::Detection2DArray RosDetectionBridge::objectsToDetections(
    const std::vector<det::Object>& objs,
    const std::vector<std::string>& class_names,
    const std_msgs::msg::Header& header)
{
    vision_msgs::msg::Detection2DArray det_array;
    det_array.header = header;

    for (const auto& obj : objs) {
        vision_msgs::msg::Detection2D det;

        det.bbox.center.position.x = obj.rect.x + obj.rect.width / 2.0;
        det.bbox.center.position.y = obj.rect.y + obj.rect.height / 2.0;
        det.bbox.size_x = obj.rect.width;
        det.bbox.size_y = obj.rect.height;

        vision_msgs::msg::ObjectHypothesisWithPose hyp;
        hyp.hypothesis.class_id =
            (obj.label >= 0 && obj.label < static_cast<int>(class_names.size()))
                ? class_names[obj.label]
                : ("class_" + std::to_string(obj.label));
        hyp.hypothesis.score = obj.prob;
        det.results.push_back(hyp);

        det_array.detections.push_back(det);
    }

    return det_array;
}
//----------------------------------------------------------------------------------------
