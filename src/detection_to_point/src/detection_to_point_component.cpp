// -------------------------------------------------------------------------------------
// detection_to_point_component.cpp
//
// Implementation of DetectionToPointNode — converts 2D bounding-box detections into
// 3D world-frame points using the pinhole camera model and TF2.
//
// -------------------------------------------------------------------------------------

#include <cstdio>

#include <detection_to_point/detection_to_point_node.hpp>
#include "tf2/exceptions.h"
#include "tf2_geometry_msgs/tf2_geometry_msgs.hpp"
#include "rclcpp_components/register_node_macro.hpp"

// -------------------------------------------------------------------------------------

/**
 * @brief Construct the DetectionToPointNode.
 *
 * Declares ROS2 parameters (object_height, object_length, average_points,
 * world_frame_id), creates the camera_info subscription, initialises
 * publishers, and sets up the TF2 buffer + listener.
 *
 * @param options  Node options forwarded by the component loader.
 */
DetectionToPointNode::DetectionToPointNode(const rclcpp::NodeOptions & options)
: Node("detection_to_point_node", options)
{
  // Declare and read parameters
  this->declare_parameter("object_height", 2.0E-3);
  this->get_parameter("object_height", object_height_);

  this->declare_parameter("object_length", 2.0E-3);
  this->get_parameter("object_length", object_length_);

  this->declare_parameter("average_points", false);
  this->get_parameter("average_points", average_points_);

  this->declare_parameter("world_frame_id", "map");
  this->get_parameter("world_frame_id", world_frame_id_);

  // Subscribe to camera_info — will be reset after the first message
  camera_info_sub_ = this->create_subscription<sensor_msgs::msg::CameraInfo>(
    "camera_info", 10,
    std::bind(&DetectionToPointNode::cameraInfoCallback, this, std::placeholders::_1));

  // Publishers
  marker_pub_     = this->create_publisher<visualization_msgs::msg::Marker>("marker", 10);
  world_pose_pub_ = this->create_publisher<geometry_msgs::msg::PoseArray>("world_poses", 10);

  // TF2
  tf_buffer_   = std::make_unique<tf2_ros::Buffer>(this->get_clock());
  tf_listener_ = std::make_unique<tf2_ros::TransformListener>(*tf_buffer_);
}

// -------------------------------------------------------------------------------------

/**
 * @brief Destructor — logs a shutdown message.
 */
DetectionToPointNode::~DetectionToPointNode()
{
  RCLCPP_INFO_STREAM(this->get_logger(), "Shutting down detection to point node\n");
}

// -------------------------------------------------------------------------------------

/**
 * @brief Callback for the first CameraInfo message.
 *
 * Caches the pinhole camera model (focal length, principal point), stores the
 * camera optical frame ID, tears down the camera_info subscription (no longer
 * needed), and creates the detection subscription so processing can begin.
 *
 * @param msg  CameraInfo containing the intrinsic matrix K.
 */
void DetectionToPointNode::cameraInfoCallback(
  const sensor_msgs::msg::CameraInfo::SharedPtr msg)
{
  // Initialise the pinhole model from the intrinsics
  camera_model_.fromCameraInfo(msg);
  camera_frame_id_     = msg->header.frame_id;
  camera_info_received_ = true;

  // No longer need camera_info — drop the subscription
  RCLCPP_INFO_STREAM(this->get_logger(),
    "Camera info received, shutting down camera info subscriber");
  camera_info_sub_.reset();

  // Now subscribe to detections
  detection_sub_ = this->create_subscription<vision_msgs::msg::Detection2DArray>(
    "detection", 10,
    std::bind(&DetectionToPointNode::detectionCallback, this, std::placeholders::_1));
  RCLCPP_INFO_STREAM(this->get_logger(), "Subscribed to detection topic");
}

// -------------------------------------------------------------------------------------

/**
 * @brief Processes each Detection2DArray frame.
 *
 * For every detection:
 *   1. Extracts bbox centre (u, v) and pixel height (H_image).
 *   2. Projects pixel to 3D ray via the cached pinhole model.
 *   3. Computes depth:  Z = (H_real * fy) / H_image.
 *   4. Scales the ray to obtain a 3D point in the camera frame.
 *   5. Transforms to the world frame via TF2.
 *
 * If average_points_ is true, all detections in the frame are averaged into a
 * single world-frame point.  Otherwise each detection produces its own pose.
 *
 * @param msg  Array of 2D detections from the inference node.
 */
void DetectionToPointNode::detectionCallback(
  const vision_msgs::msg::Detection2DArray::UniquePtr msg)
{
  // Prepare the output PoseArray header
  world_poses_msg_.header.stamp    = msg->header.stamp;
  world_poses_msg_.header.frame_id = world_frame_id_;

  geometry_msgs::msg::PointStamped point;
  geometry_msgs::msg::PointStamped point_avg;
  geometry_msgs::msg::PointStamped world_point;

  point.header.stamp    = msg->header.stamp;
  point.header.frame_id = camera_frame_id_;

  cv::Point2d uv;
  cv::Point3d ray;

  // -- RViz marker setup --
  visualization_msgs::msg::Marker marker;
  marker.header.stamp    = msg->header.stamp;
  marker.header.frame_id = camera_frame_id_;
  marker.type            = visualization_msgs::msg::Marker::POINTS;
  marker.action          = visualization_msgs::msg::Marker::MODIFY;
  marker.ns              = "detection_points";
  marker.id              = 0;
  marker.lifetime        = rclcpp::Duration(0, 0);
  marker.color.r         = 0.8;
  marker.color.g         = 0.0;
  marker.color.b         = 0.2;
  marker.color.a         = 1.0;
  marker.pose.orientation.w = 1.0;

  if (average_points_) {
    marker.scale.x = object_length_ * 15;
    marker.scale.y = object_height_ * 15;
  } else {
    marker.scale.x = object_length_ * 5;
    marker.scale.y = object_height_ * 5;
  }
  marker.scale.z = 10E-3;

  // Known real-world object height and camera focal length
  double H_real = object_height_;
  double F      = camera_model_.fy();

  geometry_msgs::msg::Pose pose;
  pose.orientation.w = 1.0;

  // -- Process each detection --
  for (const auto & detection : msg->detections) {
    // 1. Bbox centre in image coordinates
    auto image_point = detection.bbox.center.position;
    uv.x = image_point.x;
    uv.y = image_point.y;

    // 2. Project pixel to normalised 3D ray
    ray = camera_model_.projectPixelTo3dRay(uv);

    // 3. Estimate depth from bbox pixel height
    double H_image = detection.bbox.size_y;
    double Z       = H_real * F / H_image;

    // 4. Scale ray to get camera-frame 3D point
    point.point.x = ray.x * Z;
    point.point.y = ray.y * Z;
    point.point.z = ray.z * Z;

    if (!average_points_) {
      // Individual point mode — transform each detection to world frame
      marker.points.push_back(point.point);
      try {
        geometry_msgs::msg::TransformStamped transform =
          tf_buffer_->lookupTransform(world_frame_id_, camera_frame_id_, rclcpp::Time(0));
        tf2::doTransform(point, world_point, transform);
        pose.position = world_point.point;
        world_poses_msg_.poses.push_back(pose);
      } catch (tf2::TransformException & ex) {
        RCLCPP_ERROR_STREAM(this->get_logger(),
          "Transform error : " << camera_frame_id_ << " to "
          << world_frame_id_ << ": " << ex.what());
      }
    } else {
      // Averaging mode — accumulate
      point_avg.point.x += point.point.x;
      point_avg.point.y += point.point.y;
      point_avg.point.z += point.point.z;
    }
  }

  // -- If averaging, compute mean and transform once --
  if (average_points_ && msg->detections.size() > 0) {
    point_avg.point.x /= msg->detections.size();
    point_avg.point.y /= msg->detections.size();
    point_avg.point.z /= msg->detections.size();

    marker.points.push_back(point_avg.point);

    try {
      geometry_msgs::msg::TransformStamped transform =
        tf_buffer_->lookupTransform(world_frame_id_, camera_frame_id_, rclcpp::Time(0));
      tf2::doTransform(point_avg, world_point, transform);
      pose.position = world_point.point;
      world_poses_msg_.poses.push_back(pose);
    } catch (tf2::TransformException & ex) {
      RCLCPP_ERROR_STREAM(this->get_logger(),
        "Transform error : " << camera_frame_id_ << " to "
        << world_frame_id_ << ": " << ex.what());
    }
  }

  // Publish and clear for next frame
  world_pose_pub_->publish(world_poses_msg_);
  world_poses_msg_.poses.clear();
  marker_pub_->publish(marker);
}

// -------------------------------------------------------------------------------------

RCLCPP_COMPONENTS_REGISTER_NODE(DetectionToPointNode)
