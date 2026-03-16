// -------------------------------------------------------------------------------------
// detection_to_point_node.hpp
//
// Converts 2D bounding-box detections into 3D world-frame points using the pinhole
// camera model.  Depth (Z) is estimated from the known real-world object height and
// the bounding-box pixel height:  Z = (H_real * fy) / H_image.
//
// -------------------------------------------------------------------------------------

#ifndef DETECTION_TO_POINT__DETECTION_TO_POINT_NODE_HPP_
#define DETECTION_TO_POINT__DETECTION_TO_POINT_NODE_HPP_

#include <string>

#include <rclcpp/rclcpp.hpp>
#include "image_geometry/pinhole_camera_model.h"
#include "geometry_msgs/msg/point_stamped.hpp"
#include "geometry_msgs/msg/pose_array.hpp"
#include "geometry_msgs/msg/pose.hpp"
#include "sensor_msgs/msg/camera_info.hpp"
#include "vision_msgs/msg/detection2_d_array.hpp"
#include "visualization_msgs/msg/marker.hpp"
#include "tf2_ros/transform_listener.h"
#include "tf2_ros/buffer.h"

// -------------------------------------------------------------------------------------

/**
 * @brief ROS2 component that projects 2D detections into 3D world-frame points.
 *
 * Subscribes to:
 *   - camera_info  (sensor_msgs/CameraInfo)   — once, to obtain focal length
 *   - detection    (vision_msgs/Detection2DArray) — continuous, from any detector
 *
 * Publishes:
 *   - world_poses  (geometry_msgs/PoseArray)  — 3D points in the world frame
 *   - marker       (visualization_msgs/Marker) — RViz debug visualisation
 *
 * The depth for each detection is computed internally using:
 *   Z = (object_height * focal_length_y) / bbox_pixel_height
 *
 * The resulting camera-frame 3D point is then transformed to the world frame
 * via a TF2 lookup (camera_frame → world_frame_id).
 */
class DetectionToPointNode : public rclcpp::Node
{
public:
  /**
   * @brief Construct the node, declare parameters, and create subscriptions.
   * @param options  Node options forwarded by the component loader.
   */
  explicit DetectionToPointNode(const rclcpp::NodeOptions & options);

  /**
   * @brief Destructor — logs shutdown message.
   */
  ~DetectionToPointNode();

private:
  // ---- Callbacks -----------------------------------------------------------

  /**
   * @brief Receives CameraInfo once, caches the pinhole model, then subscribes
   *        to the detection topic.
   * @param msg  Camera intrinsics (focal length, principal point, distortion).
   */
  void cameraInfoCallback(const sensor_msgs::msg::CameraInfo::SharedPtr msg);

  /**
   * @brief Processes each Detection2DArray frame: computes 3D camera-frame
   *        points, transforms them to the world frame, and publishes the
   *        resulting PoseArray + RViz markers.
   * @param msg  Array of 2D bounding-box detections from the inference node.
   */
  void detectionCallback(const vision_msgs::msg::Detection2DArray::UniquePtr msg);

  // ---- Members -------------------------------------------------------------

  /// Pinhole camera model initialised from the first CameraInfo message.
  image_geometry::PinholeCameraModel camera_model_;

  /// Subscription to camera_info (reset after first message).
  rclcpp::Subscription<sensor_msgs::msg::CameraInfo>::SharedPtr camera_info_sub_;

  /// Subscription to the detection topic (created after camera_info is received).
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr detection_sub_;

  /// Publisher for RViz point markers in the camera frame.
  rclcpp::Publisher<visualization_msgs::msg::Marker>::SharedPtr marker_pub_;

  /// Publisher for 3D poses in the world frame.
  rclcpp::Publisher<geometry_msgs::msg::PoseArray>::SharedPtr world_pose_pub_;

  /// Whether camera_info has been received and cached.
  bool camera_info_received_ = false;

  /// If true, average all detections in a frame into a single point.
  bool average_points_ = false;

  /// Camera optical frame ID (read from the CameraInfo header).
  std::string camera_frame_id_;

  /// Real-world vertical extent of the detected object (metres).
  double object_height_;

  /// Real-world horizontal extent of the detected object (metres).
  double object_length_;

  /// Target TF frame for world-coordinate output (e.g. "odom", "base_link").
  std::string world_frame_id_;

  /// TF2 buffer for coordinate-frame lookups.
  std::shared_ptr<tf2_ros::Buffer> tf_buffer_;

  /// TF2 listener that populates the buffer.
  std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

  /// Reusable PoseArray message (cleared after each publish).
  geometry_msgs::msg::PoseArray world_poses_msg_;
};

#endif  // DETECTION_TO_POINT__DETECTION_TO_POINT_NODE_HPP_
