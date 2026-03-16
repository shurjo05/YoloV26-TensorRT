# -------------------------------------------------------------------------------------
# oak_inference.launch.py
#
# Full inference pipeline: OAK camera --> YOLO26 TensorRT --> DetectionToPointNode.
#
# Data flow:
#   [OAK Camera (depthai_ros_driver)]
#     |-- /oak/rgb/image_raw ---------> [yolo_detector_node]
#     |-- /oak/rgb/camera_info -------> [detection_to_point_node]
#
#   [yolo_detector_node]
#     |-- ~/detections ---------------> [detection_to_point_node]
#
#   [detection_to_point_node]
#     |-- world_poses ----------------> (available for downstream consumers)
#     |-- marker ---------------------> (RViz debug visualisation)
#
#   [static_transform_publisher]
#     |-- base_link <-> oak camera frame (placeholder until full URDF)
# -------------------------------------------------------------------------------------

import os

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    # --- Package share directories ---
    yolo_share = get_package_share_directory('yolo26_tensorrt')

    # --- Config file paths ---
    yolo_params = os.path.join(yolo_share, 'config', 'params.yaml')
    oak_params  = os.path.join(yolo_share, 'config', 'oak_camera_params.yaml')

    # --- Launch arguments ---
    # Object height in metres (used for Z depth estimation)
    object_height_arg = DeclareLaunchArgument(
        'object_height',
        default_value='4.0E-3',
        description='Real-world object height in metres (e.g. 4mm = 4.0E-3 for aphid)'
    )

    # World frame for TF transform output
    world_frame_arg = DeclareLaunchArgument(
        'world_frame_id',
        default_value='base_link',
        description='Target TF frame for world-coordinate output'
    )

    # --- OAK Camera (depthai_ros_driver) ---
    # Launches the DepthAI ROS2 driver with our config.
    # Publishes: /oak/rgb/image_raw, /oak/rgb/camera_info
    # NOTE: if depthai_ros_driver is not installed, this will fail at runtime.
    #       Install via: sudo apt install ros-humble-depthai-ros-driver
    #       Or build from source: https://github.com/luxonis/depthai-ros
    oak_camera = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            get_package_share_directory('depthai_ros_driver'),
            '/launch/camera.launch.py'
        ]),
        launch_arguments={
            'params_file': oak_params,
        }.items()
    )

    # --- YOLO26 Inference Node ---
    # Subscribes to image, runs TensorRT inference, publishes Detection2DArray.
    yolo_node = Node(
        package='yolo26_tensorrt',
        executable='yolo_detector_node',
        name='yolo_detector_node',
        parameters=[yolo_params],
        remappings=[
            ('image_raw', '/oak/rgb/image_raw'),
        ],
        output='screen',
    )

    # --- Detection To Point Node ---
    # Subscribes to Detection2DArray + CameraInfo, publishes PoseArray with Z depth.
    detection_to_point_node = Node(
        package='detection_to_point',
        executable='detection_to_point_node',
        name='detection_to_point_node',
        parameters=[{
            'object_height': LaunchConfiguration('object_height'),
            'object_length': LaunchConfiguration('object_height'),  # same as height for aphid
            'average_points': False,
            'world_frame_id': LaunchConfiguration('world_frame_id'),
        }],
        remappings=[
            ('camera_info', '/oak/rgb/camera_info'),
            ('detection', '/yolo_detector_node/detections'),
        ],
        output='screen',
    )

    # --- Static TF: base_link -> camera optical frame ---
    # Placeholder transform until the full robot URDF is integrated.
    # The camera frame ID is published by the OAK driver in the camera_info header.
    # Common OAK frame IDs: "oak_rgb_camera_optical_frame", "oak-d_rgb_camera_optical_frame"
    # Adjust the child_frame_id below to match what your OAK driver publishes.
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_to_base_tf',
        arguments=[
            '0', '0', '0',       # x, y, z translation (metres)
            '0', '0', '0',       # roll, pitch, yaw (radians)
            'base_link',         # parent frame
            'oak_rgb_camera_optical_frame',  # child frame (adjust if needed)
        ],
        output='screen',
    )

    return LaunchDescription([
        object_height_arg,
        world_frame_arg,
        oak_camera,
        yolo_node,
        detection_to_point_node,
        static_tf,
    ])
