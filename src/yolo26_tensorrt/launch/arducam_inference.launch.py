# -------------------------------------------------------------------------------------
# arducam_inference.launch.py
#
# Full inference pipeline: Arducam AR0234 (UC-788) --> YOLO26 TensorRT --> DetectionToPoint
#
# Camera hardware:
#   Luxonis OAK-D PoE with Arducam AR0234 (UC-788) sensor module.
#   Connects over Ethernet at 169.100.100.200. RGB-only pipeline (no stereo).
#
# Data flow:
#   [depthai_ros_driver (Arducam via OAK-D PoE)]
#     |-- /arducam/rgb/image_raw -------> [yolo_detector_node]
#     |-- /arducam/rgb/camera_info -----> [detection_to_point_node]
#
#   [yolo_detector_node]
#     |-- ~/detections -----------------> [detection_to_point_node]
#
#   [detection_to_point_node]
#     |-- world_poses ------------------> (available for downstream consumers)
#     |-- marker -----------------------> (RViz debug visualisation)
# -------------------------------------------------------------------------------------

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    yolo_share = get_package_share_directory('yolo26_tensorrt')

    yolo_params     = os.path.join(yolo_share, 'config', 'params.yaml')
    arducam_params  = os.path.join(yolo_share, 'config', 'arducam_camera_params.yaml')
    # Calibration injected at launch time — depthai requires absolute file:// path.
    calibration     = 'file://' + os.path.join(yolo_share, 'config', 'arducam_uc788_calibration.yaml')

    object_height_arg = DeclareLaunchArgument(
        'object_height',
        default_value='4.0E-3',
        description='Real-world object height in metres (e.g. 4mm = 4.0E-3 for aphid)'
    )

    world_frame_arg = DeclareLaunchArgument(
        'world_frame_id',
        default_value='base_link',
        description='Target TF frame for world-coordinate output'
    )

    # --- Arducam AR0234 via OAK-FFC-4P-POE (Python depthai publisher) ---
    # depthai_ros_driver C++ (v2 and v3) segfaults on OAK-FFC-4P-POE.
    # Python depthai connects directly via IP and publishes /arducam/rgb/image_raw.
    depthai_node = Node(
        package="yolo26_tensorrt",
        executable="arducam_publisher.py",
        name="arducam_publisher",
        parameters=[{'ip': '169.254.1.222'}],
        output="screen",
    )


    # --- YOLO26 Inference Node ---
    yolo_node = Node(
        package='yolo26_tensorrt',
        executable='yolo_detector_node',
        name='yolo_detector_node',
        parameters=[yolo_params],
        remappings=[
            ('image_raw', '/arducam/rgb/image_raw'),
        ],
        output='screen',
    )

    # --- Detection To Point Node ---
    detection_to_point_node = Node(
        package='detection_to_point',
        executable='detection_to_point_node',
        name='detection_to_point_node',
        parameters=[{
            'object_height': LaunchConfiguration('object_height'),
            'object_length': LaunchConfiguration('object_height'),
            'average_points': False,
            'world_frame_id': LaunchConfiguration('world_frame_id'),
        }],
        remappings=[
            ('camera_info', '/arducam/rgb/camera_info'),
            ('detection', '/yolo_detector_node/detections'),
        ],
        output='screen',
    )

    # --- Foxglove Bridge --- WebSocket on port 8765 for Foxglove Studio ---
    foxglove = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{'port': 8765}],
        output='screen',
    )

    # --- Static TF: base_link -> arducam optical frame ---
    # NOTE: verify the actual frame ID published by depthai with:
    #   ros2 topic echo /arducam/rgb/camera_info --once
    # and check header.frame_id. Update child frame below if it differs.
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_to_base_tf',
        arguments=[
            '0', '0', '0',
            '0', '0', '0',
            'base_link',
            'arducam_rgb_camera_optical_frame',
        ],
        output='screen',
    )

    return LaunchDescription([
        object_height_arg,
        world_frame_arg,
        depthai_node,
        yolo_node,
        detection_to_point_node,
        static_tf,
        foxglove,
    ])
