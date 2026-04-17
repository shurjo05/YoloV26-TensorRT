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
    # depthai_ros_driver C++ (v2 and v3) cannot find the device on OAK-FFC-4P-POE.
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
            ('image_raw/compressed', '/arducam/rgb/image_raw/compressed'),
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

    # --- Static TF: body → base_link ---
    # The Box robot URDF publishes 'body' as its root frame, but the rest of the
    # stack (nav2, localization, detection) expects 'base_link'. This zero-offset
    # identity bridge makes both conventions work without modifying hardware_ws.
    static_tf_body_to_base = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='body_to_base_link_tf',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll', '0', '--pitch', '0', '--yaw', '0',
            '--frame-id',       'body',
            '--child-frame-id', 'base_link',
        ],
        output='screen',
    )

    # --- Static TF: base_link → camera_link ---
    # Physical mount offset: 0.398m forward, 0.115m right, 0.222m up.
    # Yaw -90° because camera faces right (+X of robot = -90° yaw in base_link).
    static_tf_base_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        arguments=[
            '--x',     '0.115',    # 4.538in forward
            '--y',     '-0.398',   # 15.677in to the right
            '--z',     '0.222',    # 8.732in up
            '--roll',  '0',
            '--pitch', '0',
            '--yaw',   '-1.5708',  # camera faces right
            '--frame-id',       'base_link',
            '--child-frame-id', 'camera_link',
        ],
        output='screen',
    )

    # --- Static TF: camera_link → arducam_rgb_camera_optical_frame ---
    # Standard ROS optical frame convention: X right, Y down, Z forward.
    static_tf_camera_to_optical = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='camera_to_optical_tf',
        arguments=[
            '--x', '0', '--y', '0', '--z', '0',
            '--roll',  '-1.5708',
            '--pitch', '0',
            '--yaw',   '-1.5708',
            '--frame-id',       'camera_link',
            '--child-frame-id', 'arducam_rgb_camera_optical_frame',
        ],
        output='screen',
    )

    return LaunchDescription([
        object_height_arg,
        world_frame_arg,
        depthai_node,
        yolo_node,
        detection_to_point_node,
        static_tf_body_to_base,
        static_tf_base_to_camera,
        static_tf_camera_to_optical,
        foxglove,
    ])
