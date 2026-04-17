# -------------------------------------------------------------------------------------
# arducam_rtdetr.launch.py
#
# Full inference pipeline using the RT-DETR detector:
#   Arducam AR0234 (UC-788) --> RT-DETR TensorRT --> DetectionToPoint
#
# Mirrors arducam_inference.launch.py (yolo26_tensorrt) but swaps YOLO26 for RT-DETR.
# Reuses the arducam_publisher.py Python node from yolo26_tensorrt so we don't
# duplicate camera drivers between packages.
# -------------------------------------------------------------------------------------

import os

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    rtdetr_share = get_package_share_directory('rtdetr_tensorrt')
    yolo_share   = get_package_share_directory('yolo26_tensorrt')

    rtdetr_params = os.path.join(rtdetr_share, 'config', 'params.yaml')

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

    # --- Arducam AR0234 via OAK-FFC-4P-POE (reuses publisher from yolo26_tensorrt) ---
    depthai_node = Node(
        package="yolo26_tensorrt",
        executable="arducam_publisher.py",
        name="arducam_publisher",
        parameters=[{'ip': '169.254.1.222'}],
        output="screen",
    )

    # --- RT-DETR Inference Node ---
    rtdetr_node = Node(
        package='rtdetr_tensorrt',
        executable='rtdetr_detector_node',
        name='rtdetr_detector_node',
        parameters=[rtdetr_params],
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
            ('detection',   '/rtdetr_detector_node/detections'),
        ],
        output='screen',
    )

    # --- Foxglove Bridge ---
    foxglove = Node(
        package='foxglove_bridge',
        executable='foxglove_bridge',
        name='foxglove_bridge',
        parameters=[{'port': 8765}],
        output='screen',
    )

    # --- Static TFs (identical to arducam_inference.launch.py) ---
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

    static_tf_base_to_camera = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_camera_tf',
        arguments=[
            '--x',     '0.115',
            '--y',     '-0.398',
            '--z',     '0.222',
            '--roll',  '0',
            '--pitch', '0',
            '--yaw',   '-1.5708',
            '--frame-id',       'base_link',
            '--child-frame-id', 'camera_link',
        ],
        output='screen',
    )

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
        rtdetr_node,
        detection_to_point_node,
        static_tf_body_to_base,
        static_tf_base_to_camera,
        static_tf_camera_to_optical,
        foxglove,
    ])
