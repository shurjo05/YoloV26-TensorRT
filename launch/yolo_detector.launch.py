import os
from launch import LaunchDescription
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory('yolo26_tensorrt'),
        'config',
        'params.yaml'
    )

    return LaunchDescription([
        Node(
            package='yolo26_tensorrt',
            executable='yolo_detector_node',
            name='yolo_detector_node',
            parameters=[config],
            output='screen',
        ),
    ])
