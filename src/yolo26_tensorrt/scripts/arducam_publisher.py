#!/usr/bin/env python3
# -------------------------------------------------------------------------------------
# arducam_publisher.py
#
# Python ROS2 node that publishes frames from the Arducam AR0234 (UC-788) on an
# OAK-FFC-4P-POE over PoE, using the Python depthai API directly.
#
# Replaces depthai_ros_driver/camera_node which segfaults on this device.
# Python depthai connects successfully; the C++ driver does not.
#
# Publishes:
#   /arducam/rgb/image_raw   (sensor_msgs/Image,      RGB8, 960x600)
#   /arducam/rgb/camera_info (sensor_msgs/CameraInfo, calibration from yaml values)
# -------------------------------------------------------------------------------------

import threading

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo

import depthai as dai
import numpy as np


class ArducamPublisher(Node):

    def __init__(self):
        super().__init__('arducam')

        self.declare_parameter('ip', '169.254.1.222')
        self.declare_parameter('fps', 30.0)

        ip  = self.get_parameter('ip').get_parameter_value().string_value
        fps = self.get_parameter('fps').get_parameter_value().double_value

        self._image_pub = self.create_publisher(Image,      '/arducam/rgb/image_raw',   10)
        self._info_pub  = self.create_publisher(CameraInfo, '/arducam/rgb/camera_info', 10)
        self._camera_info = self._build_camera_info()

        pipeline = dai.Pipeline()

        cam = pipeline.create(dai.node.ColorCamera)
        cam.setBoardSocket(dai.CameraBoardSocket.CAM_A)
        cam.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1200_P)
        cam.setFps(fps)
        cam.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        # Scale down on-device before sending to host — full 1920x1200 NV12→BGR
        # conversion in Python is too slow (~0.2 Hz). Half resolution = 4x less data.
        cam.setIspScale(1, 2)   # 1920x1200 → 960x600

        xout = pipeline.create(dai.node.XLinkOut)
        xout.setStreamName('rgb')
        cam.isp.link(xout.input)

        device_info = dai.DeviceInfo(ip)
        self._device = dai.Device(pipeline, device_info)
        self._queue  = self._device.getOutputQueue('rgb', maxSize=4, blocking=False)

        self._running = True
        self._thread = threading.Thread(target=self._capture_loop, daemon=True)
        self._thread.start()
        self.get_logger().info(f'Arducam publisher connected to {ip} at {fps} fps')

    # ---------------------------------------------------------------------------------

    def _capture_loop(self):
        while self._running:
            frame_data = self._queue.get()   # blocking — wakes up when a frame arrives
            if frame_data is None:
                continue
            self._publish_frame(frame_data)

    def _publish_frame(self, frame_data):

        frame = frame_data.getCvFrame()   # numpy array, BGR ordering from depthai
        frame = frame[:, :, ::-1]         # BGR → RGB (reverse channel axis)
        stamp = self.get_clock().now().to_msg()

        img_msg              = Image()
        img_msg.header.stamp = stamp
        img_msg.header.frame_id = 'arducam_rgb_camera_optical_frame'
        img_msg.height       = frame.shape[0]
        img_msg.width        = frame.shape[1]
        img_msg.encoding     = 'rgb8'
        img_msg.is_bigendian = False
        img_msg.step         = frame.shape[1] * 3
        img_msg.data         = frame.tobytes()

        self._camera_info.header.stamp = stamp

        self._image_pub.publish(img_msg)
        self._info_pub.publish(self._camera_info)

    # ---------------------------------------------------------------------------------

    def _build_camera_info(self):
        """Build CameraInfo from arducam_uc788_calibration.yaml values."""
        info = CameraInfo()
        info.header.frame_id  = 'arducam_rgb_camera_optical_frame'
        # Calibration scaled from 1920x1200 → 960x600 (setIspScale 1/2).
        # Focal lengths and principal point scale linearly with resolution.
        info.width            = 960
        info.height           = 600
        info.distortion_model = 'plumb_bob'
        info.k = [594.21363, 0.0,       475.03598,
                  0.0,       601.36985, 280.65202,
                  0.0,       0.0,       1.0]
        info.d = [0.042264, 0.001751, -0.010125, -0.001833, 0.0]
        info.r = [1.0, 0.0, 0.0,
                  0.0, 1.0, 0.0,
                  0.0, 0.0, 1.0]
        info.p = [619.67646, 0.0,       472.98940, 0.0,
                  0.0,       622.68073, 273.27551, 0.0,
                  0.0,       0.0,       1.0,       0.0]
        return info

    # ---------------------------------------------------------------------------------

    def destroy_node(self):
        self._running = False
        self._thread.join(timeout=2.0)
        self._device.close()
        super().destroy_node()


# -------------------------------------------------------------------------------------

def main():
    rclpy.init()
    node = ArducamPublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
