FROM dustynv/ros:humble-ros-base-l4t-r36.3.0

# Refresh the ROS2 apt signing key before apt-get update (key in base image may have expired).
# curl is already present in the dustynv base image — no apt install needed.
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
       -o /usr/share/keyrings/ros-archive-keyring.gpg

# Install ROS2 packages and build tools not included in the base image.
# --force-overwrite is needed because the dustynv base image has JetPack OpenCV 4.8
# already installed, which conflicts with Ubuntu's OpenCV 4.5 pulled in by cv_bridge.
RUN apt-get update && apt-get -o Dpkg::Options::="--force-overwrite" install -y --no-install-recommends \
    ros-humble-cv-bridge \
    ros-humble-vision-msgs \
    ros-humble-image-transport \
    ros-humble-image-publisher \
    ros-humble-image-geometry \
    ros-humble-tf2-ros \
    ros-humble-tf2-geometry-msgs \
    ros-humble-visualization-msgs \
    ros-humble-rclcpp-components \
    python3-colcon-common-extensions \
    ros-humble-depthai-ros-driver \
    ros-humble-depthai-ros \
    ros-humble-camera-calibration \
    libnvinfer-dev \
    libnvinfer-plugin-dev \
    iputils-ping \
    && rm -rf /var/lib/apt/lists/*

# Auto-source ROS2 in every shell session.
# Also source the colcon install overlay if it exists (i.e. after colcon build inside container).
RUN echo "source /opt/ros/humble/setup.bash" >> /root/.bashrc && \
    echo '[ -f /workspace/install/setup.bash ] && source /workspace/install/setup.bash' >> /root/.bashrc

# Repo is volume-mounted at runtime — no source copy needed.
WORKDIR /workspace

CMD ["/bin/bash"]