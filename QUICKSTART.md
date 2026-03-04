# YOLO26 Aphid Detector — Quickstart

## Prerequisites

- Jetson with JetPack 6 (Ubuntu 22.04, TensorRT 10.x, CUDA)
- ROS2 Humble

Install ROS2 dependencies (one-time per device):
```bash
sudo apt install ros-humble-ros-base python3-colcon-common-extensions
sudo apt install ros-humble-cv-bridge ros-humble-vision-msgs ros-humble-image-transport ros-humble-image-publisher
```

---

## Step 1 — Build the TensorRT engine (one-time per device)

Engines are device-specific and not committed to the repo. Build from the ONNX on each target Jetson:

```bash
./scripts/build_engine.sh
# Takes 5-15 minutes. Saves engine to models/yolo26n-aphid.engine
```

Then set `engine_file_path` in `config/params.yaml` to the full path of the generated engine file.

## Step 3 — Build the ROS2 package

```bash
# Deactivate any Python venv first
deactivate 2>/dev/null || true

source /opt/ros/humble/setup.bash
cd /path/to/ros2_ws
colcon build --packages-select yolo26_tensorrt
source install/setup.bash
```

> Rebuild whenever source files or `config/params.yaml` change.

## Step 4 — Launch the inference node

```bash
source /opt/ros/humble/setup.bash
source /path/to/ros2_ws/install/setup.bash
ros2 launch yolo26_tensorrt yolo_detector.launch.py
```

Override the engine path without rebuilding:
```bash
ros2 launch yolo26_tensorrt yolo_detector.launch.py \
  --ros-args -p yolo_detector_node:engine_file_path:=/path/to/engine.engine
```

---

## Testing

### Standalone (no ROS2)
```bash
cd YoloV8-TensorRT-Jetson_Nano
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make yolo_standalone -j$(nproc)
./yolo_standalone ../models/yolo26n-aphid.engine /path/to/image.jpg --no-gui out.jpg
```

### ROS2 pipeline (3 terminals)

**Terminal 1** — inference node (after Step 3–4 above):
```bash
ros2 launch yolo26_tensorrt yolo_detector.launch.py
```

**Terminal 2** — publish a test image:
```bash
source /opt/ros/humble/setup.bash
ros2 run image_publisher image_publisher_node /path/to/image.jpg \
  --ros-args -p publish_rate:=5.0
```

**Terminal 3** — verify detections:
```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /yolo_detector_node/detections
```

---

## Docker

Engines must still be built on the host before running the container (same GPU, same result either way).

Find your L4T version:
```bash
cat /etc/nv_tegra_release
```

Run the inference node in a container:
```bash
docker run -it --rm \
  --runtime nvidia \
  --network host \
  -v $(pwd):/workspace \
  dustynv/ros:humble-ros-base-l4t-r36.x.x \
  bash -c "
    source /opt/ros/humble/setup.bash &&
    cd /workspace/ros2_ws &&
    colcon build --packages-select yolo26_tensorrt &&
    source install/setup.bash &&
    ros2 launch yolo26_tensorrt yolo_detector.launch.py \
      --ros-args -p yolo_detector_node:engine_file_path:=/workspace/models/yolo26n-aphid.engine
  "
```

Replace `r36.x.x` with your actual L4T version (e.g. `r36.3.0`).
