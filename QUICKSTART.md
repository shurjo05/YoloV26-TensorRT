# YOLO26 Aphid Detector — Quickstart

## What's in this repo

Two ROS2 packages in one colcon workspace:
- **`yolo26_tensorrt`** — YOLO26 TensorRT inference node. Subscribes to an image topic, publishes `Detection2DArray`.
- **`detection_to_point`** — Converts 2D detections to 3D world-frame poses using the pinhole camera model and TF2. Subscribes to `Detection2DArray` + `camera_info`, publishes `PoseArray`.

---

## First-time setup on a new Jetson

### 1 — Clone the repo

```bash
git clone -b trt10-aphid https://github.com/shurjo05/YoloV26-TensorRT.git
cd YoloV26-TensorRT
```

### 2 — Build the Docker image (one-time)

```bash
docker build -t aphid-inference .
# Takes a few minutes on first run — downloads base image and installs deps.
```

### 3 — Build the TensorRT engine (one-time per device)

Engines are device-specific and not committed to the repo. Run from **outside** the container (host has trtexec):

```bash
cd src/yolo26_tensorrt
./scripts/build_engine.sh
# Saves to: src/yolo26_tensorrt/models/yolo26n-aphid.engine
# Takes 5–15 minutes.
```

### 4 — Update the engine path in params.yaml

Edit `src/yolo26_tensorrt/config/params.yaml` and set `engine_file_path` to the full path of the generated engine:

```yaml
engine_file_path: "/workspace/src/yolo26_tensorrt/models/yolo26n-aphid.engine"
```

Use the `/workspace/...` path since that's where the repo is mounted inside the container.

---

## Daily use — Docker (recommended)

### Enter the container

```bash
bash docker/run.sh
# Drops into a shell with ROS2 sourced and the repo mounted at /workspace.
```

### Build the ROS2 packages (inside container, after any source change)

```bash
colcon build
source install/setup.bash
```

> `install/setup.bash` is auto-sourced in future shells via `.bashrc`, but you need to source it manually once in the current shell after the first build.

---

## Launch options

### A — Inference only (testing with rosbag or image_publisher)

Use this when you don't have an OAK camera connected. Feed images via `image_publisher` or a rosbag.

**Terminal 1 — inference node:**
```bash
# Inside container
colcon build && source install/setup.bash
ros2 launch yolo26_tensorrt yolo_detector.launch.py
```

**Terminal 2 — publish a test image:**
```bash
# On host or second container shell
source /opt/ros/humble/setup.bash
ros2 run image_publisher image_publisher_node /path/to/image.jpg \
  --ros-args -p publish_rate:=5.0
```

**Terminal 3 — verify detections:**
```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /yolo_detector_node/detections
```
Expected: `class_id: aphid`, `score: 0.25+`, bbox center + size populated.

---

### B — Full pipeline with OAK camera

Use this when the OAK camera is physically connected. Launches the OAK driver, YOLO26 node, and DetectionToPointNode together.

**Requires:** `ros-humble-depthai-ros-driver` installed on the host or inside the container.

```bash
# Inside container
colcon build && source install/setup.bash
ros2 launch yolo26_tensorrt oak_inference.launch.py
```

Override object height or world frame at launch time:
```bash
ros2 launch yolo26_tensorrt oak_inference.launch.py \
  object_height:=0.004 \
  world_frame_id:=base_link
```

**Verify 3D poses:**
```bash
ros2 topic echo /detection_to_point_node/world_poses
```
Expected: `PoseArray` with `position.z` values representing distance in metres.

> **OAK frame ID gotcha:** If DetectionToPointNode isn't publishing, check that the static TF frame ID in `oak_inference.launch.py` matches what the OAK driver actually publishes. Run:
> ```bash
> ros2 topic echo /oak/rgb/camera_info --once
> ```
> and check `header.frame_id`. Update the `child_frame_id` in the static TF node in `oak_inference.launch.py` if they don't match.

---

### C — Standalone (no ROS2, headless testing)

Useful for quickly verifying the engine on a single image without launching ROS2.

```bash
# On host (no container needed), from repo root:
cd src/yolo26_tensorrt
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make yolo_standalone -j$(nproc)
./yolo_standalone ../models/yolo26n-aphid.engine /path/to/image.jpg --no-gui
# Output saved to repo root as out.jpg
```

---

## Key parameters

Edit `src/yolo26_tensorrt/config/params.yaml` to change YOLO26 node settings:

| Parameter | Default | Description |
|---|---|---|
| `engine_file_path` | `""` | Path to `.engine` file **(required)** |
| `confidence_threshold` | `0.25` | Minimum detection confidence |
| `input_width` / `input_height` | `640` | Must match the engine's input dimensions |
| `class_names` | `["aphid"]` | Label list indexed by class ID |
| `topk` | `100` | Max detections per frame |

DetectionToPointNode parameters are set in `oak_inference.launch.py` (or via `--ros-args` overrides):

| Parameter | Default | Description |
|---|---|---|
| `object_height` | `4.0E-3` | Real-world object height in metres (used for Z estimation) |
| `world_frame_id` | `base_link` | TF frame for output poses |
| `average_points` | `false` | Average all detections in a frame into one point |

---

## Rebuild after source changes

```bash
# Inside container
colcon build
source install/setup.bash
```

No Docker image rebuild needed — source is volume-mounted.

> **Always rebuild after editing `params.yaml`** — the config is copied into `install/` at build time.
