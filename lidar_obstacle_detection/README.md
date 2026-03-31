# `lidar_obstacle_detection`

ROS 2 package: single node **`lidar_cloud_ingress`** that ingests a LiDAR (or any) `sensor_msgs/PointCloud2`, applies a rigid transform into the robot base frame, spatially filters and voxel-downsamples, optionally merges scans over time, and optionally runs surface-normal segmentation + DBSCAN to publish **`lidar_obstacle_detection_msgs/ObstacleList`**.

Companion definitions live in the **`lidar_obstacle_detection_msgs`** package (see its `README.md`).

---

## Python dependencies

| Source | What to install |
|--------|------------------|
| **Pip (numeric / 3D stack)** | `requirements.txt` in this package: `python3 -m pip install -r requirements.txt` (lists `numpy`, `open3d`, `scikit-learn`; versions are minimums). Same packages are mirrored in `setup.py` `install_requires` for `pip install` of the package itself. |
| **ROS 2** | `rclpy`, `sensor_msgs`, `geometry_msgs`, `std_msgs`, `visualization_msgs`, `tf2_ros`, `sensor_msgs_py`, `rcl_interfaces`, etc. — use **apt** / **`rosdep`** for your distro (`package.xml` lists dependencies). |
| **Messages** | Build **`lidar_obstacle_detection_msgs`** with colcon; no pip package for the `.msg` types. |

After install, `requirements.txt` is also copied to  
`$(ros2 pkg prefix lidar_obstacle_detection)/share/lidar_obstacle_detection/requirements.txt`.

---

## Algorithm and tuning (config file)

All tuning is done through ROS parameters. Defaults are declared in code to match the shipped YAML.

| Location | Purpose |
|----------|---------|
| **Source tree** | `lidar_obstacle_detection/config/lidar_cloud_ingress.yaml` |
| **After `colcon install`** | `$(ros2 pkg prefix lidar_obstacle_detection)/share/lidar_obstacle_detection/config/lidar_cloud_ingress.yaml` |

**Launch** loads that file by default (`launch/lidar_cloud_ingress.launch.py`, argument `ingress_params_file`). Override with:

```bash
ros2 run lidar_obstacle_detection lidar_cloud_ingress --ros-args \
  --params-file /path/to/your.yaml
```

The YAML documents groups: logging, topics, spatial filter, temporal merge, static TF (visualization), perception (normals / DBSCAN / obstacle outputs), QoS, and RViz marker sizing.

---

## Input and output topics

Names below are **defaults** from the node (`declare_parameter`); your config file can change any `*_topic` value. Message types are fixed.

### Subscriptions

| Default topic | Type | Role |
|---------------|------|------|
| `/utlidar/cloud` | `sensor_msgs/PointCloud2` | Incoming cloud (`input_topic`). Driver `header.frame_id` is not rewritten; transform uses mount/driver extrinsics in parameters. |

QoS: `input_qos_reliability` / `input_qos_depth` (default best effort, depth 5).

### Publishers (always when the node runs)

| Default topic | Type | Role |
|---------------|------|------|
| `/lidar_obstacle_detection/cloud_in_base` | `sensor_msgs/PointCloud2` | Filtered (and optionally temporally merged) cloud in `output_cloud_frame_id`, xyz float32 (`output_topic`). |

QoS: `output_cloud_qos_reliability` / `output_cloud_qos_depth`.

### Output message stamps (`output_stamps_use_node_time`)

By default the published cloud (and perception outputs that share its header) copy **`header.stamp` from the driver cloud**. RViz evaluates TF at that stamp when the **fixed frame** is not the cloud frame (e.g. fixed **`odom`**, cloud in **`base_link`**), so the transform **`odom` → `base_link`** must exist at the **same time** as that stamp. If your state estimator stamps TF on a different clock domain than the LiDAR driver, set **`output_stamps_use_node_time: true`** so outputs use **`get_clock().now()`** at publish time instead.

**Rosbag playback:** publish **`/clock`** with `ros2 bag play <bag> --clock`, and run this node (plus RViz, localization, etc.) with **`--ros-args -p use_sim_time:=true`** so **`now()`** and recorded **`/tf`** stay aligned. Static transforms on **`/tf_static`** may appear with zero-like times in **`view_frames`**; that is normal and is not the usual cause of **`odom`** fixed-frame failures.

### Optional: static TF (visualization)

If `publish_lidar_mount_static_tf` is true, the node publishes **`tf2_msgs` /tf_static** (via `tf2_ros.StaticTransformBroadcaster`) for `lidar_mount_tf_parent_frame` → `lidar_link_frame`. This does **not** drive the point transform; points use the same extrinsics numerically in `pointcloud_rigid_body`.

### Optional: perception (requires `perception_enabled: true`)

Perception publishers are **created lazily** the first time they are needed: if `publish_*` is false, that topic is not advertised on DDS.

| Default topic | Type | Role |
|---------------|------|------|
| `/lidar_obstacle_detection/cloud_segmented` | `sensor_msgs/PointCloud2` | Colored cloud by cluster (`publish_colored_segmented_cloud`). |
| `/lidar_obstacle_detection/obstacle_markers` | `visualization_msgs/MarkerArray` | AABB + surface-normal arrows (`publish_obstacle_markers`). |
| `/lidar_obstacle_detection/obstacle_list` | `lidar_obstacle_detection_msgs/ObstacleList` | Obstacles + header (`publish_obstacle_list`). |

Debug cloud + markers use `perception_debug_qos_*` (default **reliable** so RViz2’s default display QoS matches); `ObstacleList` uses `obstacle_list_qos_*`. If you use `best_effort` on the node, set each RViz display’s QoS to Best Effort as well.

---

## Code layout (by category)

Python package directory: `lidar_obstacle_detection/lidar_obstacle_detection/`.

### Executable / orchestration

| Module | Role |
|--------|------|
| `lidar_cloud_ingress_node.py` | **Entry point** (`lidar_cloud_ingress`). Declares parameters, wires subscribe/publish, runs pipeline each callback. |

### Spatial processing and accumulation

| Module | Role |
|--------|------|
| `base_link_spatial_filter.py` | Forward FOV sector, range cap, optional height/lateral/body crops, voxel downsample (`SpatialFilterParams`, `filter_and_downsample_xyz`). |
| `temporal_cloud_accumulator.py` | Rolling merge of last N scans, stride publish, optional merged voxel / point cap (`TemporalPointCloudAccumulator`). |
| `pointcloud_rigid_body.py` | Parse `PointCloud2` → numpy xyz; apply fixed R\|t (driver + mount chain). |

### Math and TF helpers

| Module | Role |
|--------|------|
| `geometry_utils.py` | Extrinsic RPY → rotation matrix; RPY → quaternion for static TF. |
| `static_mount_tf.py` | Optional `/tf_static` broadcaster for RViz alignment. |

### Perception (library, no ROS graph in core algorithm)

| Module | Role |
|--------|------|
| `surface_obstacle_segmentation.py` | Open3D normals, cosine split vs. dominant surface, DBSCAN; footprint ground normal; `build_xyz_rgba_pointcloud2` for debug cloud. |
| `obstacle_ros_msgs.py` | Builds `ObstacleList` from internal `ObstacleDetection` dataclasses. |
| `obstacle_rviz_markers.py` | Builds `MarkerArray` for boxes and normal arrows. |

### Cross-cutting

| Module | Role |
|--------|------|
| `ros_qos.py` | `reliability_from_param`, `make_volatile_qos` for declared QoS parameters. |
| `verbose_log.py` | Throttled / gated INFO when `verbose` is true. |

### Share / launch

| Path | Role |
|------|------|
| `config/lidar_cloud_ingress.yaml` | Default parameters (algorithm + topics + QoS). |
| `launch/lidar_cloud_ingress.launch.py` | Node + optional RViz; loads YAML. Use `headless:=true` to skip RViz. |
| `rviz/lidar_ingress.rviz` | Saved RViz display config. |

### Tests

| Path | Role |
|------|------|
| `test/test_flake8.py`, `test_pep257.py`, `test_copyright.py` | ament lint tests. |

---

## Build and run

```bash
cd ~/ros2_ws
colcon build --packages-up-to lidar_obstacle_detection --symlink-install
source install/setup.bash
ros2 launch lidar_obstacle_detection lidar_cloud_ingress.launch.py
# No GUI:
ros2 launch lidar_obstacle_detection lidar_cloud_ingress.launch.py headless:=true
```

Depends on **`lidar_obstacle_detection_msgs`** (build order: messages package first, or use `--packages-up-to lidar_obstacle_detection`).

---

## Dependency

- **`lidar_obstacle_detection_msgs`**: `Obstacle.msg`, `ObstacleList.msg` (see that package’s README).
