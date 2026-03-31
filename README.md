# ROS 2 LiDAR obstacle detection

Stack of two packages in this repository:

| Package | Role |
|--------|------|
| [**`lidar_obstacle_detection`**](lidar_obstacle_detection/) | Node **`lidar_cloud_ingress`**: subscribe `PointCloud2`, rigid transform into the robot base frame, spatial filter + voxel, optional temporal merge, optional surface segmentation + DBSCAN → **`ObstacleList`** and debug topics. |
| [**`lidar_obstacle_detection_msgs`**](lidar_obstacle_detection_msgs/) | Message definitions (`Obstacle`, `ObstacleList`). |

**Defaults** in [`lidar_obstacle_detection/config/lidar_cloud_ingress.yaml`](lidar_obstacle_detection/config/lidar_cloud_ingress.yaml) target a **Unitree Go2** setup: input cloud **`/utlidar/cloud`**, output in **`base_link`**, and mount extrinsics matching that robot. The published cloud is always expressed in `output_cloud_frame_id` (usually `base_link`). A static TF **`base_link` → `lidar_link`** is published only for visualization alignment in RViz; it does not change the point math.

---

## Build

From your ROS 2 workspace (parent of this `src` tree):

```bash
cd ~/ros2_ws
rosdep install --from-paths src/object_detection --ignore-src -r -y   # optional; first-time deps
python3 -m pip install -r src/object_detection/lidar_obstacle_detection/requirements.txt
colcon build --packages-up-to lidar_obstacle_detection --symlink-install
source install/setup.bash
```

---

## Launch

```bash
ros2 launch lidar_obstacle_detection lidar_cloud_ingress.launch.py
```

- **With RViz** (default): omit `headless` or set `headless:=false`.
- **Headless** (no RViz, e.g. on-robot or SSH): `headless:=true`.

Override parameters file: `ingress_params_file:=/path/to/your.yaml`. Verbose logs: `verbose:=true`.

---

## Customizing for another robot / LiDAR

### 1. Config YAML (required)

Edit **`lidar_obstacle_detection/config/lidar_cloud_ingress.yaml`** (or a copy passed as `ingress_params_file`). Fields that almost always change for a new platform:

| Parameter | What to set |
|-----------|-------------|
| **`input_topic`** | Your driver’s `sensor_msgs/PointCloud2` topic. |
| **`output_cloud_frame_id`** | Frame of the processed cloud (typically the robot base / `base_link`). |
| **`lidar_mount_tf_parent_frame`** | Must match **`output_cloud_frame_id`** so TF and cloud headers agree. |
| **`lidar_link_frame`** | Child name for the visualization static TF (often matches URDF LiDAR link). |
| **`lidar_mount_tf_xyz`**, **`lidar_mount_tf_rpy_rad`** | Parent → LiDAR extrinsic (meters, radians); same geometry the node uses to transform points. |
| **`driver_cloud_to_lidar_link_xyz`**, **`driver_cloud_to_lidar_link_rpy_rad`** | Extra rigid step if the driver’s points are not already in the frame your URDF expects before the mount (often all zeros). |
| **`publish_lidar_mount_static_tf`** | Set **`false`** if another node (e.g. `robot_state_publisher`) already publishes the same transform. |

Then tune **`fov_deg`**, **`max_depth`**, **`voxel_size`**, **`temporal_*`**, and perception (**`dbscan_*`**, **`cosine_threshold`**, etc.) for your sensor density and environment.

### 2. RViz (optional but typical)

Saved config: **`lidar_obstacle_detection/rviz/lidar_ingress.rviz`**.

- Set **Global Options → Fixed Frame** to your base frame (default **`base_link`**).
- Point each display’s **Topic** at the names you set in YAML (`output_topic`, `colored_segmented_cloud_topic`, `obstacle_markers_topic`, etc., if you renamed them).

---

## Further detail

See the package README: [`lidar_obstacle_detection/README.md`](lidar_obstacle_detection/README.md) (topics, QoS, algorithm layout, message README link).
