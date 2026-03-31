"""
ROS 2 node: driver PointCloud2 → ``base_link`` cloud (single entry).

Pipeline per message: **rigid transform** into ``output_cloud_frame_id`` → **forward
FOV / depth / optional height, lateral corridor, min forward-x crop** (see
``base_link_spatial_filter``) → **voxel downsample** → optional **temporal merge** of the last N scans (see
``temporal_cloud_accumulator``) → publish ``sensor_msgs/PointCloud2`` (xyz32) on
``output_topic``.

Optional **perception** (``surface_obstacle_segmentation``): normals oriented to
``perception_up_axis``, cosine split vs. dominant surface, DBSCAN clusters; colored
cloud, ``lidar_obstacle_detection_msgs/ObstacleList``, and RViz markers (AABB + mean
surface normal under footprint) on separate topics when enabled via parameters.

Driver ``header.frame_id`` is read-only. Mount math matches static TF:
``p_base = R_mount @ (R_drv @ p + t_drv) + t_mount``.

Optional static TF ``base_link``→``lidar_link_frame`` is **visualization only**;
it is not used to transform the cloud.

**Parameters:** Declared here with the same defaults as
``share/lidar_obstacle_detection/config/lidar_cloud_ingress.yaml``. Prefer loading
that file from launch (see ``launch/lidar_cloud_ingress.launch.py``) or
``--params-file`` so tuning stays in one place.

**QoS (graph / load):** Configurable QoS on subscription and each publisher class;
high-rate debug topics (colored cloud, markers) default off and use lazy publisher
creation so disabled topics are not advertised on the DDS graph.
"""

from __future__ import annotations

import numpy as np
import rclpy
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.node import Node
from rclpy.publisher import Publisher
from rclpy.qos import QoSProfile
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header
from visualization_msgs.msg import MarkerArray

from lidar_obstacle_detection_msgs.msg import ObstacleList

from lidar_obstacle_detection.base_link_spatial_filter import (
    SpatialFilterParams,
    filter_and_downsample_xyz,
)
from lidar_obstacle_detection.geometry_utils import extrinsic_rpy_to_rotation_matrix
from lidar_obstacle_detection.pointcloud_rigid_body import transform_pointcloud_to_xyz
from lidar_obstacle_detection.static_mount_tf import StaticMountTfPublisher
from lidar_obstacle_detection.temporal_cloud_accumulator import (
    TemporalAccumulateParams,
    TemporalPointCloudAccumulator,
)
from lidar_obstacle_detection.verbose_log import VerboseLog
from lidar_obstacle_detection.obstacle_ros_msgs import build_obstacle_list_msg
from lidar_obstacle_detection.ros_qos import make_volatile_qos, reliability_from_param
from lidar_obstacle_detection.surface_obstacle_segmentation import (
    SurfaceObstacleParams,
    build_xyz_rgba_pointcloud2,
    segment_surface_obstacles,
)
from lidar_obstacle_detection.obstacle_rviz_markers import build_obstacle_marker_array


def _translation_from_xyz(xyz: list) -> np.ndarray:
    return np.array([float(xyz[0]), float(xyz[1]), float(xyz[2])], dtype=np.float64)


def _compose_rigid(
    r_second: np.ndarray,
    t_second: np.ndarray,
    r_first: np.ndarray,
    t_first: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return R,t for p' = R_second @ (R_first @ p + t_first) + t_second."""
    r_tot = r_second @ r_first
    t_tot = r_second @ t_first + t_second
    return r_tot, t_tot


# =============================================================================
# Node
# =============================================================================


class LidarCloudIngressNode(Node):
    """Transform, spatial filter, downsample; publish one cloud in ``base_link``."""

    def __init__(self) -> None:
        super().__init__('lidar_cloud_ingress')

        # ------------------------------------------------------------------
        # Logging
        # ------------------------------------------------------------------
        self._verbose = self.declare_parameter('verbose', False).value
        self._vlog = VerboseLog(self, bool(self._verbose))

        # ------------------------------------------------------------------
        # Topics
        # ------------------------------------------------------------------
        self._in_topic = self.declare_parameter('input_topic', '/utlidar/cloud').value
        self._out_topic = self.declare_parameter(
            'output_topic',
            '/lidar_obstacle_detection/cloud_in_base',
        ).value

        # ------------------------------------------------------------------
        # Output cloud is always expressed in this frame (robot base)
        # ------------------------------------------------------------------
        self._output_frame = self.declare_parameter(
            'output_cloud_frame_id',
            'base_link',
        ).value

        self._output_stamps_use_node_time = bool(
            self.declare_parameter(
                'output_stamps_use_node_time',
                False,
                descriptor=ParameterDescriptor(
                    description=(
                        'If true, output PointCloud2, segmented cloud, markers, and '
                        'ObstacleList use get_clock().now() at publish time instead of '
                        'the driver message stamp. Use when RViz fixed frame is odom '
                        '(or map) and odom→base_link is on the same ROS clock as this '
                        'node; set use_sim_time:=true with ros2 bag play --clock for '
                        'bags. If false, stamps match the input cloud (sensor time).'
                    ),
                ),
            ).value,
        )

        # ------------------------------------------------------------------
        # QoS (match publishers, shallow queues on debug outputs)
        # ------------------------------------------------------------------
        self.declare_parameter(
            'input_qos_reliability',
            'best_effort',
            descriptor=ParameterDescriptor(
                description='PointCloud2 subscription: use best_effort to match typical LiDAR.',
            ),
        )
        self.declare_parameter(
            'input_qos_depth',
            5,
            descriptor=ParameterDescriptor(
                description='Subscription history depth (KEEP_LAST).',
            ),
        )
        self.declare_parameter(
            'output_cloud_qos_reliability',
            'reliable',
            descriptor=ParameterDescriptor(
                description='Filtered cloud publisher: reliable for recorders; best_effort to mimic sensors.',
            ),
        )
        self.declare_parameter(
            'output_cloud_qos_depth',
            5,
            descriptor=ParameterDescriptor(
                description='Output cloud history depth.',
            ),
        )
        self.declare_parameter(
            'perception_debug_qos_reliability',
            'reliable',
            descriptor=ParameterDescriptor(
                description=(
                    'Colored segmented cloud + MarkerArray. Default reliable to match '
                    'RViz2 displays; use best_effort to mimic sensor-style drops under load.'
                ),
            ),
        )
        self.declare_parameter(
            'perception_debug_qos_depth',
            2,
            descriptor=ParameterDescriptor(
                description='Shallow queue for debug visualization topics.',
            ),
        )
        self.declare_parameter(
            'obstacle_list_qos_reliability',
            'reliable',
            descriptor=ParameterDescriptor(
                description='ObstacleList publisher reliability.',
            ),
        )
        self.declare_parameter(
            'obstacle_list_qos_depth',
            5,
            descriptor=ParameterDescriptor(
                description='ObstacleList history depth.',
            ),
        )

        # ------------------------------------------------------------------
        # Spatial filter + downsample (``base_link``: +x forward; see library docstring)
        # ------------------------------------------------------------------
        self.declare_parameter('fov_deg', 90.0)
        self.declare_parameter('max_depth', 15.0)
        self.declare_parameter('voxel_size', 0.05)
        self.declare_parameter('max_height_enabled', False)
        self.declare_parameter('max_height', 2.0)
        self.declare_parameter('max_lateral_enabled', False)
        self.declare_parameter('max_lateral', 1.0)
        self.declare_parameter('min_forward_distance_enabled', False)
        self.declare_parameter('min_forward_distance', 0.3)

        # ------------------------------------------------------------------
        # Temporal accumulation (rotating LiDAR: denser merged cloud, lower publish rate)
        # ------------------------------------------------------------------
        self.declare_parameter('temporal_n_scans', 1)
        self.declare_parameter('temporal_publish_every_n_inputs', 1)
        self.declare_parameter('temporal_max_merged_points', 0)
        self.declare_parameter('temporal_merge_voxel_size', 0.0)

        try:
            self._temporal = TemporalPointCloudAccumulator(
                TemporalAccumulateParams(
                    n_scans=int(self.get_parameter('temporal_n_scans').value),
                    publish_every_n_inputs=int(
                        self.get_parameter('temporal_publish_every_n_inputs').value,
                    ),
                    max_merged_points=int(
                        self.get_parameter('temporal_max_merged_points').value,
                    ),
                    merge_voxel_size=float(
                        self.get_parameter('temporal_merge_voxel_size').value,
                    ),
                ),
            )
        except ValueError as ex:
            self.get_logger().error(f'Invalid temporal accumulation parameters: {ex}')
            raise

        # ------------------------------------------------------------------
        # Perception: normals + cosine surface split + DBSCAN (library API)
        # ------------------------------------------------------------------
        self._perception_enabled = bool(
            self.declare_parameter('perception_enabled', False).value,
        )
        self.declare_parameter('perception_min_points', 40)
        self.declare_parameter('cosine_threshold', 0.88)
        self.declare_parameter('normal_search_radius', 0.5)
        self.declare_parameter('max_nn', 30)
        self.declare_parameter('dbscan_eps', 0.15)
        self.declare_parameter('dbscan_min_points', 15)
        self.declare_parameter('obstacle_bbox_scale', 1.0)
        self.declare_parameter('surface_normal_bin_deg', 10.0)
        self.declare_parameter('perception_up_axis', [0.0, 0.0, 1.0])
        self.declare_parameter('perception_reference_normal_xyz', [0.0, 0.0, 0.0])
        self.declare_parameter(
            'publish_colored_segmented_cloud',
            False,
            descriptor=ParameterDescriptor(
                description='High-bandwidth debug cloud; keep false on robot (Phase F).',
            ),
        )
        self.declare_parameter(
            'colored_segmented_cloud_topic',
            '/lidar_obstacle_detection/cloud_segmented',
        )
        self.declare_parameter(
            'publish_obstacle_markers',
            False,
            descriptor=ParameterDescriptor(
                description='RViz MarkerArray; enable for dev, false on robot by default.',
            ),
        )
        self.declare_parameter(
            'obstacle_markers_topic',
            '/lidar_obstacle_detection/obstacle_markers',
        )
        self.declare_parameter('publish_obstacle_list', True)
        self.declare_parameter(
            'obstacle_list_topic',
            '/lidar_obstacle_detection/obstacle_list',
        )
        self.declare_parameter('obstacle_normal_arrow_length', 0.35)
        self.declare_parameter('obstacle_box_line_width', 0.02)

        # Perception publishers are created lazily so DDS does not advertise topics
        # that are disabled (reduces discovery noise and accidental subscribers).
        self._pub_seg: Publisher | None = None
        self._pub_markers: Publisher | None = None
        self._pub_obstacles: Publisher | None = None

        # ------------------------------------------------------------------
        # Viz-only: child frame for static TF (same mount numbers as below)
        # ------------------------------------------------------------------
        self._lidar_viz_frame = self.declare_parameter(
            'lidar_link_frame',
            'lidar_link',
        ).value

        # ------------------------------------------------------------------
        # Driver point coords → lidar-aligned frame (before mount; default I)
        # ------------------------------------------------------------------
        self._drv_xyz = self.declare_parameter(
            'driver_cloud_to_lidar_link_xyz',
            [0.0, 0.0, 0.0],
        ).value
        self._drv_rpy = self.declare_parameter(
            'driver_cloud_to_lidar_link_rpy_rad',
            [0.0, 0.0, 0.0],
        ).value

        # ------------------------------------------------------------------
        # Mount: output_cloud_frame @ lidar_viz (URDF fixed joint, same as TF)
        # ------------------------------------------------------------------
        self._publish_static_tf = self.declare_parameter(
            'publish_lidar_mount_static_tf',
            True,
        ).value
        self._static_parent = self.declare_parameter(
            'lidar_mount_tf_parent_frame',
            'base_link',
        ).value
        self._static_xyz = self.declare_parameter(
            'lidar_mount_tf_xyz',
            [0.28945, 0.0, -0.046825],
        ).value
        self._static_rpy = self.declare_parameter(
            'lidar_mount_tf_rpy_rad',
            [0.0, 2.8782, 0.0],
        ).value
        self._static_rep_hz = self.declare_parameter('lidar_mount_tf_republish_hz', 0.0).value

        if len(self._drv_xyz) != 3 or len(self._drv_rpy) != 3:
            self.get_logger().warn(
                'driver_cloud_to_lidar_link_xyz / _rpy_rad invalid; using identity',
            )
            self._drv_xyz = [0.0, 0.0, 0.0]
            self._drv_rpy = [0.0, 0.0, 0.0]

        rr = float(self._drv_rpy[0])
        rp = float(self._drv_rpy[1])
        ry = float(self._drv_rpy[2])
        r_drv = extrinsic_rpy_to_rotation_matrix(rr, rp, ry)
        t_drv = _translation_from_xyz(self._drv_xyz)

        if len(self._static_xyz) != 3 or len(self._static_rpy) != 3:
            self.get_logger().error(
                'lidar_mount_tf_xyz / lidar_mount_tf_rpy_rad invalid; '
                'using identity mount for point transform',
            )
            r_mount = np.eye(3, dtype=np.float64)
            t_mount = np.zeros(3, dtype=np.float64)
        else:
            sr = float(self._static_rpy[0])
            sp = float(self._static_rpy[1])
            sy = float(self._static_rpy[2])
            r_mount = extrinsic_rpy_to_rotation_matrix(sr, sp, sy)
            t_mount = _translation_from_xyz(self._static_xyz)

        self._r_total, self._t_total = _compose_rigid(r_mount, t_mount, r_drv, t_drv)

        if str(self._static_parent) != str(self._output_frame):
            self.get_logger().warn(
                'lidar_mount_tf_parent_frame should match output_cloud_frame_id '
                'so mount R|t maps into the published coordinate frame.',
            )

        try:
            self._qos_in = make_volatile_qos(
                reliability_from_param(
                    self.get_parameter('input_qos_reliability').value,
                ),
                int(self.get_parameter('input_qos_depth').value),
            )
            self._qos_out_cloud = make_volatile_qos(
                reliability_from_param(
                    self.get_parameter('output_cloud_qos_reliability').value,
                ),
                int(self.get_parameter('output_cloud_qos_depth').value),
            )
        except ValueError as ex:
            self.get_logger().error(f'Invalid QoS parameters: {ex}')
            raise

        self._pub = self.create_publisher(
            PointCloud2,
            self._out_topic,
            self._qos_out_cloud,
        )

        self._static_tf = None
        if self._publish_static_tf and len(self._static_xyz) == 3 and len(self._static_rpy) == 3:
            tx = float(self._static_xyz[0])
            ty = float(self._static_xyz[1])
            tz = float(self._static_xyz[2])
            sr = float(self._static_rpy[0])
            sp = float(self._static_rpy[1])
            sy = float(self._static_rpy[2])
            self._static_tf = StaticMountTfPublisher(
                self,
                parent_frame=str(self._static_parent),
                child_frame=str(self._lidar_viz_frame),
                translation_xyz=(tx, ty, tz),
                rpy_rad=(sr, sp, sy),
                vlog=self._vlog,
                republish_hz=float(self._static_rep_hz),
            )

        self._sub = self.create_subscription(
            PointCloud2,
            self._in_topic,
            self._on_cloud,
            self._qos_in,
        )

        in_rel = self.get_parameter('input_qos_reliability').value
        in_d = int(self.get_parameter('input_qos_depth').value)
        out_rel = self.get_parameter('output_cloud_qos_reliability').value
        out_d = int(self.get_parameter('output_cloud_qos_depth').value)
        self._vlog.info(
            f'Lidar ingress: in={self._in_topic!r} out={self._out_topic!r} '
            f'output_frame={self._output_frame!r} viz_frame={self._lidar_viz_frame!r} '
            f'static_tf={bool(self._static_tf)}',
        )
        self.get_logger().info(
            f'ROS graph (ingress): subscribe PointCloud2 {self._in_topic!r} '
            f'qos={in_rel!r} depth={in_d}; publish {self._out_topic!r} qos={out_rel!r} depth={out_d}.',
        )
        self.get_logger().info(
            f'Ingress: transform + FOV/downsample → {self._output_frame!r}; '
            f'fov_deg={self.get_parameter("fov_deg").value} '
            f'max_depth={self.get_parameter("max_depth").value} '
            f'voxel={self.get_parameter("voxel_size").value}; '
            f'temporal_n_scans={self.get_parameter("temporal_n_scans").value} '
            f'temporal_publish_every_n_inputs='
            f'{self.get_parameter("temporal_publish_every_n_inputs").value} '
            f'temporal_max_merged_points='
            f'{self.get_parameter("temporal_max_merged_points").value} '
            f'temporal_merge_voxel_size='
            f'{self.get_parameter("temporal_merge_voxel_size").value}; '
            f'static TF (viz) {self._static_parent!r}→{self._lidar_viz_frame!r} is '
            f'{"on" if self._static_tf else "off"}.',
        )
        if self._static_tf:
            self.get_logger().info(
                'If robot_state_publisher already publishes that mount TF, set '
                'publish_lidar_mount_static_tf:=false.',
            )
        if self._perception_enabled:
            self.get_logger().info(
                'Perception on: publishers for segmented cloud, markers, and ObstacleList '
                'are created lazily when their publish_* params are true (no DDS leak for '
                'disabled topics). See perception_debug_qos_* and obstacle_list_qos_*.k',
            )

    def _surface_obstacle_params(self) -> SurfaceObstacleParams:
        """Read live ROS parameters into :class:`SurfaceObstacleParams`."""
        ref_raw = list(self.get_parameter('perception_reference_normal_xyz').value)
        ref: tuple[float, float, float] | None
        if len(ref_raw) == 3 and (
            float(ref_raw[0]) != 0.0
            or float(ref_raw[1]) != 0.0
            or float(ref_raw[2]) != 0.0
        ):
            ref = (float(ref_raw[0]), float(ref_raw[1]), float(ref_raw[2]))
        else:
            ref = None
        up = list(self.get_parameter('perception_up_axis').value)
        if len(up) != 3:
            up = [0.0, 0.0, 1.0]
        return SurfaceObstacleParams(
            cosine_threshold=float(self.get_parameter('cosine_threshold').value),
            normal_search_radius=float(self.get_parameter('normal_search_radius').value),
            max_nn=int(self.get_parameter('max_nn').value),
            dbscan_eps=float(self.get_parameter('dbscan_eps').value),
            dbscan_min_points=int(self.get_parameter('dbscan_min_points').value),
            obstacle_bbox_scale=float(self.get_parameter('obstacle_bbox_scale').value),
            surface_normal_bin_deg=float(self.get_parameter('surface_normal_bin_deg').value),
            up_axis=(float(up[0]), float(up[1]), float(up[2])),
            reference_normal_xyz=ref,
        )

    def _spatial_params(self) -> SpatialFilterParams:
        mh: float | None
        if bool(self.get_parameter('max_height_enabled').value):
            mh = float(self.get_parameter('max_height').value)
        else:
            mh = None
        ml: float | None
        if bool(self.get_parameter('max_lateral_enabled').value):
            ml = float(self.get_parameter('max_lateral').value)
        else:
            ml = None
        mfd: float | None
        if bool(self.get_parameter('min_forward_distance_enabled').value):
            mfd = float(self.get_parameter('min_forward_distance').value)
        else:
            mfd = None
        return SpatialFilterParams(
            fov_deg=float(self.get_parameter('fov_deg').value),
            max_depth=float(self.get_parameter('max_depth').value),
            voxel_size=float(self.get_parameter('voxel_size').value),
            max_height=mh,
            max_lateral=ml,
            min_forward_distance=mfd,
        )

    def _perception_debug_qos_profile(self) -> QoSProfile:
        return make_volatile_qos(
            reliability_from_param(
                self.get_parameter('perception_debug_qos_reliability').value,
            ),
            int(self.get_parameter('perception_debug_qos_depth').value),
        )

    def _obstacle_list_qos_profile(self) -> QoSProfile:
        return make_volatile_qos(
            reliability_from_param(
                self.get_parameter('obstacle_list_qos_reliability').value,
            ),
            int(self.get_parameter('obstacle_list_qos_depth').value),
        )

    def _ensure_segmented_cloud_publisher(self) -> Publisher | None:
        if not self._perception_enabled:
            return None
        if not bool(self.get_parameter('publish_colored_segmented_cloud').value):
            return None
        if self._pub_seg is None:
            self._pub_seg = self.create_publisher(
                PointCloud2,
                str(self.get_parameter('colored_segmented_cloud_topic').value),
                self._perception_debug_qos_profile(),
            )
        return self._pub_seg

    def _ensure_obstacle_markers_publisher(self) -> Publisher | None:
        if not self._perception_enabled:
            return None
        if not bool(self.get_parameter('publish_obstacle_markers').value):
            return None
        if self._pub_markers is None:
            self._pub_markers = self.create_publisher(
                MarkerArray,
                str(self.get_parameter('obstacle_markers_topic').value),
                self._perception_debug_qos_profile(),
            )
        return self._pub_markers

    def _ensure_obstacle_list_publisher(self) -> Publisher | None:
        if not self._perception_enabled:
            return None
        if not bool(self.get_parameter('publish_obstacle_list').value):
            return None
        if self._pub_obstacles is None:
            self._pub_obstacles = self.create_publisher(
                ObstacleList,
                str(self.get_parameter('obstacle_list_topic').value),
                self._obstacle_list_qos_profile(),
            )
        return self._pub_obstacles

    def _on_cloud(self, msg: PointCloud2) -> None:
        """Transform → slice / crops / voxel → publish xyz32 in ``base_link``."""
        driver_fid = msg.header.frame_id
        self._vlog.info(
            f'Input driver frame_id={driver_fid!r} (read-only) '
            f'stamp={msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d} '
            f'points~={msg.width * max(msg.height, 1)}',
        )

        xyz_base = transform_pointcloud_to_xyz(
            msg,
            rotation=self._r_total,
            translation=self._t_total,
            vlog=self._vlog,
        )
        if xyz_base is None:
            self.get_logger().warn(
                'Dropping cloud: rigid transform failed (need x/y/z fields, height==1)',
                throttle_duration_sec=2.0,
            )
            return

        try:
            sp = self._spatial_params()
            out_xyz = filter_and_downsample_xyz(xyz_base, sp)
        except ValueError as ex:
            self.get_logger().error(f'Invalid spatial filter parameters: {ex}')
            return

        self._temporal.push(out_xyz)
        if not self._temporal.should_publish_after_last_push():
            self._vlog.info(
                f'Temporal buffer: push only (stride); buffered_scans='
                f'{self._temporal.num_buffered_scans()} '
                f'per_scan_points={out_xyz.shape[0]}',
            )
            return

        merged = self._temporal.merged_output_xyz()
        stamp = (
            self.get_clock().now().to_msg()
            if self._output_stamps_use_node_time
            else msg.header.stamp
        )
        header = Header()
        header.stamp = stamp
        header.frame_id = str(self._output_frame)
        out_msg = pc2.create_cloud_xyz32(header, merged.astype(np.float32))
        self._pub.publish(out_msg)

        self._vlog.info(
            f'Published {merged.shape[0]} points in {self._output_frame!r} '
            f'(per_scan={out_xyz.shape[0]} buffered_scans='
            f'{self._temporal.num_buffered_scans()}; '
            f'in={xyz_base.shape[0]} after transform; '
            f'fov={sp.fov_deg}° max_depth={sp.max_depth} voxel={sp.voxel_size}; '
            f'temporal_n={self._temporal.params.n_scans} '
            f'publish_every={self._temporal.params.publish_every_n_inputs})',
        )

        if not self._perception_enabled:
            return

        min_pts = int(self.get_parameter('perception_min_points').value)
        if merged.shape[0] < min_pts:
            self._vlog.info(
                f'Perception skipped: merged points {merged.shape[0]} < '
                f'perception_min_points={min_pts}',
            )
            return

        try:
            seg = segment_surface_obstacles(merged, self._surface_obstacle_params())
        except Exception as ex:  # noqa: BLE001 — keep ingress alive on Open3D / data errors
            self.get_logger().error(
                f'segment_surface_obstacles failed: {ex}',
                throttle_duration_sec=5.0,
            )
            return

        pub_seg = self._ensure_segmented_cloud_publisher()
        if pub_seg is not None:
            colored = build_xyz_rgba_pointcloud2(header, seg.points, seg.labels)
            pub_seg.publish(colored)

        stamp = header.stamp
        pub_markers = self._ensure_obstacle_markers_publisher()
        if pub_markers is not None:
            arr = build_obstacle_marker_array(
                str(self._output_frame),
                stamp,
                seg.obstacles,
                normal_arrow_length=float(
                    self.get_parameter('obstacle_normal_arrow_length').value,
                ),
                box_line_width=float(
                    self.get_parameter('obstacle_box_line_width').value,
                ),
            )
            pub_markers.publish(arr)

        pub_olist = self._ensure_obstacle_list_publisher()
        if pub_olist is not None:
            olist_header = Header()
            olist_header.frame_id = str(self._output_frame)
            olist_header.stamp = header.stamp
            olist = build_obstacle_list_msg(seg.obstacles, olist_header)
            pub_olist.publish(olist)

        self._vlog.info(
            f'Perception: obstacles={len(seg.obstacles)} points={seg.points.shape[0]}',
        )


# =============================================================================
# Entry point
# =============================================================================


def main(args: list[str] | None = None) -> None:
    """Initialize rclpy, spin the ingress node, then shut down."""
    rclpy.init(args=args)
    node = LidarCloudIngressNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
