"""Static TF publisher for lidar mount pose (equivalent to a URDF fixed joint)."""

from __future__ import annotations

# =============================================================================
# static_mount_tf — /tf_static broadcaster
# =============================================================================
# Disable when robot_state_publisher already publishes the same transform.

from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster

from lidar_obstacle_detection.geometry_utils import rpy_extrinsic_xyz_to_quaternion
from lidar_obstacle_detection.verbose_log import VerboseLog


class StaticMountTfPublisher:
    """Sends ``TransformStamped`` on ``/tf_static``; optional periodic republish."""

    def __init__(
        self,
        node: Node,
        *,
        parent_frame: str,
        child_frame: str,
        translation_xyz: tuple[float, float, float],
        rpy_rad: tuple[float, float, float],
        vlog: VerboseLog,
        republish_hz: float = 0.0,
    ) -> None:
        """
        Build quaternion from URDF-style RPY and start broadcasting.

        ``child_frame`` must match ``PointCloud2.header.frame_id``. If
        ``republish_hz`` > 0, republish at that rate; else publish once.
        """
        self._node = node
        self._broadcaster = StaticTransformBroadcaster(node)
        self._parent = parent_frame
        self._child = child_frame
        self._translation = translation_xyz
        self._rpy = rpy_rad
        self._vlog = vlog

        # --- Precompute message body (stamp refreshed on each send) ----------
        qx, qy, qz, qw = rpy_extrinsic_xyz_to_quaternion(rpy_rad[0], rpy_rad[1], rpy_rad[2])
        self._transform = TransformStamped()
        self._transform.header.frame_id = parent_frame
        self._transform.child_frame_id = child_frame
        self._transform.transform.translation.x = float(translation_xyz[0])
        self._transform.transform.translation.y = float(translation_xyz[1])
        self._transform.transform.translation.z = float(translation_xyz[2])
        self._transform.transform.rotation.x = qx
        self._transform.transform.rotation.y = qy
        self._transform.transform.rotation.z = qz
        self._transform.transform.rotation.w = qw

        self._timer = None
        if republish_hz > 0.0:
            self._timer = node.create_timer(1.0 / republish_hz, self._on_timer)
        self.send_once()

    def send_once(self) -> None:
        """Stamp with current time and publish one static transform."""
        self._transform.header.stamp = self._node.get_clock().now().to_msg()
        self._broadcaster.sendTransform(self._transform)
        x, y, z = self._translation
        rr, rp, ryaw = self._rpy
        self._vlog.info(
            f'Static TF {self._parent} → {self._child}: '
            f't=[{x:.5f}, {y:.5f}, {z:.5f}], rpy=[{rr:.5f}, {rp:.5f}, {ryaw:.5f}] rad',
        )

    def _on_timer(self) -> None:
        """Periodic republish (optional; helps very late RViz subscribers)."""
        self.send_once()
