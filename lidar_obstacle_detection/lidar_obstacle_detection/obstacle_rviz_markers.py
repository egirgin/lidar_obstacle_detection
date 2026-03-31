"""RViz ``MarkerArray`` helpers: obstacle AABBs and mean surface-normal arrows."""

from __future__ import annotations

import numpy as np
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Point
from visualization_msgs.msg import Marker, MarkerArray

from .surface_obstacle_segmentation import ObstacleDetection


def _delete_all_marker(frame_id: str, stamp: Time, namespace: str) -> Marker:
    m = Marker()
    m.header.frame_id = frame_id
    m.header.stamp = stamp
    m.ns = namespace
    m.id = 0
    m.action = Marker.DELETEALL
    return m


def _bbox_line_list_marker(
    frame_id: str,
    stamp: Time,
    namespace: str,
    marker_id: int,
    min_b: np.ndarray,
    max_b: np.ndarray,
    r: float,
    g: float,
    b: float,
    line_width: float,
) -> Marker:
    """LINE_LIST for an axis-aligned box given min/max corners."""
    mn = np.asarray(min_b, dtype=np.float64).reshape(3)
    mx = np.asarray(max_b, dtype=np.float64).reshape(3)
    x0, y0, z0 = float(mn[0]), float(mn[1]), float(mn[2])
    x1, y1, z1 = float(mx[0]), float(mx[1]), float(mx[2])

    corners = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y1, z0),
        (x0, y1, z0),
        (x0, y0, z1),
        (x1, y0, z1),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    edges = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 0),
        (4, 5),
        (5, 6),
        (6, 7),
        (7, 4),
        (0, 4),
        (1, 5),
        (2, 6),
        (3, 7),
    ]
    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = int(marker_id)
    marker.type = Marker.LINE_LIST
    marker.action = Marker.ADD
    marker.scale.x = float(line_width)
    marker.color.r = float(r)
    marker.color.g = float(g)
    marker.color.b = float(b)
    marker.color.a = 1.0
    for i, j in edges:
        a = corners[i]
        b_ = corners[j]
        marker.points.append(Point(x=a[0], y=a[1], z=a[2]))
        marker.points.append(Point(x=b_[0], y=b_[1], z=b_[2]))
    return marker


def _normal_arrow_marker(
    frame_id: str,
    stamp: Time,
    namespace: str,
    marker_id: int,
    anchor: np.ndarray,
    direction_unit: np.ndarray,
    length: float,
    r: float,
    g: float,
    b: float,
) -> Marker:
    anchor = np.asarray(anchor, dtype=np.float64).reshape(3)
    d = np.asarray(direction_unit, dtype=np.float64).reshape(3)
    dn = float(np.linalg.norm(d))
    if dn < 1e-9 or not np.all(np.isfinite(d)):
        m = Marker()
        m.header.frame_id = frame_id
        m.header.stamp = stamp
        m.ns = namespace
        m.id = int(marker_id)
        m.action = Marker.DELETE
        return m
    d = d / dn
    end = anchor + d * float(length)

    marker = Marker()
    marker.header.frame_id = frame_id
    marker.header.stamp = stamp
    marker.ns = namespace
    marker.id = int(marker_id)
    marker.type = Marker.ARROW
    marker.action = Marker.ADD
    marker.scale.x = 0.02
    marker.scale.y = 0.04
    marker.scale.z = 0.06
    marker.color.r = float(r)
    marker.color.g = float(g)
    marker.color.b = float(b)
    marker.color.a = 1.0
    marker.points.append(
        Point(x=float(anchor[0]), y=float(anchor[1]), z=float(anchor[2])),
    )
    marker.points.append(Point(x=float(end[0]), y=float(end[1]), z=float(end[2])))
    return marker


def build_obstacle_marker_array(
    frame_id: str,
    stamp: Time,
    obstacles: list[ObstacleDetection],
    normal_arrow_length: float = 0.35,
    box_line_width: float = 0.02,
) -> MarkerArray:
    """
    One DELETEALL per namespace, then LINE_LIST AABBs and ARROW surface normals.

    Namespaces: ``lidar_obstacle_boxes``, ``lidar_obstacle_surface_normals``.
    """
    arr = MarkerArray()
    arr.markers.append(_delete_all_marker(frame_id, stamp, 'lidar_obstacle_boxes'))
    arr.markers.append(_delete_all_marker(frame_id, stamp, 'lidar_obstacle_surface_normals'))

    for o in obstacles:
        bid = int(o.cluster_id)
        arr.markers.append(
            _bbox_line_list_marker(
                frame_id,
                stamp,
                'lidar_obstacle_boxes',
                bid,
                o.min_bound,
                o.max_bound,
                0.0,
                1.0,
                1.0,
                box_line_width,
            ),
        )
        arr.markers.append(
            _normal_arrow_marker(
                frame_id,
                stamp,
                'lidar_obstacle_surface_normals',
                bid,
                o.normal_anchor,
                o.median_surface_normal,
                normal_arrow_length,
                1.0,
                0.4,
                0.1,
            ),
        )
    return arr
