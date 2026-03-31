"""
Build ``lidar_obstacle_detection_msgs/ObstacleList`` from segmentation results.

Keeps ROS message assembly out of the pure geometry / Open3D segmentation module.
"""

from __future__ import annotations

import numpy as np
from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import Header

from lidar_obstacle_detection_msgs.msg import Obstacle, ObstacleList

from .surface_obstacle_segmentation import ObstacleDetection


def build_obstacle_list_msg(
    obstacles: list[ObstacleDetection],
    header: Header,
) -> ObstacleList:
    """
    Fill ``ObstacleList`` with AABB center, dimensions, volume, and ground normal.

    ``surface_normals`` holds at most one unit vector when the footprint median
    normal is finite; otherwise the array is empty.
    """
    out = ObstacleList()
    out.header = header
    for o in obstacles:
        w = float(o.max_bound[0] - o.min_bound[0])
        h = float(o.max_bound[1] - o.min_bound[1])
        ell = float(o.max_bound[2] - o.min_bound[2])
        vol = float(w * h * ell)
        msg = Obstacle()
        msg.position = Point(x=float(o.center[0]), y=float(o.center[1]), z=float(o.center[2]))
        msg.width = w
        msg.height = h
        msg.length = ell
        msg.volume = vol
        msg.surface_normals = []
        if np.all(np.isfinite(o.median_surface_normal)):
            vn = o.median_surface_normal.astype(np.float64)
            msg.surface_normals.append(
                Vector3(x=float(vn[0]), y=float(vn[1]), z=float(vn[2])),
            )
        out.obstacles.append(msg)
    return out
