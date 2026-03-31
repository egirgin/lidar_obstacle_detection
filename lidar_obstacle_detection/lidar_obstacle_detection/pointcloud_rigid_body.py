"""Apply a fixed rigid body to PointCloud2 xyz (no tf2; driver frame is read-only)."""

from __future__ import annotations

# =============================================================================
# pointcloud_rigid_body
# =============================================================================

from typing import Optional

import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs_py import point_cloud2 as pc2
from std_msgs.msg import Header

from lidar_obstacle_detection.verbose_log import VerboseLog


def transform_pointcloud_to_xyz(
    cloud: PointCloud2,
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    vlog: Optional[VerboseLog] = None,
) -> Optional[np.ndarray]:
    """
    Read ``x/y/z``, apply ``R @ p + t``, return ``(N, 3)`` float64.

    Same validity rules as :func:`transform_pointcloud_rigid` (``height==1``, fields).
    Empty cloud yields shape ``(0, 3)``.
    """
    if cloud.height != 1:
        if vlog is not None:
            vlog.info(
                f'Skipping rigid transform: height={cloud.height} (only height==1 supported)',
            )
        return None

    try:
        structured = pc2.read_points(cloud, skip_nans=True)
    except Exception as ex:  # noqa: BLE001
        if vlog is not None:
            vlog.info(f'read_points failed: {ex}')
        return None

    names = structured.dtype.names
    if names is None or 'x' not in names or 'y' not in names or 'z' not in names:
        if vlog is not None:
            vlog.info('PointCloud2 missing x/y/z fields; cannot apply rigid body')
        return None

    if structured.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float64).reshape(3)

    xs = structured['x'].astype(np.float64)
    ys = structured['y'].astype(np.float64)
    zs = structured['z'].astype(np.float64)
    xyz = np.stack([xs, ys, zs], axis=1)
    return (r @ xyz.T).T + t


def transform_pointcloud_rigid(
    cloud: PointCloud2,
    *,
    rotation: np.ndarray,
    translation: np.ndarray,
    output_frame_id: str,
    vlog: Optional[VerboseLog] = None,
) -> Optional[PointCloud2]:
    """
    Map each point p to R @ p + t; set ``header.frame_id`` to ``output_frame_id``.

    Preserves all PointCloud2 fields; requires structured fields x, y, z.
    Incoming ``cloud.header.frame_id`` (driver name) is ignored except for logs.
    """
    if cloud.height != 1:
        if vlog is not None:
            vlog.info(
                f'Skipping rigid transform: height={cloud.height} (only height==1 supported)',
            )
        return None

    try:
        structured = pc2.read_points(cloud, skip_nans=True)
    except Exception as ex:  # noqa: BLE001
        if vlog is not None:
            vlog.info(f'read_points failed: {ex}')
        return None

    names = structured.dtype.names
    if names is None or 'x' not in names or 'y' not in names or 'z' not in names:
        if vlog is not None:
            vlog.info('PointCloud2 missing x/y/z fields; cannot apply rigid body')
        return None

    if structured.size == 0:
        header = Header()
        header.stamp = cloud.header.stamp
        header.frame_id = output_frame_id
        return pc2.create_cloud(header, cloud.fields, [])

    r = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    t = np.asarray(translation, dtype=np.float64).reshape(3)

    xs = structured['x'].astype(np.float64)
    ys = structured['y'].astype(np.float64)
    zs = structured['z'].astype(np.float64)
    xyz = np.stack([xs, ys, zs], axis=1)
    xyz_out = (r @ xyz.T).T + t

    out_struct = structured.copy()
    out_struct['x'] = xyz_out[:, 0].astype(np.float32)
    out_struct['y'] = xyz_out[:, 1].astype(np.float32)
    out_struct['z'] = xyz_out[:, 2].astype(np.float32)

    rows = [tuple(row) for row in out_struct]
    header = Header()
    header.stamp = cloud.header.stamp
    header.frame_id = output_frame_id
    return pc2.create_cloud(header, cloud.fields, rows)
