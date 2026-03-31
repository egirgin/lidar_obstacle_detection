"""
Spatial filtering and voxel downsampling for clouds already in ``base_link``.

Convention (REP-103 typical): **+X forward**, +Y left, +Z up. The robot origin is
the slice center. Forward FOV is a horizontal sector: total aperture ``fov_deg``
centered on +X (half-angle ``fov_deg / 2`` each side in the XY plane).

**max_depth:** simplified range cap — keep points with ``x <= max_depth`` (cheap
alternative to arc length from the origin).

**max_height:** optional; when set, drop points with ``z > max_height``.

**max_lateral:** optional; when set, keep points with ``|y| <= max_lateral`` (strip
wide left/right returns past the corridor width).

**min_forward_distance:** optional; when set, drop points with ``x < min_forward_distance``
(e.g. remove hits on the robot body / mast near ``base_link`` origin).

Order: sector + depth (+ height + lateral + min forward x) → Open3D voxel downsample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import open3d as o3d


@dataclass(frozen=True)
class SpatialFilterParams:
    """Immutable parameters for slice filtering and downsampling."""

    fov_deg: float
    """Total yaw aperture (degrees), centered on +X in the XY plane."""

    max_depth: float
    """Keep points with x <= max_depth (forward axis in ``base_link``)."""

    voxel_size: float
    """Open3D voxel grid leaf size (same units as point coordinates, usually meters)."""

    max_height: Optional[float] = None
    """If not ``None``, drop points with z > max_height."""

    max_lateral: Optional[float] = None
    """If not ``None``, keep points with ``|y| <= max_lateral``."""

    min_forward_distance: Optional[float] = None
    """If not ``None``, keep points with ``x >= min_forward_distance`` (meters ahead of base)."""


def _validate_params(params: SpatialFilterParams) -> None:
    if not (0.0 < params.fov_deg <= 360.0):
        raise ValueError(f'fov_deg must be in (0, 360], got {params.fov_deg!r}')
    if params.max_depth <= 0.0:
        raise ValueError(f'max_depth must be positive, got {params.max_depth!r}')
    if params.voxel_size <= 0.0:
        raise ValueError(f'voxel_size must be positive, got {params.voxel_size!r}')
    if params.max_lateral is not None and params.max_lateral <= 0.0:
        raise ValueError(f'max_lateral must be positive when set, got {params.max_lateral!r}')
    if params.min_forward_distance is not None and params.min_forward_distance < 0.0:
        raise ValueError(
            f'min_forward_distance must be non-negative when set, got {params.min_forward_distance!r}',
        )


def mask_forward_cheesecake_slice(
    xyz: np.ndarray,
    *,
    fov_deg: float,
    max_depth: float,
    max_height: Optional[float] = None,
    max_lateral: Optional[float] = None,
    min_forward_distance: Optional[float] = None,
    min_forward_x: float = 1e-6,
) -> np.ndarray:
    """
    Boolean mask for the forward sector, depth, and optional box-like crops.

    Parameters
    ----------
    xyz
        Shape (N, 3), float.
    fov_deg, max_depth, max_height
        Same meaning as :class:`SpatialFilterParams`.
    max_lateral
        If set, require ``|y| <= max_lateral``.
    min_forward_distance
        If set, require ``x >= min_forward_distance``. If unset, require ``x > min_forward_x``
        so points on or behind the ``x = 0`` plane are dropped.
    min_forward_x
        Used only when ``min_forward_distance`` is ``None``.
    """
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f'xyz must be (N, 3), got shape {xyz.shape}')

    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]

    half = np.deg2rad(0.5 * float(fov_deg))
    theta = np.arctan2(y, x)
    if min_forward_distance is not None:
        in_forward_x = x >= float(min_forward_distance)
    else:
        in_forward_x = x > float(min_forward_x)
    in_slice = in_forward_x & (np.abs(theta) <= half)
    in_depth = x <= float(max_depth)
    mask = in_slice & in_depth

    if max_height is not None:
        mask = mask & (z <= float(max_height))

    if max_lateral is not None:
        mask = mask & (np.abs(y) <= float(max_lateral))

    return mask


def voxel_downsample_xyz(xyz: np.ndarray, voxel_size: float) -> np.ndarray:
    """Voxel-grid downsample using Open3D; returns (M, 3) float64."""
    if voxel_size <= 0.0:
        raise ValueError('voxel_size must be positive')
    n = xyz.shape[0]
    if n == 0:
        return np.zeros((0, 3), dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(xyz, dtype=np.float64))
    down = pcd.voxel_down_sample(voxel_size=float(voxel_size))
    return np.asarray(down.points, dtype=np.float64)


def filter_and_downsample_xyz(
    xyz: np.ndarray,
    params: SpatialFilterParams,
    *,
    min_forward_x: float = 1e-6,
) -> np.ndarray:
    """
    Apply cheesecake slice (+ optional height / lateral / min forward x), then voxel downsample.

    Raises
    ------
    ValueError
        If ``params`` are invalid or ``xyz`` is not (N, 3).
    """
    _validate_params(params)
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f'xyz must be (N, 3), got shape {xyz.shape}')

    mask = mask_forward_cheesecake_slice(
        xyz,
        fov_deg=params.fov_deg,
        max_depth=params.max_depth,
        max_height=params.max_height,
        max_lateral=params.max_lateral,
        min_forward_distance=params.min_forward_distance,
        min_forward_x=min_forward_x,
    )
    sliced = xyz[mask]
    return voxel_downsample_xyz(sliced, params.voxel_size)
