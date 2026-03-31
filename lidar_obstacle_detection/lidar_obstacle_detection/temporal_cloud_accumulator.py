"""
Rolling temporal merge of LiDAR scans (``base_link`` xyz), already filtered/downsampled.

Use after rigid transform + spatial filter so each stored scan is lightweight. The buffer
keeps the last ``n_scans`` arrays and concatenates them for denser input to downstream
normals / clustering.

**Publish stride:** ``publish_every_n_inputs`` lowers the output topic rate while the
buffer still advances every callback (rotating LiDAR: more slices in the merged cloud).

**Safeguards:** optional voxel on the merged cloud and/or a hard point cap (uniform
index subsample) to bound memory and CPU.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from lidar_obstacle_detection.base_link_spatial_filter import voxel_downsample_xyz


@dataclass(frozen=True)
class TemporalAccumulateParams:
    """Parameters for :class:`TemporalPointCloudAccumulator`."""

    n_scans: int = 1
    """Number of most recent scans to concatenate (>= 1)."""

    publish_every_n_inputs: int = 1
    """Publish merged output only every K-th ``push`` (>= 1)."""

    max_merged_points: int = 0
    """If > 0, subsample merged cloud to at most this many points (after merge voxel)."""

    merge_voxel_size: float = 0.0
    """If > 0, voxel-downsample the concatenated cloud with this leaf size (meters)."""


def _validate_temporal_params(p: TemporalAccumulateParams) -> None:
    if p.n_scans < 1:
        raise ValueError(f'n_scans must be >= 1, got {p.n_scans!r}')
    if p.publish_every_n_inputs < 1:
        raise ValueError(
            f'publish_every_n_inputs must be >= 1, got {p.publish_every_n_inputs!r}',
        )
    if p.max_merged_points < 0:
        raise ValueError(f'max_merged_points must be >= 0, got {p.max_merged_points!r}')
    if p.merge_voxel_size < 0.0:
        raise ValueError(f'merge_voxel_size must be >= 0, got {p.merge_voxel_size!r}')


def subsample_xyz_uniform(xyz: np.ndarray, max_points: int) -> np.ndarray:
    """Deterministically subsample (N, 3) to at most ``max_points`` rows."""
    if max_points <= 0:
        raise ValueError('max_points must be positive')
    xyz = np.asarray(xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f'xyz must be (N, 3), got shape {xyz.shape}')
    n = xyz.shape[0]
    if n <= max_points:
        return xyz
    idx = np.linspace(0, n - 1, num=max_points, dtype=np.int64)
    return xyz[idx]


def postprocess_merged_xyz(
    xyz: np.ndarray,
    *,
    merge_voxel_size: float,
    max_merged_points: int,
) -> np.ndarray:
    """
    Apply optional merge-time voxel, then optional point cap.

    Parameters
    ----------
    xyz
        Concatenated cloud, shape (N, 3).
    merge_voxel_size
        If > 0, Open3D voxel downsample with this leaf size.
    max_merged_points
        If > 0, uniform subsample to this many points after voxel step.
    """
    out = np.asarray(xyz, dtype=np.float64)
    if out.size == 0:
        return np.zeros((0, 3), dtype=np.float64)

    if merge_voxel_size > 0.0:
        out = voxel_downsample_xyz(out, merge_voxel_size)
    if max_merged_points > 0 and out.shape[0] > max_merged_points:
        out = subsample_xyz_uniform(out, max_merged_points)
    return out


class TemporalPointCloudAccumulator:
    """
    Stateful buffer: ``push`` each per-scan xyz; query merged output and publish gating.

    Not thread-safe; use from a single ROS callback or lock externally.
    """

    def __init__(self, params: TemporalAccumulateParams) -> None:
        _validate_temporal_params(params)
        self._params = params
        self._scans: deque[np.ndarray] = deque(maxlen=params.n_scans)
        self._push_count: int = 0

    @property
    def params(self) -> TemporalAccumulateParams:
        return self._params

    def clear(self) -> None:
        """Drop all buffered scans and reset the publish counter."""
        self._scans.clear()
        self._push_count = 0

    def push(self, xyz: np.ndarray) -> None:
        """
        Append one scan. Empty scans are stored as (0, 3) so stride/timing stay consistent.
        """
        arr = np.asarray(xyz, dtype=np.float64)
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f'xyz must be (N, 3), got shape {arr.shape}')
        if arr.shape[0] == 0:
            self._scans.append(np.zeros((0, 3), dtype=np.float64))
        else:
            self._scans.append(arr.copy())
        self._push_count += 1

    @property
    def push_count(self) -> int:
        """Total number of ``push`` calls since construction or ``clear``."""
        return self._push_count

    def num_buffered_scans(self) -> int:
        return len(self._scans)

    def merged_xyz(self) -> np.ndarray:
        """Concatenate non-empty buffered scans; shape (M, 3) float64."""
        if not self._scans:
            return np.zeros((0, 3), dtype=np.float64)
        parts = [s for s in self._scans if s.shape[0] > 0]
        if not parts:
            return np.zeros((0, 3), dtype=np.float64)
        return np.vstack(parts)

    def should_publish_after_last_push(self) -> bool:
        """True if the current ``push_count`` matches the publish stride."""
        k = self._params.publish_every_n_inputs
        if k <= 1:
            return True
        return self._push_count % k == 0

    def merged_output_xyz(self) -> np.ndarray:
        """
        Merged cloud after :func:`postprocess_merged_xyz` (voxel + max points).

        Call after ``push`` when ``should_publish_after_last_push()`` is true, or anytime
        for debugging.
        """
        raw = self.merged_xyz()
        return postprocess_merged_xyz(
            raw,
            merge_voxel_size=self._params.merge_voxel_size,
            max_merged_points=self._params.max_merged_points,
        )
