"""
URDF-style extrinsic RPY → quaternion (static TF / geometry_msgs).

Uses stdlib + NumPy only. Convention: roll about fixed parent X, then pitch Y,
then yaw Z → R = Rz(yaw) @ Ry(pitch) @ Rx(roll).
"""

from __future__ import annotations

# =============================================================================
# geometry_utils — implementation sections
# =============================================================================

import math

import numpy as np

# -----------------------------------------------------------------------------
# URDF extrinsic RPY → rotation matrix / quaternion
# -----------------------------------------------------------------------------


def extrinsic_rpy_to_rotation_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """
    URDF extrinsic roll-pitch-yaw (rad): R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

    Returns 3×3 float64.
    """
    cx, sx = math.cos(roll), math.sin(roll)
    cy, sy = math.cos(pitch), math.sin(pitch)
    cz, sz = math.cos(yaw), math.sin(yaw)
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, cx, -sx], [0.0, sx, cx]])
    Ry = np.array([[cy, 0.0, sy], [0.0, 1.0, 0.0], [-sy, 0.0, cy]])
    Rz = np.array([[cz, -sz, 0.0], [sz, cz, 0.0], [0.0, 0.0, 1.0]])
    return (Rz @ Ry @ Rx).astype(np.float64)


def rpy_extrinsic_xyz_to_quaternion(
    roll: float, pitch: float, yaw: float,
) -> tuple[float, float, float, float]:
    """
    Map URDF (roll, pitch, yaw) radians to a unit quaternion (x, y, z, w).

    Builds R = Rz(yaw) Ry(pitch) Rx(roll) for geometry_msgs / tf2.
    """
    r = extrinsic_rpy_to_rotation_matrix(roll, pitch, yaw)
    return rotation_matrix_to_quaternion(r)


# -----------------------------------------------------------------------------
# Rotation matrix → quaternion (Shepperd / branch-stable)
# -----------------------------------------------------------------------------


def rotation_matrix_to_quaternion(R: np.ndarray) -> tuple[float, float, float, float]:
    """
    Convert a proper 3×3 rotation matrix to a unit quaternion (x, y, z, w).

    Trace-based branch selection for numerical stability.
    """
    t = np.trace(R)
    if t > 0.0:
        s = 0.5 / math.sqrt(t + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    n = math.sqrt(x * x + y * y + z * z + w * w)
    return (x / n, y / n, z / n, w / n)
