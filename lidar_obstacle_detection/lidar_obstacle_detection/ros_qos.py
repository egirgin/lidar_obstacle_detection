"""Small QoS helpers for ingress / perception publishers and subscriptions."""

from __future__ import annotations

from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy


def reliability_from_param(value: object) -> ReliabilityPolicy:
    """Map parameter string to policy; invalid values raise ValueError."""
    s = str(value).lower().strip()
    if s in ('reliable', 'reliability_reliable'):
        return ReliabilityPolicy.RELIABLE
    if s in ('best_effort', 'besteffort', 'reliability_best_effort'):
        return ReliabilityPolicy.BEST_EFFORT
    raise ValueError(
        f'QoS reliability must be "reliable" or "best_effort", got {value!r}',
    )


def make_volatile_qos(
    reliability: ReliabilityPolicy,
    depth: int,
) -> QoSProfile:
    """Standard volatile KEEP_LAST profile (typical sensor / pipeline topics)."""
    d = int(depth)
    if d < 1:
        raise ValueError(f'QoS depth must be >= 1, got {depth}')
    return QoSProfile(
        reliability=reliability,
        durability=DurabilityPolicy.VOLATILE,
        history=HistoryPolicy.KEEP_LAST,
        depth=d,
    )
