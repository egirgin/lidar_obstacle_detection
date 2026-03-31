"""Gated INFO logging when ``verbose`` is false (use node logger for warnings)."""

from __future__ import annotations

# =============================================================================
# verbose_log
# =============================================================================

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rclpy.node import Node


class VerboseLog:
    """Forwards ``info`` to ``rclpy`` only when verbose mode is enabled."""

    def __init__(self, node: Node, enabled: bool) -> None:
        self._logger = node.get_logger()
        self._enabled = enabled

    def set_enabled(self, enabled: bool) -> None:
        """Turn verbose INFO logs on or off (e.g. from a parameter callback)."""
        self._enabled = enabled

    def info(self, msg: str) -> None:
        """Log at INFO if verbose; no-op otherwise."""
        if self._enabled:
            self._logger.info(msg)
