# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Sequence
import numpy as np


class TransportBase:
    """Abstract transport interface."""

    def __init__(self, dof: int):
        self.dof = int(dof)

    def connect(self) -> None:
        raise NotImplementedError

    def disconnect(self) -> None:
        raise NotImplementedError

    def get_q(self) -> np.ndarray:
        raise NotImplementedError

    def send_movej(
        self,
        q: Sequence[float],
        duration: float,
        speed: Optional[float] = None,
        accel: Optional[float] = None,
    ) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        raise NotImplementedError

    def wait_until_stopped(self, timeout: Optional[float] = None) -> bool:
        """Best-effort wait for motion completion.

        Returns:
            True when motion is considered complete.
            False when timeout expires before completion.

        The default implementation is a no-op for transports that cannot
        introspect motion state yet.
        """
        return True

    def set_gripper(
        self,
        open_ratio: float,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> None:
        raise NotImplementedError("set_gripper is not supported by this transport")

    def get_gripper_open_ratio(self) -> Optional[float]:
        """Return current gripper open ratio in [0, 1], if supported."""
        return None
