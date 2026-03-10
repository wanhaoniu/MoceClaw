# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Sequence
import numpy as np


class TransportBase:
    """Abstract transport interface for 5DOF arm + optional gripper."""

    def __init__(self, dof: int, *, has_gripper: bool = False):
        self.dof = int(dof)
        self.has_gripper = bool(has_gripper)

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
        return True

    def set_gripper(
        self,
        open_ratio: float,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> None:
        raise NotImplementedError("set_gripper is not supported by this transport")

    def get_gripper_open_ratio(self) -> Optional[float]:
        return None

    def is_gripper_available(self) -> bool:
        return bool(self.has_gripper)
