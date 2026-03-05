# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Sequence
import time
import numpy as np

from .base import TransportBase


class MockTransport(TransportBase):
    """Mock transport for tests and examples (no hardware)."""

    def __init__(self, dof: int):
        super().__init__(dof)
        self._connected = False
        self._q = np.zeros(self.dof, dtype=float)
        self._gripper_open_ratio = 1.0
        self._motion_end_time = 0.0

    def connect(self) -> None:
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def get_q(self) -> np.ndarray:
        if not self._connected:
            raise RuntimeError("MockTransport not connected")
        return self._q.copy()

    def send_movej(
        self,
        q: Sequence[float],
        duration: float,
        speed: Optional[float] = None,
        accel: Optional[float] = None,
    ) -> None:
        if not self._connected:
            raise RuntimeError("MockTransport not connected")
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.dof:
            raise ValueError(f"Expected {self.dof} joints, got {q.shape[0]}")
        self._q = q.copy()
        self._motion_end_time = time.monotonic() + max(0.0, float(duration))

    def stop(self) -> None:
        self._motion_end_time = time.monotonic()

    def wait_until_stopped(self, timeout: Optional[float] = None) -> bool:
        if not self._connected:
            raise RuntimeError("MockTransport not connected")
        remaining = max(0.0, self._motion_end_time - time.monotonic())
        if timeout is not None:
            timeout = max(0.0, float(timeout))
            if remaining > timeout:
                time.sleep(timeout)
                return False
        if remaining > 0.0:
            time.sleep(remaining)
        return True

    def set_gripper(self, open_ratio: float, wait: bool = True, timeout: Optional[float] = None) -> None:
        if not self._connected:
            raise RuntimeError("MockTransport not connected")
        ratio = float(open_ratio)
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("open_ratio must be within [0.0, 1.0]")
        self._gripper_open_ratio = ratio
        self._motion_end_time = time.monotonic() + 0.05
        if wait:
            self.wait_until_stopped(timeout=timeout)

    def get_gripper_open_ratio(self) -> Optional[float]:
        if not self._connected:
            raise RuntimeError("MockTransport not connected")
        return float(self._gripper_open_ratio)
