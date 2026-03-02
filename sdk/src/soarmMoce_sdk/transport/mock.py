# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Sequence
import numpy as np

from .base import TransportBase


class MockTransport(TransportBase):
    """Mock transport for tests and examples (no hardware)."""

    def __init__(self, dof: int):
        super().__init__(dof)
        self._connected = False
        self._q = np.zeros(self.dof, dtype=float)

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

    def stop(self) -> None:
        # No-op for mock
        pass
