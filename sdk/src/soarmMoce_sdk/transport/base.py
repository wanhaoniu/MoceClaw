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
