# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class JointState:
    q: np.ndarray
    dq: Optional[np.ndarray] = None
    tau: Optional[np.ndarray] = None


@dataclass
class Pose:
    xyz: np.ndarray
    rpy: np.ndarray
