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
    names: Optional[list[str]] = None


@dataclass
class Pose:
    xyz: np.ndarray
    rpy: np.ndarray


@dataclass
class GripperState:
    available: bool
    installed: bool = True
    open_ratio: Optional[float] = None
    moving: Optional[bool] = None


@dataclass
class PermissionState:
    allow_motion: bool = True
    allow_gripper: bool = True
    allow_home: bool = True
    allow_stop: bool = True


@dataclass
class TwinState:
    source: str
    joint_state: JointState
    tcp_pose: Pose
    gripper_state: Optional[GripperState] = None


@dataclass
class RobotState:
    connected: bool
    joint_state: JointState
    tcp_pose: Pose
    gripper_state: Optional[GripperState] = None
    permissions: Optional[PermissionState] = None
    timestamp: Optional[float] = None
    actual: Optional[TwinState] = None
    twin: Optional[TwinState] = None
