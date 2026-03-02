# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import numpy as np

from .frames import transform_from_xyz_rpy, transform_rot, transform_trans
from .urdf_loader import RobotModel


def fk(robot: RobotModel, q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape[0] != robot.dof:
        raise ValueError(f"Expected {robot.dof} joint values, got {q.shape[0]}")

    T = np.eye(4, dtype=float)
    qi = 0
    for j in robot.chain_joints:
        T = T @ transform_from_xyz_rpy(j.origin_xyz, j.origin_rpy)
        if j.jtype in ("revolute", "continuous"):
            T = T @ transform_rot(j.axis, float(q[qi]))
            qi += 1
        elif j.jtype == "prismatic":
            T = T @ transform_trans(j.axis * float(q[qi]))
            qi += 1
    return T


def jacobian(robot: RobotModel, q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(-1)
    if q.shape[0] != robot.dof:
        raise ValueError(f"Expected {robot.dof} joint values, got {q.shape[0]}")

    T = np.eye(4, dtype=float)
    p_ee = fk(robot, q)[:3, 3]
    J = np.zeros((6, robot.dof), dtype=float)

    qi = 0
    col = 0
    for j in robot.chain_joints:
        T = T @ transform_from_xyz_rpy(j.origin_xyz, j.origin_rpy)
        if j.jtype in ("revolute", "continuous", "prismatic"):
            axis_base = T[:3, :3] @ j.axis
            p_joint = T[:3, 3]
            if j.jtype in ("revolute", "continuous"):
                J[:3, col] = np.cross(axis_base, (p_ee - p_joint))
                J[3:, col] = axis_base
                T = T @ transform_rot(j.axis, float(q[qi]))
            else:
                J[:3, col] = axis_base
                J[3:, col] = 0.0
                T = T @ transform_trans(j.axis * float(q[qi]))
            qi += 1
            col += 1
    return J


def matrix_to_rpy(R: np.ndarray) -> np.ndarray:
    sy = -R[2, 0]
    cy = (1.0 - sy * sy) ** 0.5

    if cy < 1e-9:
        yaw = math.atan2(-R[0, 1], R[1, 1])
        pitch = math.asin(sy)
        roll = 0.0
    else:
        yaw = math.atan2(R[1, 0], R[0, 0])
        pitch = math.asin(sy)
        roll = math.atan2(R[2, 1], R[2, 2])

    return np.array([roll, pitch, yaw], dtype=float)
