# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import numpy as np


def rot_x(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, ca, -sa],
        [0.0, sa, ca],
    ], dtype=float)


def rot_y(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([
        [ca, 0.0, sa],
        [0.0, 1.0, 0.0],
        [-sa, 0.0, ca],
    ], dtype=float)


def rot_z(a: float) -> np.ndarray:
    ca, sa = math.cos(a), math.sin(a)
    return np.array([
        [ca, -sa, 0.0],
        [sa, ca, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=float)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = float(rpy[0]), float(rpy[1]), float(rpy[2])
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def axis_angle_to_matrix(axis: np.ndarray, theta: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=float)
    norm = np.linalg.norm(axis)
    if norm < 1e-12:
        return np.eye(3)
    axis = axis / norm
    x, y, z = axis
    K = np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ], dtype=float)
    I = np.eye(3)
    return I + math.sin(theta) * K + (1.0 - math.cos(theta)) * (K @ K)


def rotvec_from_matrix(R: np.ndarray) -> np.ndarray:
    trace = float(np.trace(R))
    cos_theta = max(-1.0, min(1.0, (trace - 1.0) / 2.0))
    theta = math.acos(cos_theta)
    if theta < 1e-9:
        return np.zeros(3, dtype=float)
    sin_theta = math.sin(theta)
    if abs(sin_theta) < 1e-9:
        return np.zeros(3, dtype=float)
    axis = np.array([
        R[2, 1] - R[1, 2],
        R[0, 2] - R[2, 0],
        R[1, 0] - R[0, 1],
    ], dtype=float) / (2.0 * sin_theta)
    return axis * theta


def transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = rpy_to_matrix(rpy)
    T[:3, 3] = xyz
    return T


def transform_rot(axis: np.ndarray, theta: float) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, :3] = axis_angle_to_matrix(axis, theta)
    return T


def transform_trans(vec: np.ndarray) -> np.ndarray:
    T = np.eye(4, dtype=float)
    T[:3, 3] = vec
    return T
