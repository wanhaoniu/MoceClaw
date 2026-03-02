# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np

from .frames import rotvec_from_matrix, transform_from_xyz_rpy
from .fk import fk, jacobian
from .urdf_loader import RobotModel


@dataclass
class IKSolution:
    success: bool
    q: np.ndarray
    reason: str
    iterations: int
    pos_err: float
    rot_err: float


def solve_ik(
    robot: RobotModel,
    target_xyz: np.ndarray,
    target_rpy: np.ndarray,
    q0: Optional[np.ndarray] = None,
    rpy_in_degrees: bool = False,
    max_iters: int = 200,
    damping: float = 1e-2,
    step_scale: float = 1.0,
    pos_tol: float = 1e-3,
    rot_tol: float = 1e-2,
    rot_weight: float = 1.0,
    max_step: float = 0.3,
    clamp_limits: bool = True,
) -> IKSolution:
    if rpy_in_degrees:
        target_rpy = np.deg2rad(target_rpy)

    target_T = transform_from_xyz_rpy(np.asarray(target_xyz, dtype=float), np.asarray(target_rpy, dtype=float))

    if q0 is None:
        q = np.zeros(robot.dof, dtype=float)
    else:
        q = np.asarray(q0, dtype=float).reshape(-1).copy()
        if q.shape[0] != robot.dof:
            raise ValueError(f"q0 size mismatch: expected {robot.dof}, got {q.shape[0]}")

    lower = np.array([l for l, _ in robot.joint_limits], dtype=float)
    upper = np.array([u for _, u in robot.joint_limits], dtype=float)
    limit_margin = 1e-6

    reason = "numerical_not_converged"
    lam = float(damping)

    for it in range(1, max_iters + 1):
        T = fk(robot, q)
        pos = T[:3, 3]
        R = T[:3, :3]

        pos_err_vec = target_T[:3, 3] - pos
        R_err = target_T[:3, :3] @ R.T
        rot_err_vec = rotvec_from_matrix(R_err)

        pos_err = float(np.linalg.norm(pos_err_vec))
        rot_err = float(np.linalg.norm(rot_err_vec))

        if pos_err < pos_tol and rot_err < rot_tol:
            return IKSolution(True, q, "success", it, pos_err, rot_err)

        err = np.hstack([pos_err_vec, rot_err_vec * rot_weight])
        J = jacobian(robot, q)
        Jw = J.copy()
        Jw[3:, :] *= rot_weight

        JJt = Jw @ Jw.T
        damp = (lam ** 2) * np.eye(JJt.shape[0])
        dq = Jw.T @ np.linalg.solve(JJt + damp, err)

        if max_step is not None:
            max_abs = float(np.max(np.abs(dq)))
            if max_abs > max_step:
                dq = dq * (max_step / max_abs)

        q_trial = q + step_scale * dq
        if clamp_limits:
            q_trial = np.minimum(np.maximum(q_trial, lower), upper)

        T_trial = fk(robot, q_trial)
        pos_trial = T_trial[:3, 3]
        R_trial = T_trial[:3, :3]
        pos_err_trial = float(np.linalg.norm(target_T[:3, 3] - pos_trial))
        R_err_trial = target_T[:3, :3] @ R_trial.T
        rot_err_trial = float(np.linalg.norm(rotvec_from_matrix(R_err_trial)))
        err_norm = pos_err_trial + rot_err_trial

        if err_norm < (pos_err + rot_err):
            q = q_trial
            lam = max(lam * 0.7, 1e-6)
            step_scale = min(step_scale * 1.05, 1.0)
        else:
            lam = min(lam * 2.0, 1e2)
            step_scale = max(step_scale * 0.5, 1e-3)
            if step_scale <= 1e-3 and lam >= 1e2:
                reason = "stuck_or_unreachable"
                break

    if np.any(q < lower - limit_margin) or np.any(q > upper + limit_margin):
        reason = "joint_limit_violation"

    T = fk(robot, q)
    pos_err = float(np.linalg.norm(target_T[:3, 3] - T[:3, 3]))
    R_err = target_T[:3, :3] @ T[:3, :3].T
    rot_err = float(np.linalg.norm(rotvec_from_matrix(R_err)))

    return IKSolution(False, q, reason, max_iters, pos_err, rot_err)
