# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np

from .fk import fk, jacobian
from .frames import rpy_to_matrix, rotvec_from_matrix
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
    target_rpy: Optional[np.ndarray] = None,
    q0: Optional[np.ndarray] = None,
    preferred_q: Optional[np.ndarray] = None,
    locked_joint_targets: Optional[Dict[str, float]] = None,
    rpy_in_degrees: bool = False,
    max_iters: int = 200,
    damping: float = 5e-2,
    step_scale: float = 0.8,
    pos_tol: float = 1e-3,
    rot_tol: float = 1e-1,
    max_step: float = 0.15,
    seed_bias: float = 0.02,
    position_weight: float = 1.0,
    orientation_weight: float = 0.0,
    clamp_limits: bool = True,
) -> IKSolution:
    target_xyz = np.asarray(target_xyz, dtype=float).reshape(3)
    if target_rpy is not None:
        target_rpy = np.asarray(target_rpy, dtype=float).reshape(3)
        if rpy_in_degrees:
            target_rpy = np.deg2rad(target_rpy)
        target_rot = rpy_to_matrix(target_rpy)
    else:
        target_rot = None

    if q0 is None:
        q = np.zeros(robot.dof, dtype=float)
    else:
        q = np.asarray(q0, dtype=float).reshape(-1).copy()
        if q.shape[0] != robot.dof:
            raise ValueError(f"q0 size mismatch: expected {robot.dof}, got {q.shape[0]}")

    if preferred_q is None:
        preferred = q.copy()
    else:
        preferred = np.asarray(preferred_q, dtype=float).reshape(-1).copy()
        if preferred.shape[0] != robot.dof:
            raise ValueError(f"preferred_q size mismatch: expected {robot.dof}, got {preferred.shape[0]}")

    lower = np.array([l for l, _ in robot.joint_limits], dtype=float)
    upper = np.array([u for _, u in robot.joint_limits], dtype=float)

    locked_joint_targets = dict(locked_joint_targets or {})
    locked_indices = []
    for joint_name, joint_target in locked_joint_targets.items():
        idx = robot.resolve_joint_index(joint_name)
        q[idx] = float(joint_target)
        preferred[idx] = float(joint_target)
        locked_indices.append(idx)
    active_indices = [i for i in range(robot.dof) if i not in locked_indices]
    if not active_indices:
        raise ValueError("No active joints left for IK")

    best_q = q.copy()
    best_pos_err = float("inf")
    best_rot_err = float("inf")
    reason = "numerical_not_converged"
    lam = float(max(1e-6, damping))
    identity_joint = np.eye(len(active_indices), dtype=float)
    position_weight = float(max(0.0, position_weight))
    orientation_weight = float(max(0.0, orientation_weight))
    use_position_task = bool(position_weight > 0.0)
    use_orientation_task = bool(target_rot is not None and orientation_weight > 0.0)
    if not use_position_task and not use_orientation_task:
        raise ValueError("No active tasks left for IK")
    task_dim = (3 if use_position_task else 0) + (3 if use_orientation_task else 0)
    identity_task = np.eye(task_dim, dtype=float)

    for it in range(1, max_iters + 1):
        T = fk(robot, q)
        pos = T[:3, 3]
        pos_err_vec = target_xyz - pos
        pos_err = float(np.linalg.norm(pos_err_vec))

        if target_rot is None:
            rot_err_vec = np.zeros(3, dtype=float)
            rot_err = 0.0
        else:
            rot_err_vec = rotvec_from_matrix(target_rot @ T[:3, :3].T)
            rot_err = float(np.linalg.norm(rot_err_vec))

        if use_position_task and use_orientation_task:
            is_better = pos_err < best_pos_err or (abs(pos_err - best_pos_err) < 1e-9 and rot_err < best_rot_err)
        elif use_position_task:
            is_better = pos_err < best_pos_err
        else:
            is_better = rot_err < best_rot_err
        if is_better:
            best_q = q.copy()
            best_pos_err = pos_err
            best_rot_err = rot_err

        pos_ok = (not use_position_task) or pos_err <= pos_tol
        rot_ok = (not use_orientation_task) or rot_err <= rot_tol
        if pos_ok and rot_ok:
            return IKSolution(True, q, "success", it, pos_err, rot_err)

        J_active = jacobian(robot, q)[:, active_indices]
        task_blocks = []
        task_errors = []
        if use_position_task:
            task_blocks.append(position_weight * J_active[:3, :])
            task_errors.append(position_weight * pos_err_vec)
        if use_orientation_task:
            task_blocks.append(orientation_weight * J_active[3:, :])
            task_errors.append(orientation_weight * rot_err_vec)
        J_task = np.vstack(task_blocks)
        task_err = np.concatenate(task_errors)
        JJt = J_task @ J_task.T
        damped = JJt + (lam**2) * identity_task
        try:
            primary_step = J_task.T @ np.linalg.solve(damped, task_err)
            null_projector = identity_joint - (J_task.T @ np.linalg.solve(damped, J_task))
            secondary_step = null_projector @ (preferred[active_indices] - q[active_indices])
        except np.linalg.LinAlgError:
            reason = "singular"
            break

        dq_active = (primary_step + seed_bias * secondary_step) * float(step_scale)
        dq_active = np.clip(dq_active, -float(max_step), float(max_step))
        if not np.all(np.isfinite(dq_active)):
            reason = "non_finite_step"
            break
        if np.linalg.norm(dq_active) < 1e-9:
            reason = "stuck_or_unreachable"
            break

        q[active_indices] = q[active_indices] + dq_active
        for idx in locked_indices:
            q[idx] = preferred[idx]
        if clamp_limits:
            q = np.minimum(np.maximum(q, lower), upper)

    T_best = fk(robot, best_q)
    if target_rot is None:
        rot_err = 0.0
    else:
        rot_err = float(np.linalg.norm(rotvec_from_matrix(target_rot @ T_best[:3, :3].T)))
    success = ((not use_position_task) or best_pos_err <= pos_tol) and ((not use_orientation_task) or rot_err <= rot_tol)
    return IKSolution(success, best_q, reason, max_iters, best_pos_err, rot_err)
