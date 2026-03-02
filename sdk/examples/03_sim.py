#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PyBullet IK teaching demo (ik1-style): sliders for xyz/rpy.
Run from repo root so ./configs/soarm_moce.yaml is found.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
import importlib.resources as resources

import numpy as np

from soarmMoce_sdk.config import load_config


def _resolve_pkg_path(value: str) -> Path:
    if value.startswith("pkg://"):
        rel = value[len("pkg://"):]
        pkg, rel_path = rel.split("/", 1)
        res = resources.files(pkg) / rel_path
        with resources.as_file(res) as p:
            return Path(p)
    return Path(value)


def _resolve_urdf_for_pybullet(urdf_path: Path, mesh_dir: Path) -> Path:
    text = urdf_path.read_text(encoding="utf-8")
    mesh_dir_str = str(mesh_dir).replace("\\", "/")
    text = text.replace("meshes/", mesh_dir_str + "/")

    out_dir = Path(__file__).resolve().parent / ".cache"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{urdf_path.stem}_resolved.urdf"
    out_path.write_text(text, encoding="utf-8")
    return out_path


def build_joint_lists(p, robot_id):
    movable_joint_indices = []
    movable_joint_names = []
    lower_limits = []
    upper_limits = []
    joint_ranges = []
    rest_poses = []

    num_joints = p.getNumJoints(robot_id)
    for ji in range(num_joints):
        info = p.getJointInfo(robot_id, ji)
        joint_type = info[2]
        joint_name = info[1].decode("utf-8")
        ll = info[8]
        ul = info[9]

        if joint_type in (p.JOINT_REVOLUTE, p.JOINT_PRISMATIC):
            movable_joint_indices.append(ji)
            movable_joint_names.append(joint_name)

            if (not np.isfinite(ll)) or (not np.isfinite(ul)) or (ll > ul):
                ll, ul = -np.pi, np.pi

            lower_limits.append(float(ll))
            upper_limits.append(float(ul))
            joint_ranges.append(float(ul - ll) if ul > ll else float(2 * np.pi))
            rest_poses.append(float(0.5 * (ll + ul)))

    return (movable_joint_indices, movable_joint_names,
            lower_limits, upper_limits, joint_ranges, rest_poses)


def auto_detect_ee_link(p, robot_id):
    num_joints = p.getNumJoints(robot_id)
    parents = set()
    all_links = set(range(num_joints))
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        parent_idx = info[16]
        if parent_idx >= 0:
            parents.add(parent_idx)

    leaf_links = sorted(list(all_links - parents))
    if len(leaf_links) == 0:
        return num_joints - 1 if num_joints > 0 else -1
    return leaf_links[-1]


def find_link_index_by_name(p, robot_id, link_name):
    num_joints = p.getNumJoints(robot_id)
    for j in range(num_joints):
        info = p.getJointInfo(robot_id, j)
        child_link_name = info[12].decode("utf-8")
        if child_link_name == link_name:
            return j
    return None


def set_robot_joints(p, robot_id, movable_joint_indices, q):
    for idx, ji in enumerate(movable_joint_indices):
        p.resetJointState(robot_id, ji, float(q[idx]))


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="./configs/soarm_moce.yaml")
    parser.add_argument("--urdf", default=None)
    parser.add_argument("--ee_link", default=None)
    parser.add_argument("--deg", action="store_true", help="use degrees for rpy (default)")
    parser.add_argument("--rad", action="store_true", help="use radians for rpy")
    parser.add_argument("--pos_tol", type=float, default=0.01)
    parser.add_argument("--ori_tol", type=float, default=0.15)
    args = parser.parse_args()

    cfg = load_config(args.config)
    urdf_cfg = args.urdf or cfg.get("urdf", {}).get("path")
    mesh_cfg = cfg.get("meshes", {}).get("path")

    urdf_path = _resolve_pkg_path(urdf_cfg)
    mesh_path = _resolve_pkg_path(mesh_cfg) if mesh_cfg else urdf_path.parent
    urdf_for_bullet = _resolve_urdf_for_pybullet(urdf_path, mesh_path)

    use_deg = True
    if args.rad:
        use_deg = False
    if args.deg:
        use_deg = True

    import pybullet as p
    import pybullet_data

    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.resetDebugVisualizerCamera(cameraDistance=1.2, cameraYaw=45, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.2])
    p.loadURDF("plane.urdf")

    robot_id = p.loadURDF(str(urdf_for_bullet), basePosition=[0, 0, 0], baseOrientation=[0, 0, 0, 1], useFixedBase=True)

    movable_joint_indices, movable_joint_names, lower_limits, upper_limits, joint_ranges, rest_poses = build_joint_lists(p, robot_id)
    if len(movable_joint_indices) == 0:
        raise RuntimeError("No movable joints found. Check URDF.")

    if args.ee_link is not None:
        ee_idx = find_link_index_by_name(p, robot_id, args.ee_link)
        if ee_idx is None:
            ee_idx = auto_detect_ee_link(p, robot_id)
    else:
        ee_idx = auto_detect_ee_link(p, robot_id)

    # Initial EE pose
    ee_state = p.getLinkState(robot_id, ee_idx, computeForwardKinematics=True)
    ee_pos = ee_state[4]
    ee_quat = ee_state[5]
    ee_rpy = p.getEulerFromQuaternion(ee_quat)

    x0, y0, z0 = ee_pos
    x_id = p.addUserDebugParameter("target_x", x0 - 0.4, x0 + 0.4, x0)
    y_id = p.addUserDebugParameter("target_y", y0 - 0.4, y0 + 0.4, y0)
    z_id = p.addUserDebugParameter("target_z", max(0.0, z0 - 0.4), z0 + 0.4, z0)

    if use_deg:
        r0, p0, y0_ = np.rad2deg(ee_rpy)
        rr_id = p.addUserDebugParameter("target_roll(deg)", -180.0, 180.0, r0)
        pp_id = p.addUserDebugParameter("target_pitch(deg)", -180.0, 180.0, p0)
        yy_id = p.addUserDebugParameter("target_yaw(deg)", -180.0, 180.0, y0_)
    else:
        r0, p0, y0_ = ee_rpy
        rr_id = p.addUserDebugParameter("target_roll(rad)", -3.14, 3.14, r0)
        pp_id = p.addUserDebugParameter("target_pitch(rad)", -3.14, 3.14, p0)
        yy_id = p.addUserDebugParameter("target_yaw(rad)", -3.14, 3.14, y0_)

    while p.isConnected():
        tx = p.readUserDebugParameter(x_id)
        ty = p.readUserDebugParameter(y_id)
        tz = p.readUserDebugParameter(z_id)
        tr = p.readUserDebugParameter(rr_id)
        tp = p.readUserDebugParameter(pp_id)
        tyaw = p.readUserDebugParameter(yy_id)

        if use_deg:
            tr, tp, tyaw = np.deg2rad([tr, tp, tyaw])

        target_pos = [tx, ty, tz]
        target_quat = p.getQuaternionFromEuler([tr, tp, tyaw])

        q_full = p.calculateInverseKinematics(
            robot_id,
            ee_idx,
            targetPosition=target_pos,
            targetOrientation=target_quat,
            lowerLimits=lower_limits,
            upperLimits=upper_limits,
            jointRanges=joint_ranges,
            restPoses=rest_poses,
            maxNumIterations=200,
            residualThreshold=1e-4,
        )
        q = np.array(q_full[:len(movable_joint_indices)], dtype=float)
        set_robot_joints(p, robot_id, movable_joint_indices, q)

        # error display
        ee_state = p.getLinkState(robot_id, ee_idx, computeForwardKinematics=True)
        pos_fk = np.array(ee_state[4])
        quat_fk = np.array(ee_state[5])
        pos_err = float(np.linalg.norm(pos_fk - np.array(target_pos)))
        ori_err = float(np.linalg.norm(np.array(p.getDifferenceQuaternion(target_quat, quat_fk))))

        status = "REACH" if (pos_err <= args.pos_tol and ori_err <= args.ori_tol) else "UNREACH"
        p.addUserDebugText(
            f"{status}  pos_err={pos_err:.4f}  ori_err={ori_err:.4f}",
            [0, 0, 1.0],
            textColorRGB=[0, 1, 0] if status == "REACH" else [1, 0.2, 0.2],
            textSize=1.1,
            lifeTime=0.0,
        )

        p.stepSimulation()
        time.sleep(1.0 / 240.0)


if __name__ == "__main__":
    run()
