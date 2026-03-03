#!/usr/bin/env python3
"""OpenClaw tools for SO-ARM-Moce control.

This module provides:
- move_robot_arm(x, y, z, ...)
- get_robot_state()
- get_camera_frame(...)
- stop_robot()

Default backend is local PyBullet simulation for safety and portability.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import threading
import time
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None
import numpy as np
try:
    import pybullet as pb  # type: ignore
except Exception:
    pb = None


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = Path("/tmp/mocearm_openclaw_frames")
DEFAULT_URDF_CANDIDATES = [
    REPO_ROOT / "sdk" / "src" / "soarmMoce_sdk" / "resources" / "urdf" / "soarmoce_urdf.urdf",
    REPO_ROOT / "Urdf" / "urdf" / "soarmoce_urdf.urdf",
    REPO_ROOT / "Urdf" / "urdf" / "soarmoce_purple.urdf",
    REPO_ROOT / "Soarm101" / "SO101" / "so101_new_calib.urdf",
]


def _find_urdf_path() -> Path:
    raw_env = os.environ.get("MOCEARM_URDF_PATH")
    if raw_env is not None and raw_env.strip():
        env_path = Path(raw_env.strip()).expanduser()
        if env_path.exists() and env_path.is_file():
            return env_path.resolve()
        raise FileNotFoundError(f"MOCEARM_URDF_PATH is not a valid URDF file: {env_path}")
    for path in DEFAULT_URDF_CANDIDATES:
        if path.exists() and path.is_file():
            return path.resolve()
    raise FileNotFoundError("No usable URDF found. Set MOCEARM_URDF_PATH.")


def _safe_float(v: Any, name: str) -> float:
    try:
        return float(v)
    except Exception as exc:
        raise ValueError(f"{name} must be a number, got {v!r}") from exc


def _safe_int(v: Any, name: str) -> int:
    try:
        return int(v)
    except Exception as exc:
        raise ValueError(f"{name} must be an integer, got {v!r}") from exc


def _rotation_from_quat(quat_xyzw: Tuple[float, float, float, float]) -> np.ndarray:
    return np.array(pb.getMatrixFromQuaternion(quat_xyzw), dtype=float).reshape(3, 3)


@dataclass
class SimJointInfo:
    joint_indices: List[int]
    joint_names: List[str]
    joint_limits: List[Tuple[float, float]]
    ee_link_index: int


class MoceArmSimBackend:
    """Local simulation backend using PyBullet DIRECT."""

    def __init__(self, urdf_path: Optional[Path] = None):
        if pb is None:
            raise RuntimeError("pybullet is not available; install pybullet to use this backend")
        self.urdf_path = urdf_path.resolve() if urdf_path else _find_urdf_path()
        self._lock = threading.RLock()
        self._halted = False
        self._client = pb.connect(pb.DIRECT)
        self._robot_id = pb.loadURDF(
            str(self.urdf_path),
            basePosition=[0.0, 0.0, 0.0],
            baseOrientation=[0.0, 0.0, 0.0, 1.0],
            useFixedBase=True,
            physicsClientId=self._client,
        )
        self._joint_info = self._init_joint_info()
        self._joint_values = np.zeros(len(self._joint_info.joint_indices), dtype=float)
        self._reset_to_zero()

    def _init_joint_info(self) -> SimJointInfo:
        joint_indices: List[int] = []
        joint_names: List[str] = []
        joint_limits: List[Tuple[float, float]] = []

        num_joints = pb.getNumJoints(self._robot_id, physicsClientId=self._client)
        for idx in range(num_joints):
            info = pb.getJointInfo(self._robot_id, idx, physicsClientId=self._client)
            jtype = int(info[2])
            if jtype not in (pb.JOINT_REVOLUTE, pb.JOINT_PRISMATIC):
                continue
            name = info[1].decode("utf-8")
            lo, hi = float(info[8]), float(info[9])
            if (not math.isfinite(lo)) or (not math.isfinite(hi)) or lo >= hi:
                lo, hi = -math.pi, math.pi
            joint_indices.append(idx)
            joint_names.append(name)
            joint_limits.append((lo, hi))

        if not joint_indices:
            raise RuntimeError("No movable joints found in URDF")

        ee_idx = self._detect_ee_link_index()
        if ee_idx is None:
            raise RuntimeError("Failed to detect end-effector link index")

        return SimJointInfo(
            joint_indices=joint_indices,
            joint_names=joint_names,
            joint_limits=joint_limits,
            ee_link_index=ee_idx,
        )

    def _detect_ee_link_index(self) -> Optional[int]:
        num_joints = pb.getNumJoints(self._robot_id, physicsClientId=self._client)
        preferred = ("wrist_roll", "wrist", "gripper", "tool0", "ee", "end_effector")
        found_by_name: Optional[int] = None
        parents = set()
        for j in range(num_joints):
            info = pb.getJointInfo(self._robot_id, j, physicsClientId=self._client)
            jname = info[1].decode("utf-8").strip().lower()
            if found_by_name is None and jname in preferred:
                found_by_name = j
            pidx = int(info[16])
            if pidx >= 0:
                parents.add(pidx)
        if found_by_name is not None:
            return found_by_name
        leaves = sorted(set(range(num_joints)) - parents)
        if leaves:
            return int(leaves[-1])
        return int(num_joints - 1) if num_joints > 0 else None

    def _reset_to_zero(self):
        for i, joint_idx in enumerate(self._joint_info.joint_indices):
            lo, hi = self._joint_info.joint_limits[i]
            q0 = float(np.clip(0.0, lo, hi))
            pb.resetJointState(self._robot_id, joint_idx, q0, physicsClientId=self._client)
            self._joint_values[i] = q0
        pb.stepSimulation(physicsClientId=self._client)

    def _ee_pose(self) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        state = pb.getLinkState(
            self._robot_id,
            int(self._joint_info.ee_link_index),
            computeForwardKinematics=True,
            physicsClientId=self._client,
        )
        xyz = np.array(state[4], dtype=float)
        quat = tuple(float(x) for x in state[5])
        return xyz, quat

    def stop_robot(self) -> Dict[str, Any]:
        with self._lock:
            self._halted = True
            return {"ok": True, "stopped": True, "backend": "pybullet-sim"}

    def move_robot_arm(
        self,
        x: float,
        y: float,
        z: float,
        frame: str = "base",
        duration: float = 2.0,
        wait: bool = True,
    ) -> Dict[str, Any]:
        x = _safe_float(x, "x")
        y = _safe_float(y, "y")
        z = _safe_float(z, "z")
        duration = float(np.clip(_safe_float(duration, "duration"), 0.2, 20.0))
        if frame not in ("base", "tool"):
            raise ValueError("frame must be 'base' or 'tool'")

        # Conservative workspace guard rails (meters).
        if not (-0.60 <= x <= 0.60 and -0.60 <= y <= 0.60 and 0.00 <= z <= 0.80):
            raise ValueError("Target is outside safety workspace")

        with self._lock:
            if self._halted:
                self._halted = False

            xyz_now, quat_now = self._ee_pose()
            target_xyz = np.array([x, y, z], dtype=float)
            target_quat = quat_now

            lower = [float(lo) for lo, _ in self._joint_info.joint_limits]
            upper = [float(hi) for _, hi in self._joint_info.joint_limits]
            ranges = [max(1e-4, hi - lo) for lo, hi in self._joint_info.joint_limits]
            rest = [float(v) for v in self._joint_values.tolist()]

            q_full = pb.calculateInverseKinematics(
                self._robot_id,
                int(self._joint_info.ee_link_index),
                targetPosition=[float(target_xyz[0]), float(target_xyz[1]), float(target_xyz[2])],
                targetOrientation=target_quat,
                lowerLimits=lower,
                upperLimits=upper,
                jointRanges=ranges,
                restPoses=rest,
                maxNumIterations=220,
                residualThreshold=1e-5,
                physicsClientId=self._client,
            )

            q_vals = list(q_full) if q_full is not None else []
            q_out = np.array(rest, dtype=float)
            for i, ji in enumerate(self._joint_info.joint_indices):
                if ji < len(q_vals):
                    raw = float(q_vals[ji])
                elif i < len(q_vals):
                    raw = float(q_vals[i])
                else:
                    raw = float(q_out[i])
                lo, hi = self._joint_info.joint_limits[i]
                q_out[i] = float(np.clip(raw, lo, hi))

            steps = max(1, int(duration * 60.0)) if wait else 1
            q_start = np.array(self._joint_values, dtype=float)
            for s in range(1, steps + 1):
                if self._halted:
                    break
                alpha = float(s) / float(steps)
                q_interp = q_start + (q_out - q_start) * alpha
                for i, joint_idx in enumerate(self._joint_info.joint_indices):
                    pb.resetJointState(
                        self._robot_id,
                        int(joint_idx),
                        float(q_interp[i]),
                        physicsClientId=self._client,
                    )
                pb.stepSimulation(physicsClientId=self._client)
                if wait:
                    time.sleep(duration / float(steps))

            self._joint_values = np.array(q_out, dtype=float)
            xyz_after, quat_after = self._ee_pose()
            err = float(np.linalg.norm(xyz_after - target_xyz))

            return {
                "ok": err <= 0.10,
                "backend": "pybullet-sim",
                "target_xyz_m": [float(x), float(y), float(z)],
                "actual_xyz_m": [float(xyz_after[0]), float(xyz_after[1]), float(xyz_after[2])],
                "position_error_m": err,
                "joint_values_rad": [float(v) for v in self._joint_values.tolist()],
                "ee_quat_xyzw": [float(v) for v in quat_after],
            }

    def get_robot_state(self) -> Dict[str, Any]:
        with self._lock:
            xyz, quat = self._ee_pose()
            joints = {
                self._joint_info.joint_names[i]: float(self._joint_values[i])
                for i in range(len(self._joint_info.joint_names))
            }
            return {
                "ok": True,
                "backend": "pybullet-sim",
                "connected": True,
                "mode": "simulation",
                "halted": bool(self._halted),
                "joints_rad": joints,
                "ee_xyz_m": [float(xyz[0]), float(xyz[1]), float(xyz[2])],
                "ee_quat_xyzw": [float(v) for v in quat],
                "ee_link_index": int(self._joint_info.ee_link_index),
            }

    def _render_camera_bgr(self, source: str, width: int, height: int) -> np.ndarray:
        w = max(160, min(1920, _safe_int(width, "width")))
        h = max(120, min(1080, _safe_int(height, "height")))

        if source == "eye_in_hand":
            xyz, quat = self._ee_pose()
            R = _rotation_from_quat(quat)
            cam_pos = xyz + R @ np.array([0.060, 0.000, 0.028], dtype=float)
            focal = cam_pos + R @ np.array([0.320, 0.000, 0.000], dtype=float)
            up = R @ np.array([0.000, 0.000, 1.000], dtype=float)
        elif source == "scene":
            cam_pos = np.array([0.76, -1.08, 0.66], dtype=float)
            focal = np.array([0.00, 0.00, 0.18], dtype=float)
            up = np.array([0.00, 0.00, 1.00], dtype=float)
        else:
            raise ValueError("source must be 'eye_in_hand' or 'scene'")

        view = pb.computeViewMatrix(
            cameraEyePosition=[float(cam_pos[0]), float(cam_pos[1]), float(cam_pos[2])],
            cameraTargetPosition=[float(focal[0]), float(focal[1]), float(focal[2])],
            cameraUpVector=[float(up[0]), float(up[1]), float(up[2])],
        )
        proj = pb.computeProjectionMatrixFOV(
            fov=62.0,
            aspect=float(w) / float(h),
            nearVal=0.01,
            farVal=5.0,
        )

        _w, _h, rgba, _depth, _mask = pb.getCameraImage(
            width=w,
            height=h,
            viewMatrix=view,
            projectionMatrix=proj,
            renderer=pb.ER_TINY_RENDERER,
            physicsClientId=self._client,
        )
        arr = np.asarray(rgba, dtype=np.uint8).reshape(h, w, 4)
        rgb = np.ascontiguousarray(arr[:, :, :3])
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        return bgr

    def get_camera_frame(
        self,
        source: str = "eye_in_hand",
        width: int = 960,
        height: int = 720,
        format: str = "jpg",
        return_mode: str = "path",
    ) -> Dict[str, Any]:
        if cv2 is None:
            raise RuntimeError("opencv-python is not available; install opencv-python to encode frames")
        fmt = str(format).strip().lower()
        if fmt not in ("jpg", "png"):
            raise ValueError("format must be 'jpg' or 'png'")
        mode = str(return_mode).strip().lower()
        if mode not in ("path", "base64"):
            raise ValueError("return_mode must be 'path' or 'base64'")

        with self._lock:
            frame_bgr = self._render_camera_bgr(source=source, width=width, height=height)
            ext = ".jpg" if fmt == "jpg" else ".png"
            ok, encoded = cv2.imencode(ext, frame_bgr)
            if not ok:
                raise RuntimeError("Failed to encode camera frame")
            blob = bytes(encoded.tobytes())

            out: Dict[str, Any] = {
                "ok": True,
                "backend": "pybullet-sim",
                "source": source,
                "width": int(frame_bgr.shape[1]),
                "height": int(frame_bgr.shape[0]),
                "format": fmt,
                "timestamp": time.time(),
            }
            if mode == "path":
                DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
                fpath = DEFAULT_OUTPUT_DIR / f"frame_{int(time.time() * 1000)}{ext}"
                fpath.write_bytes(blob)
                out["path"] = str(fpath)
            else:
                out["base64"] = base64.b64encode(blob).decode("ascii")
            return out

    def close(self):
        with self._lock:
            try:
                pb.disconnect(self._client)
            except Exception:
                pass


class MoceArmToolService:
    """High-level tool service that OpenClaw can call."""

    def __init__(self):
        self._backend = MoceArmSimBackend()

    def move_robot_arm(
        self,
        x: float,
        y: float,
        z: float,
        frame: str = "base",
        duration: float = 2.0,
        wait: bool = True,
    ) -> Dict[str, Any]:
        return self._backend.move_robot_arm(x=x, y=y, z=z, frame=frame, duration=duration, wait=bool(wait))

    def get_robot_state(self) -> Dict[str, Any]:
        return self._backend.get_robot_state()

    def get_camera_frame(
        self,
        source: str = "eye_in_hand",
        width: int = 960,
        height: int = 720,
        format: str = "jpg",
        return_mode: str = "path",
    ) -> Dict[str, Any]:
        return self._backend.get_camera_frame(
            source=source,
            width=width,
            height=height,
            format=format,
            return_mode=return_mode,
        )

    def stop_robot(self) -> Dict[str, Any]:
        return self._backend.stop_robot()

    def close(self):
        self._backend.close()


TOOL_SCHEMAS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "move_robot_arm",
            "description": "Move robot TCP in Cartesian space.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "number", "description": "target x in meters, base frame"},
                    "y": {"type": "number", "description": "target y in meters, base frame"},
                    "z": {"type": "number", "description": "target z in meters, base frame"},
                    "frame": {"type": "string", "enum": ["base", "tool"], "default": "base"},
                    "duration": {"type": "number", "minimum": 0.2, "maximum": 20.0, "default": 2.0},
                    "wait": {"type": "boolean", "default": True},
                },
                "required": ["x", "y", "z"],
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_robot_state",
            "description": "Read robot connection and kinematic state.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_camera_frame",
            "description": "Capture one frame from simulation camera.",
            "parameters": {
                "type": "object",
                "properties": {
                    "source": {
                        "type": "string",
                        "enum": ["eye_in_hand", "scene"],
                        "default": "eye_in_hand",
                    },
                    "width": {"type": "integer", "minimum": 160, "maximum": 1920, "default": 960},
                    "height": {"type": "integer", "minimum": 120, "maximum": 1080, "default": 720},
                    "format": {"type": "string", "enum": ["jpg", "png"], "default": "jpg"},
                    "return_mode": {"type": "string", "enum": ["path", "base64"], "default": "path"},
                },
                "additionalProperties": False,
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "stop_robot",
            "description": "Immediately stop current robot action.",
            "parameters": {"type": "object", "properties": {}, "additionalProperties": False},
        },
    },
]


_SERVICE: Optional[MoceArmToolService] = None
_SERVICE_LOCK = threading.Lock()


def _get_service() -> MoceArmToolService:
    global _SERVICE
    with _SERVICE_LOCK:
        if _SERVICE is None:
            _SERVICE = MoceArmToolService()
        return _SERVICE


def move_robot_arm(
    x: float,
    y: float,
    z: float,
    frame: str = "base",
    duration: float = 2.0,
    wait: bool = True,
) -> Dict[str, Any]:
    return _get_service().move_robot_arm(x=x, y=y, z=z, frame=frame, duration=duration, wait=wait)


def get_robot_state() -> Dict[str, Any]:
    return _get_service().get_robot_state()


def get_camera_frame(
    source: str = "eye_in_hand",
    width: int = 960,
    height: int = 720,
    format: str = "jpg",
    return_mode: str = "path",
) -> Dict[str, Any]:
    return _get_service().get_camera_frame(
        source=source,
        width=width,
        height=height,
        format=format,
        return_mode=return_mode,
    )


def stop_robot() -> Dict[str, Any]:
    return _get_service().stop_robot()


TOOL_FUNCTIONS = {
    "move_robot_arm": move_robot_arm,
    "get_robot_state": get_robot_state,
    "get_camera_frame": get_camera_frame,
    "stop_robot": stop_robot,
}


def call_tool(name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    if name not in TOOL_FUNCTIONS:
        raise ValueError(f"Unknown tool: {name}")
    fn = TOOL_FUNCTIONS[name]
    return fn(**arguments)


def _cli():
    parser = argparse.ArgumentParser(description="MOCEARM OpenClaw tools")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("schema", help="Print tool schemas as JSON")

    call = sub.add_parser("call", help="Call one tool")
    call.add_argument("--name", required=True)
    call.add_argument("--args", default="{}")

    args = parser.parse_args()
    if args.cmd == "schema":
        print(json.dumps(TOOL_SCHEMAS, ensure_ascii=False, indent=2))
        return
    if args.cmd == "call":
        payload = json.loads(args.args)
        result = call_tool(args.name, payload)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    _cli()
