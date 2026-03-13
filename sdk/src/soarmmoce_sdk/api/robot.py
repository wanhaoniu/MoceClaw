# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.resources as resources
import time
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Union

import numpy as np

from ..config import load_calibration_json
from .errors import (
    CapabilityError,
    ConnectionError,
    IKError,
    LimitError,
    PermissionError as SDKPermissionError,
    SoarmMoceError,
    TimeoutError as SDKTimeoutError,
)
from .types import GripperState, JointState, PermissionState, Pose, RobotState, TwinState
from ..config import load_config
from ..kinematics import RobotModel, fk, matrix_to_rpy, solve_ik
from ..kinematics.frames import rotvec_from_matrix, rpy_to_matrix
from ..transport import MockTransport, SerialTransport, TCPTransport, TransportBase

_PACKAGE_NAME = "soarmmoce_sdk"
_DEFAULT_END_LINK = "wrist_roll"
_DEFAULT_JOINT_ALIASES = {
    "shoulder_pan": "shoulder",
    "shoulder_lift": "shoulder_lift",
    "elbow_flex": "elbow",
    "wrist_flex": "wrist",
    "wrist_roll": "wrist_roll",
}
_DEFAULT_HOME_DEG = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0,
}
_DEFAULT_GUI_ROTVEC_TO_JOINT = {
    "wrist_roll": [0.0, 0.0, 1.0],
}


def _default_urdf_path() -> Path:
    res = resources.files(_PACKAGE_NAME) / "resources" / "urdf" / "soarmoce_urdf.urdf"
    with resources.as_file(res) as p:
        return Path(p)


def _resolve_pkg_resource_uri(uri: str) -> Path:
    rel = str(uri[len("pkg://") :]).strip()
    if not rel or "/" not in rel:
        raise ValueError(f"Invalid pkg URI: {uri!r}")
    pkg, rel_path = rel.split("/", 1)
    pkg = _PACKAGE_NAME
    try:
        res = resources.files(pkg) / rel_path
    except ModuleNotFoundError as exc:
        raise FileNotFoundError(f"Package not found for URI {uri!r}") from exc
    with resources.as_file(res) as p:
        return Path(p)


def _resolve_urdf_path(urdf_path: Optional[str]) -> Path:
    if urdf_path is None:
        return _default_urdf_path()
    raw = str(urdf_path).strip()
    if raw.startswith("pkg://"):
        return _resolve_pkg_resource_uri(raw)
    return Path(raw).expanduser()


class Robot:
    """5DOF SoarmMoce SDK main API."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        transport: Optional[TransportBase] = None,
        urdf_path: Optional[str] = None,
        base_link: Optional[str] = None,
        end_link: Optional[str] = None,
    ):
        self.config = load_config(config_path)
        self.urdf_path = _resolve_urdf_path(urdf_path or self.config.get("urdf", {}).get("path"))
        robot_cfg = self.config.get("robot", {}) if isinstance(self.config, dict) else {}
        joint_aliases = dict(robot_cfg.get("joint_name_aliases") or _DEFAULT_JOINT_ALIASES)
        resolved_end_link = end_link or robot_cfg.get("end_link") or _DEFAULT_END_LINK
        resolved_base_link = base_link or robot_cfg.get("base_link")
        self.robot_model = RobotModel(
            self.urdf_path,
            base_link=resolved_base_link,
            end_link=resolved_end_link,
            joint_name_aliases=joint_aliases,
        )
        self._transport = transport
        self._connected = False
        self._permissions = self._resolve_permissions(self.config.get("permissions", {}))
        self._robot_cfg = robot_cfg if isinstance(robot_cfg, dict) else {}
        self._ik_cfg = self.config.get("ik", {}) if isinstance(self.config, dict) else {}
        self._control_cfg = self.config.get("control", {}) if isinstance(self.config, dict) else {}
        self._calibration_payload = self._load_calibration_payload()
        self._twin_q: Optional[np.ndarray] = None
        self._twin_gripper_ratio: Optional[float] = None
        self._cartesian_locked_joints = [
            str(name) for name in (self._robot_cfg.get("cartesian_locked_joints") or ["wrist_roll"])
            if str(name) in self.robot_model.joint_name_to_index
        ]
        self._gui_rotvec_to_joint = self._resolve_gui_rotation_map(
            self._robot_cfg.get("gui_rotvec_to_joint") or _DEFAULT_GUI_ROTVEC_TO_JOINT
        )

    @classmethod
    def from_config(
        cls,
        path: str,
        transport: Optional[TransportBase] = None,
        urdf_path: Optional[str] = None,
        base_link: Optional[str] = None,
        end_link: Optional[str] = None,
    ) -> "Robot":
        return cls(
            config_path=path,
            transport=transport,
            urdf_path=urdf_path,
            base_link=base_link,
            end_link=end_link,
        )

    @property
    def connected(self) -> bool:
        return bool(self._connected)

    @property
    def permissions(self) -> PermissionState:
        p = self._permissions
        return PermissionState(
            allow_motion=bool(p.allow_motion),
            allow_gripper=bool(p.allow_gripper),
            allow_home=bool(p.allow_home),
            allow_stop=bool(p.allow_stop),
        )

    def set_permissions(
        self,
        *,
        allow_motion: Optional[bool] = None,
        allow_gripper: Optional[bool] = None,
        allow_home: Optional[bool] = None,
        allow_stop: Optional[bool] = None,
    ) -> PermissionState:
        if allow_motion is not None:
            self._permissions.allow_motion = bool(allow_motion)
        if allow_gripper is not None:
            self._permissions.allow_gripper = bool(allow_gripper)
        if allow_home is not None:
            self._permissions.allow_home = bool(allow_home)
        if allow_stop is not None:
            self._permissions.allow_stop = bool(allow_stop)
        return self.permissions

    def connect(self) -> None:
        if self._transport is None:
            self._transport = self._create_transport_from_config()
        try:
            self._transport.connect()
        except Exception as exc:
            self._raise_transport_error(exc, "connect failed")
        self._connected = True
        self._sync_twin_from_actual()

    def disconnect(self) -> None:
        if self._transport is None:
            self._connected = False
            return
        try:
            self._transport.disconnect()
        except Exception as exc:
            self._raise_transport_error(exc, "disconnect failed")
        finally:
            self._connected = False

    def get_joint_state(self) -> JointState:
        q = self._protocol_get_q()
        return JointState(q=np.asarray(q, dtype=float).copy(), names=list(self.robot_model.joint_names))

    def get_end_effector_pose(self, q: Optional[Sequence[float]] = None) -> Pose:
        if q is None:
            q = self.get_joint_state().q
        T = fk(self.robot_model, np.asarray(q, dtype=float))
        xyz = T[:3, 3]
        rpy = matrix_to_rpy(T[:3, :3])
        return Pose(xyz=xyz, rpy=rpy)

    def get_gripper_state(self) -> GripperState:
        if not self._transport:
            return GripperState(available=False, installed=False, open_ratio=None, moving=None)
        installed = bool(self._transport.is_gripper_available())
        if not installed:
            return GripperState(available=False, installed=False, open_ratio=None, moving=None)
        try:
            ratio = self._transport.get_gripper_open_ratio()
        except Exception as exc:
            self._raise_transport_error(exc, "get gripper state failed")
            return GripperState(available=False, installed=False, open_ratio=None, moving=None)
        if ratio is None:
            return GripperState(available=False, installed=False, open_ratio=None, moving=None)
        return GripperState(available=True, installed=True, open_ratio=float(min(1.0, max(0.0, ratio))), moving=None)

    def get_state(self) -> RobotState:
        joint_state = self.get_joint_state()
        tcp_pose = self.get_end_effector_pose(joint_state.q)
        gripper_state = self.get_gripper_state()
        source = self._transport_source_name()
        actual = TwinState(source=source, joint_state=joint_state, tcp_pose=tcp_pose, gripper_state=gripper_state)
        twin = self._build_twin_state(actual_joint_state=joint_state, actual_gripper_state=gripper_state)
        return RobotState(
            connected=self.connected,
            joint_state=joint_state,
            tcp_pose=tcp_pose,
            gripper_state=gripper_state,
            permissions=self.permissions,
            timestamp=time.time(),
            actual=actual,
            twin=twin,
        )

    def move_joints(
        self,
        q: Sequence[float],
        duration: float = 2.0,
        wait: bool = True,
        timeout: Optional[float] = None,
        speed: Optional[float] = None,
        accel: Optional[float] = None,
    ) -> np.ndarray:
        self._require_permission("motion")
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        if q_arr.shape[0] != self.robot_model.dof:
            raise ValueError(f"Expected {self.robot_model.dof} joints, got {q_arr.shape[0]}")
        self._check_limits(q_arr)
        self._protocol_send_movej(q_arr, duration, speed=speed, accel=accel)
        self._twin_q = q_arr.copy()
        if wait:
            self.wait_until_stopped(timeout=timeout)
        return q_arr

    def move_pose(
        self,
        xyz: Sequence[float],
        rpy: Optional[Sequence[float]],
        q0: Optional[Sequence[float]] = None,
        seed_policy: str = "current",
        duration: float = 2.0,
        wait: bool = True,
        timeout: Optional[float] = None,
        speed: Optional[float] = None,
        accel: Optional[float] = None,
    ) -> np.ndarray:
        self._require_permission("motion")
        q_seed = np.asarray(q0, dtype=float).reshape(-1) if q0 is not None else self._seed_from_policy(seed_policy)
        current_pose = self.get_end_effector_pose(q_seed)
        target_xyz = np.asarray(xyz, dtype=float).reshape(3)
        target_rpy = None if rpy is None else np.asarray(rpy, dtype=float).reshape(3)

        preferred_q = q_seed.copy()
        locked_joint_targets: Dict[str, float] = {}
        position_weight = 1.0
        orientation_weight = 0.0
        if target_rpy is None:
            for joint_name in self._cartesian_locked_joints:
                idx = self.robot_model.resolve_joint_index(joint_name)
                locked_joint_targets[joint_name] = float(q_seed[idx])
        else:
            preferred_q = self._apply_gui_rotation_mapping(q_seed, current_pose.rpy, target_rpy)
            pure_orientation_threshold = float(self._ik_cfg.get("orientation_only_pos_threshold_m", 1e-6))
            if float(np.linalg.norm(target_xyz - current_pose.xyz)) <= pure_orientation_threshold:
                position_weight = float(self._ik_cfg.get("orientation_only_position_weight", 0.0))
                orientation_weight = float(self._ik_cfg.get("orientation_only_weight", 1.0))
            else:
                orientation_weight = float(self._ik_cfg.get("orientation_weight", 0.35))

        res = solve_ik(
            self.robot_model,
            target_xyz,
            target_rpy=target_rpy,
            q0=q_seed,
            preferred_q=preferred_q,
            locked_joint_targets=locked_joint_targets,
            max_iters=int(self._ik_cfg.get("max_iters", 200)),
            damping=float(self._ik_cfg.get("damping", 0.05)),
            step_scale=float(self._ik_cfg.get("step_scale", 0.8)),
            pos_tol=float(self._ik_cfg.get("pos_tol", 0.001)),
            rot_tol=float(self._ik_cfg.get("rot_tol", 0.02)),
            max_step=float(self._ik_cfg.get("max_step_rad", 0.15)),
            seed_bias=float(self._ik_cfg.get("seed_bias", 0.02)),
            position_weight=position_weight,
            orientation_weight=orientation_weight,
        )
        max_pos_err = float(self._ik_cfg.get("max_pos_error_m", 0.03))
        if (not res.success) and res.pos_err > max_pos_err:
            raise IKError(f"IK failed: {res.reason} (pos_err={res.pos_err:.4f}, rot_err={res.rot_err:.4f})")

        self.move_joints(
            res.q,
            duration=duration,
            wait=wait,
            timeout=timeout,
            speed=speed,
            accel=accel,
        )
        return res.q

    def move_tcp(
        self,
        x: float,
        y: float,
        z: float,
        rpy: Optional[Sequence[float]] = None,
        frame: str = "base",
        duration: float = 2.0,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> np.ndarray:
        frame_norm = str(frame or "base").strip().lower()
        if frame_norm not in ("base", "tool"):
            raise ValueError("frame must be 'base' or 'tool'")

        current_pose = self.get_end_effector_pose()
        target_rpy = None if rpy is None else np.asarray(rpy, dtype=float).reshape(3)
        if frame_norm == "base":
            target_xyz = np.asarray([x, y, z], dtype=float)
        else:
            delta_tool = np.asarray([x, y, z], dtype=float)
            target_xyz = current_pose.xyz + rpy_to_matrix(current_pose.rpy) @ delta_tool

        return self.move_pose(
            xyz=target_xyz,
            rpy=target_rpy,
            duration=duration,
            wait=wait,
            timeout=timeout,
        )

    def home(
        self,
        duration: float = 2.0,
        wait: bool = True,
        timeout: Optional[float] = None,
    ) -> np.ndarray:
        self._require_permission("home")
        q_home = self._resolve_home_q()
        self.move_joints(q_home, duration=duration, wait=wait, timeout=timeout)
        return q_home

    def set_gripper(self, open_ratio: float, wait: bool = True, timeout: Optional[float] = None) -> None:
        self._require_permission("gripper")
        ratio = float(open_ratio)
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("open_ratio must be within [0.0, 1.0]")
        if not self._transport:
            raise ConnectionError("Transport not initialized")
        if not self._transport.is_gripper_available():
            return
        try:
            self._transport.set_gripper(open_ratio=ratio, wait=wait, timeout=timeout)
        except Exception as exc:
            if isinstance(exc, NotImplementedError):
                raise CapabilityError("set_gripper is unsupported by current transport") from exc
            self._raise_transport_error(exc, "set_gripper failed")
        self._twin_gripper_ratio = ratio

    def rotate_joint(
        self,
        joint: Union[int, str],
        *,
        delta_deg: Optional[float] = None,
        target_deg: Optional[float] = None,
        duration: float = 1.0,
        wait: bool = True,
        timeout: Optional[float] = None,
        speed: Optional[float] = None,
        accel: Optional[float] = None,
    ) -> np.ndarray:
        self._require_permission("motion")
        if (delta_deg is None) == (target_deg is None):
            raise ValueError("Exactly one of delta_deg or target_deg must be provided")

        idx = self._resolve_joint_index(joint)
        current_q = self.get_joint_state().q.copy()
        lo, hi = self.robot_model.joint_limits[idx]
        raw_target = (
            float(np.deg2rad(float(target_deg)))
            if target_deg is not None
            else float(current_q[idx] + np.deg2rad(float(delta_deg)))
        )
        current_q[idx] = float(np.clip(raw_target, float(lo), float(hi)))
        self.move_joints(
            current_q,
            duration=duration,
            wait=wait,
            timeout=timeout,
            speed=speed,
            accel=accel,
        )
        return current_q

    def wait_until_stopped(self, timeout: Optional[float] = None) -> None:
        if not self._transport:
            raise ConnectionError("Transport not initialized")
        try:
            completed = bool(self._transport.wait_until_stopped(timeout=timeout))
        except Exception as exc:
            self._raise_transport_error(exc, "wait_until_stopped failed")
            return
        if not completed:
            raise SDKTimeoutError("wait_until_stopped timeout exceeded")

    def stop(self) -> None:
        self._require_permission("stop")
        self._protocol_stop()
        self._sync_twin_from_actual()

    def _protocol_get_q(self) -> np.ndarray:
        if not self._transport:
            raise ConnectionError("Transport not initialized")
        try:
            return self._transport.get_q()
        except Exception as exc:
            self._raise_transport_error(exc, "get_q failed")

    def _protocol_send_movej(
        self,
        q: np.ndarray,
        duration: float,
        speed: Optional[float] = None,
        accel: Optional[float] = None,
    ) -> None:
        if not self._transport:
            raise ConnectionError("Transport not initialized")
        try:
            self._transport.send_movej(q, duration, speed=speed, accel=accel)
        except Exception as exc:
            self._raise_transport_error(exc, "send_movej failed")

    def _protocol_stop(self) -> None:
        if not self._transport:
            return
        try:
            self._transport.stop()
        except Exception as exc:
            self._raise_transport_error(exc, "stop failed")

    def _create_transport_from_config(self) -> TransportBase:
        tcfg = self.config.get("transport", {}) if isinstance(self.config, dict) else {}
        control_cfg = self.config.get("control", {}) if isinstance(self.config, dict) else {}
        ttype = str(tcfg.get("type", "mock") or "mock").strip().lower()
        gripper_available = self._to_bool(
            tcfg.get("gripper_available"),
            True if ttype == "mock" else False,
        )
        if ttype == "mock":
            return MockTransport(self.robot_model.dof, has_gripper=gripper_available)
        if ttype == "serial":
            return SerialTransport(
                self.robot_model.dof,
                joint_names=self.robot_model.joint_names,
                port=str(tcfg.get("port", "/dev/ttyACM0")),
                baudrate=int(tcfg.get("baudrate", 115200)),
                timeout=float(tcfg.get("timeout", 1.0)),
                robot_id=str(tcfg.get("robot_id", "follower_moce")),
                calibration_path=str(tcfg.get("calibration_path", "") or "") or None,
                joint_scales=self._resolve_joint_scales(),
                multi_turn_joint_names=self._resolve_multi_turn_joint_names(),
                has_gripper=gripper_available,
                arm_p_coefficient=int(tcfg.get("arm_p_coefficient", 16)),
                arm_d_coefficient=int(tcfg.get("arm_d_coefficient", 8)),
                update_hz=float(tcfg.get("update_hz", control_cfg.get("hz", 25.0))),
            )
        if ttype == "tcp":
            proto = self.config.get("protocol", {}) if isinstance(self.config, dict) else {}
            return TCPTransport(
                self.robot_model.dof,
                host=str(tcfg.get("host", "127.0.0.1")),
                port=int(tcfg.get("port", 6666)),
                timeout=float(tcfg.get("timeout", 2.0)),
                joint_names=self.robot_model.joint_names,
                joint_map=proto.get("sdk_to_server_map", {}),
                unit=str(proto.get("unit", "deg")),
                max_retries=int(proto.get("max_retries", 1)),
                use_seq=bool(proto.get("use_seq", False)),
            )
        raise ConnectionError(f"Unknown transport type: {ttype}")

    def _resolve_joint_scales(self) -> Dict[str, float]:
        robot_cfg = self._robot_cfg if isinstance(self._robot_cfg, dict) else {}
        raw = robot_cfg.get("joint_scales") or {}
        out = {name: 1.0 for name in self.robot_model.joint_names}
        for name, value in raw.items():
            if name in out:
                out[name] = float(value)
        return out

    def _resolve_multi_turn_joint_names(self) -> list[str]:
        robot_cfg = self._robot_cfg if isinstance(self._robot_cfg, dict) else {}
        raw = robot_cfg.get("multi_turn_joints") or ["shoulder_lift", "elbow_flex"]
        return [str(name) for name in raw if str(name) in self.robot_model.joint_name_to_index]

    def _resolve_home_q(self) -> np.ndarray:
        meta = self._calibration_payload.get("_meta")
        if isinstance(meta, dict):
            home_joint_deg = meta.get("home_joint_deg")
            if isinstance(home_joint_deg, dict):
                resolved = np.array(
                    [float(home_joint_deg.get(name, _DEFAULT_HOME_DEG.get(name, 0.0))) for name in self.robot_model.joint_names],
                    dtype=float,
                )
                return np.deg2rad(resolved)
        return np.deg2rad(
            np.array([_DEFAULT_HOME_DEG.get(name, 0.0) for name in self.robot_model.joint_names], dtype=float)
        )

    def _load_calibration_payload(self) -> Dict[str, Any]:
        tcfg = self.config.get("transport", {}) if isinstance(self.config, dict) else {}
        ccfg = self.config.get("calibration", {}) if isinstance(self.config, dict) else {}

        candidates: list[Path] = []
        explicit = str(ccfg.get("path", "") or tcfg.get("calibration_path", "") or "").strip()
        if explicit:
            candidates.append(Path(explicit).expanduser())

        robot_id = str(tcfg.get("robot_id", "follower_moce") or "follower_moce").strip()
        try:
            from ..transport.serial import _candidate_calibration_paths

            candidates.extend(_candidate_calibration_paths(robot_id))
        except Exception:
            pass

        seen: set[str] = set()
        for path in candidates:
            key = str(path)
            if key in seen:
                continue
            seen.add(key)
            if not path.exists():
                continue
            try:
                payload = load_calibration_json(str(path))
            except Exception:
                continue
            if isinstance(payload, dict):
                return payload
        return {}

    def _check_limits(self, q: np.ndarray) -> None:
        lower = np.array([l for l, _ in self.robot_model.joint_limits], dtype=float)
        upper = np.array([u for _, u in self.robot_model.joint_limits], dtype=float)
        if np.any(q < lower) or np.any(q > upper):
            raise LimitError("Joint limits exceeded")

    def _seed_from_policy(self, policy: str) -> np.ndarray:
        policy = str(policy or "current").lower()
        if policy == "zeros":
            return np.zeros(self.robot_model.dof, dtype=float)
        return self.get_joint_state().q

    def _apply_gui_rotation_mapping(self, q_seed: np.ndarray, current_rpy: np.ndarray, target_rpy: np.ndarray) -> np.ndarray:
        out = np.asarray(q_seed, dtype=float).reshape(-1).copy()
        current_R = rpy_to_matrix(np.asarray(current_rpy, dtype=float))
        target_R = rpy_to_matrix(np.asarray(target_rpy, dtype=float))
        delta_rotvec = rotvec_from_matrix(target_R @ current_R.T)
        for joint_name, weights in self._gui_rotvec_to_joint.items():
            if joint_name not in self.robot_model.joint_name_to_index:
                continue
            idx = self.robot_model.joint_name_to_index[joint_name]
            out[idx] = out[idx] + float(np.dot(weights, delta_rotvec))
        return out

    def _resolve_joint_index(self, joint: Union[int, str]) -> int:
        if isinstance(joint, (int, np.integer)):
            idx = int(joint)
            if idx < 0 or idx >= self.robot_model.dof:
                raise ValueError(f"joint index out of range: {idx}")
            return idx
        key = str(joint or "").strip().lower()
        if not key:
            raise ValueError("joint name is required")
        for idx, name in enumerate(self.robot_model.joint_names):
            if str(name).strip().lower() == key:
                return idx
        for idx, name in enumerate(self.robot_model.joint_names):
            if key in str(name).strip().lower():
                return idx
        raise ValueError(f"joint not found: {joint}")

    def _transport_source_name(self) -> str:
        if self._transport is None:
            return "uninitialized"
        return type(self._transport).__name__.replace("Transport", "").lower()

    def _build_twin_state(
        self,
        *,
        actual_joint_state: JointState,
        actual_gripper_state: Optional[GripperState],
    ) -> TwinState:
        if self._twin_q is None or np.asarray(self._twin_q).shape != actual_joint_state.q.shape:
            twin_q = np.asarray(actual_joint_state.q, dtype=float).copy()
        else:
            twin_q = np.asarray(self._twin_q, dtype=float).copy()
        twin_joint_state = JointState(q=twin_q, names=list(self.robot_model.joint_names))
        twin_pose = self.get_end_effector_pose(twin_q)
        twin_gripper_state = self._build_twin_gripper_state(actual_gripper_state)
        return TwinState(
            source="kinematic_twin",
            joint_state=twin_joint_state,
            tcp_pose=twin_pose,
            gripper_state=twin_gripper_state,
        )

    def _build_twin_gripper_state(self, actual: Optional[GripperState]) -> Optional[GripperState]:
        if actual is None:
            return None
        if not actual.installed or not actual.available:
            return GripperState(available=False, installed=False, open_ratio=None, moving=None)
        ratio = actual.open_ratio if self._twin_gripper_ratio is None else self._twin_gripper_ratio
        if ratio is None:
            return GripperState(available=False, installed=True, open_ratio=None, moving=None)
        return GripperState(
            available=True,
            installed=True,
            open_ratio=float(min(1.0, max(0.0, ratio))),
            moving=actual.moving,
        )

    def _sync_twin_from_actual(self) -> None:
        if not self._connected or self._transport is None:
            return
        try:
            self._twin_q = np.asarray(self._protocol_get_q(), dtype=float).reshape(-1).copy()
        except Exception:
            self._twin_q = None
        try:
            gripper = self.get_gripper_state()
        except Exception:
            gripper = None
        if gripper is None or not gripper.installed or not gripper.available or gripper.open_ratio is None:
            self._twin_gripper_ratio = None
        else:
            self._twin_gripper_ratio = float(gripper.open_ratio)

    @staticmethod
    def _to_bool(value: object, default: bool) -> bool:
        if value is None:
            return bool(default)
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            raw = value.strip().lower()
            if raw in {"1", "true", "yes", "y", "on"}:
                return True
            if raw in {"0", "false", "no", "n", "off"}:
                return False
        return bool(default)

    @classmethod
    def _resolve_permissions(cls, raw: object) -> PermissionState:
        data = raw if isinstance(raw, dict) else {}
        return PermissionState(
            allow_motion=cls._to_bool(data.get("allow_motion"), True),
            allow_gripper=cls._to_bool(data.get("allow_gripper"), True),
            allow_home=cls._to_bool(data.get("allow_home"), True),
            allow_stop=cls._to_bool(data.get("allow_stop"), True),
        )

    @staticmethod
    def _resolve_gui_rotation_map(raw: object) -> Dict[str, np.ndarray]:
        data = raw if isinstance(raw, dict) else {}
        mapping: Dict[str, np.ndarray] = {}
        for joint_name, vector in data.items():
            try:
                arr = np.asarray(vector, dtype=float).reshape(3)
            except Exception:
                continue
            mapping[str(joint_name)] = arr
        return mapping

    def _require_permission(self, operation: str) -> None:
        op = str(operation or "").strip().lower()
        if op == "motion" and not self._permissions.allow_motion:
            raise SDKPermissionError("motion commands are disabled by SDK permissions")
        if op == "gripper" and not self._permissions.allow_gripper:
            raise SDKPermissionError("gripper commands are disabled by SDK permissions")
        if op == "home" and not self._permissions.allow_home:
            raise SDKPermissionError("home command is disabled by SDK permissions")
        if op == "stop" and not self._permissions.allow_stop:
            raise SDKPermissionError("stop command is disabled by SDK permissions")

    @staticmethod
    def _raise_transport_error(exc: Exception, default_message: str) -> None:
        if isinstance(exc, SoarmMoceError):
            raise exc
        if isinstance(exc, PermissionError):
            raise SDKPermissionError(str(exc) or default_message) from exc
        if isinstance(exc, NotImplementedError):
            raise CapabilityError(str(exc) or default_message) from exc
        if isinstance(exc, TimeoutError):
            raise SDKTimeoutError(str(exc) or default_message) from exc
        raise ConnectionError(str(exc) or default_message) from exc
