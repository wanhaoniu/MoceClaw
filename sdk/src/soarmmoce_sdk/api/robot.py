# -*- coding: utf-8 -*-
from __future__ import annotations

import importlib.resources as resources
import time
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

from .errors import (
    CapabilityError,
    ConnectionError,
    IKError,
    LimitError,
    PermissionError as SDKPermissionError,
    SoarmMoceError,
    TimeoutError as SDKTimeoutError,
)
from .types import GripperState, JointState, PermissionState, Pose, RobotState
from ..config import load_config
from ..kinematics import RobotModel, fk, matrix_to_rpy, solve_ik
from ..kinematics.frames import rpy_to_matrix
from ..transport import MockTransport, SerialTransport, TCPTransport, TransportBase

_PACKAGE_NAME = "soarmmoce_sdk"


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
    """SoarmMoce SDK main API.

    Notes:
        - If ``config_path`` is None, package default config
          ``resources/configs/soarm_moce.yaml`` is used.
        - ``wait``/``timeout`` are best-effort. When transport cannot verify real
          motion completion, ``wait_until_stopped`` resolves deterministically as a no-op.
    """

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
        self.robot_model = RobotModel(self.urdf_path, base_link=base_link, end_link=end_link)
        self._transport = transport
        self._connected = False
        self._permissions = self._resolve_permissions(self.config.get("permissions", {}))

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

    # ----- public API -----

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
        return JointState(q=q)

    def get_end_effector_pose(self, q: Optional[Sequence[float]] = None) -> Pose:
        if q is None:
            q = self.get_joint_state().q
        T = fk(self.robot_model, np.asarray(q, dtype=float))
        xyz = T[:3, 3]
        rpy = matrix_to_rpy(T[:3, :3])
        return Pose(xyz=xyz, rpy=rpy)

    def get_state(self) -> RobotState:
        joint_state = self.get_joint_state()
        tcp_pose = self.get_end_effector_pose(joint_state.q)
        return RobotState(
            connected=self.connected,
            joint_state=joint_state,
            tcp_pose=tcp_pose,
            gripper_state=self.get_gripper_state(),
            permissions=self.permissions,
            timestamp=time.time(),
        )

    def get_gripper_state(self) -> GripperState:
        if not self._transport:
            return GripperState(available=False, open_ratio=None, moving=None)
        try:
            ratio = self._transport.get_gripper_open_ratio()
        except NotImplementedError:
            ratio = None
        except Exception as exc:
            self._raise_transport_error(exc, "get gripper state failed")
            return GripperState(available=False, open_ratio=None, moving=None)
        if ratio is None:
            return GripperState(available=False, open_ratio=None, moving=None)
        ratio_f = float(max(0.0, min(1.0, float(ratio))))
        return GripperState(available=True, open_ratio=ratio_f, moving=None)

    def move_joints(
        self,
        q: Sequence[float],
        duration: float = 2.0,
        wait: bool = True,
        timeout: Optional[float] = None,
        speed: Optional[float] = None,
        accel: Optional[float] = None,
    ) -> None:
        self._require_permission("motion")
        q_arr = np.asarray(q, dtype=float).reshape(-1)
        if q_arr.shape[0] != self.robot_model.dof:
            raise ValueError(f"Expected {self.robot_model.dof} joints, got {q_arr.shape[0]}")
        self._check_limits(q_arr)
        self._protocol_send_movej(q_arr, duration, speed=speed, accel=accel)
        if wait:
            self.wait_until_stopped(timeout=timeout)

    def move_pose(
        self,
        xyz: Sequence[float],
        rpy: Sequence[float],
        q0: Optional[Sequence[float]] = None,
        seed_policy: str = "current",
        duration: float = 2.0,
        wait: bool = True,
        timeout: Optional[float] = None,
        speed: Optional[float] = None,
        accel: Optional[float] = None,
    ) -> np.ndarray:
        self._require_permission("motion")
        if q0 is None:
            q0 = self._seed_from_policy(seed_policy)

        res = solve_ik(
            self.robot_model,
            np.asarray(xyz, dtype=float),
            np.asarray(rpy, dtype=float),
            q0=np.asarray(q0, dtype=float),
        )
        if not res.success:
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

        if rpy is None:
            target_rpy = current_pose.rpy
        else:
            target_rpy = np.asarray(rpy, dtype=float).reshape(-1)
            if target_rpy.shape[0] != 3:
                raise ValueError("rpy must contain exactly 3 values")

        if frame_norm == "base":
            target_xyz = np.asarray([x, y, z], dtype=float)
        else:
            delta_tool = np.asarray([x, y, z], dtype=float)
            rot = rpy_to_matrix(current_pose.rpy)
            target_xyz = current_pose.xyz + rot @ delta_tool

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
        home_cfg = self.config.get("home", {}) if isinstance(self.config, dict) else {}
        joints = None
        if isinstance(home_cfg, dict):
            joints = home_cfg.get("joints", home_cfg.get("q"))

        if joints is None:
            q_home = np.zeros(self.robot_model.dof, dtype=float)
        else:
            q_home = np.asarray(joints, dtype=float).reshape(-1)
            if q_home.shape[0] != self.robot_model.dof:
                raise ValueError(
                    f"Home joint count mismatch: expected {self.robot_model.dof}, got {q_home.shape[0]}"
                )

        self.move_joints(q_home, duration=duration, wait=wait, timeout=timeout)
        return q_home

    def set_gripper(self, open_ratio: float, wait: bool = True, timeout: Optional[float] = None) -> None:
        self._require_permission("gripper")
        ratio = float(open_ratio)
        if ratio < 0.0 or ratio > 1.0:
            raise ValueError("open_ratio must be within [0.0, 1.0]")
        if not self._transport:
            raise ConnectionError("Transport not initialized")

        try:
            self._transport.set_gripper(open_ratio=ratio, wait=False, timeout=timeout)
        except Exception as exc:
            if isinstance(exc, NotImplementedError):
                raise CapabilityError("set_gripper is unsupported by current transport") from exc
            self._raise_transport_error(exc, "set_gripper failed")

        if wait:
            self.wait_until_stopped(timeout=timeout)

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

    # ----- protocol wrappers -----

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

    # ----- helpers -----

    def _create_transport_from_config(self) -> TransportBase:
        tcfg = self.config.get("transport", {})
        ttype = tcfg.get("type", "mock")
        if ttype == "mock":
            return MockTransport(self.robot_model.dof)
        if ttype == "serial":
            return SerialTransport(
                self.robot_model.dof,
                port=tcfg.get("port", "/dev/ttyACM0"),
                baudrate=tcfg.get("baudrate", 115200),
                timeout=tcfg.get("timeout", 1.0),
            )
        if ttype == "tcp":
            proto = self.config.get("protocol", {})
            return TCPTransport(
                self.robot_model.dof,
                host=tcfg.get("host", "127.0.0.1"),
                port=tcfg.get("port", 6666),
                timeout=tcfg.get("timeout", 2.0),
                joint_names=self.robot_model.joint_names,
                joint_map=proto.get("sdk_to_server_map", {}),
                unit=proto.get("unit", "deg"),
                max_retries=int(proto.get("max_retries", 1)),
                use_seq=bool(proto.get("use_seq", False)),
            )
        raise ConnectionError(f"Unknown transport type: {ttype}")

    def _check_limits(self, q: np.ndarray) -> None:
        lower = np.array([l for l, _ in self.robot_model.joint_limits], dtype=float)
        upper = np.array([u for _, u in self.robot_model.joint_limits], dtype=float)
        if np.any(q < lower) or np.any(q > upper):
            raise LimitError("Joint limits exceeded")

    def _seed_from_policy(self, policy: str) -> np.ndarray:
        policy = policy.lower()
        if policy == "current":
            return self.get_joint_state().q
        if policy == "zeros":
            return np.zeros(self.robot_model.dof, dtype=float)
        return self.get_joint_state().q

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
