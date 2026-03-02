# -*- coding: utf-8 -*-
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence
import importlib.resources as resources
import numpy as np

from .types import JointState, Pose
from .errors import ConnectionError, IKError, LimitError
from ..config import load_config
from ..kinematics import RobotModel, fk, matrix_to_rpy, solve_ik
from ..transport import MockTransport, SerialTransport, TCPTransport, TransportBase


def _default_urdf_path() -> Path:
    pkg = "soarmMoce_sdk"
    res = resources.files(pkg) / "resources" / "urdf" / "soarmoce_urdf.urdf"
    with resources.as_file(res) as p:
        return Path(p)


def _resolve_urdf_path(urdf_path: Optional[str]) -> Path:
    if urdf_path is None:
        return _default_urdf_path()
    if urdf_path.startswith("pkg://"):
        rel = urdf_path[len("pkg://"):]
        pkg, rel_path = rel.split("/", 1)
        res = resources.files(pkg) / rel_path
        with resources.as_file(res) as p:
            return Path(p)
    return Path(urdf_path)


class Robot:
    """SoarmMoce SDK main API."""

    def __init__(
        self,
        config_path: Optional[str] = None,
        transport: Optional[TransportBase] = None,
        urdf_path: Optional[str] = None,
        base_link: Optional[str] = None,
        end_link: Optional[str] = None,
    ):
        if config_path is None:
            raise ValueError("Robot() requires a config path. Use Robot.from_config(path).")
        self.config = load_config(config_path)
        self.urdf_path = _resolve_urdf_path(urdf_path or self.config.get("urdf", {}).get("path"))
        self.robot_model = RobotModel(self.urdf_path, base_link=base_link, end_link=end_link)
        self._transport = transport
        self._connected = False

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

    def connect(self) -> None:
        if self._transport is None:
            self._transport = self._create_transport_from_config()
        try:
            self._transport.connect()
        except Exception as e:
            raise ConnectionError(str(e)) from e
        self._connected = True

    def disconnect(self) -> None:
        if self._transport is None:
            return
        try:
            self._transport.disconnect()
        finally:
            self._connected = False

    def get_joint_state(self) -> JointState:
        q = self._protocol_get_q()
        return JointState(q=q)

    def move_joints(self, q: Sequence[float], duration: float = 2.0, speed: Optional[float] = None,
                    accel: Optional[float] = None) -> None:
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.robot_model.dof:
            raise ValueError(f"Expected {self.robot_model.dof} joints, got {q.shape[0]}")
        self._check_limits(q)
        self._protocol_send_movej(q, duration, speed=speed, accel=accel)

    def get_end_effector_pose(self, q: Optional[Sequence[float]] = None) -> Pose:
        if q is None:
            q = self.get_joint_state().q
        T = fk(self.robot_model, np.asarray(q, dtype=float))
        xyz = T[:3, 3]
        rpy = matrix_to_rpy(T[:3, :3])
        return Pose(xyz=xyz, rpy=rpy)

    def move_pose(
        self,
        xyz: Sequence[float],
        rpy: Sequence[float],
        q0: Optional[Sequence[float]] = None,
        seed_policy: str = "current",
        duration: float = 2.0,
    ) -> np.ndarray:
        if q0 is None:
            q0 = self._seed_from_policy(seed_policy)
        res = solve_ik(self.robot_model, np.asarray(xyz, dtype=float), np.asarray(rpy, dtype=float), q0=q0)
        if not res.success:
            raise IKError(f"IK failed: {res.reason} (pos_err={res.pos_err:.4f}, rot_err={res.rot_err:.4f})")
        self.move_joints(res.q, duration=duration)
        return res.q

    def stop(self) -> None:
        self._protocol_stop()

    # ----- protocol wrappers -----

    def _protocol_get_q(self) -> np.ndarray:
        if not self._transport:
            raise ConnectionError("Transport not initialized")
        return self._transport.get_q()

    def _protocol_send_movej(self, q: np.ndarray, duration: float, speed=None, accel=None) -> None:
        if not self._transport:
            raise ConnectionError("Transport not initialized")
        self._transport.send_movej(q, duration, speed=speed, accel=accel)

    def _protocol_stop(self) -> None:
        if not self._transport:
            return
        self._transport.stop()

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
        if policy == "random":
            lower = np.array([l for l, _ in self.robot_model.joint_limits], dtype=float)
            upper = np.array([u for _, u in self.robot_model.joint_limits], dtype=float)
            return lower + (upper - lower) * np.random.random(self.robot_model.dof)
        return self.get_joint_state().q
