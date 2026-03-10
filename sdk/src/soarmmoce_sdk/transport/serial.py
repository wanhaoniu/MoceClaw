# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import threading
import time
import types
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np

from .base import TransportBase

REPO_ROOT = Path(__file__).resolve().parents[4]
MULTI_TURN_RAW_RANGE = 900000
RAW_COUNTS_PER_REV = 4096.0
DEFAULT_MULTI_TURN_JOINTS = ("shoulder_lift", "elbow_flex")
DEFAULT_MOTOR_IDS = {
    "shoulder_pan": 1,
    "shoulder_lift": 2,
    "elbow_flex": 3,
    "wrist_flex": 4,
    "wrist_roll": 5,
    "gripper": 6,
}


def _make_passthrough_unnormalize(original_method, passthrough_ids: set[int]):
    def hybrid_unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:
        result = {}
        for motor_id, value in ids_values.items():
            if motor_id in passthrough_ids:
                result[motor_id] = int(value)
            else:
                result.update(original_method({motor_id: value}))
        return result

    return hybrid_unnormalize


def _smooth_fraction(fraction: float) -> float:
    t = min(1.0, max(0.0, float(fraction)))
    return t * t * (3.0 - 2.0 * t)


def _load_calibration(path: Path):
    from lerobot.motors import MotorCalibration

    data = json.loads(path.read_text(encoding="utf-8"))
    calib: Dict[str, MotorCalibration] = {}
    for name, payload in data.items():
        if not isinstance(payload, dict):
            continue
        try:
            calib[str(name)] = MotorCalibration(
                id=int(payload["id"]),
                drive_mode=int(payload.get("drive_mode", 0)),
                homing_offset=int(payload.get("homing_offset", 0)),
                range_min=int(payload.get("range_min", 0)),
                range_max=int(payload.get("range_max", 4095)),
            )
        except Exception:
            continue
    return calib


def _candidate_calibration_paths(robot_id: str) -> list[Path]:
    return [
        REPO_ROOT / "Software/Slave/calibration/robots/so101_follower" / f"{robot_id}.json",
        REPO_ROOT / "Software/Master/calibration/robots/so101_follower" / f"{robot_id}.json",
        Path.cwd() / "Software/Slave/calibration/robots/so101_follower" / f"{robot_id}.json",
        Path.home() / "Code/SO-ARM-Moce/Software/Slave/calibration/robots/so101_follower" / f"{robot_id}.json",
    ]


class SerialTransport(TransportBase):
    """5DOF real serial transport using the Feetech bus."""

    def __init__(
        self,
        dof: int,
        *,
        joint_names: Sequence[str],
        port: str,
        baudrate: int = 115200,
        timeout: float = 1.0,
        robot_id: str = "follower_moce",
        calibration_path: Optional[str] = None,
        joint_scales: Optional[Dict[str, float]] = None,
        multi_turn_joint_names: Optional[Sequence[str]] = None,
        has_gripper: bool = False,
        arm_p_coefficient: int = 16,
        arm_d_coefficient: int = 8,
        update_hz: float = 25.0,
    ):
        super().__init__(dof, has_gripper=has_gripper)
        self.joint_names = [str(name) for name in joint_names]
        if len(self.joint_names) != self.dof:
            raise ValueError(f"joint_names mismatch: expected {self.dof}, got {len(self.joint_names)}")
        self.port = str(port)
        self.baudrate = int(baudrate)
        self.timeout = float(timeout)
        self.robot_id = str(robot_id)
        self.calibration_path = Path(calibration_path).expanduser() if calibration_path else None
        self.joint_scales = {name: 1.0 for name in self.joint_names}
        for name, value in (joint_scales or {}).items():
            if name in self.joint_scales:
                self.joint_scales[name] = float(value)
        self.multi_turn_joint_names = tuple(
            name for name in (multi_turn_joint_names or DEFAULT_MULTI_TURN_JOINTS) if name in self.joint_names
        )
        self.arm_p_coefficient = int(arm_p_coefficient)
        self.arm_d_coefficient = int(arm_d_coefficient)
        self.update_hz = max(1.0, float(update_hz))
        self._bus = None
        self._connected = False
        self._gripper_open_ratio = None if not self.has_gripper else 1.0
        self._io_lock = threading.RLock()
        self._motion_cond = threading.Condition()
        self._motion_idle = threading.Event()
        self._motion_idle.set()
        self._motion_shutdown = False
        self._motion_version = 0
        self._pending_motion: Optional[tuple[Dict[str, float], float]] = None
        self._motion_thread: Optional[threading.Thread] = None

    def connect(self) -> None:
        try:
            from lerobot.motors import Motor, MotorNormMode
            from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "SerialTransport requires lerobot hardware dependencies"
            ) from exc

        calib_path = self._resolve_calibration_path()
        calibration = _load_calibration(calib_path)
        for name in self.multi_turn_joint_names:
            if name in calibration:
                calibration[name].range_min = -MULTI_TURN_RAW_RANGE
                calibration[name].range_max = MULTI_TURN_RAW_RANGE

        motors = {
            name: Motor(
                DEFAULT_MOTOR_IDS.get(name, i + 1),
                "sts3215",
                MotorNormMode.DEGREES,
            )
            for i, name in enumerate(self.joint_names)
        }
        if self.has_gripper:
            motors["gripper"] = Motor(DEFAULT_MOTOR_IDS["gripper"], "sts3215", MotorNormMode.RANGE_0_100)

        bus = FeetechMotorsBus(port=self.port, motors=motors, calibration=calibration)
        passthrough_ids = {bus.motors[name].id for name in self.multi_turn_joint_names if name in bus.motors}
        bus._unnormalize = types.MethodType(_make_passthrough_unnormalize(bus._unnormalize, passthrough_ids), bus)
        with self._io_lock:
            bus.connect()
            with bus.torque_disabled():
                bus.configure_motors()
                for name in self.joint_names:
                    if name in self.multi_turn_joint_names:
                        bus.write("Lock", name, 0)
                        time.sleep(0.02)
                        bus.write("Min_Position_Limit", name, 0)
                        bus.write("Max_Position_Limit", name, 0)
                        bus.write("Operating_Mode", name, 3)
                        time.sleep(0.02)
                        bus.write("Lock", name, 1)
                    else:
                        bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
                        bus.write("P_Coefficient", name, self.arm_p_coefficient)
                        bus.write("I_Coefficient", name, 0)
                        bus.write("D_Coefficient", name, self.arm_d_coefficient)
                if self.has_gripper and "gripper" in bus.motors:
                    bus.write("Operating_Mode", "gripper", OperatingMode.POSITION.value)
            bus.enable_torque()
        self._bus = bus
        self._connected = True
        self._start_motion_worker()
        self.stop()

    def disconnect(self) -> None:
        self._stop_motion_worker()
        if self._bus is not None:
            disconnect = getattr(self._bus, "disconnect", None)
            if callable(disconnect):
                try:
                    with self._io_lock:
                        disconnect()
                except Exception:
                    pass
        self._bus = None
        self._connected = False

    def get_q(self) -> np.ndarray:
        joint_deg = self._read_joint_state_deg()
        return np.deg2rad(np.array([joint_deg[name] for name in self.joint_names], dtype=float))

    def send_movej(
        self,
        q: Sequence[float],
        duration: float,
        speed: Optional[float] = None,
        accel: Optional[float] = None,
    ) -> None:
        target = np.rad2deg(np.asarray(q, dtype=float).reshape(-1))
        if target.shape[0] != self.dof:
            raise ValueError(f"Expected {self.dof} joints, got {target.shape[0]}")
        target_deg = {name: float(target[i]) for i, name in enumerate(self.joint_names)}
        self._start_motion_worker()
        with self._motion_cond:
            self._motion_version += 1
            self._pending_motion = (target_deg, max(0.0, float(duration)))
            self._motion_idle.clear()
            self._motion_cond.notify_all()

    def stop(self) -> None:
        with self._motion_cond:
            self._motion_version += 1
            self._pending_motion = None
            self._motion_idle.set()
            self._motion_cond.notify_all()
        if not self._connected or self._bus is None:
            return
        state = self._read_joint_state_deg()
        hold = {name: float(state[name]) for name in self.joint_names if name not in self.multi_turn_joint_names}
        if hold:
            with self._io_lock:
                self._bus.sync_write(
                    "Goal_Position",
                    {name: self._joint_to_motor_deg(name, value) for name, value in hold.items()},
                )

    def wait_until_stopped(self, timeout: Optional[float] = None) -> bool:
        if timeout is None:
            self._motion_idle.wait()
            return True
        return bool(self._motion_idle.wait(max(0.0, float(timeout))))

    def set_gripper(self, open_ratio: float, wait: bool = True, timeout: Optional[float] = None) -> None:
        if not self.has_gripper or self._bus is None or "gripper" not in self._bus.motors:
            return
        ratio = float(min(1.0, max(0.0, open_ratio)))
        self._gripper_open_ratio = ratio
        with self._io_lock:
            self._bus.write("Goal_Position", "gripper", ratio * 100.0)
        if wait:
            time.sleep(min(max(0.0, self.timeout), 0.2))

    def get_gripper_open_ratio(self) -> Optional[float]:
        if not self.has_gripper or self._bus is None or "gripper" not in self._bus.motors:
            return None
        try:
            with self._io_lock:
                ratio = float(self._bus.read("Present_Position", "gripper")) / 100.0
        except Exception:
            ratio = self._gripper_open_ratio if self._gripper_open_ratio is not None else 1.0
        return float(min(1.0, max(0.0, ratio)))

    def _resolve_calibration_path(self) -> Path:
        if self.calibration_path and self.calibration_path.exists():
            return self.calibration_path.resolve()
        for path in _candidate_calibration_paths(self.robot_id):
            if path.exists():
                return path.resolve()
        raise FileNotFoundError(
            f"Calibration file not found for {self.robot_id}. Set calibration.path explicitly."
        )

    def _assert_bus(self):
        if self._bus is None or not self._connected:
            raise RuntimeError("SerialTransport not connected")
        return self._bus

    def _start_motion_worker(self) -> None:
        with self._motion_cond:
            if self._motion_thread is not None and self._motion_thread.is_alive():
                return
            self._motion_shutdown = False
            self._motion_thread = threading.Thread(
                target=self._motion_loop,
                name="soarmmoce-serial-motion",
                daemon=True,
            )
            self._motion_thread.start()

    def _stop_motion_worker(self) -> None:
        thread = None
        with self._motion_cond:
            self._motion_shutdown = True
            self._motion_version += 1
            self._pending_motion = None
            self._motion_idle.set()
            self._motion_cond.notify_all()
            thread = self._motion_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=max(0.2, self.timeout))
        with self._motion_cond:
            self._motion_thread = None
            self._motion_shutdown = False

    def _motion_loop(self) -> None:
        while True:
            with self._motion_cond:
                while (not self._motion_shutdown) and self._pending_motion is None:
                    self._motion_idle.set()
                    self._motion_cond.wait()
                if self._motion_shutdown:
                    self._motion_idle.set()
                    return
                motion = self._pending_motion
                version = self._motion_version
                self._pending_motion = None
                self._motion_idle.clear()

            if motion is None:
                continue
            target_deg, duration = motion
            interrupted = self._run_motion(target_deg, duration, version)
            with self._motion_cond:
                if self._motion_shutdown:
                    self._motion_idle.set()
                    return
                if interrupted:
                    continue
                if self._pending_motion is None and version == self._motion_version:
                    self._motion_idle.set()

    def _run_motion(self, target_deg: Dict[str, float], duration: float, version: int) -> bool:
        start = self._read_joint_state_deg()
        max_change = max(abs(float(target_deg[name]) - float(start[name])) for name in self.joint_names)
        steps = max(
            1,
            int(np.ceil(max(0.0, float(duration)) * self.update_hz)),
            int(np.ceil(max_change / 5.0)),
        )
        step_duration = max(0.0, float(duration)) / steps if steps else 0.0
        for step_index in range(1, steps + 1):
            with self._motion_cond:
                if self._motion_shutdown or version != self._motion_version:
                    return True
            alpha = _smooth_fraction(float(step_index) / float(steps))
            waypoint = {
                name: float(start[name]) + (float(target_deg[name]) - float(start[name])) * alpha
                for name in self.joint_names
            }
            self._write_joint_targets_deg(waypoint)
            if step_duration > 0.0:
                with self._motion_cond:
                    interrupted = self._motion_cond.wait_for(
                        lambda: self._motion_shutdown or version != self._motion_version,
                        timeout=step_duration,
                    )
                if interrupted:
                    return True
        return False

    def _joint_to_motor_deg(self, joint_name: str, joint_deg: float) -> float:
        scale = float(self.joint_scales.get(joint_name, 1.0))
        return float(joint_deg) * scale

    def _motor_to_joint_deg(self, joint_name: str, motor_deg: float) -> float:
        scale = float(self.joint_scales.get(joint_name, 1.0))
        return float(motor_deg) / scale

    def _read_joint_state_deg(self) -> Dict[str, float]:
        bus = self._assert_bus()
        with self._io_lock:
            current_motor = bus.sync_read("Present_Position", self.joint_names)
        return {
            name: self._motor_to_joint_deg(name, float(current_motor.get(name, 0.0)))
            for name in self.joint_names
        }

    def _build_bus_command(self, target_joint_deg: Dict[str, float], current_joint_deg: Dict[str, float]) -> Dict[str, float]:
        cmd: Dict[str, float] = {}
        for name, target in target_joint_deg.items():
            if name in self.multi_turn_joint_names:
                current = float(current_joint_deg[name])
                delta_motor_deg = self._joint_to_motor_deg(name, float(target) - current)
                raw_step = int(delta_motor_deg * RAW_COUNTS_PER_REV / 360.0)
                if raw_step != 0:
                    cmd[name] = float(raw_step)
            else:
                cmd[name] = self._joint_to_motor_deg(name, float(target))
        return cmd

    def _write_joint_targets_deg(self, target_joint_deg: Dict[str, float]) -> None:
        bus = self._assert_bus()
        current_joint_deg = self._read_joint_state_deg()
        cmd = self._build_bus_command(target_joint_deg, current_joint_deg)
        if cmd:
            with self._io_lock:
                bus.sync_write("Goal_Position", cmd)
