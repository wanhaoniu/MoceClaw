#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local record/play tool for a directly connected follower arm.

Supports two modes:
- moce: soarmMoce follower mapping (with multi-turn joints)
- 101:  soarm101 follower mapping

Examples:
  python3 local_record_play.py --mode moce --port /dev/ttyACM0 record demo --duration 10
  python3 local_record_play.py --mode moce --port /dev/ttyACM0 play demo --times 2
  python3 local_record_play.py --mode 101 list
"""

from __future__ import annotations

import argparse
import json
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

from lerobot.motors import Motor, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

from so101_utils import load_calibration


JOINTS = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]


@dataclass(frozen=True)
class ModeConfig:
    name: str
    follower_id: str
    scale: List[float]
    offset: List[float]
    multi_turn_joints: List[str]
    gripper_norm_mode: str


MODE_CONFIGS = {
    "moce": ModeConfig(
        name="moce",
        follower_id="follower_moce",
        scale=[1.0, 5.6, 5.6, -1.0, 1.0, 5.6],
        offset=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        multi_turn_joints=["shoulder_lift", "elbow_flex", "gripper"],
        gripper_norm_mode="degrees",
    ),
    "101": ModeConfig(
        name="101",
        follower_id="brown_arm_follower",
        scale=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        offset=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        multi_turn_joints=[],
        gripper_norm_mode="range_0_100",
    ),
}


def make_hybrid_unnormalize(original_method, passthrough_ids: set[int]):
    def hybrid_unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:
        result = {}
        for mid, val in ids_values.items():
            if mid in passthrough_ids:
                result[mid] = int(val)
            else:
                partial = original_method({mid: val})
                result.update(partial)
        return result

    return hybrid_unnormalize


def recordings_dir(base: Path, mode: str) -> Path:
    return (base / mode).resolve()


def save_recording(base: Path, mode: str, name: str, payload: Dict) -> Path:
    out_dir = recordings_dir(base, mode)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_path


def load_recording(base: Path, mode: str, name: str) -> Dict | None:
    path = recordings_dir(base, mode) / f"{name}.json"
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def list_recordings(base: Path, mode: str) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    folder = recordings_dir(base, mode)
    if not folder.exists():
        return out

    for fp in sorted(folder.glob("*.json")):
        try:
            data = json.loads(fp.read_text(encoding="utf-8"))
            frames = data.get("frames", [])
            duration = float(frames[-1].get("t", 0.0)) if frames else 0.0
            out[fp.stem] = {
                "frames": len(frames),
                "duration": duration,
                "created_at": data.get("created_at", ""),
            }
        except Exception:
            continue
    return out


def delete_recording(base: Path, mode: str, name: str) -> bool:
    path = recordings_dir(base, mode) / f"{name}.json"
    if not path.exists():
        return False
    path.unlink()
    return True


def compute_joint_spans(frames: List[Dict]) -> Dict[str, float]:
    spans: Dict[str, float] = {}
    for j in JOINTS:
        vals = [float(f.get("joints", {}).get(j, 0.0)) for f in frames]
        if vals:
            spans[j] = max(vals) - min(vals)
        else:
            spans[j] = 0.0
    return spans


def _unwrap_mod_delta(delta: int, modulo: int) -> int:
    """Return shortest signed delta under modular counter space."""
    if modulo <= 0:
        return delta
    half = modulo // 2
    if delta > half:
        delta -= modulo
    elif delta < -half:
        delta += modulo
    return delta


def _sanitize_single_turn_wrap(frames: List[Dict], multi_turn: List[str], rec_scale: List[float], gripper_norm_mode: str) -> List[Dict]:
    """Unwrap single-turn joint trajectories (legacy recordings may contain +/-360 jumps)."""
    if not frames:
        return frames

    single_turn_deg = []
    for j in JOINTS:
        if j in multi_turn:
            continue
        if j == "gripper" and gripper_norm_mode == "range_0_100":
            continue
        single_turn_deg.append(j)

    if not single_turn_deg:
        return frames

    prev: Dict[str, float] = {}
    out: List[Dict] = []
    for frame in frames:
        joints = frame.get("joints", {})
        if not isinstance(joints, dict):
            out.append(frame)
            continue

        fixed = dict(joints)
        for j in single_turn_deg:
            if j not in joints:
                continue
            try:
                curr = float(joints[j])
            except Exception:
                continue

            if j not in prev:
                prev[j] = curr
                fixed[j] = curr
                continue

            idx = JOINTS.index(j)
            scale = abs(float(rec_scale[idx])) if float(rec_scale[idx]) != 0.0 else 1.0
            period = 360.0 / scale
            half = 0.5 * period

            # Repeatedly fold into nearest-neighbor branch.
            while curr - prev[j] > half:
                curr -= period
            while curr - prev[j] < -half:
                curr += period

            prev[j] = curr
            fixed[j] = curr

        out.append({"t": frame.get("t", 0.0), "joints": fixed})

    return out


class LocalArmRecordPlay:
    def __init__(
        self,
        mode: str,
        port: str,
        follower_id: str,
        calib_dir: Path,
        record_hz: float,
        rec_dir: Path,
    ):
        self.mode_cfg = MODE_CONFIGS[mode]
        self.mode = mode
        self.port = port
        self.follower_id = follower_id
        self.calib_dir = calib_dir
        self.record_hz = max(1.0, float(record_hz))
        self.record_dt = 1.0 / self.record_hz
        self.rec_dir = rec_dir

        self.scale = list(self.mode_cfg.scale)
        self.offset = list(self.mode_cfg.offset)
        self.multi_turn = list(self.mode_cfg.multi_turn_joints)

        self.bus: FeetechMotorsBus | None = None
        self.slave_start_pos: Dict[str, float] = {}
        self.keep_torque_on_close: bool = False

    def _configure_moce_for_record(self) -> None:
        if self.mode != "moce" or not self.multi_turn:
            return
        bus = self._assert_bus()
        with bus.torque_disabled():
            for name in self.multi_turn:
                bus.write("Lock", name, 0)
                time.sleep(0.02)
                # 临时切回位置模式用于人工拖动录制（读取 Present_Position）
                bus.write("Min_Position_Limit", name, 0)
                bus.write("Max_Position_Limit", name, 4095)
                bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
                bus.write("P_Coefficient", name, 16)
                bus.write("I_Coefficient", name, 0)
                bus.write("D_Coefficient", name, 16)
                time.sleep(0.02)
                bus.write("Lock", name, 1)
        print(f"[MODE] moce record mode enabled for {self.multi_turn}")

    def _configure_moce_for_play(self) -> None:
        if self.mode != "moce" or not self.multi_turn:
            return
        bus = self._assert_bus()
        with bus.torque_disabled():
            for name in self.multi_turn:
                bus.write("Lock", name, 0)
                time.sleep(0.02)
                bus.write("Min_Position_Limit", name, 0)
                bus.write("Max_Position_Limit", name, 0)
                bus.write("Operating_Mode", name, 3)
                time.sleep(0.02)
                bus.write("Lock", name, 1)
        print(f"[MODE] moce play mode enabled for {self.multi_turn}")

    def connect(self) -> None:
        calib = load_calibration(self.follower_id, calib_dir=self.calib_dir)

        if self.mode == "moce":
            for name in self.multi_turn:
                if name in calib:
                    calib[name].range_min = -900000
                    calib[name].range_max = 900000

        gripper_norm = (
            MotorNormMode.RANGE_0_100
            if self.mode_cfg.gripper_norm_mode == "range_0_100"
            else MotorNormMode.DEGREES
        )

        bus = FeetechMotorsBus(
            port=self.port,
            motors={
                "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
                "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
                "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
                "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
                "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
                "gripper": Motor(6, "sts3215", gripper_norm),
            },
            calibration=calib,
        )

        if self.mode == "moce":
            passthrough_ids = {JOINTS.index(name) + 1 for name in self.multi_turn}
            bus._unnormalize = types.MethodType(make_hybrid_unnormalize(bus._unnormalize, passthrough_ids), bus)

        bus.connect()
        with bus.torque_disabled():
            bus.configure_motors()
            for name in JOINTS:
                if name in self.multi_turn:
                    bus.write("Lock", name, 0)
                    time.sleep(0.05)
                    bus.write("Min_Position_Limit", name, 0)
                    bus.write("Max_Position_Limit", name, 0)
                    bus.write("Operating_Mode", name, 3)
                    time.sleep(0.05)
                    bus.write("Lock", name, 1)
                else:
                    bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
                    bus.write("P_Coefficient", name, 32)
                    bus.write("I_Coefficient", name, 0)
                    bus.write("D_Coefficient", name, 16)

        bus.enable_torque()
        self.bus = bus

        self.slave_start_pos = {}
        for name in JOINTS:
            if name not in self.multi_turn:
                self.slave_start_pos[name] = float(self.bus.read("Present_Position", name))
        print(f"[CONNECT] mode={self.mode} follower_id={self.follower_id} port={self.port}")

    def _align_to_record_start_normalized(self, rec_start_pos: Dict[str, float], duration: float, hz: float) -> None:
        bus = self._assert_bus()
        steps = max(1, int(max(0.02, duration) * max(1.0, hz)))
        dt = max(0.001, duration / steps)

        current = bus.sync_read("Present_Position")
        start_pos = {j: float(current.get(j, 0.0)) for j in JOINTS if j in rec_start_pos}
        target_pos = {j: float(rec_start_pos[j]) for j in JOINTS if j in rec_start_pos}
        if not target_pos:
            return

        t_begin = time.perf_counter()
        for i in range(1, steps + 1):
            ratio = i / steps
            cmd = {j: start_pos[j] + (target_pos[j] - start_pos[j]) * ratio for j in target_pos}
            if cmd:
                bus.sync_write("Goal_Position", cmd)

            target_tick = t_begin + i * dt
            now = time.perf_counter()
            if now < target_tick:
                time.sleep(target_tick - now)

    def close(self) -> None:
        if self.bus is None:
            return
        keep_torque = bool(self.keep_torque_on_close)
        try:
            self.bus.disconnect(disable_torque=not keep_torque)
        except Exception:
            pass
        self.keep_torque_on_close = False
        self.bus = None

    def _assert_bus(self) -> FeetechMotorsBus:
        if self.bus is None:
            raise RuntimeError("Bus is not connected")
        return self.bus

    def _hold_last_pose(self, last_cmd_written: Dict[str, float]) -> None:
        """Hold target at end of playback while keeping torque enabled."""
        bus = self._assert_bus()
        current = bus.sync_read("Present_Position")
        cmd: Dict[str, float] = {}

        if self.multi_turn:
            # For multi-turn joints in mode=3, Goal_Position is incremental.
            # Do not write absolute hold for these joints here.
            hold_joints = [j for j in JOINTS if j not in self.multi_turn]
        else:
            hold_joints = list(JOINTS)

        for j in hold_joints:
            if j in last_cmd_written:
                cmd[j] = float(last_cmd_written[j])
            else:
                cmd[j] = float(current.get(j, 0.0))

        if cmd:
            bus.sync_write("Goal_Position", cmd)

    def record(self, name: str, duration: float | None) -> bool:
        bus = self._assert_bus()
        self._configure_moce_for_record()

        print(f"[REC] mode={self.mode} name={name} hz={self.record_hz:.1f}")
        if duration is None:
            print("[REC] 手动停止: Ctrl+C")
        else:
            print(f"[REC] 自动停止: {duration:.2f}s")

        frames: List[Dict] = []

        try:
            with bus.torque_disabled():
                start_pos = bus.sync_read("Present_Position")
                start_pos_raw = bus.sync_read("Present_Position", normalize=False)
                last_raws = {k: int(start_pos_raw.get(k, 0)) for k in JOINTS}
                last_norm = {k: float(start_pos.get(k, 0.0)) for k in JOINTS}
                accum = {k: 0.0 for k in JOINTS}

                # t=0 帧
                frames.append({"t": 0.0, "joints": {k: 0.0 for k in JOINTS}})

                t0 = time.perf_counter()
                try:
                    while True:
                        now = time.perf_counter()
                        elapsed = now - t0

                        if duration is not None and elapsed >= duration:
                            break

                        curr_pos = bus.sync_read("Present_Position")
                        curr_pos_raw = bus.sync_read("Present_Position", normalize=False)
                        q_local: Dict[str, float] = {}

                        for k in JOINTS:
                            idx = JOINTS.index(k)
                            scale = float(self.scale[idx]) if float(self.scale[idx]) != 0.0 else 1.0

                            # 101 模式下 gripper 使用 RANGE_0_100 归一化，直接在归一化域累计。
                            if k == "gripper" and self.mode_cfg.gripper_norm_mode == "range_0_100":
                                curr_norm = float(curr_pos.get(k, last_norm[k]))
                                delta_norm = curr_norm - last_norm[k]
                                accum[k] += delta_norm / scale
                                last_norm[k] = curr_norm
                                q_local[k] = accum[k]
                                continue

                            curr_raw = int(curr_pos_raw.get(k, last_raws[k]))
                            if k in self.multi_turn:
                                # multi-turn 关节按 16-bit 计数解环绕
                                diff_raw = _unwrap_mod_delta(curr_raw - last_raws[k], 65536)
                            else:
                                # single-turn 关节按 12-bit(0~4095) 解环绕
                                diff_raw = _unwrap_mod_delta(curr_raw - last_raws[k], 4096)
                            delta_motor_deg = diff_raw * 360.0 / 4096.0
                            accum[k] += delta_motor_deg / scale
                            last_raws[k] = curr_raw
                            q_local[k] = accum[k]

                        frames.append({"t": elapsed, "joints": q_local})
                        time.sleep(self.record_dt)
                except KeyboardInterrupt:
                    print("\n[REC] 手动停止")
        finally:
            # 录制完成后切回回放模式，避免后续命令模式不一致
            self._configure_moce_for_play()

        if len(frames) <= 1:
            print("[REC] 没有有效帧")
            return False

        if self.multi_turn:
            for j in self.multi_turn:
                vals = [float(f.get("joints", {}).get(j, 0.0)) for f in frames]
                span = (max(vals) - min(vals)) if vals else 0.0
                if abs(span) < 1e-6:
                    print(f"[REC][WARN] {j} 轨迹无变化（跨度={span:.6f}），回放时该关节可能不动")

        payload = {
            "name": name,
            "mode": self.mode,
            "hz": self.record_hz,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "meta": {
                "joint_order": JOINTS,
                "multi_turn_joints": self.multi_turn,
                "scale": self.scale,
                "offset": self.offset,
                "record_start_pos": {k: float(v) for k, v in start_pos.items()},
                "record_start_raw": {k: int(v) for k, v in start_pos_raw.items()},
            },
            "frames": frames,
        }

        path = save_recording(self.rec_dir, self.mode, name, payload)
        print(f"[REC] 保存完成: {path} ({len(frames)} frames, {frames[-1]['t']:.2f}s)")
        return True

    def _align_to_record_start(self, rec_start_raw: Dict[str, int], duration: float, hz: float) -> None:
        bus = self._assert_bus()
        steps = max(1, int(max(0.02, duration) * max(1.0, hz)))
        dt = max(0.001, duration / steps)

        current_raw = bus.sync_read("Present_Position", normalize=False)
        start_raw = {j: int(current_raw.get(j, 0)) for j in JOINTS if j in rec_start_raw}
        target_raw = {j: int(rec_start_raw[j]) for j in JOINTS if j in rec_start_raw}
        if not target_raw:
            return

        # last_abs 仅用于 multi-turn 关节把“绝对目标”转换为“每步增量”。
        last_abs = start_raw.copy()
        t_begin = time.perf_counter()

        for i in range(1, steps + 1):
            ratio = i / steps
            cmd: Dict[str, int] = {}
            for j in target_raw:
                s = start_raw[j]
                t = target_raw[j]
                desired_abs = int(round(s + (t - s) * ratio))

                if j in self.multi_turn:
                    delta = desired_abs - last_abs[j]
                    if delta < -32768:
                        delta += 65536
                    elif delta > 32768:
                        delta -= 65536
                    if delta != 0:
                        cmd[j] = int(delta)
                        last_abs[j] += int(delta)
                else:
                    cmd[j] = desired_abs
                    last_abs[j] = desired_abs

            if cmd:
                bus.sync_write("Goal_Position", cmd, normalize=False)

            target_tick = t_begin + i * dt
            now = time.perf_counter()
            if now < target_tick:
                time.sleep(target_tick - now)

    def play(
        self,
        name: str,
        times: int = 1,
        align_duration: float = 2.0,
        align_hz: float = 120.0,
        no_align: bool = True,
        speed: float = 0.65,
        wrist_roll_max_step: float = 8.0,
        hold_last: bool = True,
    ) -> bool:
        bus = self._assert_bus()
        self._configure_moce_for_play()
        data = load_recording(self.rec_dir, self.mode, name)
        if data is None:
            print(f"[PLAY] 未找到录制: {name}")
            return False

        frames = data.get("frames", [])
        if not frames:
            print(f"[PLAY] 录制为空: {name}")
            return False

        spans = compute_joint_spans(frames)
        if self.mode == "moce" and self.multi_turn:
            dead = [j for j in self.multi_turn if abs(spans.get(j, 0.0)) < 1e-6]
            if dead:
                print(f"[PLAY][WARN] 多圈关节轨迹全0: {dead}")
                print("[PLAY][WARN] 将继续执行回放（这些关节会保持不动）")

        meta = data.get("meta", {}) if isinstance(data.get("meta", {}), dict) else {}
        rec_start_pos = meta.get("record_start_pos", {}) if isinstance(meta.get("record_start_pos", {}), dict) else {}
        rec_start_raw = meta.get("record_start_raw", {}) if isinstance(meta.get("record_start_raw", {}), dict) else {}
        rec_scale = meta.get("scale", self.scale)
        if not isinstance(rec_scale, list) or len(rec_scale) != len(JOINTS):
            rec_scale = self.scale
        frames = _sanitize_single_turn_wrap(frames, self.multi_turn, rec_scale, self.mode_cfg.gripper_norm_mode)

        speed = max(0.05, float(speed))
        wrist_roll_max_step = max(0.1, float(wrist_roll_max_step))

        print(f"[PLAY] mode={self.mode} name={name} loops={times} frames={len(frames)} speed={speed:.2f}x")
        if no_align:
            print("[PLAY] 默认最小模式：跳过预对齐，按当前姿态相对回放")
        elif self.mode == "moce":
            # moce 多圈场景下，multi-turn 起点对齐不稳定；
            # 仅对齐单圈关节，避免起始姿态飘偏。
            single_turn = [j for j in JOINTS if j not in self.multi_turn]
            if rec_start_raw:
                rec_start_raw_single = {j: int(rec_start_raw[j]) for j in single_turn if j in rec_start_raw}
                if rec_start_raw_single:
                    print(
                        f"[PLAY] moce 仅对齐单圈关节: {list(rec_start_raw_single.keys())} "
                        f"(duration={align_duration:.2f}s, hz={align_hz:.1f})"
                    )
                    self._align_to_record_start(
                        rec_start_raw_single,
                        duration=max(0.02, align_duration),
                        hz=max(1.0, align_hz),
                    )
                else:
                    print("[PLAY] moce 录制缺少单圈起点 raw 信息，跳过预对齐")
            elif rec_start_pos:
                rec_start_pos_single = {j: float(rec_start_pos[j]) for j in single_turn if j in rec_start_pos}
                if rec_start_pos_single:
                    print(
                        f"[PLAY] moce 使用旧录制回退对齐单圈关节: {list(rec_start_pos_single.keys())} "
                        f"(duration={align_duration:.2f}s, hz={align_hz:.1f})"
                    )
                    self._align_to_record_start_normalized(
                        rec_start_pos_single,
                        duration=max(0.02, align_duration),
                        hz=max(1.0, align_hz),
                    )
                else:
                    print("[PLAY] moce 录制缺少单圈起点信息，跳过预对齐")
            else:
                print("[PLAY] moce 录制不含起点信息，跳过预对齐（建议手动摆到接近起点后播放）")
        elif rec_start_raw:
            print(f"[PLAY] 对齐到录制初始点: duration={align_duration:.2f}s, hz={align_hz:.1f}")
            self._align_to_record_start(rec_start_raw, duration=max(0.02, align_duration), hz=max(1.0, align_hz))
        elif rec_start_pos and not self.multi_turn:
            print(f"[PLAY] 使用旧录制回退对齐(归一化): duration={align_duration:.2f}s, hz={align_hz:.1f}")
            self._align_to_record_start_normalized(rec_start_pos, duration=max(0.02, align_duration), hz=max(1.0, align_hz))
        else:
            print("[PLAY] 警告: 该录制不含起点信息，跳过预对齐")

        # 每次播放重置 multi-turn 内部状态
        last_cmd_written: Dict[str, float] = {}
        for loop in range(times):
            print(f"[PLAY] loop {loop + 1}/{times}")
            start_time = time.perf_counter()
            last_virtual = {k: 0.0 for k in self.multi_turn}
            first_runs = {k: True for k in self.multi_turn}
            loop_base_pos = bus.sync_read("Present_Position")
            last_single_cmd = dict(loop_base_pos)

            for frame in frames:
                frame_t = float(frame.get("t", 0.0))
                target_t = start_time + (frame_t / speed)
                now = time.perf_counter()
                if now < target_t:
                    time.sleep(target_t - now)

                joints = frame.get("joints", {})
                if not isinstance(joints, dict):
                    continue

                cmd: Dict[str, float] = {}
                for j in JOINTS:
                    if j not in joints:
                        continue

                    try:
                        lj = float(joints[j])
                    except Exception:
                        continue

                    idx = JOINTS.index(j)
                    if j in self.multi_turn:
                        if first_runs[j]:
                            last_virtual[j] = lj
                            first_runs[j] = False
                        else:
                            delta_virtual = lj - last_virtual[j]
                            delta_motor = delta_virtual * float(rec_scale[idx])
                            raw_step = int(delta_motor * 4096.0 / 360.0)
                            if raw_step != 0:
                                cmd[j] = float(raw_step)
                                sent_motor = raw_step * 360.0 / 4096.0
                                if float(rec_scale[idx]) != 0:
                                    sent_virtual = sent_motor / float(rec_scale[idx])
                                    last_virtual[j] += sent_virtual
                    else:
                        if no_align:
                            base = float(loop_base_pos.get(j, self.slave_start_pos.get(j, 0.0)))
                        else:
                            base = float(rec_start_pos.get(j, loop_base_pos.get(j, self.slave_start_pos.get(j, 0.0))))
                        target_abs = base + (lj * float(rec_scale[idx])) + self.offset[idx]
                        if j == "wrist_roll":
                            prev = float(last_single_cmd.get(j, target_abs))
                            step = target_abs - prev
                            if step > wrist_roll_max_step:
                                target_abs = prev + wrist_roll_max_step
                            elif step < -wrist_roll_max_step:
                                target_abs = prev - wrist_roll_max_step
                        cmd[j] = target_abs
                        last_single_cmd[j] = target_abs

                if cmd:
                    bus.sync_write("Goal_Position", cmd)
                    last_cmd_written.update(cmd)

        if hold_last:
            self._hold_last_pose(last_cmd_written)
            self.keep_torque_on_close = True
            print("[PLAY] 已锁定到最后一帧，并保持力矩上锁")
        else:
            self.keep_torque_on_close = False

        print("[PLAY] 播放完成")
        return True

    def probe(self, seconds: float = 8.0, hz: float = 10.0) -> bool:
        bus = self._assert_bus()
        hz = max(1.0, float(hz))
        dt = 1.0 / hz
        seconds = max(0.5, float(seconds))

        self._configure_moce_for_record()
        print(f"[PROBE] mode={self.mode} seconds={seconds:.1f} hz={hz:.1f}")
        print(f"[PROBE] joints={self.multi_turn if self.multi_turn else JOINTS}")
        print("[PROBE] move joints by hand and observe values...")

        t0 = time.perf_counter()
        try:
            with bus.torque_disabled():
                while True:
                    now = time.perf_counter()
                    elapsed = now - t0
                    if elapsed >= seconds:
                        break
                    pos = bus.sync_read("Present_Position")
                    raw = bus.sync_read("Present_Position", normalize=False)
                    if self.multi_turn:
                        parts = []
                        for j in self.multi_turn:
                            parts.append(f"{j}:deg={float(pos.get(j, 0.0)):.3f},raw={int(raw.get(j, 0))}")
                        print("[PROBE] " + " | ".join(parts))
                    else:
                        parts = []
                        for j in JOINTS:
                            parts.append(f"{j}:{float(pos.get(j, 0.0)):.2f}")
                        print("[PROBE] " + " | ".join(parts))
                    time.sleep(dt)
        finally:
            self._configure_moce_for_play()
        return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Local record/play for soarmMoce or soarm101.")
    parser.add_argument("--mode", choices=["101", "moce"], required=True, help="Robot mode: 101 or moce")
    parser.add_argument("--port", default="/dev/ttyACM1", help="Follower serial port")
    parser.add_argument("--follower-id", default="", help="Override follower calibration id")
    parser.add_argument(
        "--calib-dir",
        default="",
        help="Calibration directory (default: SO102/Software/Slave/calibration/robots/so101_follower)",
    )
    parser.add_argument("--record-hz", type=float, default=20.0, help="Record sampling rate")
    parser.add_argument(
        "--recordings-dir",
        default="",
        help="Directory for saved recordings (default: SO102/Software/Slave/recordings_local)",
    )

    sub = parser.add_subparsers(dest="cmd", required=True)

    p_record = sub.add_parser("record", help="Record a new trajectory")
    p_record.add_argument("name", help="Recording name")
    p_record.add_argument("--duration", type=float, default=None, help="Record duration in seconds (omit for Ctrl+C stop)")
    p_record.add_argument("--port", dest="port_sub", default=None, help="Follower serial port (subcommand override)")

    p_play = sub.add_parser("play", help="Play a saved trajectory")
    p_play.add_argument("name", help="Recording name")
    p_play.add_argument("--times", type=int, default=1, help="Loop count")
    p_play.add_argument("--align", action="store_true", help="Enable pre-alignment before playback (default: off)")
    p_play.add_argument("--no-align", action="store_true", help="Skip pre-alignment before playback (legacy; default already off)")
    p_play.add_argument("--port", dest="port_sub", default=None, help="Follower serial port (subcommand override)")
    p_play.add_argument("--speed", type=float, default=0.65, help="Playback speed ratio, 1.0=original, <1 slower")
    p_play.add_argument("--wrist-roll-max-step", type=float, default=8.0, help="Per-frame max step for wrist_roll (degrees)")
    p_play.add_argument("--no-hold-last", action="store_true", help="Do not hold/lock last frame after playback")
    p_play.add_argument("--align-duration", type=float, default=2.0, help="Pre-align duration to recording start (seconds)")
    p_play.add_argument("--align-hz", type=float, default=120.0, help="Pre-align interpolation rate (Hz)")

    p_list = sub.add_parser("list", help="List recordings")
    p_list.add_argument("--port", dest="port_sub", default=None, help="Follower serial port (subcommand override)")
    p_list.add_argument("--verbose", action="store_true", help="Print details")

    p_delete = sub.add_parser("delete", help="Delete recording")
    p_delete.add_argument("--port", dest="port_sub", default=None, help="Follower serial port (subcommand override)")
    p_delete.add_argument("name", help="Recording name")

    p_analyze = sub.add_parser("analyze", help="Analyze trajectory joint spans")
    p_analyze.add_argument("--port", dest="port_sub", default=None, help="Follower serial port (subcommand override)")
    p_analyze.add_argument("name", help="Recording name")

    p_probe = sub.add_parser("probe", help="Probe live joint readback for debugging")
    p_probe.add_argument("--port", dest="port_sub", default=None, help="Follower serial port (subcommand override)")
    p_probe.add_argument("--seconds", type=float, default=8.0, help="Probe duration in seconds")
    p_probe.add_argument("--hz", type=float, default=10.0, help="Probe print rate")

    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    mode_cfg = MODE_CONFIGS[args.mode]
    follower_id = args.follower_id.strip() if args.follower_id else mode_cfg.follower_id
    effective_port = getattr(args, "port_sub", None) or args.port

    script_dir = Path(__file__).resolve().parent
    calib_dir = Path(args.calib_dir).expanduser().resolve() if args.calib_dir else (script_dir / "calibration" / "robots" / "so101_follower")
    rec_dir = Path(args.recordings_dir).expanduser().resolve() if args.recordings_dir else (script_dir / "recordings_local")

    if args.cmd == "list":
        rows = list_recordings(rec_dir, args.mode)
        if not rows:
            print(f"[LIST] 无录制 (mode={args.mode})")
            return 0
        for name, info in rows.items():
            if args.verbose:
                print(f"- {name}: {info['frames']} frames, {info['duration']:.2f}s, created={info.get('created_at','')}")
            else:
                print(f"- {name}: {info['frames']} frames, {info['duration']:.2f}s")
        return 0

    if args.cmd == "delete":
        ok = delete_recording(rec_dir, args.mode, args.name)
        print(f"[DELETE] {'ok' if ok else 'not found'}: {args.name}")
        return 0 if ok else 1

    if args.cmd == "analyze":
        data = load_recording(rec_dir, args.mode, args.name)
        if data is None:
            print(f"[ANALYZE] 未找到录制: {args.name}")
            return 1
        frames = data.get("frames", [])
        if not frames:
            print(f"[ANALYZE] 录制为空: {args.name}")
            return 1
        spans = compute_joint_spans(frames)
        print(f"[ANALYZE] mode={args.mode} name={args.name} frames={len(frames)}")
        for j in JOINTS:
            print(f"- {j}: span={spans.get(j, 0.0):.6f}")
        return 0

    app = LocalArmRecordPlay(
        mode=args.mode,
        port=effective_port,
        follower_id=follower_id,
        calib_dir=calib_dir,
        record_hz=args.record_hz,
        rec_dir=rec_dir,
    )

    try:
        app.connect()

        if args.cmd == "record":
            ok = app.record(args.name, duration=args.duration)
            return 0 if ok else 1

        if args.cmd == "play":
            no_align = (not bool(args.align)) or bool(args.no_align)
            ok = app.play(
                args.name,
                times=max(1, int(args.times)),
                align_duration=max(0.02, float(args.align_duration)),
                align_hz=max(1.0, float(args.align_hz)),
                no_align=no_align,
                speed=float(args.speed),
                wrist_roll_max_step=float(args.wrist_roll_max_step),
                hold_last=(not bool(args.no_hold_last)),
            )
            return 0 if ok else 1

        if args.cmd == "probe":
            ok = app.probe(seconds=float(args.seconds), hz=float(args.hz))
            return 0 if ok else 1

        parser.error(f"Unknown command: {args.cmd}")
        return 2
    finally:
        app.close()


if __name__ == "__main__":
    raise SystemExit(main())
