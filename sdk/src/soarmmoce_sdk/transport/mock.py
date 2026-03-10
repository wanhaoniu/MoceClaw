# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .base import TransportBase

try:
    import fcntl  # type: ignore
except Exception:  # pragma: no cover
    fcntl = None


_MOCK_SHARED_FILE_ENV = "SOARMMOCE_MOCK_SHARED_STATE_FILE"


class MockTransport(TransportBase):
    """Stateful mock transport with shared wall-clock interpolation."""

    def __init__(self, dof: int, *, has_gripper: bool = True):
        super().__init__(dof, has_gripper=has_gripper)
        self._connected = False
        self._q_start = np.zeros(self.dof, dtype=float)
        self._q_target = np.zeros(self.dof, dtype=float)
        self._motion_start_time = 0.0
        self._motion_end_time = 0.0
        self._gripper_open_ratio = 1.0 if self.has_gripper else None
        raw_shared = str(os.getenv(_MOCK_SHARED_FILE_ENV, "")).strip()
        self._shared_state_path = Path(raw_shared).expanduser() if raw_shared else None

    def connect(self) -> None:
        self._connected = True
        state = self._shared_read_state()
        if state is not None:
            self._apply_state(state)

    def disconnect(self) -> None:
        self._connected = False

    def get_q(self) -> np.ndarray:
        self._ensure_connected()
        state = self._shared_read_state()
        if state is not None:
            self._apply_state(state)
        return self._compute_current_q(time.time())

    def send_movej(
        self,
        q: Sequence[float],
        duration: float,
        speed: Optional[float] = None,
        accel: Optional[float] = None,
    ) -> None:
        self._ensure_connected()
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.dof:
            raise ValueError(f"Expected {self.dof} joints, got {q.shape[0]}")

        now = time.time()
        current_q = self._compute_current_q(now)
        self._q_start = current_q.copy()
        self._q_target = q.copy()
        self._motion_start_time = now
        self._motion_end_time = now + max(0.0, float(duration))
        self._shared_write_state()

    def stop(self) -> None:
        now = time.time()
        current_q = self._compute_current_q(now)
        self._q_start = current_q.copy()
        self._q_target = current_q.copy()
        self._motion_start_time = now
        self._motion_end_time = now
        self._shared_write_state()

    def wait_until_stopped(self, timeout: Optional[float] = None) -> bool:
        self._ensure_connected()
        state = self._shared_read_state()
        if state is not None:
            self._apply_state(state)
        remaining = max(0.0, self._motion_end_time - time.time())
        if timeout is not None:
            timeout = max(0.0, float(timeout))
            if remaining > timeout:
                time.sleep(timeout)
                return False
        if remaining > 0.0:
            time.sleep(remaining)
        return True

    def set_gripper(self, open_ratio: float, wait: bool = True, timeout: Optional[float] = None) -> None:
        self._ensure_connected()
        if not self.has_gripper:
            return
        ratio = float(min(1.0, max(0.0, open_ratio)))
        self._gripper_open_ratio = ratio
        self._shared_write_state()
        if wait:
            self.wait_until_stopped(timeout=timeout)

    def get_gripper_open_ratio(self) -> Optional[float]:
        self._ensure_connected()
        state = self._shared_read_state()
        if state is not None:
            self._apply_state(state)
        if not self.has_gripper:
            return None
        return float(self._gripper_open_ratio if self._gripper_open_ratio is not None else 1.0)

    def _ensure_connected(self) -> None:
        if not self._connected:
            raise RuntimeError("MockTransport not connected")

    @staticmethod
    def _smooth_fraction(fraction: float) -> float:
        t = min(1.0, max(0.0, float(fraction)))
        return t * t * (3.0 - 2.0 * t)

    def _compute_current_q(self, now: float) -> np.ndarray:
        if self._motion_end_time <= self._motion_start_time or now >= self._motion_end_time:
            return self._q_target.copy()
        if now <= self._motion_start_time:
            return self._q_start.copy()
        span = max(1e-9, self._motion_end_time - self._motion_start_time)
        alpha = self._smooth_fraction((now - self._motion_start_time) / span)
        return self._q_start + (self._q_target - self._q_start) * alpha

    def _default_state(self) -> dict:
        return {
            "dof": int(self.dof),
            "q_start": [0.0 for _ in range(self.dof)],
            "q_target": [0.0 for _ in range(self.dof)],
            "motion_start_time": 0.0,
            "motion_end_time": 0.0,
            "has_gripper": bool(self.has_gripper),
            "gripper_open_ratio": 1.0 if self.has_gripper else None,
        }

    def _normalize_state(self, raw: object) -> dict:
        data = raw if isinstance(raw, dict) else {}

        def _vec(key: str) -> np.ndarray:
            try:
                q = np.asarray(data.get(key, []), dtype=float).reshape(-1)
            except Exception:
                q = np.zeros(self.dof, dtype=float)
            if q.shape[0] != self.dof:
                q = np.zeros(self.dof, dtype=float)
            return q

        try:
            motion_start = float(data.get("motion_start_time", 0.0))
        except Exception:
            motion_start = 0.0
        try:
            motion_end = float(data.get("motion_end_time", 0.0))
        except Exception:
            motion_end = 0.0

        has_gripper = bool(data.get("has_gripper", self.has_gripper))
        ratio = data.get("gripper_open_ratio", 1.0 if has_gripper else None)
        if has_gripper and ratio is not None:
            try:
                ratio = float(min(1.0, max(0.0, float(ratio))))
            except Exception:
                ratio = 1.0
        else:
            ratio = None

        return {
            "dof": int(self.dof),
            "q_start": [float(v) for v in _vec("q_start").tolist()],
            "q_target": [float(v) for v in _vec("q_target").tolist()],
            "motion_start_time": max(0.0, motion_start),
            "motion_end_time": max(0.0, motion_end),
            "has_gripper": has_gripper,
            "gripper_open_ratio": ratio,
        }

    def _apply_state(self, state: dict) -> None:
        self._q_start = np.asarray(state.get("q_start", []), dtype=float).reshape(-1)
        if self._q_start.shape[0] != self.dof:
            self._q_start = np.zeros(self.dof, dtype=float)
        self._q_target = np.asarray(state.get("q_target", []), dtype=float).reshape(-1)
        if self._q_target.shape[0] != self.dof:
            self._q_target = self._q_start.copy()
        self._motion_start_time = float(state.get("motion_start_time", 0.0))
        self._motion_end_time = float(state.get("motion_end_time", 0.0))
        self.has_gripper = bool(state.get("has_gripper", self.has_gripper))
        ratio = state.get("gripper_open_ratio", None)
        self._gripper_open_ratio = None if ratio is None else float(ratio)

    def _shared_read_state(self) -> Optional[dict]:
        if self._shared_state_path is None:
            return None
        return self._shared_load_or_update(None)

    def _shared_write_state(self) -> None:
        if self._shared_state_path is None:
            return

        def _updater(current: dict) -> dict:
            current["q_start"] = [float(v) for v in self._q_start.tolist()]
            current["q_target"] = [float(v) for v in self._q_target.tolist()]
            current["motion_start_time"] = float(self._motion_start_time)
            current["motion_end_time"] = float(self._motion_end_time)
            current["has_gripper"] = bool(self.has_gripper)
            current["gripper_open_ratio"] = self._gripper_open_ratio
            return current

        self._shared_load_or_update(_updater)

    def _shared_load_or_update(self, updater) -> dict:
        assert self._shared_state_path is not None
        path = self._shared_state_path
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("", encoding="utf-8")

        with path.open("r+", encoding="utf-8") as f:
            if fcntl is not None:
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                f.seek(0)
                raw = f.read().strip()
                if raw:
                    try:
                        state = self._normalize_state(json.loads(raw))
                    except Exception:
                        state = self._default_state()
                else:
                    state = self._default_state()

                if updater is not None:
                    state = self._normalize_state(updater(dict(state)))
                    f.seek(0)
                    f.truncate()
                    f.write(json.dumps(state, ensure_ascii=False))
                    f.flush()
            finally:
                if fcntl is not None:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        return state
