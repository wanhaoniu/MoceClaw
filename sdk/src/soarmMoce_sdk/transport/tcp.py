# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Sequence, Dict, List
import socket
import json
import numpy as np

from .base import TransportBase
from ..api.errors import TimeoutError as SDKTimeoutError, ProtocolError, ConnectionError as SDKConnectionError


def _set_sockopts_rx(s: socket.socket):
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)


def _send_all(conn: socket.socket, b: bytes):
    mv = memoryview(b)
    while mv:
        n = conn.send(mv)
        mv = mv[n:]


def _send_json(conn: socket.socket, obj: dict):
    data = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
    _send_all(conn, data)


def _recv_json_line(conn: socket.socket, buf: bytes, timeout: float | None = None):
    if timeout is not None:
        conn.settimeout(timeout)
    while True:
        i = buf.find(b"\n")
        if i >= 0:
            line = buf[:i]
            buf = buf[i + 1 :]
            if line:
                try:
                    return json.loads(line.decode("utf-8")), buf
                except json.JSONDecodeError:
                    snippet = line[:200].decode("utf-8", errors="replace")
                    raise ProtocolError(f"JSON decode failed: {snippet}")
        try:
            chunk = conn.recv(4096)
        except socket.timeout as e:
            raise SDKTimeoutError("recv timeout") from e
        if not chunk:
            raise SDKConnectionError("socket closed")
        buf += chunk


class TCPTransport(TransportBase):
    """TCP transport using the existing JSON protocol (cmd/ack)."""

    def __init__(
        self,
        dof: int,
        host: str,
        port: int,
        timeout: float = 2.0,
        joint_names: Optional[List[str]] = None,
        joint_map: Optional[Dict[str, str]] = None,
        unit: str = "deg",
        max_retries: int = 1,
        use_seq: bool = False,
    ):
        super().__init__(dof)
        self.host = host
        self.port = int(port)
        self.timeout = float(timeout)
        self._sock: Optional[socket.socket] = None
        self._buf = b""
        self.joint_names = joint_names or [f"j{i}" for i in range(self.dof)]
        self.joint_map = joint_map or {}
        self.unit = unit
        self.max_retries = int(max_retries)
        self.use_seq = bool(use_seq)
        self._seq = 0
        self._last_q = np.zeros(self.dof, dtype=float)

        self._validate_joint_map()

    def _validate_joint_map(self) -> None:
        if not self.joint_map:
            self.joint_map = {name: name for name in self.joint_names}
            return
        missing = [name for name in self.joint_names if name not in self.joint_map]
        if missing:
            raise ProtocolError(f"Joint map missing entries: {missing}")

    def connect(self) -> None:
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        _set_sockopts_rx(self._sock)
        self._sock.settimeout(self.timeout)
        self._sock.connect((self.host, self.port))
        self._buf = b""

    def disconnect(self) -> None:
        if self._sock:
            try:
                self._sock.close()
            except Exception:
                pass
        self._sock = None
        self._buf = b""

    def get_q(self) -> np.ndarray:
        if not self._sock:
            raise SDKConnectionError("TCPTransport not connected")
        # Trigger an ack by sending the last command (or zeros on first call)
        self.send_movej(self._last_q, duration=0.0)
        return self._last_q.copy()

    def send_movej(
        self,
        q: Sequence[float],
        duration: float,
        speed: Optional[float] = None,
        accel: Optional[float] = None,
    ) -> None:
        if not self._sock:
            raise SDKConnectionError("TCPTransport not connected")
        q = np.asarray(q, dtype=float).reshape(-1)
        if q.shape[0] != self.dof:
            raise ValueError(f"Expected {self.dof} joints, got {q.shape[0]}")

        q_send = q.copy()
        if self.unit == "deg":
            q_send = np.rad2deg(q_send)

        qL = {}
        for i, name in enumerate(self.joint_names):
            proto_name = self.joint_map.get(name, name)
            qL[proto_name] = float(q_send[i])

        last_err = None
        for attempt in range(self.max_retries + 1):
            payload = {"type": "cmd", "tsL_send": 0.0, "qL": qL}
            if self.use_seq:
                self._seq += 1
                payload["seq"] = self._seq
                payload["request_id"] = self._seq
            _send_json(self._sock, payload)
            try:
                msg, self._buf = _recv_json_line(self._sock, self._buf, timeout=self.timeout)
                if msg.get("type") != "ack":
                    raise ProtocolError(f"Unexpected message type: {msg.get('type')}")
                qF = msg.get("qF", {})
                q_recv = []
                for i, name in enumerate(self.joint_names):
                    proto_name = self.joint_map.get(name, name)
                    q_recv.append(float(qF.get(proto_name, q_send[i])))
                q_recv = np.asarray(q_recv, dtype=float)
                if self.unit == "deg":
                    q_recv = np.deg2rad(q_recv)
                self._last_q = q_recv
                return
            except (SDKTimeoutError, ProtocolError, SDKConnectionError) as e:
                last_err = e
                if attempt >= self.max_retries:
                    raise

        if last_err:
            raise last_err

        self._last_q = q.copy()

    def stop(self) -> None:
        if self._sock:
            try:
                self._sock.shutdown(socket.SHUT_RDWR)
            except Exception:
                pass
