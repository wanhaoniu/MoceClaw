#!/usr/bin/env python3

"""
arm_server4.py - 超轻量从臂控制器

设计原则：
- Server 端"傻瓜式执行"：只接收命令，只执行动作
- 无文件存储：所有数据管理在 Client 端
- 最小 CPU 占用：适合 RK3562 等边缘设备

支持命令：
1. cmd: 实时控制（主要命令）
2. goto: 平滑运动到目标位置（目标数据在命令中）
"""

from __future__ import annotations

import json
import math
import socket
import threading
import time
import types
import sys, site
from pathlib import Path
from typing import Sequence, Dict, Optional

site.addsitedir("/home/forlinx/.local/lib/python3.10/site-packages")

# 导入摄像头模块
from cam_server_pipeline import HardwareStreamer

# ==================== 摄像头配置 ====================
ENABLE_CAM1     = True      # 摄像头 1 开关
CAM1_PATH       = "/dev/v4l/by-path/platform-fed00000.usb-usb-0:1.2:1.0-video-index0"
CAM1_DEV        = 40
CAM1_PORT       = 6000
CAM1_WIDTH      = 640
CAM1_HEIGHT     = 480
CAM1_FPS        = 30
TARGET_PC_IP    = "192.168.66.212"  # 接收端 PC 的 IP

# ==================== 1. 核心 Patch ====================
def make_hybrid_unnormalize(original_method):
    def hybrid_unnormalize(self, ids_values: dict[int, float]) -> dict[int, int]:
        result = {}
        for mid, val in ids_values.items():
            if mid in [2, 3, 6]:  # Lift, Elbow, Gripper
                result[mid] = int(val) 
            else:
                partial_res = original_method({mid: val})
                result.update(partial_res)
        return result
    return hybrid_unnormalize

# ==================== 2. 通信函数 ====================
def set_sockopts_tx(s: socket.socket):
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4 * 1024 * 1024)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

def send_all(conn: socket.socket, b: bytes):
    mv = memoryview(b)
    while mv:
        n = conn.send(mv)
        mv = mv[n:]

def send_json(conn: socket.socket, obj: dict):
    data = (json.dumps(obj, separators=(",", ":")) + "\n").encode("utf-8")
    send_all(conn, data)

def recv_json_line(conn: socket.socket, buf: bytes, timeout: float | None = None):
    if timeout is not None:
        conn.settimeout(timeout)
    while True:
        i = buf.find(b"\n")
        if i >= 0:
            line = buf[:i]
            buf = buf[i + 1:]
            if line:
                return json.loads(line.decode("utf-8")), buf
        chunk = conn.recv(4096)
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk

# ==================== 3. 主程序 ====================

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]

MULTI_TURN_JOINTS = ["shoulder_lift", "elbow_flex", "gripper"]

class ArmServer:
    def __init__(self,
                 bind_host: str = "0.0.0.0",
                 ctl_port: int = 6666,
                 follower_port: str = "/dev/ttyACM0",
                 follower_id: str = "brown_arm_follower",
                 calib_dir: str = "calibration/robots/so101_follower",
                 scale: Sequence[float] | None = None,
                 offset: Sequence[float] | None = None, 
                 mirror: Sequence[float] | None = None,
                 alpha: float = 0.2, deadband_deg: float = 0.5, 
                 max_step_deg: float = 10.0, gripper_min: float = 0.0, gripper_max: float = 100.0,
                 enable_cam: bool = ENABLE_CAM1,
                 cam_path: str = CAM1_PATH,
                 cam_port: int = CAM1_PORT,
                 target_ip: str = TARGET_PC_IP):

        self.bind_host = bind_host
        self.ctl_port = ctl_port
        
        # 摄像头配置
        self.enable_cam = enable_cam
        self.cam_path = cam_path
        self.cam_port = cam_port
        self.target_ip = target_ip
        self.cam_server = None
        
        self.scale = list(scale) if scale is not None else [1.0] * len(JOINTS)
        self.offset = list(offset) if offset is not None else [0.0] * len(JOINTS)
        self.mirror = list(mirror) if mirror is not None else [1.0] * len(JOINTS)

        self._stop = threading.Event()
        
        # 硬件初始化
        from so101_utils import load_calibration
        from lerobot.motors import Motor, MotorNormMode
        from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode

        calib = load_calibration(follower_id, calib_dir=Path(calib_dir))
        
        for name in MULTI_TURN_JOINTS:
            if name in calib:
                calib[name].range_min = -900000
                calib[name].range_max = 900000

        bus = FeetechMotorsBus(
            port=follower_port,
            motors={
                name: Motor(i+1, "sts3215", MotorNormMode.DEGREES) 
                for i, name in enumerate(JOINTS)
            },
            calibration=calib,
        )

        bus._unnormalize = types.MethodType(make_hybrid_unnormalize(bus._unnormalize), bus)

        bus.connect()
        with bus.torque_disabled():
            bus.configure_motors()
            
            print(">>> Configuring Motors...")
            for name in JOINTS:
                if name in MULTI_TURN_JOINTS:
                    print(f"  - {name}: Mode 3 (Multi-turn)")
                    bus.write("Lock", name, 0)
                    time.sleep(0.05)
                    bus.write("Min_Position_Limit", name, 0)
                    bus.write("Max_Position_Limit", name, 0)
                    bus.write("Operating_Mode", name, 3)
                    time.sleep(0.05)
                    bus.write("Lock", name, 1)
                else:
                    print(f"  - {name}: Mode 0 (Position)")
                    bus.write("Operating_Mode", name, OperatingMode.POSITION.value)
                    bus.write("P_Coefficient", name, 32)
                    bus.write("I_Coefficient", name, 0)
                    bus.write("D_Coefficient", name, 16)

        bus.enable_torque()
        self._bus = bus
        
        self.slave_start_pos = {}
        for name in JOINTS:
            if name not in MULTI_TURN_JOINTS:
                self.slave_start_pos[name] = bus.read("Present_Position", name)
        print(f">>> Slave Start Pos (Mode 0 Only): {self.slave_start_pos}")
        print(">>> Server Ready (Ultra Lightweight Mode - No File Storage)")

    def start_server(self):
        self._stop.clear()
        
        # 启动摄像头服务器
        if self.enable_cam:
            try:
                print(f"[CAM] 启动硬件加速摄像头 ({self.cam_path})...")
                self.cam_server = HardwareStreamer(
                    dev_path=self.cam_path,
                    port=self.cam_port,
                    target_ip=self.target_ip,
                    width=CAM1_WIDTH,
                    height=CAM1_HEIGHT,
                    fps=CAM1_FPS,
                )
                self.cam_server.start()
                time.sleep(2.0)
                print(f"[CAM] 摄像头已启动，推流到 {self.target_ip}:{self.cam_port}")
            except Exception as e:
                print(f"[CAM] 摄像头启动失败: {e}")
                self.cam_server = None
        
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        set_sockopts_tx(srv)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.bind_host, self.ctl_port))
        srv.listen(1)
        srv.settimeout(1.0)
        self._srv = srv

        print(f"[SERVER] Listening on {self.ctl_port}")

        try:
            while not self._stop.is_set():
                try:
                    conn, addr = srv.accept()
                except socket.timeout: continue
                except OSError: break

                set_sockopts_tx(conn)
                print(f"[SERVER] Connected: {addr}")
                
                # === 连接初始化 ===
                buf = b""
                _last_virtual = {k: 0.0 for k in MULTI_TURN_JOINTS}
                _first_runs = {k: True for k in MULTI_TURN_JOINTS}
                
                # goto 任务状态（可选，用于平滑插值）
                _goto_task: Optional[Dict] = None
                
                try:
                    while not self._stop.is_set():
                        # === 处理 goto 任务（如果有） ===
                        if _goto_task is not None:
                            done = self._goto_step(_goto_task, _last_virtual)
                            if done:
                                final_virtual = self._get_current_virtual(_last_virtual)
                                try:
                                    send_json(conn, {
                                        "type": "goto_done",
                                        "virtual": final_virtual,
                                    })
                                except: pass
                                _goto_task = None
                        
                        # === 接收消息 ===
                        try:
                            msg, buf = recv_json_line(conn, buf, timeout=2.0)
                        except socket.timeout: continue
                        except ConnectionError: 
                            print("[SERVER] Client disconnected")
                            break
                        
                        msg_type = msg.get("type", "")
                        
                        # ========== 实时控制命令 ==========
                        if msg_type == "cmd":
                            # 收到实时命令时取消 goto
                            _goto_task = None
                            
                            tsL_send = float(msg.get("tsL_send", 0.0))
                            qL = msg.get("qL", {})
                            
                            cmd = {}
                            
                            for j in JOINTS:
                                lj = float(qL.get(j, float("nan")))
                                if math.isnan(lj): continue

                                idx = JOINTS.index(j)
                                
                                if j in MULTI_TURN_JOINTS:
                                    if _first_runs[j]:
                                        _last_virtual[j] = lj
                                        _first_runs[j] = False
                                    else:
                                        delta_virtual = lj - _last_virtual[j]
                                        delta_motor = delta_virtual * self.scale[idx]
                                        raw_step = int(delta_motor * 4096.0 / 360.0)
                                        
                                        if raw_step != 0:
                                            cmd[j] = raw_step
                                            sent_motor = raw_step * 360.0 / 4096.0
                                            sent_virtual = sent_motor / self.scale[idx]
                                            _last_virtual[j] += sent_virtual
                                else:
                                    target_abs = self.slave_start_pos[j] + (lj * self.scale[idx]) + self.offset[idx]
                                    cmd[j] = target_abs

                            if cmd:
                                try:
                                    self._bus.sync_write("Goal_Position", cmd)
                                except Exception as e:
                                    print(f"[Err] Write: {e}")

                            qF = {}
                            try:
                                qF = self._bus.sync_read("Present_Position")
                            except Exception: pass

                            tsU_recv = time.perf_counter()
                            tsU_send = time.perf_counter()
                            try:
                                send_json(conn, {
                                    "type": "ack", "tsL_send": tsL_send,
                                    "tsU_recv": tsU_recv, "tsU_send": tsU_send, "qF": qF,
                                })
                            except BrokenPipeError: break
                        
                        # ========== 跳转位置命令（目标数据在命令中）==========
                        elif msg_type == "goto":
                            duration = float(msg.get("duration", 2.0))
                            target = msg.get("target", {})  # Client 发送目标数据
                            
                            if not target:
                                try:
                                    send_json(conn, {"type": "error", "msg": "No target data"})
                                except: pass
                                continue
                            
                            current = self._get_current_virtual(_last_virtual)
                            
                            _goto_task = {
                                "start": current,
                                "target": target,
                                "duration": max(0.1, duration),
                                "start_time": time.perf_counter(),
                            }
                            
                            try:
                                send_json(conn, {"type": "goto_start"})
                            except: pass

                finally:
                    conn.close()
                    print("[SERVER] Connection closed")

        except KeyboardInterrupt:
            print("\n[SERVER] Stopped by user")
        finally:
            if self._bus: self._bus.disable_torque()
            if self._srv: self._srv.close()
            if self.cam_server:
                print("[CAM] 正在停止摄像头...")
                self.cam_server.stop()
    
    def stop(self):
        """停止服务器"""
        self._stop.set()
        if self.cam_server:
            self.cam_server.stop()

    def _get_current_virtual(self, _last_virtual: Dict[str, float]) -> Dict[str, float]:
        """获取当前虚拟坐标"""
        current = {}
        for j in JOINTS:
            if j in MULTI_TURN_JOINTS:
                current[j] = _last_virtual[j]
            else:
                try:
                    pos = self._bus.read("Present_Position", j)
                    idx = JOINTS.index(j)
                    current[j] = (pos - self.slave_start_pos[j] - self.offset[idx]) / self.scale[idx]
                except:
                    current[j] = 0.0
        return current

    def _goto_step(self, task: Dict, _last_virtual: Dict[str, float]) -> bool:
        """执行 goto 一步（平滑插值）"""
        elapsed = time.perf_counter() - task["start_time"]
        progress = min(1.0, elapsed / task["duration"])
        
        t = progress
        smooth = t * t * (3.0 - 2.0 * t)
        
        cmd = {}
        for j in JOINTS:
            start_val = task["start"].get(j, 0.0)
            target_val = task["target"].get(j, 0.0)
            current_virtual = start_val + (target_val - start_val) * smooth
            
            idx = JOINTS.index(j)
            
            if j in MULTI_TURN_JOINTS:
                delta_virtual = current_virtual - _last_virtual[j]
                delta_motor = delta_virtual * self.scale[idx]
                raw_step = int(delta_motor * 4096.0 / 360.0)
                if raw_step != 0:
                    cmd[j] = raw_step
                    sent_motor = raw_step * 360.0 / 4096.0
                    sent_virtual = sent_motor / self.scale[idx]
                    _last_virtual[j] += sent_virtual
            else:
                target_abs = self.slave_start_pos[j] + (current_virtual * self.scale[idx]) + self.offset[idx]
                cmd[j] = target_abs
        
        if cmd:
            try:
                self._bus.sync_write("Goal_Position", cmd)
            except Exception as e:
                print(f"[Err] Goto: {e}")
        
        return progress >= 1.0


if __name__ == "__main__":
    print("=" * 60)
    print("  ARM SERVER v4.0 - Ultra Lightweight Executor")
    print("=" * 60)
    print()
    print("Design: No file storage, minimal CPU usage")
    print("Perfect for: RK3562, Raspberry Pi, ESP32, etc.")
    print()
    print(f"Camera: {'Enabled' if ENABLE_CAM1 else 'Disabled'}")
    if ENABLE_CAM1:
        print(f"  - Path: {CAM1_PATH}")
        print(f"  - Port: {CAM1_PORT}")
        print(f"  - Target: {TARGET_PC_IP}")
    print()
    
#    s = ArmServer(scale=[1.0, 5.6, 5.6, -1.0, 1.0, 5.6])
    s = ArmServer(scale=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    s.start_server()
