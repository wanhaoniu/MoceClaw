#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
arm_client4.py - 智能主臂控制器（本地存储版）

设计原则：
- Client 端"智能控制"：管理所有数据，控制回放节奏
- 本地文件存储：positions/ 和 recordings/
- Server 端无负担：只发送执行命令

功能：
1. 实时遥操作
2. 位置保存与跳转（本地存储）
3. 动作录制与回放（本地存储，逐帧发送）
"""

from __future__ import annotations

import json
import socket
import threading
import time
import math
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, List

# ==================== 配置 ====================

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex",
          "wrist_flex", "wrist_roll", "gripper"]

MULTI_TURN_CLIENT = ["shoulder_lift", "elbow_flex", "gripper"]

# 本地存储目录
LOCAL_POSITIONS_DIR = Path("positions")
LOCAL_RECORDINGS_DIR = Path("recordings")

# ==================== 网络工具 ====================

def set_sockopts_rx(s: socket.socket):
    s.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 4 * 1024 * 1024)

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
            buf = buf[i + 1 :]
            if line:
                return json.loads(line.decode("utf-8")), buf
        chunk = conn.recv(4096)
        if not chunk:
            raise ConnectionError("socket closed")
        buf += chunk

# ==================== 本地文件管理 ====================

def load_position(name: str) -> Optional[Dict[str, float]]:
    """从本地加载位置"""
    fpath = LOCAL_POSITIONS_DIR / f"{name}.json"
    if not fpath.exists():
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def save_position(name: str, joints: Dict[str, float]):
    """保存位置到本地"""
    LOCAL_POSITIONS_DIR.mkdir(parents=True, exist_ok=True)
    fpath = LOCAL_POSITIONS_DIR / f"{name}.json"
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(joints, f, indent=2)
    print(f"[LOCAL] Position '{name}' saved to {fpath}")

def list_positions() -> List[str]:
    """列出所有本地位置"""
    if not LOCAL_POSITIONS_DIR.exists():
        return []
    return [f.stem for f in LOCAL_POSITIONS_DIR.glob("*.json")]

def delete_position(name: str) -> bool:
    """删除本地位置"""
    fpath = LOCAL_POSITIONS_DIR / f"{name}.json"
    if fpath.exists():
        fpath.unlink()
        return True
    return False

def load_recording(name: str) -> Optional[Dict]:
    """从本地加载录制"""
    fpath = LOCAL_RECORDINGS_DIR / f"{name}.json"
    if not fpath.exists():
        return None
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return None

def save_recording(name: str, data: Dict):
    """保存录制到本地"""
    LOCAL_RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)
    fpath = LOCAL_RECORDINGS_DIR / f"{name}.json"
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    frames = len(data.get("frames", []))
    duration = data.get("frames", [{}])[-1].get("t", 0) if data.get("frames") else 0
    print(f"[LOCAL] Recording '{name}' saved to {fpath} ({frames} frames, {duration:.1f}s)")

def list_recordings() -> Dict[str, Dict]:
    """列出所有本地录制"""
    if not LOCAL_RECORDINGS_DIR.exists():
        return {}
    
    recordings = {}
    for fpath in LOCAL_RECORDINGS_DIR.glob("*.json"):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
                frames = data.get("frames", [])
                recordings[fpath.stem] = {
                    "frames": len(frames),
                    "duration": frames[-1].get("t", 0) if frames else 0,
                }
        except:
            pass
    return recordings

def delete_recording(name: str) -> bool:
    """删除本地录制"""
    fpath = LOCAL_RECORDINGS_DIR / f"{name}.json"
    if fpath.exists():
        fpath.unlink()
        return True
    return False

# ==================== 主客户端类 ====================

class ArmClient:
    """
    智能主臂控制器
    - 管理所有文件（本地存储）
    - 控制回放节奏（逐帧发送）
    - Server 端无负担（只执行命令）
    """

    def __init__(
        self,
        name: str,
        server_ip: str,
        ctl_port: int,
        leader_port: str = "/dev/ttyACM0",
        leader_id: str = "black_arm_leader",
        calib_dir: str = "calibration/teleoperators/so101_leader",
        hz: float = 100.0,
        window: float = 30.0,
        rtt_buf: Optional[deque] = None,
        save_csv: bool = False,
        csv_path: Optional[str] = None,
        record_hz: float = 10.0,
    ):
        # --- 参数保存 ---
        self.name = name
        self.server_ip = server_ip
        self.ctl_port = int(ctl_port)
        self.leader_port = leader_port
        self.leader_id = leader_id
        self.calib_dir = calib_dir
        self.hz = max(1.0, float(hz))
        self.dt = 1.0 / self.hz
        self.record_hz = max(1.0, float(record_hz))
        self.record_dt = 1.0 / self.record_hz

        # RTT buffer
        if rtt_buf is None:
            self.rtt_ms_buf = deque(maxlen=int(window * self.hz))
        else:
            self.rtt_ms_buf = rtt_buf

        # 状态缓存
        self.qL_last: Dict[str, float] = {k: 0.0 for k in JOINTS}
        self.qF_last: Dict[str, float] = {k: float("nan") for k in JOINTS}

        # 线程控制
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._sock: Optional[socket.socket] = None
        self._buf = b""
        
        # 多圈累积器
        self._accumulators: Dict[str, float] = {k: 0.0 for k in MULTI_TURN_CLIENT}
        self._accumulators_lock = threading.Lock()
        
        # 零点位置（启动时锁定的姿态）
        self._zero_point: Optional[Dict[str, float]] = None
        self._zero_point_lock = threading.Lock()
        
        # goto/play 状态
        self._goto_active = False
        self._play_active = False
        self._play_event = threading.Event()  # 用于通知 play 完成
        
        # 录制状态
        self._recording = False
        self._record_name: Optional[str] = None
        self._record_frames: List[Dict] = []
        self._record_start_time: float = 0.0
        self._record_last_sample: float = 0.0

        # CSV 日志
        self._csv_file = None
        if save_csv and csv_path:
            p = Path(csv_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            self._csv_file = p.open("w", encoding="utf-8")
            self._csv_file.write(
                "arm,iso,t,tsL_send,tsL_recv,tsU_recv,tsU_send,rtt_ms,"
                "L_shoulder_pan,L_shoulder_lift,L_elbow_flex,L_wrist_flex,L_wrist_roll,L_gripper,"
                "F_shoulder_pan,F_shoulder_lift,F_elbow_flex,F_wrist_flex,F_wrist_roll,F_gripper\n"
            )
            self._csv_file.flush()

        # --- 硬件初始化 ---
        from so101_utils import load_calibration, setup_leader_bus
        calib = load_calibration(self.leader_id, calib_dir=Path(self.calib_dir))
        self.leader_bus = setup_leader_bus(self.leader_port, calib)
        
        try:
            self.leader_bus.connect()
            self.leader_bus.disable_torque()
        except Exception as e:
            print(f"[{self.name}] Bus init warning: {e}")

        self._t0 = time.perf_counter()

    # ==================== 对外接口 ====================

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        """
        停止客户端（不回零点，回零点应该在调用 stop 前完成）
        """
        print(f"[{self.name}] Stopping...")
        
        # 停止控制线程
        self._stop.set()
        
        # 等待控制线程结束（最多2秒）
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)
        
        if self._sock:
            try: self._sock.close()
            except: pass
            self._sock = None
            
        if self._csv_file:
            try: self._csv_file.close()
            except: pass
            self._csv_file = None
        
        print(f"[{self.name}] Stopped")

    def get_latest_joints(self) -> list[float]:
        """获取从机最新角度(弧度)"""
        vals = []
        for name in JOINTS:
            deg = self.qF_last.get(name, 0.0)
            if math.isnan(deg): deg = 0.0
            vals.append(math.radians(deg))
        return vals
    
    def get_accumulators(self) -> Dict[str, float]:
        """获取当前的累积角度"""
        with self._accumulators_lock:
            return self._accumulators.copy()
    
    # ==================== 位置命令 ====================
    
    def savepos(self, name: str) -> bool:
        """保存当前位置到本地"""
        with self._accumulators_lock:
            current = self.qL_last.copy()
        save_position(name, current)
        return True
    
    def goto(self, name: str, duration: float = 2.0, timeout: float = 30.0) -> bool:
        """跳转到本地保存的位置"""
        # 1. 读取本地位置
        target = load_position(name)
        if target is None:
            print(f"[{self.name}] Position '{name}' not found locally")
            return False
        
        # 2. 发送 goto 命令（带目标数据）
        self._goto_active = True
        
        try:
            send_json(self._sock, {
                "type": "goto",
                "target": target,
                "duration": duration,
            })
        except Exception as e:
            print(f"[{self.name}] Failed to send goto: {e}")
            self._goto_active = False
            return False
        
        # 3. 等待完成
        start = time.time()
        while time.time() - start < timeout:
            if not self._goto_active:
                return True
            time.sleep(0.05)
        
        self._goto_active = False
        return False
    
    def listpos(self) -> List[str]:
        """列出所有本地位置"""
        return list_positions()
    
    def delpos(self, name: str) -> bool:
        """删除本地位置"""
        return delete_position(name)
    
    def return_to_zero(self, duration: float = 3.0) -> bool:
        """
        回到零点（在程序正常运行时调用）
        :param duration: 运动时间（秒）
        :return: 是否成功
        """
        print(f"[{self.name}] Returning to zero point...")
        
        # 零点对应的虚拟坐标就是 {所有关节: 0.0}
        zero_target = {k: 0.0 for k in JOINTS}
        
        # 临时保存为 "_zero" 位置
        save_position("_zero", zero_target)
        try:
            success = self.goto("_zero", duration=duration, timeout=duration + 5.0)
            return success
        finally:
            delete_position("_zero")
    
    # ==================== 录制命令 ====================
    
    def start_record(self, name: str) -> bool:
        """开始录制"""
        if self._recording:
            print(f"[{self.name}] Already recording!")
            return False
        
        self._recording = True
        self._record_name = name
        self._record_frames = []
        self._record_start_time = time.perf_counter()
        self._record_last_sample = 0.0
        
        print(f"[{self.name}] Recording '{name}' started (@ {self.record_hz} Hz)")
        return True
    
    def stop_record(self) -> bool:
        """停止录制并保存到本地"""
        if not self._recording:
            print(f"[{self.name}] Not recording!")
            return False
        
        self._recording = False
        
        if not self._record_frames:
            print(f"[{self.name}] No frames recorded!")
            return False
        
        # 保存到本地
        rec_data = {
            "name": self._record_name,
            "hz": self.record_hz,
            "frames": self._record_frames,
        }
        
        save_recording(self._record_name, rec_data)
        return True
    
    def play(self, name: str, times: int = 1) -> bool:
        """回放本地录制（逐帧发送）"""
        # 1. 读取本地录制
        rec_data = load_recording(name)
        if rec_data is None:
            print(f"[{self.name}] Recording '{name}' not found locally")
            return False
        
        frames = rec_data.get("frames", [])
        if not frames:
            print(f"[{self.name}] Recording '{name}' has no frames")
            return False
        
        print(f"[{self.name}] Playing '{name}' × {times} ({len(frames)} frames)")
        
        # 2. 标记进入回放模式
        self._play_active = True
        self._play_event.clear()
        
        # 3. 逐帧发送（在独立线程中）
        def _play_thread():
            try:
                for loop in range(times):
                    if self._stop.is_set():
                        break
                    
                    print(f"[{self.name}] Loop {loop + 1}/{times}")
                    start_time = time.perf_counter()
                    
                    for i, frame in enumerate(frames):
                        if self._stop.is_set():
                            break
                        
                        # 等待到达时间点
                        target_time = start_time + frame["t"]
                        now = time.perf_counter()
                        if now < target_time:
                            time.sleep(target_time - now)
                        
                        # 发送该帧的关节位置（就像实时控制一样）
                        joints = frame["joints"]
                        
                        try:
                            send_json(self._sock, {
                                "type": "cmd",
                                "tsL_send": time.perf_counter(),
                                "qL": joints,
                            })
                        except Exception as e:
                            print(f"[{self.name}] Play send failed: {e}")
                            break
                        
                        # 接收 ACK（但不阻塞太久）
                        try:
                            msg, self._buf = recv_json_line(self._sock, self._buf, timeout=0.01)
                            if msg.get("type") == "ack":
                                # 更新从机状态
                                qF_raw = msg.get("qF", {})
                                for k in JOINTS:
                                    if k in qF_raw:
                                        self.qF_last[k] = float(qF_raw[k])
                        except socket.timeout:
                            pass
                        except Exception:
                            break
                
                # 回放完成后同步累积器
                # 从最后一帧获取虚拟坐标
                last_frame = frames[-1]
                with self._accumulators_lock:
                    for k in MULTI_TURN_CLIENT:
                        if k in last_frame["joints"]:
                            self._accumulators[k] = last_frame["joints"][k]
                
                print(f"[{self.name}] Play completed, accumulators synced")
                
            finally:
                self._play_active = False
                self._play_event.set()
        
        thread = threading.Thread(target=_play_thread, daemon=True)
        thread.start()
        
        # 等待完成
        self._play_event.wait()
        return True
    
    def list_recordings(self) -> Dict[str, Dict]:
        """列出所有本地录制"""
        return list_recordings()
    
    def delete_recording(self, name: str) -> bool:
        """删除本地录制"""
        return delete_recording(name)
    
    # ==================== 内部实现 ====================

    def _loop(self):
        # 1. 建立连接
        while not self._stop.is_set():
            try:
                self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                set_sockopts_rx(self._sock)
                print(f"[{self.name}] Connecting to {self.server_ip}:{self.ctl_port}...")
                self._sock.connect((self.server_ip, self.ctl_port))
                print(f"[{self.name}] Connected!")
                break
            except Exception as e:
                print(f"[{self.name}] Connect failed: {e}, retrying in 3s...")
                time.sleep(3)

        # 2. 读取零点（保存启动时的实际姿态）
        print(f"[{self.name}] Reading Zero Point...")
        start_pos = {}
        try:
            start_pos = self.leader_bus.sync_read("Present_Position")
            print(f"[{self.name}] Zero Point Locked: {start_pos}")
            
            # 【关键】保存零点姿态（这是真正的零点，不是全0）
            with self._zero_point_lock:
                self._zero_point = start_pos.copy()
            
        except Exception as e:
            print(f"[{self.name}] Failed to read start pos: {e}")
            return

        # 初始化多圈状态
        last_raws = {k: start_pos.get(k, 0.0) for k in MULTI_TURN_CLIENT}
        with self._accumulators_lock:
            self._accumulators = {k: 0.0 for k in MULTI_TURN_CLIENT}

        self._buf = b""

        try:
            while not self._stop.is_set():
                loop_start = time.perf_counter()
                
                # === 如果 goto/play 正在执行，跳过实时控制 ===
                if self._goto_active or self._play_active:
                    # 只接收消息
                    try:
                        msg, self._buf = recv_json_line(self._sock, self._buf, timeout=0.1)
                        
                        if msg.get("type") == "goto_done":
                            # goto 完成，同步累积器
                            virtual = msg.get("virtual", {})
                            with self._accumulators_lock:
                                for j in MULTI_TURN_CLIENT:
                                    if j in virtual:
                                        self._accumulators[j] = float(virtual[j])
                            print(f"[{self.name}] Goto done, accumulators synced")
                            self._goto_active = False
                            
                            # 重新读取主臂位置
                            try:
                                curr_pos = self.leader_bus.sync_read("Present_Position")
                                for k in MULTI_TURN_CLIENT:
                                    last_raws[k] = curr_pos.get(k, 0.0)
                            except:
                                pass
                    
                    except socket.timeout:
                        pass
                    except (ConnectionError, OSError):
                        print(f"[{self.name}] Connection lost")
                        break
                    continue

                # --- 读取主臂姿态 ---
                try:
                    curr_pos = self.leader_bus.sync_read("Present_Position")
                except:
                    time.sleep(0.005)
                    continue

                packet_qL = {}

                # --- 计算逻辑 ---
                with self._accumulators_lock:
                    for k in JOINTS:
                        curr = curr_pos.get(k, 0.0)
                        start = start_pos.get(k, 0.0)

                        if k in MULTI_TURN_CLIENT:
                            diff = curr - last_raws[k]
                            if diff < -180: diff += 360
                            elif diff > 180: diff -= 360
                            self._accumulators[k] += diff
                            last_raws[k] = curr
                            packet_qL[k] = self._accumulators[k]
                        else:
                            packet_qL[k] = curr - start

                self.qL_last = packet_qL.copy()
                
                # === 录制采样 ===
                if self._recording:
                    now = time.perf_counter()
                    elapsed = now - self._record_start_time
                    if elapsed - self._record_last_sample >= self.record_dt:
                        self._record_frames.append({
                            "t": elapsed,
                            "joints": packet_qL.copy(),
                        })
                        self._record_last_sample = elapsed
                
                tsL_send = time.perf_counter()

                # --- 发送 JSON ---
                try:
                    send_json(self._sock, {"type": "cmd", "tsL_send": tsL_send, "qL": packet_qL})
                except (BrokenPipeError, ConnectionError):
                    print(f"[{self.name}] Connection lost during send")
                    break

                # --- 接收 ACK ---
                try:
                    msg, self._buf = recv_json_line(self._sock, self._buf, timeout=0.5)
                    
                    if msg.get("type") == "ack":
                        tsL_recv = time.perf_counter()
                        tsU_recv = float(msg.get("tsU_recv", 0.0))
                        tsU_send = float(msg.get("tsU_send", 0.0))
                        
                        qF_raw = msg.get("qF", {})
                        for k in JOINTS:
                            if k in qF_raw:
                                self.qF_last[k] = float(qF_raw[k])

                        rtt_ms = (tsL_recv - tsL_send) * 1000.0
                        self.rtt_ms_buf.append(rtt_ms)

                        if self._csv_file:
                            self._write_csv(tsL_send, tsL_recv, tsU_recv, tsU_send, rtt_ms)
                            
                except socket.timeout:
                    pass
                except (ConnectionError, OSError):
                    print(f"[{self.name}] Connection lost during recv")
                    break

                # --- 控频 ---
                elapsed = time.perf_counter() - loop_start
                sleep_time = max(0.0, self.dt - elapsed)
                time.sleep(sleep_time)

        except Exception as e:
            print(f"[{self.name}] Critical error: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self._sock:
                try: self._sock.close()
                except: pass
            print(f"[{self.name}] Stopped")

    def _write_csv(self, tsL_send, tsL_recv, tsU_recv, tsU_send, rtt_ms):
        try:
            iso = datetime.now().isoformat(timespec="seconds")
            tnow = time.perf_counter() - self._t0
            def fmt(val): return f"{val:.3f}"
            
            line = (
                f"{self.name},{iso},{tnow:.3f},{tsL_send:.6f},{tsL_recv:.6f},{tsU_recv:.6f},{tsU_send:.6f},{rtt_ms:.3f},"
                + ",".join(fmt(self.qL_last.get(k, 0.0)) for k in JOINTS) + ","
                + ",".join(fmt(self.qF_last.get(k, 0.0)) for k in JOINTS)
                + "\n"
            )
            self._csv_file.write(line)
        except: pass


# ==================== 独立运行模式 ====================

if __name__ == "__main__":
    import argparse
    import cv2
    import numpy as np
    
    # 尝试导入视频客户端
    try:
        from video_client_h264 import H264VideoClient as VideoClient
        VIDEO_CLIENT_AVAILABLE = True
    except ImportError:
        VIDEO_CLIENT_AVAILABLE = False
        print("[Warning] Video client not available")
    
    parser = argparse.ArgumentParser(description="ARM Client v4 - Local Storage")
    parser.add_argument("--ip", default="172.18.29.159")
    parser.add_argument("--port", type=int, default=6666)
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument("--leader-id", default="black_arm_leader")
    parser.add_argument("--hz", type=float, default=100.0)
    parser.add_argument("--record-hz", type=float, default=10.0)
    parser.add_argument("--cam-port", type=int, default=6000, help="Camera port (default: 6000)")
    parser.add_argument("--no-cam", action="store_true", help="Disable camera display")
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("  ARM CLIENT v4.0 - Smart Controller with Local Storage")
    print("=" * 70)
    print()
    print("✅ All files stored locally (PC side)")
    print("✅ Server has zero file operations (Edge device friendly)")
    print()
    print(f"📁 Positions:  {LOCAL_POSITIONS_DIR.absolute()}/")
    print(f"📁 Recordings: {LOCAL_RECORDINGS_DIR.absolute()}/")
    print()
    
    # ==================== 摄像头显示线程 ====================
    
    video_client = None
    video_running = True
    
    def video_display_loop():
        """摄像头显示线程"""
        global video_running
        
        cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Camera", 640, 480)
        
        while video_running:
            if video_client:
                frame, latency, fps = video_client.get_latest()
                
                if frame is not None:
                    # 添加 FPS 和延迟信息
                    info_text = f"FPS: {int(fps)}  Latency: {latency:.1f}ms"
                    cv2.putText(frame, info_text, (10, 25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # 如果正在录制，显示红点
                    if client._recording:
                        cv2.circle(frame, (frame.shape[1] - 30, 30), 15, (0, 0, 255), -1)
                        cv2.putText(frame, "REC", (frame.shape[1] - 70, 60),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    
                    cv2.imshow("Camera", frame)
                else:
                    # 无信号时显示黑屏
                    black = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(black, "No Signal", (240, 240),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
                    cv2.imshow("Camera", black)
            
            key = cv2.waitKey(30) & 0xFF
            if key == ord('q'):
                video_running = False
                break
        
        cv2.destroyAllWindows()
    
    # 启动摄像头
    if VIDEO_CLIENT_AVAILABLE and not args.no_cam:
        print(f"📹 Camera: {args.ip}:{args.cam_port}")
        video_client = VideoClient(server_ip=args.ip, video_port=args.cam_port)
        video_thread = threading.Thread(target=video_client.start, daemon=True)
        video_thread.start()
        
        # 启动显示线程
        display_thread = threading.Thread(target=video_display_loop, daemon=True)
        display_thread.start()
        print("📹 Camera display started (press 'q' in window to close)")
    else:
        print("📹 Camera: Disabled")
    
    print()
    
    # ==================== 机械臂客户端 ====================
    
    client = ArmClient(
        name="arm",
        server_ip=args.ip,
        ctl_port=args.port,
        leader_port=args.leader_port,
        leader_id=args.leader_id,
        hz=args.hz,
        record_hz=args.record_hz,
    )
    
    client.start()
    
    print("Commands:")
    print("  Position:  savepos <name>, goto <name> [dur], listpos, delpos <name>")
    print("  Recording: record <name>, stop, play <name> [times], recordings, delrec <name>")
    print("  Other:     status, home, quit")
    print()
    print("💡 Tips:")
    print("  - Type 'home' to return to zero point")
    print("  - Type 'quit' to exit (will auto return home)")
    print()
    
    try:
        while True:
            if client._recording:
                prompt = f"[REC {len(client._record_frames)}] >>> "
            else:
                prompt = ">>> "
            
            cmd = input(prompt).strip()
            if not cmd:
                continue
            
            parts = cmd.split()
            action = parts[0].lower()
            
            if action in ("quit", "exit", "q"):
                break
            
            elif action == "savepos" and len(parts) >= 2:
                client.savepos(parts[1])
                print(f"✓ Saved locally")
            
            elif action == "goto" and len(parts) >= 2:
                duration = float(parts[2]) if len(parts) >= 3 else 2.0
                success = client.goto(parts[1], duration)
                print(f"✓ {'Success' if success else 'Failed'}")
            
            elif action == "listpos":
                pos = client.listpos()
                print(f"Local positions: {', '.join(pos) if pos else 'None'}")
            
            elif action == "delpos" and len(parts) >= 2:
                success = client.delpos(parts[1])
                print(f"✓ {'Deleted' if success else 'Not found'}")
            
            elif action == "record" and len(parts) >= 2:
                client.start_record(parts[1])
            
            elif action == "stop":
                client.stop_record()
            
            elif action == "play" and len(parts) >= 2:
                times = int(parts[2]) if len(parts) >= 3 else 1
                client.play(parts[1], times)
            
            elif action == "recordings":
                recs = client.list_recordings()
                if recs:
                    print("Local recordings:")
                    for name, info in recs.items():
                        print(f"  - {name}: {info['frames']} frames, {info['duration']:.1f}s")
                else:
                    print("No recordings")
            
            elif action == "delrec" and len(parts) >= 2:
                success = client.delete_recording(parts[1])
                print(f"✓ {'Deleted' if success else 'Not found'}")
            
            elif action == "status":
                acc = client.get_accumulators()
                print(f"Accumulators: {acc}")
                buf = list(client.rtt_ms_buf)
                if buf:
                    print(f"RTT: min={min(buf):.1f}, max={max(buf):.1f}, avg={sum(buf)/len(buf):.1f} ms")
                if client._recording:
                    print(f"Recording: {client._record_name} ({len(client._record_frames)} frames)")
            
            elif action == "home":
                success = client.return_to_zero(duration=3.0)
                print(f"✓ {'Success' if success else 'Failed'}")
            
            else:
                print(f"Unknown: {cmd}")
                
    except KeyboardInterrupt:
        print("\n\n[!] Interrupted by user")
        print("[!] Returning to zero point before exit...")
        # 先回零点（在程序还正常运行时）
        try:
            client.return_to_zero(duration=3.0)
        except Exception as e:
            print(f"[!] Failed to return home: {e}")
    finally:
        # 停止视频显示
        video_running = False
        if video_client:
            try:
                video_client.stop()
            except:
                pass
        cv2.destroyAllWindows()
        
        # 然后停止程序
        client.stop()
