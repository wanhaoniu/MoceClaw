import cv2
import threading
import time
import numpy as np

class H264VideoClient:
    """
    PC端 H.264 接收客户端 (基于 GStreamer + OpenCV)
    对应移动端的 HardwareStreamer
    """
    def __init__(self, video_port, server_ip=None):
        # server_ip 在 UDP 接收端其实没用(因为是 bind 本地)，但为了保持接口一致性保留
        self.port = video_port
        self.frame = None
        self.running = False
        self.fps = 0.0
        self.latency = 0 # H.264 难以通过 OpenCV 直接获取发送端时间戳，这里暂置 0
        self.lock = threading.Lock()
        
        # 统计 FPS 用
        self._frame_count = 0
        self._last_time = time.time()

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if hasattr(self, 'thread'):
            self.thread.join(timeout=1.0)

    def get_latest(self):
        """
        返回: (frame, latency, fps)
        """
        with self.lock:
            return self.frame, self.latency, self.fps

    def _capture_loop(self):
        # -----------------------------------------------------------
        # GStreamer 接收管道 (终极低延迟版)
        # 对应发送端的 rtph264pay payload=96
        # -----------------------------------------------------------
        pipeline = (
            f"udpsrc port={self.port} ! "
            "application/x-rtp, encoding-name=H264, payload=96 ! "
            "rtph264depay ! "
            "avdec_h264 ! "       # 软件解码 H.264
            "videoconvert ! "     # 颜色转换
            "video/x-raw, format=BGR ! " 
            "appsink sync=false drop=true max-buffers=1" # 关键：丢弃旧帧，只要最新的
        )

        print(f"[H264-Client] 正在监听端口 {self.port} ...")
        
        # 必须指定 backend 为 CAP_GSTREAMER
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)

        if not cap.isOpened():
            print(f"[Error] 无法打开端口 {self.port} 的 GStreamer 管道！请检查 OpenCV 是否支持 GStreamer。")
            self.running = False
            return

        while self.running:
            ret, img = cap.read()
            if ret:
                with self.lock:
                    self.frame = img
                
                # 计算 FPS
                self._frame_count += 1
                if self._frame_count % 30 == 0:
                    now = time.time()
                    dt = now - self._last_time
                    if dt > 0:
                        self.fps = 30 / dt
                    self._last_time = now
            else:
                # 如果没读到数据（比如机器人还没发），稍微等一下避免死循环占满 CPU
                time.sleep(0.01)

        cap.release()
        print(f"[H264-Client] 端口 {self.port} 已停止。")