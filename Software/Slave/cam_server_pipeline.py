#import subprocess
#import signal
#import os
#
## ==========================================
## [配置] 接收端 PC 的 IP 地址
## 注意：这是你电脑(虚拟机)的 IP，机器人会把视频推送到这里
## ==========================================
#TARGET_PC_IP = "192.168.66.212"  # <--- 请根据实际情况修改
#
#class HardwareStreamer:
#    """
#    使用 RK3562 硬件加速 (VPU) 进行 H.264 推流
#    替代原来的 OpenCV 软编码 CameraServer
#    """
#    def __init__(self, dev_path, port, target_ip, width=1280, height=720, fps=30):
#        self.dev_path = dev_path
#        self.port = port
#        self.target_ip = target_ip
#        self.width = width
#        self.height = height
#        self.fps = fps
#        self.process = None
#
#    def start(self):
#        """启动 GStreamer 子进程 (非阻塞)"""
#        # 构建命令：V4L2(MJPEG) -> MPP解码 -> MPP编码(H.264) -> RTP -> UDP
#        cmd = [
#            "gst-launch-1.0",
#            "v4l2src", f"device={self.dev_path}", "!",
#            f"image/jpeg,width={self.width},height={self.height},framerate={self.fps}/1", "!",
#            "mppjpegdec", "!",         # 硬件解码
#            "mpph264enc", "!",         # 硬件编码
#            "rtph264pay", "config-interval=1", "pt=96", "!",
#            "udpsink", f"host={self.target_ip}", f"port={self.port}"
#        ]
#        
#        print(f"[HW-Video] 启动硬件加速推流: {self.dev_path} -> {self.target_ip}:{self.port}")
#        # 使用 subprocess 启动，不占用 Python 主线程
#        # stdout/stderr 设为 DEVNULL 防止日志刷屏，调试时可去掉
#        self.process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
#
#    def stop(self):
#        """停止子进程"""
#        if self.process:
#            print(f"[HW-Video] 正在停止 {self.dev_path} ...")
#            self.process.send_signal(signal.SIGINT)
#            try:
#                self.process.wait(timeout=1.0)
#            except subprocess.TimeoutExpired:
#                self.process.kill()
#            self.process = None
import subprocess
import signal
import os
import time
import threading

class HardwareStreamer:
    """
    使用 RK3562 硬件加速 (VPU) 进行 H.264 推流 (终极版)
    特性：
    1. 自动保活 (Auto-Respawn)
    2. 智能资源清理 (Smart Cleanup) - 崩溃后自动释放被锁死的 /dev/video 设备
    """
    def __init__(self, dev_path, port, target_ip, width=1280, height=720, fps=30, format="MJPG"):
        self.dev_path = dev_path
        self.port = port
        self.target_ip = target_ip
        self.width = width
        self.height = height
        self.fps = fps
        self.format = format
        
        self.process = None
        self.running = False
        self._monitor_thread = None

    def start(self):
        """启动保活监控线程"""
        if self.running:
            return
            
        self.running = True
        self._monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._monitor_thread.start()

    def stop(self):
        """停止推流并退出监控"""
        self.running = False
        # 主动停止时，也执行一次清理
        self._kill_process()
        self._force_cleanup_resources()
        
        if self._monitor_thread:
            self._monitor_thread.join(timeout=1.0)

    def _monitor_loop(self):
        retry_count = 0
        
        while self.running:
            # --- 1. 启动进程 ---
            if self.process is None:
                self._launch_process()
                retry_count += 1
                # 启动后观察 2 秒，看是不是“闪退”
                time.sleep(2.0) 

            # --- 2. 检查进程状态 ---
            poll_ret = self.process.poll()
            
            if poll_ret is None:
                # 进程健康运行中
                retry_count = 0
                time.sleep(1.0)
            else:
                # >>> 进程意外挂了 (Device busy 或 抢占失败) <<<
                print(f"[HW-Video] {self.dev_path} 意外退出 (Code: {poll_ret})")
                self.process = None 
                
                # [关键步骤] 在重试之前，执行“外科手术式”清理
                print(f"[HW-Video] 正在清理 {self.dev_path} 的残留占用...")
                self._force_cleanup_resources()
                
                # 策略：失败次数越多，冷却时间越长 (退避算法)
                # 第一次失败等 2秒，第二次等 3秒...
                cooldown = min(2.0 + retry_count, 8.0) 
                print(f"[HW-Video] 将在 {cooldown:.1f} 秒后尝试第 {retry_count} 次重启...")
                
                # 冷却等待 (期间也要响应 stop 信号)
                steps = int(cooldown * 10)
                for _ in range(steps):
                    if not self.running: break
                    time.sleep(0.1)

    def _launch_process(self):
        """构建命令并启动"""
        # 构建 GStreamer 管道 (保持不变)
        if self.format == "MJPG":
            source_caps = f"image/jpeg,width={self.width},height={self.height},framerate={self.fps}/1"
            pipeline_elements = [
                "v4l2src", f"device={self.dev_path}", "!",
                source_caps, "!",
                "mppjpegdec", "!", 
                "mpph264enc", "!" 
            ]
        elif self.format == "YUYV":
            source_caps = f"video/x-raw,format=YUY2,width={self.width},height={self.height},framerate={self.fps}/1"
            pipeline_elements = [
                "v4l2src", f"device={self.dev_path}", "!",
                source_caps, "!",
                "mpph264enc", "!" 
            ]
        else:
            print(f"[Error] 未知格式: {self.format}")
            return

        common_tail = [
            "rtph264pay", "config-interval=1", "pt=96", "!",
            "udpsink", f"host={self.target_ip}", f"port={self.port}"
        ]

        cmd = ["gst-launch-1.0"] + pipeline_elements + common_tail
        
        print(f"[HW-Video] 尝试启动: {self.dev_path} -> :{self.port}")
        self.process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,  # <--- 修改这里
            text=True                # <--- 加上这个以便读取文本
        )
        def log_reader(proc, name):
            try:
                # 只读取 stderr
                for line in proc.stderr:
                    print(f"[{name} ERROR] {line.strip()}")
            except:
                pass
        t = threading.Thread(target=log_reader, args=(self.process, self.dev_path), daemon=True)
        t.start()
        
    def _kill_process(self):
        if self.process:
            self.process.send_signal(signal.SIGINT)
            try:
                self.process.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.process.kill()
            self.process = None

    def _force_cleanup_resources(self):
        """
        [新增] 针对性清理资源
        使用 fuser 命令强制杀掉任何占用当前设备 (/dev/videoX) 的进程
        """
        try:
            # 1. 杀掉占用摄像头的进程 (精准打击)
            # fuser -k -9 /dev/video40
            cmd_dev = f"fuser -k -9 {self.dev_path}"
            os.system(cmd_dev + " > /dev/null 2>&1")
            
            # 2. (可选) 清理网络端口，防止 GStreamer 没释放 UDP
            # cmd_net = f"fuser -k -n udp {self.port}"
            # os.system(cmd_net + " > /dev/null 2>&1")
            
            # 3. 给内核一点时间回收文件句柄
            time.sleep(3.0)
            
        except Exception as e:
            print(f"[HW-Video] 清理资源出错: {e}")