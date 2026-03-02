# -*- coding: utf-8 -*-
from pathlib import Path
import time
import draccus

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode


def load_calibration(robot_name: str, calib_dir: Path | None = None) -> dict[str, MotorCalibration]:
    """
    读取标定JSON为 {joint_name -> MotorCalibration} 映射。
    默认从 ./calibration_files/{robot_name}.json 读取；你也可以传入自定义目录。
    """
    if calib_dir is None:
        calib_dir = Path(__file__).parent / "calibration_files"
        
    fpath = calib_dir / f"{robot_name}.json"
    print('[Debug] load_calibration fpath:', fpath)
    with open(fpath, "r", encoding="utf-8") as f, draccus.config_type("json"):
        calibration = draccus.load(dict[str, MotorCalibration], f)
    return calibration


def setup_leader_bus(port: str, calibration: dict[str, MotorCalibration]) -> FeetechMotorsBus:
    """
    连接 Leader 的 6 个电机，并应用标定。
    关节命名保持与教程一致：shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
    """
    norm_deg = MotorNormMode.DEGREES
    bus = FeetechMotorsBus(
        port=port,
        motors={
            "shoulder_pan":  Motor(1, "sts3215", norm_deg),
            "shoulder_lift": Motor(2, "sts3215", norm_deg),
            "elbow_flex":    Motor(3, "sts3215", norm_deg),
            "wrist_flex":    Motor(4, "sts3215", norm_deg),
            "wrist_roll":    Motor(5, "sts3215", norm_deg),
            "gripper":       Motor(6, "sts3215", MotorNormMode.RANGE_0_100),  # 夹爪常用 0~100 归一化范围
        },
        calibration=calibration,
    )
    # 打开串口并检查总线
    bus.connect()

    # 一般把 Leader 当“编码器”用，默认关闭力矩；但先在无力矩状态下做一次寄存器配置
    with bus.torque_disabled():
        bus.configure_motors()
        for motor in bus.motors:
            # 设为位置模式；即使不发位置命令，读取位置寄存器也正常
            bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
            # 一些实践里会把 P 调小避免抖动；I=0；D 默认 32，如仍有抖动可尝试设为 0
            bus.write("P_Coefficient", motor, 16)
            bus.write("I_Coefficient", motor, 0)
            bus.write("D_Coefficient", motor, 32)  # 抖动时可尝试 0

    # Leader 读取角度前，通常关闭力矩方便手动带动
    bus.disable_torque()
    return bus


def read_joint_angles_deg(bus: FeetechMotorsBus) -> dict[str, float]:
    """
    以“度”为单位读取 6 关节当前角度；夹爪返回 0~100 的开合量。
    """
    return bus.sync_read("Present_Position")  # 返回 {joint_name: value}
