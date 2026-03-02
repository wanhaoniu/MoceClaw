#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from soarmMoce_sdk.api.robot import Robot


def main():
    robot = Robot.from_config("./configs/soarm_moce.yaml")
    robot.connect()
    js = robot.get_joint_state()
    print("Joint state q:", js.q)
    robot.move_joints(js.q, duration=1.0)
    robot.disconnect()


if __name__ == "__main__":
    main()
