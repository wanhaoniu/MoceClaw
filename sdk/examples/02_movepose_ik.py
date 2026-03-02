#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from soarmMoce_sdk.api.robot import Robot


def main():
    robot = Robot.from_config("./configs/soarm_moce.yaml")
    robot.connect()

    # Use current pose as a reachable target for demonstration
    current_pose = robot.get_end_effector_pose()
    target_xyz = current_pose.xyz
    target_rpy = current_pose.rpy

    q = robot.move_pose(target_xyz, target_rpy, q0=robot.get_joint_state().q, duration=2.0)
    print("IK q:", q)

    pose = robot.get_end_effector_pose(q)
    print("FK pose xyz:", pose.xyz)
    print("FK pose rpy:", pose.rpy)

    robot.disconnect()


if __name__ == "__main__":
    main()
