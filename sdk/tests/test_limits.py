import numpy as np
import pytest
from pathlib import Path

from soarmMoce_sdk.api.robot import Robot
from soarmMoce_sdk.api.errors import LimitError


def test_limits_violation():
    cfg = Path(__file__).resolve().parents[2] / "configs" / "soarm_moce.yaml"
    robot = Robot.from_config(str(cfg))
    robot.connect()

    q = robot.get_joint_state().q
    upper = np.array([u for _, u in robot.robot_model.joint_limits], dtype=float)
    q[0] = upper[0] + 0.1

    with pytest.raises(LimitError):
        robot.move_joints(q)

    robot.disconnect()
