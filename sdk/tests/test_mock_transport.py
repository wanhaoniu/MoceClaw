import numpy as np
from pathlib import Path

from soarmMoce_sdk.api.robot import Robot


def test_mock_transport_movej():
    cfg = Path(__file__).resolve().parents[2] / "configs" / "soarm_moce.yaml"
    robot = Robot.from_config(str(cfg))
    robot.connect()

    lower = np.array([l for l, _ in robot.robot_model.joint_limits], dtype=float)
    upper = np.array([u for _, u in robot.robot_model.joint_limits], dtype=float)
    q = lower + 0.5 * (upper - lower)

    robot.move_joints(q, duration=1.0)
    js = robot.get_joint_state()

    assert np.allclose(js.q, q)

    robot.disconnect()
