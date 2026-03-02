import numpy as np
from pathlib import Path

from soarmMoce_sdk.api.robot import Robot
from soarmMoce_sdk.kinematics import fk, matrix_to_rpy, solve_ik


def test_fk_ik_roundtrip():
    cfg = Path(__file__).resolve().parents[2] / "configs" / "soarm_moce.yaml"
    robot = Robot.from_config(str(cfg))
    robot.connect()

    lower = np.array([l for l, _ in robot.robot_model.joint_limits], dtype=float)
    upper = np.array([u for _, u in robot.robot_model.joint_limits], dtype=float)
    rng = np.random.default_rng(0)
    q = lower + (upper - lower) * rng.random(robot.robot_model.dof)

    T = fk(robot.robot_model, q)
    rpy = matrix_to_rpy(T[:3, :3])
    res = solve_ik(robot.robot_model, T[:3, 3], rpy, q0=q)

    assert res.success
    assert np.allclose(res.q, q, atol=1e-2)

    robot.disconnect()
