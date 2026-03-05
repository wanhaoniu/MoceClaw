import numpy as np
import pytest

from soarmmoce_sdk import CapabilityError, PermissionError, Robot, TimeoutError


def test_get_state_default_config():
    robot = Robot()
    robot.connect()

    state = robot.get_state()

    assert state.connected is True
    assert state.joint_state.q.shape[0] == robot.robot_model.dof
    assert state.tcp_pose.xyz.shape == (3,)
    assert state.tcp_pose.rpy.shape == (3,)
    assert state.gripper_state is not None
    assert state.gripper_state.available is True
    assert state.gripper_state.open_ratio == pytest.approx(1.0)
    assert state.permissions is not None
    assert state.permissions.allow_motion is True
    assert isinstance(state.timestamp, float)

    robot.disconnect()


def test_home_uses_zero_fallback():
    robot = Robot()
    robot.connect()

    lower = np.array([l for l, _ in robot.robot_model.joint_limits], dtype=float)
    upper = np.array([u for _, u in robot.robot_model.joint_limits], dtype=float)
    q_mid = lower + 0.3 * (upper - lower)
    robot.move_joints(q_mid, duration=0.01)

    q_home = robot.home(duration=0.01)
    assert np.allclose(q_home, np.zeros(robot.robot_model.dof))
    assert np.allclose(robot.get_joint_state().q, q_home)

    robot.disconnect()


def test_move_tcp_keeps_orientation_when_rpy_none():
    robot = Robot()
    robot.connect()

    current = robot.get_end_effector_pose()
    robot.move_tcp(
        float(current.xyz[0]),
        float(current.xyz[1]),
        float(current.xyz[2]),
        rpy=None,
        frame="base",
        duration=0.01,
    )

    after = robot.get_end_effector_pose()
    assert np.allclose(after.rpy, current.rpy, atol=1e-2)

    robot.disconnect()


def test_set_gripper_mock_supported():
    robot = Robot()
    robot.connect()

    robot.set_gripper(0.25, wait=True, timeout=0.5)

    robot.disconnect()


def test_set_gripper_unsupported_capability_error():
    robot = Robot()
    robot.connect()

    original = robot._transport.set_gripper  # type: ignore[attr-defined]

    def _unsupported(*args, **kwargs):
        raise NotImplementedError("unsupported")

    robot._transport.set_gripper = _unsupported  # type: ignore[attr-defined]
    with pytest.raises(CapabilityError):
        robot.set_gripper(0.4)

    robot._transport.set_gripper = original  # type: ignore[attr-defined]
    robot.disconnect()


def test_wait_timeout_best_effort():
    robot = Robot()
    robot.connect()

    q = robot.get_joint_state().q
    with pytest.raises(TimeoutError):
        robot.move_joints(q, duration=0.1, wait=True, timeout=0.01)

    robot.disconnect()


def test_permission_blocks_motion():
    robot = Robot()
    robot.connect()
    robot.set_permissions(allow_motion=False)
    q = robot.get_joint_state().q

    with pytest.raises(PermissionError):
        robot.move_joints(q, duration=0.01)

    robot.disconnect()


def test_rotate_joint_changes_one_joint():
    robot = Robot()
    robot.connect()

    q0 = robot.get_joint_state().q.copy()
    q1 = robot.rotate_joint("wrist_roll", delta_deg=5.0, duration=0.01, wait=True, timeout=0.1)
    idx = robot.robot_model.joint_names.index("wrist_roll")

    assert q1.shape == q0.shape
    assert q1[idx] != pytest.approx(q0[idx])

    robot.disconnect()
