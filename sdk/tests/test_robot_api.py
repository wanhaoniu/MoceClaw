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
    assert state.actual is not None
    assert state.actual.source == "mock"
    assert state.twin is not None
    assert state.twin.source == "kinematic_twin"
    assert np.allclose(state.actual.joint_state.q, state.joint_state.q)
    assert np.allclose(state.twin.joint_state.q, state.joint_state.q)

    robot.disconnect()


def test_home_uses_urdf_zero_pose():
    robot = Robot()
    robot.connect()

    lower = np.array([l for l, _ in robot.robot_model.joint_limits], dtype=float)
    upper = np.array([u for _, u in robot.robot_model.joint_limits], dtype=float)
    q_mid = lower + 0.3 * (upper - lower)
    robot.move_joints(q_mid, duration=0.01)

    q_home = robot.home(duration=0.01)
    expected = np.zeros(robot.robot_model.dof, dtype=float)
    assert np.allclose(q_home, expected)
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


def test_move_pose_pure_ry_rotation_changes_end_effector_pitch():
    robot = Robot()
    robot.connect()

    current = robot.get_end_effector_pose()
    target_rpy = current.rpy + np.array([0.0, 0.08, 0.0], dtype=float)
    robot.move_pose(
        xyz=current.xyz,
        rpy=target_rpy,
        duration=0.01,
        wait=True,
        timeout=0.2,
    )

    after = robot.get_end_effector_pose()
    delta = after.rpy - current.rpy
    assert delta[1] > 0.05

    robot.disconnect()


def test_move_pose_pure_rz_rotation_changes_end_effector_yaw():
    robot = Robot()
    robot.connect()

    current = robot.get_end_effector_pose()
    target_rpy = current.rpy + np.array([0.0, 0.0, 0.08], dtype=float)
    robot.move_pose(
        xyz=current.xyz,
        rpy=target_rpy,
        duration=0.01,
        wait=True,
        timeout=0.2,
    )

    after = robot.get_end_effector_pose()
    delta = after.rpy - current.rpy
    assert delta[2] > 0.05

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


def test_twin_tracks_command_target_during_motion():
    robot = Robot()
    robot.connect()

    q0 = robot.get_joint_state().q.copy()
    q1 = q0.copy()
    q1[-1] = q1[-1] + np.deg2rad(8.0)

    robot.move_joints(q1, duration=0.3, wait=False)
    state = robot.get_state()

    assert state.actual is not None
    assert state.twin is not None
    assert np.allclose(state.twin.joint_state.q, q1)
    assert not np.allclose(state.actual.joint_state.q, q1, atol=1e-4)

    robot.wait_until_stopped(timeout=1.0)
    settled = robot.get_state()
    assert settled.actual is not None
    assert np.allclose(settled.actual.joint_state.q, q1, atol=1e-4)

    robot.disconnect()
