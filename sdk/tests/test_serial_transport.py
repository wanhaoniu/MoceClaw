import time

import numpy as np

from soarmmoce_sdk.transport.serial import SerialTransport


def test_serial_transport_send_movej_is_async():
    transport = SerialTransport(
        5,
        joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
        port="/dev/null",
        update_hz=20.0,
    )

    state_deg = {name: 0.0 for name in transport.joint_names}
    writes = []

    transport._connected = True
    transport._bus = object()

    def fake_read_joint_state_deg():
        return {name: float(state_deg[name]) for name in transport.joint_names}

    def fake_write_joint_targets_deg(target_joint_deg):
        writes.append((time.time(), dict(target_joint_deg)))
        for name, value in target_joint_deg.items():
            state_deg[name] = float(value)

    transport._read_joint_state_deg = fake_read_joint_state_deg  # type: ignore[method-assign]
    transport._write_joint_targets_deg = fake_write_joint_targets_deg  # type: ignore[method-assign]
    transport._start_motion_worker()

    try:
        q_target = np.deg2rad(np.array([0.0, 10.0, 10.0, 0.0, 0.0], dtype=float))
        t0 = time.time()
        transport.send_movej(q_target, duration=0.2)
        elapsed = time.time() - t0

        assert elapsed < 0.05
        assert transport.wait_until_stopped(timeout=0.01) is False
        assert transport.wait_until_stopped(timeout=1.0) is True
        assert len(writes) >= 2
        assert state_deg["shoulder_lift"] > 0.0
        assert state_deg["elbow_flex"] > 0.0
    finally:
        transport.disconnect()
