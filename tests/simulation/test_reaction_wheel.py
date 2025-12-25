from unittest.mock import Mock

from conops.simulation.reaction_wheel import ReactionWheel


def test_accel_limit_and_reserve_and_apply():
    rw = ReactionWheel(max_torque=0.1, max_momentum=1.0)
    # For a reasonable moi, accel should be positive
    accel = rw.accel_limit_deg(5.0)
    assert accel > 0

    # If motion_time * requested_torque > available momentum, reserve_impulse reduces torque
    requested = 0.1
    motion_time = 20.0
    adjusted = rw.reserve_impulse(requested, motion_time)
    assert abs(adjusted) < abs(requested)

    # Applying torque updates current_momentum and clamps at capacity
    rw.apply_torque(adjusted, motion_time)
    assert abs(rw.current_momentum) <= rw.max_momentum + 1e-12


def test_accel_with_mock_moi_returns_inf_when_torque_positive():
    rw = ReactionWheel(max_torque=0.1, max_momentum=1.0)
    mock_moi = Mock()
    val = rw.accel_limit_deg(mock_moi)
    assert val == float("inf")
