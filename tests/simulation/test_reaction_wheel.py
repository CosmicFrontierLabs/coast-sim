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


def test_power_draw_idle():
    """Test that idle power is returned when no torque applied."""
    rw = ReactionWheel(
        max_torque=0.1, max_momentum=1.0, idle_power_w=10.0, torque_power_coeff=100.0
    )
    # No torque applied yet
    assert rw.power_draw() == 10.0


def test_power_draw_with_torque():
    """Test power draw includes torque-dependent component."""
    rw = ReactionWheel(
        max_torque=0.1, max_momentum=1.0, idle_power_w=10.0, torque_power_coeff=100.0
    )
    # Apply 0.05 N*m torque
    rw.apply_torque(0.05, dt=1.0)
    # Power = idle + coeff * |torque| = 10 + 100 * 0.05 = 15 W
    assert rw.power_draw() == 15.0


def test_power_draw_negative_torque():
    """Test power draw uses absolute torque value."""
    rw = ReactionWheel(
        max_torque=0.1, max_momentum=1.0, idle_power_w=5.0, torque_power_coeff=50.0
    )
    rw.apply_torque(-0.02, dt=1.0)
    # Power = 5 + 50 * 0.02 = 6 W
    assert rw.power_draw() == 6.0
