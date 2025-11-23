import math

from conops.spacecraft_bus import AttitudeControlSystem, SpacecraftBus


def test_defaults_present_on_bus():
    bus = SpacecraftBus()
    assert isinstance(bus.attitude_control, AttitudeControlSystem)
    assert bus.attitude_control.max_slew_rate > 0
    assert bus.attitude_control.settle_time >= 0


def test_triangular_profile_time():
    acs = AttitudeControlSystem(
        slew_acceleration=1.0, max_slew_rate=0.5, settle_time=90.0
    )
    angle = 0.1  # deg
    expected_no_settle = 2 * math.sqrt(angle / acs.slew_acceleration)
    expected = expected_no_settle + acs.settle_time
    assert abs(acs.slew_time(angle) - expected) < 1e-6


def test_trapezoidal_profile_time():
    acs = AttitudeControlSystem(
        slew_acceleration=1.0, max_slew_rate=0.5, settle_time=90.0
    )
    angle = 90.0
    t_accel = acs.max_slew_rate / acs.slew_acceleration
    d_accel = 0.5 * acs.slew_acceleration * t_accel**2
    assert 2 * d_accel < angle
    d_cruise = angle - 2 * d_accel
    expected_no_settle = 2 * t_accel + d_cruise / acs.max_slew_rate
    expected = expected_no_settle + acs.settle_time
    assert abs(acs.slew_time(angle) - expected) < 1e-6


def test_zero_angle_time():
    acs = AttitudeControlSystem()
    assert acs.slew_time(0) == 0.0
