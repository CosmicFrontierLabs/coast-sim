import math

from conops import AttitudeControlSystem


class TestAttitudeControlSystem:
    def test_defaults_present_on_bus_instance(self, bus):
        assert isinstance(bus.attitude_control, AttitudeControlSystem)

    def test_defaults_present_on_bus_max_slew_rate(self, bus):
        assert bus.attitude_control.max_slew_rate > 0

    def test_defaults_present_on_bus_settle_time(self, bus):
        assert bus.attitude_control.settle_time >= 0

    def test_triangular_profile_time(self, acs_config):
        angle = 0.1  # deg
        expected_no_settle = 2 * math.sqrt(angle / acs_config.slew_acceleration)
        expected = expected_no_settle + acs_config.settle_time
        assert abs(acs_config.slew_time(angle) - expected) < 1e-6

    def test_trapezoidal_profile_time(self, acs_config):
        angle = 90.0
        t_accel = acs_config.max_slew_rate / acs_config.slew_acceleration
        d_accel = 0.5 * acs_config.slew_acceleration * t_accel**2
        assert 2 * d_accel < angle
        d_cruise = angle - 2 * d_accel
        expected_no_settle = 2 * t_accel + d_cruise / acs_config.max_slew_rate
        expected = expected_no_settle + acs_config.settle_time
        assert abs(acs_config.slew_time(angle) - expected) < 1e-6

    def test_zero_angle_time(self, default_acs):
        assert default_acs.slew_time(0) == 0.0
