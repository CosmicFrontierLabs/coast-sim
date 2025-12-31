import math

from conops import AttitudeControlSystem


class TestAttitudeControlSystem:
    def test_defaults_present_on_bus_instance(self, bus):
        assert isinstance(bus.attitude_control, AttitudeControlSystem)

    def test_defaults_present_on_bus_max_slew_rate(self, bus):
        # max_slew_rate is now optional (None means no cap, use physics)
        # Check that it's either None or a positive number
        rate = bus.attitude_control.max_slew_rate
        assert rate is None or rate > 0

    def test_defaults_present_on_bus_settle_time(self, bus):
        # settle_time is deprecated and defaults to 0
        assert bus.attitude_control.settle_time >= 0

    def test_triangular_profile_time(self, acs_config):
        """Test triangular slew profile timing (settle_time no longer added)."""
        angle = 0.1  # deg
        accel = acs_config.slew_acceleration
        # Triangular profile: motion_time = 2 * sqrt(angle / accel)
        expected = 2 * math.sqrt(angle / accel)
        assert abs(acs_config.slew_time(angle) - expected) < 1e-6

    def test_trapezoidal_profile_time(self, acs_config):
        """Test trapezoidal slew profile timing (settle_time no longer added)."""
        angle = 90.0
        accel = acs_config.slew_acceleration
        vmax = acs_config.max_slew_rate
        t_accel = vmax / accel
        d_accel = 0.5 * accel * t_accel**2
        assert 2 * d_accel < angle
        d_cruise = angle - 2 * d_accel
        expected = 2 * t_accel + d_cruise / vmax
        assert abs(acs_config.slew_time(angle) - expected) < 1e-6

    def test_zero_angle_time(self, default_acs):
        assert default_acs.slew_time(0) == 0.0
