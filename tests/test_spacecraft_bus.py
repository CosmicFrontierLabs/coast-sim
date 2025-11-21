"""Unit tests for spacecraft_bus module."""

import numpy as np

from conops.spacecraft_bus import AttitudeControlSystem, PowerDraw, SpacecraftBus


class TestPowerDraw:
    """Tests for PowerDraw class."""

    def test_initialization_defaults(self):
        """Test PowerDraw initializes with default values."""
        pd = PowerDraw()
        assert pd.nominal_power == 200
        assert pd.peak_power == 300
        assert pd.power_mode == {}

    def test_initialization_custom(self):
        """Test PowerDraw initializes with custom values."""
        pd = PowerDraw(nominal_power=150, peak_power=250, power_mode={1: 175, 2: 225})
        assert pd.nominal_power == 150
        assert pd.peak_power == 250
        assert pd.power_mode == {1: 175, 2: 225}

    def test_power_no_mode(self):
        """Test power() returns nominal_power when no mode specified."""
        pd = PowerDraw(nominal_power=180)
        assert pd.power() == 180
        assert pd.power(None) == 180

    def test_power_with_mode(self):
        """Test power() returns mode-specific power when mode is defined."""
        pd = PowerDraw(nominal_power=100, power_mode={1: 120, 2: 140})
        assert pd.power(1) == 120
        assert pd.power(2) == 140

    def test_power_undefined_mode(self):
        """Test power() returns nominal_power for undefined modes."""
        pd = PowerDraw(nominal_power=100, power_mode={1: 120})
        assert pd.power(99) == 100


class TestAttitudeControlSystem:
    """Tests for AttitudeControlSystem class."""

    def test_initialization_defaults(self):
        """Test ACS initializes with default values."""
        acs = AttitudeControlSystem()
        assert acs.slew_acceleration == 0.5
        assert acs.max_slew_rate == 0.25
        assert acs.slew_accuracy == 0.01
        assert acs.settle_time == 120.0

    def test_initialization_custom(self):
        """Test ACS initializes with custom values."""
        acs = AttitudeControlSystem(
            slew_acceleration=1.0,
            max_slew_rate=0.5,
            slew_accuracy=0.05,
            settle_time=60.0,
        )
        assert acs.slew_acceleration == 1.0
        assert acs.max_slew_rate == 0.5
        assert acs.slew_accuracy == 0.05
        assert acs.settle_time == 60.0

    def test_motion_time_zero_angle(self):
        """Test motion_time returns 0 for zero or negative angles."""
        acs = AttitudeControlSystem()
        assert acs.motion_time(0) == 0.0
        assert acs.motion_time(-10) == 0.0

    def test_motion_time_invalid_params(self):
        """Test motion_time returns 0 for invalid parameters."""
        acs = AttitudeControlSystem(slew_acceleration=0)
        assert acs.motion_time(10) == 0.0

        acs = AttitudeControlSystem(max_slew_rate=0)
        assert acs.motion_time(10) == 0.0

    def test_motion_time_triangular_profile(self):
        """Test motion_time for small angle (triangular velocity profile)."""
        # With a=0.5, vmax=0.25: t_accel=0.5, d_accel=0.0625
        # 2*d_accel = 0.125, so angles < 0.125 use triangular profile
        acs = AttitudeControlSystem(slew_acceleration=0.5, max_slew_rate=0.25)
        angle = 0.1  # Less than 2*d_accel
        motion_time = acs.motion_time(angle)
        # For triangular: t_peak = sqrt(angle/a), total = 2*t_peak
        expected = 2 * np.sqrt(angle / 0.5)
        assert abs(motion_time - expected) < 1e-6

    def test_motion_time_trapezoidal_profile(self):
        """Test motion_time for large angle (trapezoidal velocity profile)."""
        acs = AttitudeControlSystem(slew_acceleration=0.5, max_slew_rate=0.25)
        angle = 10.0  # Much larger than 2*d_accel
        motion_time = acs.motion_time(angle)
        # t_accel = 0.5, d_accel = 0.0625
        # d_cruise = 10 - 2*0.0625 = 9.875
        # t_cruise = 9.875 / 0.25 = 39.5
        # total = 2*0.5 + 39.5 = 40.5
        expected = 2 * 0.5 + 9.875 / 0.25
        assert abs(motion_time - expected) < 1e-6

    def test_s_of_t_zero_conditions(self):
        """Test s_of_t returns 0 for zero angle or time."""
        acs = AttitudeControlSystem()
        assert acs.s_of_t(0, 10) == 0.0
        assert acs.s_of_t(10, 0) == 0.0
        assert acs.s_of_t(-5, 10) == 0.0

    def test_s_of_t_invalid_params(self):
        """Test s_of_t fallback for invalid parameters."""
        acs = AttitudeControlSystem(slew_acceleration=0, max_slew_rate=0.25)
        # Should use fallback: min(max(0, t*vmax), angle)
        result = acs.s_of_t(10, 5)
        expected = min(5 * 0.25, 10)
        assert abs(result - expected) < 1e-6

    def test_s_of_t_triangular_acceleration(self):
        """Test s_of_t during acceleration phase of triangular profile."""
        acs = AttitudeControlSystem(slew_acceleration=0.5, max_slew_rate=0.25)
        angle = 0.1
        t_peak = np.sqrt(angle / 0.5)
        t = t_peak * 0.5  # Halfway through acceleration
        s = acs.s_of_t(angle, t)
        expected = 0.5 * 0.5 * t**2
        assert abs(s - expected) < 1e-6

    def test_s_of_t_triangular_deceleration(self):
        """Test s_of_t during deceleration phase of triangular profile."""
        acs = AttitudeControlSystem(slew_acceleration=0.5, max_slew_rate=0.25)
        angle = 0.1
        t_peak = np.sqrt(angle / 0.5)
        motion_time = 2 * t_peak
        t = motion_time * 0.75  # During deceleration
        s = acs.s_of_t(angle, t)
        expected = angle - 0.5 * 0.5 * (motion_time - t) ** 2
        assert abs(s - expected) < 1e-6

    def test_s_of_t_trapezoidal_acceleration(self):
        """Test s_of_t during acceleration phase of trapezoidal profile."""
        acs = AttitudeControlSystem(slew_acceleration=0.5, max_slew_rate=0.25)
        angle = 10.0
        t = 0.25  # During acceleration
        s = acs.s_of_t(angle, t)
        expected = 0.5 * 0.5 * t**2
        assert abs(s - expected) < 1e-6

    def test_s_of_t_trapezoidal_cruise(self):
        """Test s_of_t during cruise phase of trapezoidal profile."""
        acs = AttitudeControlSystem(slew_acceleration=0.5, max_slew_rate=0.25)
        angle = 10.0
        t_accel = 0.5
        d_accel = 0.0625
        t = 5.0  # During cruise
        s = acs.s_of_t(angle, t)
        expected = d_accel + 0.25 * (t - t_accel)
        assert abs(s - expected) < 1e-6

    def test_s_of_t_trapezoidal_deceleration(self):
        """Test s_of_t during deceleration phase of trapezoidal profile."""
        acs = AttitudeControlSystem(slew_acceleration=0.5, max_slew_rate=0.25)
        angle = 10.0
        t_accel = 0.5
        d_accel = 0.0625
        d_cruise = angle - 2 * d_accel
        t_cruise = d_cruise / 0.25
        t = t_accel + t_cruise + 0.1  # During deceleration
        s = acs.s_of_t(angle, t)
        t_dec = t - (t_accel + t_cruise)
        expected = d_accel + d_cruise + 0.25 * t_dec - 0.5 * 0.5 * t_dec**2
        assert abs(s - expected) < 1e-6

    def test_s_of_t_after_motion_complete(self):
        """Test s_of_t returns full angle after motion is complete."""
        acs = AttitudeControlSystem(slew_acceleration=0.5, max_slew_rate=0.25)
        angle = 10.0
        motion_time = acs.motion_time(angle)
        s = acs.s_of_t(angle, motion_time + 100)
        assert abs(s - angle) < 1e-6

    def test_slew_time(self):
        """Test slew_time includes motion time and settle time."""
        acs = AttitudeControlSystem(slew_acceleration=0.5, settle_time=60.0)
        angle = 10.0
        motion_time = acs.motion_time(angle)
        total_time = acs.slew_time(angle)
        assert abs(total_time - (motion_time + 60.0)) < 1e-6

    def test_slew_time_zero_angle(self):
        """Test slew_time returns 0 for zero or negative angles."""
        acs = AttitudeControlSystem()
        assert acs.slew_time(0) == 0.0
        assert acs.slew_time(-5) == 0.0

    def test_predict_slew_same_position(self):
        """Test predict_slew with identical start and end positions."""
        acs = AttitudeControlSystem()
        slewdist, slewpath = acs.predict_slew(0, 0, 0, 0)
        assert abs(slewdist) < 1e-6
        ra_path, dec_path = slewpath
        # great_circle returns steps+2 points (includes start and end)
        assert len(ra_path) == 22  # 20 steps + start + end
        assert len(dec_path) == 22

    def test_predict_slew_different_positions(self):
        """Test predict_slew calculates distance and path correctly."""
        acs = AttitudeControlSystem()
        # Simple case: 90 degree separation along equator
        slewdist, slewpath = acs.predict_slew(0, 0, 90, 0)
        assert abs(slewdist - 90.0) < 0.1  # Should be ~90 degrees
        ra_path, dec_path = slewpath
        # great_circle returns steps+2 points (includes start and end)
        assert len(ra_path) == 22  # 20 steps + start + end
        assert len(dec_path) == 22
        # Check path starts and ends at correct positions
        assert abs(ra_path[0] - 0) < 1e-6
        assert abs(dec_path[0] - 0) < 1e-6
        assert abs(ra_path[-1] - 90) < 1e-6
        assert abs(dec_path[-1] - 0) < 1e-6

    def test_predict_slew_custom_steps(self):
        """Test predict_slew with custom number of steps."""
        acs = AttitudeControlSystem()
        slewdist, slewpath = acs.predict_slew(0, 0, 45, 45, steps=10)
        ra_path, dec_path = slewpath
        # great_circle returns steps+2 points (includes start and end)
        assert len(ra_path) == 12  # 10 steps + start + end
        assert len(dec_path) == 12

    def test_predict_slew_across_meridian(self):
        """Test predict_slew across the 0/360 degree meridian."""
        acs = AttitudeControlSystem()
        slewdist, slewpath = acs.predict_slew(350, 0, 10, 0)
        assert slewdist > 0  # Should find the shorter path
        assert slewdist < 180  # Should not go the long way


class TestSpacecraftBus:
    """Tests for SpacecraftBus class."""

    def test_initialization_defaults(self):
        """Test SpacecraftBus initializes with default values."""
        bus = SpacecraftBus()
        assert bus.name == "Default Bus"
        assert isinstance(bus.power_draw, PowerDraw)
        assert isinstance(bus.attitude_control, AttitudeControlSystem)

    def test_initialization_custom(self):
        """Test SpacecraftBus initializes with custom values."""
        pd = PowerDraw(nominal_power=150)
        acs = AttitudeControlSystem(slew_acceleration=1.0)
        bus = SpacecraftBus(name="Custom Bus", power_draw=pd, attitude_control=acs)
        assert bus.name == "Custom Bus"
        assert bus.power_draw.nominal_power == 150
        assert bus.attitude_control.slew_acceleration == 1.0

    def test_power_delegates_to_power_draw(self):
        """Test SpacecraftBus.power() delegates to PowerDraw."""
        pd = PowerDraw(nominal_power=100, power_mode={1: 120})
        bus = SpacecraftBus(power_draw=pd)
        assert bus.power() == 100
        assert bus.power(1) == 120
        assert bus.power(99) == 100
