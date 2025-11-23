"""Unit tests for spacecraft_bus module."""

import numpy as np

from conops import AttitudeControlSystem, PowerDraw, SpacecraftBus


class TestPowerDraw:
    """Tests for PowerDraw class."""

    def test_initialization_defaults_nominal_power(self, default_power_draw):
        """Test PowerDraw initializes with default nominal_power."""
        assert default_power_draw.nominal_power == 200

    def test_initialization_defaults_peak_power(self, default_power_draw):
        """Test PowerDraw initializes with default peak_power."""
        assert default_power_draw.peak_power == 300

    def test_initialization_defaults_power_mode(self, default_power_draw):
        """Test PowerDraw initializes with default power_mode."""
        assert default_power_draw.power_mode == {}

    def test_initialization_custom_nominal_power(self, custom_power_draw):
        """Test PowerDraw initializes with custom nominal_power."""
        assert custom_power_draw.nominal_power == 150

    def test_initialization_custom_peak_power(self, custom_power_draw):
        """Test PowerDraw initializes with custom peak_power."""
        assert custom_power_draw.peak_power == 250

    def test_initialization_custom_power_mode(self, custom_power_draw):
        """Test PowerDraw initializes with custom power_mode."""
        assert custom_power_draw.power_mode == {1: 175, 2: 225}

    def test_power_no_mode(self):
        """Test power() returns nominal_power when no mode specified."""
        pd = PowerDraw(nominal_power=180)
        assert pd.power() == 180

    def test_power_none_mode(self):
        """Test power() returns nominal_power when mode is None."""
        pd = PowerDraw(nominal_power=180)
        assert pd.power(None) == 180

    def test_power_with_mode_1(self):
        """Test power() returns mode-specific power for mode 1."""
        pd = PowerDraw(nominal_power=100, power_mode={1: 120, 2: 140})
        assert pd.power(1) == 120

    def test_power_with_mode_2(self):
        """Test power() returns mode-specific power for mode 2."""
        pd = PowerDraw(nominal_power=100, power_mode={1: 120, 2: 140})
        assert pd.power(2) == 140

    def test_power_undefined_mode(self):
        """Test power() returns nominal_power for undefined modes."""
        pd = PowerDraw(nominal_power=100, power_mode={1: 120})
        assert pd.power(99) == 100


class TestAttitudeControlSystem:
    """Tests for AttitudeControlSystem class."""

    def test_initialization_defaults_slew_acceleration(self, default_acs):
        """Test ACS initializes with default slew_acceleration."""
        assert default_acs.slew_acceleration == 0.5

    def test_initialization_defaults_max_slew_rate(self, default_acs):
        """Test ACS initializes with default max_slew_rate."""
        assert default_acs.max_slew_rate == 0.25

    def test_initialization_defaults_slew_accuracy(self, default_acs):
        """Test ACS initializes with default slew_accuracy."""
        assert default_acs.slew_accuracy == 0.01

    def test_initialization_defaults_settle_time(self, default_acs):
        """Test ACS initializes with default settle_time."""
        assert default_acs.settle_time == 120.0

    def test_initialization_custom_slew_acceleration(self, custom_acs):
        """Test ACS initializes with custom slew_acceleration."""
        assert custom_acs.slew_acceleration == 1.0

    def test_initialization_custom_max_slew_rate(self, custom_acs):
        """Test ACS initializes with custom max_slew_rate."""
        assert custom_acs.max_slew_rate == 0.5

    def test_initialization_custom_slew_accuracy(self, custom_acs):
        """Test ACS initializes with custom slew_accuracy."""
        assert custom_acs.slew_accuracy == 0.05

    def test_initialization_custom_settle_time(self, custom_acs):
        """Test ACS initializes with custom settle_time."""
        assert custom_acs.settle_time == 60.0

    def test_motion_time_zero_angle(self, default_acs):
        """Test motion_time returns 0 for zero angle."""
        assert default_acs.motion_time(0) == 0.0

    def test_motion_time_negative_angle(self, default_acs):
        """Test motion_time returns 0 for negative angles."""
        assert default_acs.motion_time(-10) == 0.0

    def test_motion_time_invalid_params_zero_acceleration(self):
        """Test motion_time returns 0 for zero acceleration."""
        acs = AttitudeControlSystem(slew_acceleration=0)
        assert acs.motion_time(10) == 0.0

    def test_motion_time_invalid_params_zero_rate(self):
        """Test motion_time returns 0 for zero max_slew_rate."""
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

    def test_s_of_t_zero_angle(self, default_acs):
        """Test s_of_t returns 0 for zero angle."""
        assert default_acs.s_of_t(0, 10) == 0.0

    def test_s_of_t_zero_time(self, default_acs):
        """Test s_of_t returns 0 for zero time."""
        assert default_acs.s_of_t(10, 0) == 0.0

    def test_s_of_t_negative_angle(self, default_acs):
        """Test s_of_t returns 0 for negative angle."""
        assert default_acs.s_of_t(-5, 10) == 0.0

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

    def test_slew_time_zero_angle(self, default_acs):
        """Test slew_time returns 0 for zero angle."""
        assert default_acs.slew_time(0) == 0.0

    def test_slew_time_negative_angle(self, default_acs):
        """Test slew_time returns 0 for negative angles."""
        assert default_acs.slew_time(-5) == 0.0

    def test_predict_slew_same_position_distance(self, default_acs):
        """Test predict_slew distance with identical start and end positions."""
        slewdist, slewpath = default_acs.predict_slew(0, 0, 0, 0)
        assert abs(slewdist) < 1e-6

    def test_predict_slew_same_position_path_length_ra(self, default_acs):
        """Test predict_slew path length for ra with identical positions."""
        slewdist, slewpath = default_acs.predict_slew(0, 0, 0, 0)
        ra_path, dec_path = slewpath
        # great_circle returns steps+2 points (includes start and end)
        assert len(ra_path) == 22  # 20 steps + start + end

    def test_predict_slew_same_position_path_length_dec(self, default_acs):
        """Test predict_slew path length for dec with identical positions."""
        slewdist, slewpath = default_acs.predict_slew(0, 0, 0, 0)
        ra_path, dec_path = slewpath
        # great_circle returns steps+2 points (includes start and end)
        assert len(dec_path) == 22

    def test_predict_slew_different_positions_distance(self, default_acs):
        """Test predict_slew calculates distance correctly."""
        # Simple case: 90 degree separation along equator
        slewdist, slewpath = default_acs.predict_slew(0, 0, 90, 0)
        assert abs(slewdist - 90.0) < 0.1  # Should be ~90 degrees

    def test_predict_slew_different_positions_path_length_ra(self, default_acs):
        """Test predict_slew path length for ra."""
        # Simple case: 90 degree separation along equator
        slewdist, slewpath = default_acs.predict_slew(0, 0, 90, 0)
        ra_path, dec_path = slewpath
        # great_circle returns steps+2 points (includes start and end)
        assert len(ra_path) == 22  # 20 steps + start + end

    def test_predict_slew_different_positions_path_length_dec(self, default_acs):
        """Test predict_slew path length for dec."""
        # Simple case: 90 degree separation along equator
        slewdist, slewpath = default_acs.predict_slew(0, 0, 90, 0)
        ra_path, dec_path = slewpath
        # great_circle returns steps+2 points (includes start and end)
        assert len(dec_path) == 22

    def test_predict_slew_different_positions_start_ra(self, default_acs):
        """Test predict_slew path starts at correct ra."""
        # Simple case: 90 degree separation along equator
        slewdist, slewpath = default_acs.predict_slew(0, 0, 90, 0)
        ra_path, dec_path = slewpath
        # Check path starts and ends at correct positions
        assert abs(ra_path[0] - 0) < 1e-6

    def test_predict_slew_different_positions_start_dec(self, default_acs):
        """Test predict_slew path starts at correct dec."""
        # Simple case: 90 degree separation along equator
        slewdist, slewpath = default_acs.predict_slew(0, 0, 90, 0)
        ra_path, dec_path = slewpath
        # Check path starts and ends at correct positions
        assert abs(dec_path[0] - 0) < 1e-6

    def test_predict_slew_different_positions_end_ra(self, default_acs):
        """Test predict_slew path ends at correct ra."""
        # Simple case: 90 degree separation along equator
        slewdist, slewpath = default_acs.predict_slew(0, 0, 90, 0)
        ra_path, dec_path = slewpath
        # Check path starts and ends at correct positions
        assert abs(ra_path[-1] - 90) < 1e-6

    def test_predict_slew_different_positions_end_dec(self, default_acs):
        """Test predict_slew path ends at correct dec."""
        # Simple case: 90 degree separation along equator
        slewdist, slewpath = default_acs.predict_slew(0, 0, 90, 0)
        ra_path, dec_path = slewpath
        # Check path starts and ends at correct positions
        assert abs(dec_path[-1] - 0) < 1e-6

    def test_predict_slew_custom_steps_path_length_ra(self, default_acs):
        """Test predict_slew with custom steps path length for ra."""
        slewdist, slewpath = default_acs.predict_slew(0, 0, 45, 45, steps=10)
        ra_path, dec_path = slewpath
        # great_circle returns steps+2 points (includes start and end)
        assert len(ra_path) == 12  # 10 steps + start + end

    def test_predict_slew_custom_steps_path_length_dec(self, default_acs):
        """Test predict_slew with custom steps path length for dec."""
        slewdist, slewpath = default_acs.predict_slew(0, 0, 45, 45, steps=10)
        ra_path, dec_path = slewpath
        # great_circle returns steps+2 points (includes start and end)
        assert len(dec_path) == 12

    def test_predict_slew_across_meridian_distance_positive(self, default_acs):
        """Test predict_slew across meridian has positive distance."""
        slewdist, slewpath = default_acs.predict_slew(350, 0, 10, 0)
        assert slewdist > 0  # Should find the shorter path

    def test_predict_slew_across_meridian_distance_less_than_180(self, default_acs):
        """Test predict_slew across meridian distance less than 180."""
        slewdist, slewpath = default_acs.predict_slew(350, 0, 10, 0)
        assert slewdist < 180  # Should not go the long way


class TestSpacecraftBus:
    """Tests for SpacecraftBus class."""

    def test_initialization_defaults_name(self, default_bus):
        """Test SpacecraftBus initializes with default name."""
        assert default_bus.name == "Default Bus"

    def test_initialization_defaults_power_draw_type(self, default_bus):
        """Test SpacecraftBus initializes with PowerDraw instance."""
        assert isinstance(default_bus.power_draw, PowerDraw)

    def test_initialization_defaults_attitude_control_type(self, default_bus):
        """Test SpacecraftBus initializes with AttitudeControlSystem instance."""
        assert isinstance(default_bus.attitude_control, AttitudeControlSystem)

    def test_initialization_custom_name(self, custom_bus):
        """Test SpacecraftBus initializes with custom name."""
        assert custom_bus.name == "Custom Bus"

    def test_initialization_custom_power_draw_nominal_power(self, custom_bus):
        """Test SpacecraftBus initializes with custom power_draw nominal_power."""
        assert custom_bus.power_draw.nominal_power == 150

    def test_initialization_custom_attitude_control_slew_acceleration(self, custom_bus):
        """Test SpacecraftBus initializes with custom attitude_control slew_acceleration."""
        assert custom_bus.attitude_control.slew_acceleration == 1.0

    def test_power_delegates_to_power_draw_no_mode(self):
        """Test SpacecraftBus.power() delegates to PowerDraw for no mode."""
        pd = PowerDraw(nominal_power=100, power_mode={1: 120})
        bus = SpacecraftBus(power_draw=pd)
        assert bus.power() == 100

    def test_power_delegates_to_power_draw_mode_1(self):
        """Test SpacecraftBus.power() delegates to PowerDraw for mode 1."""
        pd = PowerDraw(nominal_power=100, power_mode={1: 120})
        bus = SpacecraftBus(power_draw=pd)
        assert bus.power(1) == 120

    def test_power_delegates_to_power_draw_undefined_mode(self):
        """Test SpacecraftBus.power() delegates to PowerDraw for undefined mode."""
        pd = PowerDraw(nominal_power=100, power_mode={1: 120})
        bus = SpacecraftBus(power_draw=pd)
        assert bus.power(99) == 100


class TestSpacecraftBusEclipse:
    """Test eclipse-aware power consumption for spacecraft bus."""

    def test_bus_with_heater_eclipse_sunlight(self):
        """Test bus power with heater in sunlight."""
        from conops import Heater

        bus = SpacecraftBus(
            name="Test Bus",
            power_draw=PowerDraw(nominal_power=150.0),
            heater=Heater(
                name="Bus Heater",
                power_draw=PowerDraw(nominal_power=15.0, eclipse_power=35.0),
            ),
        )

        # Sunlight: base + heater
        assert bus.power(in_eclipse=False) == 165.0

    def test_bus_with_heater_eclipse_eclipse(self):
        """Test bus power with heater in eclipse."""
        from conops import Heater

        bus = SpacecraftBus(
            name="Test Bus",
            power_draw=PowerDraw(nominal_power=150.0),
            heater=Heater(
                name="Bus Heater",
                power_draw=PowerDraw(nominal_power=15.0, eclipse_power=35.0),
            ),
        )

        # Eclipse: base + higher heater power
        assert bus.power(in_eclipse=True) == 185.0

    def test_bus_base_power_with_eclipse_sunlight(self):
        """Test bus base power draw in sunlight."""
        bus = SpacecraftBus(
            name="Detector Bus",
            power_draw=PowerDraw(
                nominal_power=200.0,
                eclipse_power=210.0,  # Slightly higher in eclipse
            ),
        )

        assert bus.power(in_eclipse=False) == 200.0

    def test_bus_base_power_with_eclipse_eclipse(self):
        """Test bus base power draw in eclipse."""
        bus = SpacecraftBus(
            name="Detector Bus",
            power_draw=PowerDraw(
                nominal_power=200.0,
                eclipse_power=210.0,  # Slightly higher in eclipse
            ),
        )

        assert bus.power(in_eclipse=True) == 210.0

    def test_bus_full_eclipse_configuration_nominal_sunlight(self):
        """Test bus with both base and heater eclipse power in nominal sunlight."""
        from conops import Heater

        bus = SpacecraftBus(
            name="Science Bus",
            power_draw=PowerDraw(
                nominal_power=180.0,
                eclipse_power=190.0,
                power_mode={2: 250.0},
                eclipse_power_mode={2: 270.0},
            ),
            heater=Heater(
                name="Bus Heater",
                power_draw=PowerDraw(
                    nominal_power=20.0,
                    eclipse_power=50.0,
                    power_mode={2: 25.0},
                    eclipse_power_mode={2: 55.0},
                ),
            ),
        )

        # Nominal mode
        assert bus.power(in_eclipse=False) == 200.0  # 180 + 20

    def test_bus_full_eclipse_configuration_nominal_eclipse(self):
        """Test bus with both base and heater eclipse power in nominal eclipse."""
        from conops import Heater

        bus = SpacecraftBus(
            name="Science Bus",
            power_draw=PowerDraw(
                nominal_power=180.0,
                eclipse_power=190.0,
                power_mode={2: 250.0},
                eclipse_power_mode={2: 270.0},
            ),
            heater=Heater(
                name="Bus Heater",
                power_draw=PowerDraw(
                    nominal_power=20.0,
                    eclipse_power=50.0,
                    power_mode={2: 25.0},
                    eclipse_power_mode={2: 55.0},
                ),
            ),
        )

        # Nominal mode
        assert bus.power(in_eclipse=True) == 240.0  # 190 + 50

    def test_bus_full_eclipse_configuration_mode_2_sunlight(self):
        """Test bus with both base and heater eclipse power in mode 2 sunlight."""
        from conops import Heater

        bus = SpacecraftBus(
            name="Science Bus",
            power_draw=PowerDraw(
                nominal_power=180.0,
                eclipse_power=190.0,
                power_mode={2: 250.0},
                eclipse_power_mode={2: 270.0},
            ),
            heater=Heater(
                name="Bus Heater",
                power_draw=PowerDraw(
                    nominal_power=20.0,
                    eclipse_power=50.0,
                    power_mode={2: 25.0},
                    eclipse_power_mode={2: 55.0},
                ),
            ),
        )

        # Mode 2
        assert bus.power(mode=2, in_eclipse=False) == 275.0  # 250 + 25

    def test_bus_full_eclipse_configuration_mode_2_eclipse(self):
        """Test bus with both base and heater eclipse power in mode 2 eclipse."""
        from conops import Heater

        bus = SpacecraftBus(
            name="Science Bus",
            power_draw=PowerDraw(
                nominal_power=180.0,
                eclipse_power=190.0,
                power_mode={2: 250.0},
                eclipse_power_mode={2: 270.0},
            ),
            heater=Heater(
                name="Bus Heater",
                power_draw=PowerDraw(
                    nominal_power=20.0,
                    eclipse_power=50.0,
                    power_mode={2: 25.0},
                    eclipse_power_mode={2: 55.0},
                ),
            ),
        )

        # Mode 2
        assert bus.power(mode=2, in_eclipse=True) == 325.0  # 270 + 55

    def test_bus_no_heater_eclipse_sunlight(self):
        """Test bus without heater in sunlight."""
        bus = SpacecraftBus(
            name="Simple Bus",
            power_draw=PowerDraw(nominal_power=175.0, eclipse_power=185.0),
        )

        assert bus.power(in_eclipse=False) == 175.0

    def test_bus_no_heater_eclipse_eclipse(self):
        """Test bus without heater in eclipse."""
        bus = SpacecraftBus(
            name="Simple Bus",
            power_draw=PowerDraw(nominal_power=175.0, eclipse_power=185.0),
        )

        assert bus.power(in_eclipse=True) == 185.0

    def test_bus_eclipse_backward_compatible(self):
        """Test that not passing in_eclipse works (defaults to False)."""
        from conops import Heater

        bus = SpacecraftBus(
            power_draw=PowerDraw(nominal_power=150.0, eclipse_power=200.0),
            heater=Heater(
                name="Heater",
                power_draw=PowerDraw(nominal_power=10.0, eclipse_power=30.0),
            ),
        )

        # Default should be sunlight (in_eclipse=False)
        assert bus.power() == 160.0  # 150 + 10 (not eclipse values)
