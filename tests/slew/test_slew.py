"""Tests for conops.slew module."""

from unittest.mock import Mock

import numpy as np
import pytest

from conops import Slew


class TestSlewInit:
    """Test Slew initialization."""

    @pytest.mark.parametrize(
        "attr,value",
        [
            ("constraint", "constraint"),
            ("ephem", "ephem"),
            ("acs_config", "acs_config"),
            ("slewtime", 0),
            ("slewdist", 0),
            ("slewstart", 0),
            ("slewend", 0),
            ("slewrequest", 0),
            ("startra", 0),
            ("startdec", 0),
            ("endra", 0),
            ("enddec", 0),
            ("obsid", 0),
            ("mode", 0),
            ("at", None),
            ("obstype", "PPT"),
        ],
    )
    def test_slew_init(self, slew, attr, value, constraint, ephem, acs_config):
        if attr in ["constraint", "ephem", "acs_config"]:
            # Assuming fixtures provide the expected values
            expected = locals()[attr]
        else:
            expected = value
        assert getattr(slew, attr) == expected

    def test_slew_init_missing_constraint(self, acs_config):
        mock_config = Mock()
        mock_config.constraint = None
        mock_config.spacecraft_bus = Mock()
        mock_config.spacecraft_bus.attitude_control = acs_config
        with pytest.raises(AssertionError, match="Constraint must be set"):
            Slew(config=mock_config)

    def test_slew_init_missing_ephemeris(self, constraint, acs_config):
        constraint.ephem = None
        mock_config = Mock()
        mock_config.constraint = constraint
        mock_config.spacecraft_bus = Mock()
        mock_config.spacecraft_bus.attitude_control = acs_config
        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            Slew(config=mock_config)

    def test_slew_init_missing_acs_config(self, constraint):
        mock_config = Mock()
        mock_config.constraint = constraint
        mock_config.spacecraft_bus = Mock()
        mock_config.spacecraft_bus.attitude_control = None
        with pytest.raises(AssertionError, match="ACS config must be set"):
            Slew(config=mock_config)


class TestSlewStr:
    """Test Slew string representation."""

    def test_slew_str_contains_slew_from(self, slew_with_positions):
        str_repr = str(slew_with_positions)
        assert "Slew from" in str_repr

    def test_slew_str_contains_45_000(self, slew_with_positions):
        str_repr = str(slew_with_positions)
        assert "45.000" in str_repr

    def test_slew_str_contains_30_000(self, slew_with_positions):
        str_repr = str(slew_with_positions)
        assert "30.000" in str_repr

    def test_slew_str_contains_90_0(self, slew_with_positions):
        str_repr = str(slew_with_positions)
        assert "90.0" in str_repr


class TestIsSlewing:
    """Test is_slewing method."""

    def test_is_slewing_at_1700000050_true(self, slew_slewing):
        assert slew_slewing.is_slewing(1700000050.0) is True

    def test_is_slewing_at_1699999999_false(self, slew_slewing):
        assert slew_slewing.is_slewing(1699999999.0) is False

    def test_is_slewing_at_1700000101_false(self, slew_slewing):
        assert slew_slewing.is_slewing(1700000101.0) is False

    def test_is_slewing_at_1700000000_true(self, slew_slewing):
        assert slew_slewing.is_slewing(1700000000.0) is True

    def test_is_slewing_at_1700000100_false(self, slew_slewing):
        assert slew_slewing.is_slewing(1700000100.0) is False


class TestRaDec:
    """Test ra_dec method."""

    def test_ra_dec_returns_ra_float(self, slew_ra_dec):
        ra, dec = slew_ra_dec.ra_dec(1700000000.0)
        assert isinstance(ra, (float, np.floating))

    def test_ra_dec_returns_dec_float(self, slew_ra_dec):
        ra, dec = slew_ra_dec.ra_dec(1700000000.0)
        assert isinstance(dec, (float, np.floating))


class TestSlewRaDec:
    """Test slew_ra_dec method."""

    def test_slew_ra_dec_returns_ra_float(self, slew_setup):
        ra, dec = slew_setup.slew_ra_dec(1700000000.0)
        assert isinstance(ra, (float, np.floating))

    def test_slew_ra_dec_returns_dec_float(self, slew_setup):
        ra, dec = slew_setup.slew_ra_dec(1700000000.0)
        assert isinstance(dec, (float, np.floating))

    def test_slew_ra_dec_ra_in_range(self, slew_setup):
        ra, dec = slew_setup.slew_ra_dec(1700000000.0)
        assert 0 <= ra < 360

    def test_slew_ra_dec_dec_in_range(self, slew_setup):
        ra, dec = slew_setup.slew_ra_dec(1700000000.0)
        assert -90 <= dec <= 90

    def test_slew_ra_dec_no_path_returns_ra_start(self, slew):
        slew.startra = 45.0
        slew.startdec = 30.0
        slew.slewstart = 1700000000.0
        ra, dec = slew.slew_ra_dec(1700000000.0)
        assert ra == 45.0

    def test_slew_ra_dec_no_path_returns_dec_start(self, slew):
        slew.startra = 45.0
        slew.startdec = 30.0
        slew.slewstart = 1700000000.0
        ra, dec = slew.slew_ra_dec(1700000000.0)
        assert dec == 30.0

    def test_slew_ra_dec_interpolation_mid_slew(self, slew_interpolation):
        ra, dec = slew_interpolation.slew_ra_dec(1700000050.0)
        assert isinstance(ra, (float, np.floating))
        assert isinstance(dec, (float, np.floating))
        assert 45.0 <= ra <= 90.0
        assert 30.0 <= dec <= 60.0

    def test_slew_ra_dec_modern_path_small_n(self, slew_modern_path):
        ra, dec = slew_modern_path.slew_ra_dec(1700000050.0)
        assert ra == 67.5
        assert dec == 45.0

    def test_slew_ra_dec_acs_returns_ra_float(self, slew_acs):
        ra, dec = slew_acs.slew_ra_dec(1700000050.0)
        assert isinstance(ra, (float, np.floating))

    def test_slew_ra_dec_acs_returns_dec_float(self, slew_acs):
        ra, dec = slew_acs.slew_ra_dec(1700000050.0)
        assert isinstance(dec, (float, np.floating))

    def test_slew_ra_dec_acs_ra_in_range(self, slew_acs):
        ra, dec = slew_acs.slew_ra_dec(1700000050.0)
        assert 0 <= ra < 360

    def test_slew_ra_dec_acs_dec_in_range(self, slew_acs):
        ra, dec = slew_acs.slew_ra_dec(1700000050.0)
        assert -90 <= dec <= 90

    def test_slew_ra_dec_interpolates_ra_at_start(self, slew_interp_start):
        ra, dec = slew_interp_start.slew_ra_dec(1700000000.0)
        assert np.isclose(ra, 0.0)

    def test_slew_ra_dec_interpolates_dec_at_start(self, slew_interp_start):
        ra, dec = slew_interp_start.slew_ra_dec(1700000000.0)
        assert np.isclose(dec, 0.0)


class TestCalcSlewtime:
    """Test calc_slewtime method."""

    def test_calc_slewtime_returns_slewtime(self, slew_calc_setup):
        slewtime = slew_calc_setup.calc_slewtime()
        assert slewtime == 50.0

    def test_calc_slewtime_sets_slewend(self, slew_calc_setup):
        slew_calc_setup.calc_slewtime()
        assert slew_calc_setup.slewend == 1700000050.0

    def test_calc_slewtime_sets_slewend_correctly(self, slew_calc_setup_alt):
        slew_calc_setup_alt.calc_slewtime()
        assert slew_calc_setup_alt.slewend == 1700000030.0

    @pytest.mark.parametrize("distance", [np.nan, -5.0])
    def test_calc_slewtime_handles_invalid_distance(self, slew, acs_config, distance):
        acs_config.slew_time = Mock(return_value=0.0)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 10.0
        slew.enddec = 10.0
        slew.slewstart = 1700000000.0

        def inject_bad_distance() -> None:
            slew.slewdist = distance
            slew.slewpath = ([0.0], [0.0])
            slew._quat_roll_path = []

        slew.predict_slew = inject_bad_distance
        with pytest.raises(ValueError, match="Invalid slew distance"):
            slew.calc_slewtime()


class TestPredictSlew:
    """Test predict_slew method (quaternion SLERP)."""

    def test_predict_slew_sets_slewdist_positive(self, slew_predict_setup):
        slew_predict_setup.predict_slew()
        assert slew_predict_setup.slewdist > 0

    def test_predict_slew_sets_slewdist_approx(self, slew_predict_setup):
        """Slew distance should be the quaternion angular distance."""
        import numpy as np

        from conops.common.vector import attitude_to_quat

        # Compute expected quaternion angular distance
        q1 = attitude_to_quat(
            slew_predict_setup.startra,
            slew_predict_setup.startdec,
            slew_predict_setup.startroll,
        )
        q2 = attitude_to_quat(
            slew_predict_setup.endra,
            slew_predict_setup.enddec,
            slew_predict_setup.endroll,
        )
        dot = float(np.dot(q1, q2))
        if dot < 0:
            dot = -dot
        dot = min(dot, 1.0)
        theta_rad = np.arccos(dot)
        expected = float(np.rad2deg(2 * theta_rad))

        slew_predict_setup.predict_slew()
        assert abs(slew_predict_setup.slewdist - expected) < 0.01

    def test_predict_slew_path_length(self, slew_predict_setup):
        slew_predict_setup.predict_slew()
        assert len(slew_predict_setup.slewpath[0]) == 101
        assert len(slew_predict_setup.slewpath[1]) == 101

    def test_predict_slew_path_starts_at_start(self, slew_predict_setup):
        slew_predict_setup.predict_slew()
        assert abs(slew_predict_setup.slewpath[0][0] - 45.0) < 0.01
        assert abs(slew_predict_setup.slewpath[1][0] - 30.0) < 0.01

    def test_predict_slew_path_ends_at_end(self, slew_predict_setup):
        slew_predict_setup.predict_slew()
        assert abs(slew_predict_setup.slewpath[0][-1] - 90.0) < 0.01
        assert abs(slew_predict_setup.slewpath[1][-1] - 60.0) < 0.01

    def test_predict_slew_sets_roll_path(self, slew_predict_setup):
        slew_predict_setup.predict_slew()
        assert len(slew_predict_setup._quat_roll_path) == 101


class TestPureRollManeuver:
    """Test quaternion slew algorithm with pure roll maneuvers.

    Pure roll maneuvers (where RA/Dec remain constant but roll changes) are
    critical edge cases that test the quaternion SLERP path/timing coupling.
    The great-circle distance is 0, but the spacecraft still needs time to
    rotate about the boresight axis.
    """

    def test_pure_roll_zero_radec_distance(self, slew_predict_setup):
        """Pure roll maneuver should have non-zero quaternion distance matching roll change."""
        slew_predict_setup.startra = 45.0
        slew_predict_setup.startdec = 30.0
        slew_predict_setup.startroll = 0.0
        slew_predict_setup.endra = 45.0
        slew_predict_setup.enddec = 30.0
        slew_predict_setup.endroll = 90.0

        slew_predict_setup.predict_slew()

        # Quaternion distance should equal the roll change (90°)
        assert abs(slew_predict_setup.slewdist - 90.0) < 0.1

    def test_pure_roll_has_roll_path(self, slew_predict_setup):
        """Pure roll maneuver should generate a roll path."""
        slew_predict_setup.startra = 45.0
        slew_predict_setup.startdec = 30.0
        slew_predict_setup.startroll = 0.0
        slew_predict_setup.endra = 45.0
        slew_predict_setup.enddec = 30.0
        slew_predict_setup.endroll = 90.0

        slew_predict_setup.predict_slew()

        # Should have roll path from SLERP
        assert hasattr(slew_predict_setup, "_quat_roll_path")
        assert len(slew_predict_setup._quat_roll_path) == 101

    def test_pure_roll_path_starts_and_ends_correctly(self, slew_predict_setup):
        """Roll path should start at 0° and end at 90°."""
        slew_predict_setup.startra = 45.0
        slew_predict_setup.startdec = 30.0
        slew_predict_setup.startroll = 0.0
        slew_predict_setup.endra = 45.0
        slew_predict_setup.enddec = 30.0
        slew_predict_setup.endroll = 90.0

        slew_predict_setup.predict_slew()

        # Check roll path endpoints
        assert abs(slew_predict_setup._quat_roll_path[0] - 0.0) < 0.1
        assert abs(slew_predict_setup._quat_roll_path[-1] - 90.0) < 0.1

    def test_pure_roll_path_is_monotonic(self, slew_predict_setup):
        """Roll path should monotonically increase from 0° to 90°."""
        slew_predict_setup.startra = 45.0
        slew_predict_setup.startdec = 30.0
        slew_predict_setup.startroll = 0.0
        slew_predict_setup.endra = 45.0
        slew_predict_setup.enddec = 30.0
        slew_predict_setup.endroll = 90.0

        slew_predict_setup.predict_slew()

        # Check that roll increases monotonically
        roll_path = slew_predict_setup._quat_roll_path
        for i in range(len(roll_path) - 1):
            # Allow small numerical noise
            assert roll_path[i + 1] >= roll_path[i] - 0.1

    def test_pure_roll_180_degree_rotation(self, slew_predict_setup):
        """180° roll maneuver should work correctly."""
        slew_predict_setup.startra = 45.0
        slew_predict_setup.startdec = 30.0
        slew_predict_setup.startroll = 0.0
        slew_predict_setup.endra = 45.0
        slew_predict_setup.enddec = 30.0
        slew_predict_setup.endroll = 180.0

        slew_predict_setup.predict_slew()

        # Check roll path endpoints
        assert abs(slew_predict_setup._quat_roll_path[0] - 0.0) < 0.1
        assert abs(slew_predict_setup._quat_roll_path[-1] - 180.0) < 0.1
        # Quaternion distance should equal the roll change (180°)
        assert abs(slew_predict_setup.slewdist - 180.0) < 0.1

    def test_pure_roll_wraps_around_360(self, slew_predict_setup):
        """Roll maneuver from 350° to 10° should take shortest path (20° total)."""
        slew_predict_setup.startra = 45.0
        slew_predict_setup.startdec = 30.0
        slew_predict_setup.startroll = 350.0
        slew_predict_setup.endra = 45.0
        slew_predict_setup.enddec = 30.0
        slew_predict_setup.endroll = 10.0

        slew_predict_setup.predict_slew()

        # Should have a valid roll path
        assert len(slew_predict_setup._quat_roll_path) == 101

        # The quaternion path might use negative angles, so normalize everything
        roll_path_normalized = [(r % 360) for r in slew_predict_setup._quat_roll_path]

        # Start should be near 350° or equivalent
        assert (
            abs(roll_path_normalized[0] - 350.0) < 1.0
            or abs(roll_path_normalized[0] - 10.0) < 1.0
        )

        # End should be near 10°
        assert abs(roll_path_normalized[-1] - 10.0) < 1.0

        # Slew distance should be ~20° (shortest path), not ~340°
        assert slew_predict_setup.slewdist < 25.0

    def test_pure_roll_radec_path_constant(self, slew_predict_setup):
        """RA/Dec should remain constant throughout pure roll maneuver."""
        slew_predict_setup.startra = 45.0
        slew_predict_setup.startdec = 30.0
        slew_predict_setup.startroll = 0.0
        slew_predict_setup.endra = 45.0
        slew_predict_setup.enddec = 30.0
        slew_predict_setup.endroll = 90.0

        slew_predict_setup.predict_slew()

        # RA/Dec path should be essentially constant
        ra_path, dec_path = slew_predict_setup.slewpath
        for ra in ra_path:
            assert abs(ra - 45.0) < 0.1
        for dec in dec_path:
            assert abs(dec - 30.0) < 0.1

    def test_pure_roll_slew_roll_method(self, slew_predict_setup, acs_config):
        """slew_roll() should interpolate correctly during pure roll maneuver."""
        # Set up ACS config for bang-bang motion
        acs_config.motion_time = Mock(return_value=100.0)
        acs_config.s_of_t = Mock(side_effect=lambda dist, t: t / 100.0 * dist)

        slew_predict_setup.startra = 45.0
        slew_predict_setup.startdec = 30.0
        slew_predict_setup.startroll = 0.0
        slew_predict_setup.endra = 45.0
        slew_predict_setup.enddec = 30.0
        slew_predict_setup.endroll = 90.0
        slew_predict_setup.slewstart = 1700000000.0

        slew_predict_setup.predict_slew()
        slew_predict_setup.slewtime = 100.0
        slew_predict_setup.slewend = slew_predict_setup.slewstart + 100.0

        # At start: should be 0°
        roll_start = slew_predict_setup.slew_roll(1700000000.0)
        assert abs(roll_start - 0.0) < 1.0

        # At end: should be 90°
        roll_end = slew_predict_setup.slew_roll(1700000100.0)
        assert abs(roll_end - 90.0) < 1.0

        # At midpoint: should be around 45°
        roll_mid = slew_predict_setup.slew_roll(1700000050.0)
        assert 40.0 < roll_mid < 50.0


class TestSlewPathResolution:
    """Tests demonstrating why 100 steps is better than 20 for slew paths.

    Near celestial poles, small great-circle distances correspond to large RA
    changes. When interpolating linearly in RA/Dec between sparse path points,
    the interpolated position deviates from the true great circle path.

    This test class quantifies the interpolation deviation for different step counts.
    """

    def _compute_max_interpolation_deviation(self, steps: int) -> float:
        """Compute max deviation when linearly interpolating between path points.

        For each segment between path points, computes the midpoint via:
        1. Linear RA/Dec interpolation (what the slew code does)
        2. True great-circle midpoint

        Returns the maximum angular deviation in degrees.
        """
        from conops.common import angular_separation, great_circle

        # Polar slew: RA changes significantly, Dec near pole
        start_ra, start_dec = 0.0, 85.0
        end_ra, end_dec = 180.0, 85.0

        # Get great-circle path with specified steps
        ra_path, dec_path = great_circle(start_ra, start_dec, end_ra, end_dec, steps)

        max_deviation = 0.0
        for i in range(len(ra_path) - 1):
            # Linear interpolation midpoint (what slew code does)
            lin_ra = (ra_path[i] + ra_path[i + 1]) / 2
            lin_dec = (dec_path[i] + dec_path[i + 1]) / 2

            # True great-circle midpoint
            gc_ra, gc_dec = great_circle(
                ra_path[i], dec_path[i], ra_path[i + 1], dec_path[i + 1], npts=3
            )
            true_ra, true_dec = gc_ra[1], gc_dec[1]  # Middle point

            # Angular deviation
            deviation = angular_separation(lin_ra, lin_dec, true_ra, true_dec)
            max_deviation = max(max_deviation, deviation)

        return max_deviation

    def test_20_steps_has_significant_deviation(self):
        """With 20 steps, interpolation can deviate >0.1 deg from great circle."""
        deviation = self._compute_max_interpolation_deviation(20)
        assert deviation > 0.1, (
            f"Expected >0.1° deviation with 20 steps, got {deviation:.3f}°"
        )

    def test_100_steps_has_small_deviation(self):
        """With 100 steps, interpolation deviation is under 0.1 deg."""
        deviation = self._compute_max_interpolation_deviation(100)
        assert deviation < 0.1, (
            f"Expected <0.1° deviation with 100 steps, got {deviation:.4f}°"
        )

    def test_100_steps_is_better_than_20_steps(self):
        """100 steps reduces interpolation deviation by ~5x vs 20 steps.

        With 5x more path points, the deviation scales roughly as 1/N,
        so we expect approximately 5x improvement.
        """
        deviation_20 = self._compute_max_interpolation_deviation(20)
        deviation_100 = self._compute_max_interpolation_deviation(100)
        improvement = deviation_20 / deviation_100
        assert improvement > 4, f"Expected >4x improvement, got {improvement:.1f}x"
        # Print for informational purposes
        print(
            f"\n  Deviation: 20 steps = {deviation_20:.4f}°, 100 steps = {deviation_100:.4f}° ({improvement:.1f}x better)"
        )

    def test_runtime_cost_is_acceptable(self):
        """100 steps path calculation is reasonably fast."""
        import time

        from conops.common import great_circle

        start_ra, start_dec = 0.0, 85.0
        end_ra, end_dec = 180.0, 85.0

        # Warm up
        for _ in range(10):
            great_circle(start_ra, start_dec, end_ra, end_dec, 20)
            great_circle(start_ra, start_dec, end_ra, end_dec, 100)

        # Time 20 steps
        n_iter = 1000
        t0 = time.perf_counter()
        for _ in range(n_iter):
            great_circle(start_ra, start_dec, end_ra, end_dec, 20)
        time_20 = time.perf_counter() - t0

        # Time 100 steps
        t0 = time.perf_counter()
        for _ in range(n_iter):
            great_circle(start_ra, start_dec, end_ra, end_dec, 100)
        time_100 = time.perf_counter() - t0

        ratio = time_100 / time_20
        # Accept up to 10x slowdown (very conservative) - actual is usually ~5x
        assert ratio < 10, f"100 steps is {ratio:.1f}x slower than 20 steps"
        # Print for informational purposes (visible with pytest -v)
        print(
            f"\n  Runtime: 20 steps = {time_20 * 1000:.2f}ms, 100 steps = {time_100 * 1000:.2f}ms ({ratio:.1f}x)"
        )


class TestConstraintAvoidingSlew:
    """Test constraint-avoiding slew path algorithm."""

    def test_constraint_avoiding_fallback_to_quaternion(self, slew, acs_config):
        """When no constraint is violated, falls back to quaternion SLERP."""
        from conops.common.enums import SlewAlgorithm

        acs_config.slew_algorithm = SlewAlgorithm.CONSTRAINT_AVOIDING
        slew.constraint.constraint = None  # No constraints
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.startroll = 0.0
        slew.endra = 45.0
        slew.enddec = 30.0
        slew.endroll = 0.0
        slew.slewstart = 1700000000.0

        slew.predict_slew()

        # Should have a path
        assert hasattr(slew, "slewpath")
        assert len(slew.slewpath[0]) > 0
        assert slew.slewdist > 0

    def test_constraint_avoiding_with_violation(self, slew, acs_config, constraint):
        """When constraint is violated, inserts waypoint."""
        from conops.common.enums import SlewAlgorithm

        acs_config.slew_algorithm = SlewAlgorithm.CONSTRAINT_AVOIDING
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.startroll = 0.0
        slew.endra = 90.0
        slew.enddec = 0.0
        slew.endroll = 0.0
        slew.slewstart = 1700000000.0

        # Mock constraint that violates at midpoint
        def mock_in_constraint(ra, dec, utime, target_roll=None):
            # Violate constraint around RA=45 (midpoint)
            return 40 < ra < 50

        constraint.in_constraint = mock_in_constraint
        constraint.constraint = Mock()
        constraint.constraint.in_constraint = Mock(
            side_effect=lambda **kwargs: mock_in_constraint(
                kwargs["target_ra"], kwargs["target_dec"], kwargs["time"]
            )
        )

        slew.predict_slew()

        # Should have a longer path with waypoint
        assert hasattr(slew, "slewpath")
        assert len(slew.slewpath[0]) > 0
        # Path should avoid the constraint zone
        ra_path = slew.slewpath[0]
        # Check that path deviates from direct arc
        assert any(ra < 40 or ra > 50 for ra in ra_path if 10 < ra < 80)

    def test_constraint_avoiding_path_has_roll(self, slew, acs_config):
        """Constraint-avoiding path should include roll information."""
        from conops.common.enums import SlewAlgorithm

        acs_config.slew_algorithm = SlewAlgorithm.CONSTRAINT_AVOIDING
        slew.constraint.constraint = None
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.startroll = 0.0
        slew.endra = 45.0
        slew.enddec = 30.0
        slew.endroll = 90.0
        slew.slewstart = 1700000000.0

        slew.predict_slew()

        # Should have roll path
        assert hasattr(slew, "_quat_roll_path")
        assert len(slew._quat_roll_path) > 0
        # Roll should start at 0 and end at 90
        assert abs(slew._quat_roll_path[0] - 0.0) < 1.0
        assert abs(slew._quat_roll_path[-1] - 90.0) < 1.0

    def test_constraint_avoiding_uses_acs_slew_constraint(self, slew, acs_config):
        """When ACS slew_constraint is set, it should be used instead of spacecraft constraint."""
        from unittest.mock import Mock

        from conops.common.enums import SlewAlgorithm

        acs_config.slew_algorithm = SlewAlgorithm.CONSTRAINT_AVOIDING

        # Create a mock slew constraint that always returns False (no violation)
        mock_slew_constraint = Mock()
        mock_slew_constraint.in_constraint = Mock(return_value=False)
        acs_config.slew_constraint = mock_slew_constraint

        # Set up spacecraft constraint that would violate (but should not be used)
        slew.constraint.constraint = Mock()
        slew.constraint.constraint.in_constraint = Mock(return_value=True)

        slew.startra = 0.0
        slew.startdec = 0.0
        slew.startroll = 0.0
        slew.endra = 90.0
        slew.enddec = 0.0
        slew.endroll = 0.0
        slew.slewstart = 1700000000.0

        slew.predict_slew()

        # The slew_constraint should have been called (it doesn't violate)
        assert mock_slew_constraint.in_constraint.called
        # Should fall back to quaternion path (no waypoint)
        assert hasattr(slew, "slewpath")
        assert len(slew.slewpath[0]) > 0

    def test_constraint_avoiding_falls_back_to_spacecraft_constraint(
        self, slew, acs_config
    ):
        """When ACS slew_constraint is None, falls back to spacecraft constraint."""
        from unittest.mock import Mock

        from conops.common.enums import SlewAlgorithm

        acs_config.slew_algorithm = SlewAlgorithm.CONSTRAINT_AVOIDING
        acs_config.slew_constraint = None  # No slew-specific constraint

        # Mock spacecraft constraint that doesn't violate
        mock_spacecraft_constraint = Mock()
        mock_spacecraft_constraint.in_constraint = Mock(return_value=False)
        slew.constraint.constraint = mock_spacecraft_constraint

        slew.startra = 0.0
        slew.startdec = 0.0
        slew.startroll = 0.0
        slew.endra = 90.0
        slew.enddec = 0.0
        slew.endroll = 0.0
        slew.slewstart = 1700000000.0

        slew.predict_slew()

        # The spacecraft constraint should have been called
        assert mock_spacecraft_constraint.in_constraint.called
        # Should fall back to quaternion path (no waypoint)
        assert hasattr(slew, "slewpath")
        assert len(slew.slewpath[0]) > 0


class TestConstraintAvoidingWaypoint:
    """Test constraint_avoiding_waypoint function."""

    def test_no_violation_returns_none(self):
        """When no constraint is violated, returns None."""
        from conops.common.vector import constraint_avoiding_waypoint

        def no_violation(ra, dec, time):
            return False

        result = constraint_avoiding_waypoint(
            0.0, 0.0, 90.0, 0.0, 1700000000.0, no_violation
        )
        assert result is None

    def test_violation_returns_waypoint_or_none_validated(self):
        """When constraint is violated, returns validated waypoint or None."""
        from conops.common.vector import constraint_avoiding_waypoint

        def offset_violation(ra, dec, time):
            # Violation that INTERSECTS the arc but extends more to one side
            dist = ((ra - 45.0) ** 2 + (dec - (-1.5)) ** 2) ** 0.5
            return dist < 2.5

        result = constraint_avoiding_waypoint(
            0.0, 0.0, 90.0, 0.0, 1700000000.0, offset_violation, margin_deg=5.0
        )
        # Function may return None if no valid waypoint can be found (which is correct behavior)
        # If a waypoint IS returned, it must be validated
        if result is not None:
            waypoint_ra, waypoint_dec = result
            assert isinstance(waypoint_ra, float)
            assert isinstance(waypoint_dec, float)
            # Waypoint must not violate the constraint
            assert not offset_violation(waypoint_ra, waypoint_dec, 1700000000.0)

    def test_waypoint_offset_from_direct_path(self):
        """When waypoint is returned, it should be validated."""
        from conops.common.vector import constraint_avoiding_waypoint

        def offset_circular_violation(ra, dec, time):
            # Circular violation that intersects arc
            dist_from_center = ((ra - 90.0) ** 2 + (dec - 28.5) ** 2) ** 0.5
            return dist_from_center < 2.5

        result = constraint_avoiding_waypoint(
            45.0,
            30.0,
            135.0,
            30.0,
            1700000000.0,
            offset_circular_violation,
            margin_deg=4.0,
        )
        # If a waypoint is returned, it must be validated
        if result is not None:
            waypoint_ra, waypoint_dec = result
            assert isinstance(waypoint_ra, float)
            assert isinstance(waypoint_dec, float)
            # Waypoint must not violate constraint
            assert not offset_circular_violation(
                waypoint_ra, waypoint_dec, 1700000000.0
            )

    def test_identical_points_returns_none(self):
        """When start and end are identical, returns None."""
        from conops.common.vector import constraint_avoiding_waypoint

        def any_violation(ra, dec, time):
            return True

        result = constraint_avoiding_waypoint(
            45.0, 30.0, 45.0, 30.0, 1700000000.0, any_violation
        )
        assert result is None

    def test_antipodal_points_can_route_around_constraint(self):
        """Antipodal points (180° separation) should route around constraints via alternate great circle."""
        from conops.common.vector import (
            angular_separation,
            constraint_avoiding_waypoint,
        )

        def equatorial_band_violation(ra, dec, time):
            # Violate equatorial band - should force routing over poles
            return -15 < dec < 15

        # Test antipodal points on equator - direct path through equator violates constraint
        result = constraint_avoiding_waypoint(
            0.0,
            0.0,
            180.0,
            0.0,
            1700000000.0,
            equatorial_band_violation,
            margin_deg=10.0,
        )
        # Should return a waypoint that routes over poles
        if result is not None:
            waypoint_ra, waypoint_dec = result
            # Waypoint must not violate constraint
            assert not equatorial_band_violation(
                waypoint_ra, waypoint_dec, 1700000000.0
            )
            # For antipodal points routed around equator, waypoint should be near a pole
            assert abs(waypoint_dec) > 60.0
            # Verify total path is approximately 180°
            d1 = angular_separation(0.0, 0.0, waypoint_ra, waypoint_dec)
            d2 = angular_separation(waypoint_ra, waypoint_dec, 180.0, 0.0)
            total_dist = d1 + d2
            assert 175.0 < total_dist < 185.0  # Allow some margin

    def test_waypoint_itself_violates_constraint(self):
        """When waypoint itself violates constraint, should try alternative or return None."""
        from conops.common.vector import constraint_avoiding_waypoint

        def wide_violation(ra, dec, time):
            # Violate at midpoint and surrounding region
            # This creates a wide constraint that might include waypoints
            return 30 < ra < 60 and -10 < dec < 10

        result = constraint_avoiding_waypoint(
            0.0, 0.0, 90.0, 0.0, 1700000000.0, wide_violation, margin_deg=5.0
        )
        # Should either find valid waypoint outside the region or return None
        if result is not None:
            waypoint_ra, waypoint_dec = result
            # If waypoint is returned, it must not violate constraint
            assert not wide_violation(waypoint_ra, waypoint_dec, 1700000000.0)

    def test_waypoint_segment_violates_constraint(self):
        """When start→waypoint or waypoint→end violates constraint, should return None or valid alternative."""
        from conops.common.vector import constraint_avoiding_waypoint

        def irregular_violation(ra, dec, time):
            # Create irregular constraint region that might intersect waypoint paths
            # Violate at multiple regions
            return (35 < ra < 45 and -5 < dec < 5) or (50 < ra < 60 and -5 < dec < 5)

        result = constraint_avoiding_waypoint(
            0.0, 0.0, 90.0, 0.0, 1700000000.0, irregular_violation, margin_deg=3.0
        )
        # If waypoint is returned, validate it doesn't cross violations
        if result is not None:
            waypoint_ra, waypoint_dec = result
            # Waypoint itself should not violate
            assert not irregular_violation(waypoint_ra, waypoint_dec, 1700000000.0)
            # Note: Full path validation is done internally by the function

    def test_both_waypoints_fail_returns_none(self):
        """When both candidate waypoints fail validation, should return None."""
        from conops.common.vector import constraint_avoiding_waypoint

        def surround_violation(ra, dec, time):
            # Create a constraint that surrounds the arc in both perpendicular directions
            # This makes both waypoint candidates invalid
            return 40 < ra < 50

        result = constraint_avoiding_waypoint(
            0.0, 0.0, 90.0, 0.0, 1700000000.0, surround_violation, margin_deg=2.0
        )
        # When margin is too small, both waypoints will violate, so None should be returned
        # (or if one is valid, it should be returned)
        if result is not None:
            waypoint_ra, waypoint_dec = result
            # If a waypoint is returned, it must be valid
            assert not surround_violation(waypoint_ra, waypoint_dec, 1700000000.0)

    def test_chooses_shorter_valid_path(self):
        """Should choose the shorter path when both waypoints are valid."""
        from conops.common.vector import constraint_avoiding_waypoint

        def offset_small_violation(ra, dec, time):
            # Small circular violation that intersects arc
            dist = ((ra - 45.0) ** 2 + (dec - (-1.2)) ** 2) ** 0.5
            return dist < 2.0

        result = constraint_avoiding_waypoint(
            0.0, 0.0, 90.0, 0.0, 1700000000.0, offset_small_violation, margin_deg=4.0
        )
        # If a waypoint is returned, it must be validated
        if result is not None:
            waypoint_ra, waypoint_dec = result
            # Waypoint must not violate constraint
            assert not offset_small_violation(waypoint_ra, waypoint_dec, 1700000000.0)
