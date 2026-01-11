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
        acs_config.predict_slew = Mock(
            return_value=(distance, (np.array([0.0]), np.array([0.0])))
        )
        acs_config.slew_time = Mock(return_value=0.0)
        slew.startra = 0.0
        slew.startdec = 0.0
        slew.endra = 10.0
        slew.enddec = 10.0
        slew.slewstart = 1700000000.0
        with pytest.raises(ValueError, match="Invalid slew distance"):
            slew.calc_slewtime()


class TestPredictSlew:
    """Test predict_slew method."""

    def test_predict_slew_calls_acs_predict_slew(self, slew_predict_setup):
        slew, ra_path, dec_path = slew_predict_setup
        slew.predict_slew()
        slew.acs_config.predict_slew.assert_called_once_with(
            45.0, 30.0, 90.0, 60.0, steps=100
        )

    def test_predict_slew_sets_slewdist(self, slew_predict_setup):
        slew, ra_path, dec_path = slew_predict_setup
        slew.predict_slew()
        assert slew.slewdist == 14.142

    def test_predict_slew_sets_path_ra_length(self, slew_predict_setup):
        slew, ra_path, dec_path = slew_predict_setup
        slew.predict_slew()
        assert len(slew.slewpath[0]) == 100

    def test_predict_slew_sets_path_dec_length(self, slew_predict_setup):
        slew, ra_path, dec_path = slew_predict_setup
        slew.predict_slew()
        assert len(slew.slewpath[1]) == 100

    def test_predict_slew_sets_path_ra_values(self, slew_predict_setup):
        slew, ra_path, dec_path = slew_predict_setup
        slew.predict_slew()
        assert np.allclose(slew.slewpath[0], ra_path)

    def test_predict_slew_sets_path_dec_values(self, slew_predict_setup):
        slew, ra_path, dec_path = slew_predict_setup
        slew.predict_slew()
        assert np.allclose(slew.slewpath[1], dec_path)

    def test_predict_slew_zero_distance_sets_default_axis(self, slew, acs_config):
        acs_config.predict_slew = Mock(return_value=(0.0, ([0.0] * 100, [0.0] * 100)))
        slew.startra = 10.0
        slew.startdec = -5.0
        slew.endra = 10.0
        slew.enddec = -5.0

        slew.predict_slew()

        assert slew.rotation_axis == (0.0, 0.0, 1.0)


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
