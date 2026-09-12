"""Tests for conops.roll module."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from conops import optimum_roll, optimum_roll_sidemount
from conops.common.vector import attitude_to_quat, quaternion_rotate_vector
from conops.config import Telescope


class TestOptimumRoll:
    """Test optimum_roll function."""

    def test_optimum_roll_without_solar_panel(self, mock_ephem, mock_sun_coord):
        """Test optimum_roll without solar panel (analytic solution)."""
        mock_ephem.sun = [mock_sun_coord]
        ra, dec, utime = 45.0, 30.0, 1700000000.0
        roll = optimum_roll(ra, dec, utime, mock_ephem, solar_panel=None)
        assert isinstance(roll, float) and 0 <= roll < 360

    def test_optimum_roll_returns_float(self, mock_ephem, mock_sun_coord):
        """Test that optimum_roll returns a float."""
        mock_ephem.sun = [mock_sun_coord]
        mock_sun_coord.cartesian.xyz.to_value = Mock(
            return_value=np.array([1000, 200, 600])
        )
        ra, dec, utime = 90.0, 0.0, 1700000000.0
        roll = optimum_roll(ra, dec, utime, mock_ephem, solar_panel=None)
        assert isinstance(roll, float)

    def test_optimum_roll_with_solar_panel(
        self, mock_ephem, mock_sun_coord, mock_solar_panel_single
    ):
        """Test optimum_roll with solar panel (weighted optimization)."""
        mock_ephem.sun = [mock_sun_coord]
        mock_sun_coord.cartesian.xyz.to_value = Mock(
            return_value=np.array([1000, 300, 700])
        )
        ra, dec, utime = 45.0, 30.0, 1700000000.0
        roll = optimum_roll(
            ra, dec, utime, mock_ephem, solar_panel=mock_solar_panel_single
        )
        assert isinstance(roll, float) and 0 <= roll < 360

    def test_constrained_sidemount_scan_uses_right_handed_roll(self, mock_ephem):
        mock_ephem.sun_pv.position = [np.array([0.0, 0.0, 1.0])]

        with patch(
            "conops.simulation.roll._roll_valid_mask",
            return_value=np.ones(360, dtype=bool),
        ):
            roll = optimum_roll(
                0.0,
                0.0,
                1700000000.0,
                mock_ephem,
                constraint=Mock(),
            )

        assert roll == 90.0

    def test_optimum_roll_with_multiple_panels(
        self, mock_ephem, mock_sun_coord, mock_solar_panel_multiple
    ):
        """Test optimum_roll with multiple solar panels."""
        mock_ephem.sun = [mock_sun_coord]
        mock_sun_coord.cartesian.xyz.to_value = Mock(
            return_value=np.array([1000, 400, 600])
        )
        ra, dec, utime = 60.0, 20.0, 1700000000.0
        roll = optimum_roll(
            ra, dec, utime, mock_ephem, solar_panel=mock_solar_panel_multiple
        )
        assert isinstance(roll, float) and 0 <= roll < 360

    def test_optimum_roll_with_canted_panels(
        self, mock_ephem, mock_sun_coord, mock_solar_panel_canted
    ):
        """Test optimum_roll with canted solar panels."""
        mock_ephem.sun = [mock_sun_coord]
        mock_sun_coord.cartesian.xyz.to_value = Mock(
            return_value=np.array([800, 300, 700])
        )
        ra, dec, utime = 30.0, 45.0, 1700000000.0
        roll = optimum_roll(
            ra, dec, utime, mock_ephem, solar_panel=mock_solar_panel_canted
        )
        assert isinstance(roll, float) and 0 <= roll < 360

    def test_optimum_roll_restricts_search_to_reachable_rolls(
        self, mock_ephem, mock_sun_coord
    ):
        valid_mask = np.zeros(360, dtype=bool)
        valid_mask[209] = True
        valid_mask[345] = True

        with patch("conops.simulation.roll._roll_valid_mask", return_value=valid_mask):
            roll = optimum_roll(
                45.0,
                30.0,
                1700000000.0,
                mock_ephem,
                constraint=Mock(),
                reference_roll=215.0,
                max_roll_delta=88.0,
            )

        assert roll == 209.0

    def test_optimum_roll_holds_reference_when_no_candidate_is_reachable(
        self, mock_ephem, mock_sun_coord
    ):
        valid_mask = np.zeros(360, dtype=bool)
        valid_mask[345] = True

        with patch("conops.simulation.roll._roll_valid_mask", return_value=valid_mask):
            roll = optimum_roll(
                45.0,
                30.0,
                1700000000.0,
                mock_ephem,
                constraint=Mock(),
                reference_roll=215.5,
                max_roll_delta=0.0,
            )

        assert roll == 215.5

    def test_mounted_roll_optimizes_panel_in_physical_body_frame(
        self, mock_ephem, mock_solar_panel_single
    ):
        mock_ephem.sun_pv.position = [np.array([0.0, 0.0, 1.0])]
        mock_solar_panel_single.panels[0].normal = (1.0, 0.0, 0.0)
        telescope = Telescope(boresight=(0.0, 1.0, 0.0))

        roll = optimum_roll(
            0.0,
            0.0,
            1700000000.0,
            mock_ephem,
            solar_panel=mock_solar_panel_single,
            telescope=telescope,
        )

        body_attitude = telescope.target_body_attitude(0.0, 0.0, roll)
        sun_body = quaternion_rotate_vector(
            attitude_to_quat(*body_attitude), (0.0, 0.0, 1.0)
        )
        assert roll == 270.0
        assert sun_body == pytest.approx((1.0, 0.0, 0.0), abs=1e-10)

    def test_mounted_roll_uses_zero_for_equal_power_without_reference(
        self, mock_ephem, mock_solar_panel_single
    ) -> None:
        telescope = Telescope(boresight=(0.0, 1.0, 0.0))
        mock_solar_panel_single.panels[0].normal = (0.0, -1.0, 0.0)

        roll = optimum_roll(
            0.0,
            0.0,
            1700000000.0,
            mock_ephem,
            solar_panel=mock_solar_panel_single,
            telescope=telescope,
        )

        assert roll == 0.0

    def test_mounted_roll_uses_reference_for_equal_power(
        self, mock_ephem, mock_solar_panel_single
    ) -> None:
        telescope = Telescope(boresight=(0.0, 1.0, 0.0))
        mock_solar_panel_single.panels[0].normal = (0.0, -1.0, 0.0)

        roll = optimum_roll(
            0.0,
            0.0,
            1700000000.0,
            mock_ephem,
            solar_panel=mock_solar_panel_single,
            telescope=telescope,
            reference_roll=10.0,
            max_roll_delta=5.0,
        )

        assert roll == 10.0


class TestOptimumRollSidemount:
    """Test optimum_roll_sidemount function."""

    def test_optimum_roll_sidemount_basic(self, mock_ephem_sidemount):
        """Test basic optimum_roll_sidemount calculation."""
        ra, dec, utime = 45.0, 30.0, 1700000000.0
        roll = optimum_roll_sidemount(ra, dec, utime, mock_ephem_sidemount)
        assert isinstance(roll, float) and 0 <= roll < 360

    def test_optimum_roll_sidemount_zero_sun(self, mock_ephem_sidemount):
        """Test optimum_roll_sidemount with sun directly ahead."""
        mock_ephem_sidemount.sunvec = [np.array([1000, 0, 0])]
        ra, dec, utime = 0.0, 0.0, 1700000000.0
        roll = optimum_roll_sidemount(ra, dec, utime, mock_ephem_sidemount)
        assert isinstance(roll, float) and 0 <= roll < 360

    def test_optimum_roll_sidemount_different_positions(self, mock_ephem_sidemount):
        """Test optimum_roll_sidemount with different RA/Dec positions."""
        mock_ephem_sidemount.sunvec = [np.array([800, 400, 600])]
        ra, dec, utime = 90.0, -30.0, 1700000000.0
        roll = optimum_roll_sidemount(ra, dec, utime, mock_ephem_sidemount)
        assert isinstance(roll, float) and 0 <= roll < 360

    def test_optimum_roll_sidemount_returns_float(self, mock_ephem_sidemount):
        """Test that optimum_roll_sidemount returns a float."""
        mock_ephem_sidemount.sunvec = [np.array([900, 200, 700])]
        ra, dec, utime = 180.0, 45.0, 1700000000.0
        roll = optimum_roll_sidemount(ra, dec, utime, mock_ephem_sidemount)
        assert isinstance(roll, float) and 0 <= roll < 360

    def test_optimum_roll_sidemount_wraps_to_360(self, mock_ephem_sidemount):
        """Test that roll angle wraps correctly to [0, 360)."""
        mock_ephem_sidemount.sunvec = [np.array([1000, 0, 0])]
        ra, dec, utime = 0.0, 0.0, 1700000000.0
        roll = optimum_roll_sidemount(ra, dec, utime, mock_ephem_sidemount)
        assert 0 <= roll < 360
