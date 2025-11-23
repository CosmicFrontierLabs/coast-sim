"""Tests for conops.roll module."""

from unittest.mock import Mock

import numpy as np

from conops import optimum_roll, optimum_roll_sidemount


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
