"""Tests for conops.roll module."""

from unittest.mock import Mock

import numpy as np

from conops.roll import optimum_roll, optimum_roll_sidemount


class TestOptimumRoll:
    """Test optimum_roll function."""

    def test_optimum_roll_without_solar_panel(self):
        """Test optimum_roll without solar panel (analytic solution)."""
        # Create mock ephemeris
        ephem = Mock()
        ephem.index = Mock(return_value=0)

        # Create mock sun coordinate with cartesian
        sun_coord = Mock()
        sun_coord.cartesian.xyz.to_value = Mock(
            return_value=np.array([1000, 500, 800])  # km
        )
        ephem.sun = [sun_coord]

        ra = 45.0
        dec = 30.0
        utime = 1700000000.0

        # Call the function - should use analytic path since solar_panel is None
        roll = optimum_roll(ra, dec, utime, ephem, solar_panel=None)

        # Result should be a float in [0, 360)
        assert isinstance(roll, float)
        assert 0 <= roll < 360

    def test_optimum_roll_returns_float(self):
        """Test that optimum_roll returns a float."""
        ephem = Mock()
        ephem.index = Mock(return_value=0)

        sun_coord = Mock()
        sun_coord.cartesian.xyz.to_value = Mock(return_value=np.array([1000, 200, 600]))
        ephem.sun = [sun_coord]

        ra = 90.0
        dec = 0.0
        utime = 1700000000.0

        roll = optimum_roll(ra, dec, utime, ephem, solar_panel=None)
        assert isinstance(roll, float)

    def test_optimum_roll_with_solar_panel(self):
        """Test optimum_roll with solar panel (weighted optimization)."""
        ephem = Mock()
        ephem.index = Mock(return_value=0)

        sun_coord = Mock()
        sun_coord.cartesian.xyz.to_value = Mock(return_value=np.array([1000, 300, 700]))
        ephem.sun = [sun_coord]

        # Create mock solar panel
        solar_panel = Mock()
        mock_panel = Mock()
        mock_panel.sidemount = True
        mock_panel.cant_x = None
        mock_panel.cant_y = None
        mock_panel.conversion_efficiency = 0.3
        mock_panel.max_power = 800.0
        mock_panel.azimuth_deg = 0.0

        solar_panel._effective_panels = Mock(return_value=[mock_panel])
        solar_panel.conversion_efficiency = 0.3

        ra = 45.0
        dec = 30.0
        utime = 1700000000.0

        roll = optimum_roll(ra, dec, utime, ephem, solar_panel=solar_panel)

        # Result should be a float in [0, 360)
        assert isinstance(roll, float)
        assert 0 <= roll < 360

    def test_optimum_roll_with_multiple_panels(self):
        """Test optimum_roll with multiple solar panels."""
        ephem = Mock()
        ephem.index = Mock(return_value=0)

        sun_coord = Mock()
        sun_coord.cartesian.xyz.to_value = Mock(return_value=np.array([1000, 400, 600]))
        ephem.sun = [sun_coord]

        # Create mock solar panel with multiple panels
        solar_panel = Mock()
        mock_panel1 = Mock()
        mock_panel1.sidemount = True
        mock_panel1.cant_x = 0.0
        mock_panel1.cant_y = 0.0
        mock_panel1.conversion_efficiency = 0.3
        mock_panel1.max_power = 800.0
        mock_panel1.azimuth_deg = 0.0

        mock_panel2 = Mock()
        mock_panel2.sidemount = False
        mock_panel2.cant_x = 0.0
        mock_panel2.cant_y = 0.0
        mock_panel2.conversion_efficiency = 0.3
        mock_panel2.max_power = 600.0
        mock_panel2.azimuth_deg = 90.0

        solar_panel._effective_panels = Mock(return_value=[mock_panel1, mock_panel2])
        solar_panel.conversion_efficiency = 0.3

        ra = 60.0
        dec = 20.0
        utime = 1700000000.0

        roll = optimum_roll(ra, dec, utime, ephem, solar_panel=solar_panel)

        assert isinstance(roll, float)
        assert 0 <= roll < 360

    def test_optimum_roll_with_canted_panels(self):
        """Test optimum_roll with canted solar panels."""
        ephem = Mock()
        ephem.index = Mock(return_value=0)

        sun_coord = Mock()
        sun_coord.cartesian.xyz.to_value = Mock(return_value=np.array([800, 300, 700]))
        ephem.sun = [sun_coord]

        solar_panel = Mock()
        mock_panel = Mock()
        mock_panel.sidemount = True
        mock_panel.cant_x = 10.0  # Canted in X
        mock_panel.cant_y = 5.0  # Canted in Y
        mock_panel.conversion_efficiency = 0.3
        mock_panel.max_power = 800.0
        mock_panel.azimuth_deg = 45.0

        solar_panel._effective_panels = Mock(return_value=[mock_panel])
        solar_panel.conversion_efficiency = 0.3

        ra = 30.0
        dec = 45.0
        utime = 1700000000.0

        roll = optimum_roll(ra, dec, utime, ephem, solar_panel=solar_panel)

        assert isinstance(roll, float)
        assert 0 <= roll < 360


class TestOptimumRollSidemount:
    """Test optimum_roll_sidemount function."""

    def test_optimum_roll_sidemount_basic(self):
        """Test basic optimum_roll_sidemount calculation."""
        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sunvec = [np.array([1000, 500, 800])]  # Sun vector in km

        ra = 45.0
        dec = 30.0
        utime = 1700000000.0

        roll = optimum_roll_sidemount(ra, dec, utime, ephem)

        # Result should be a float in [0, 360)
        assert isinstance(roll, float)
        assert 0 <= roll < 360

    def test_optimum_roll_sidemount_zero_sun(self):
        """Test optimum_roll_sidemount with sun directly ahead."""
        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sunvec = [np.array([1000, 0, 0])]  # Sun along X-axis

        ra = 0.0
        dec = 0.0
        utime = 1700000000.0

        roll = optimum_roll_sidemount(ra, dec, utime, ephem)

        assert isinstance(roll, float)
        assert 0 <= roll < 360

    def test_optimum_roll_sidemount_different_positions(self):
        """Test optimum_roll_sidemount with different RA/Dec positions."""
        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sunvec = [np.array([800, 400, 600])]

        ra = 90.0
        dec = -30.0
        utime = 1700000000.0

        roll = optimum_roll_sidemount(ra, dec, utime, ephem)

        assert isinstance(roll, float)
        assert 0 <= roll < 360

    def test_optimum_roll_sidemount_returns_float(self):
        """Test that optimum_roll_sidemount returns a float."""
        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sunvec = [np.array([900, 200, 700])]

        ra = 180.0
        dec = 45.0
        utime = 1700000000.0

        roll = optimum_roll_sidemount(ra, dec, utime, ephem)

        assert isinstance(roll, float)
        assert 0 <= roll < 360

    def test_optimum_roll_sidemount_wraps_to_360(self):
        """Test that roll angle wraps correctly to [0, 360)."""
        ephem = Mock()
        ephem.index = Mock(return_value=0)
        ephem.sunvec = [np.array([1000, 0, 0])]

        ra = 0.0
        dec = 0.0
        utime = 1700000000.0

        roll = optimum_roll_sidemount(ra, dec, utime, ephem)

        # Verify wrap-around works correctly
        assert 0 <= roll < 360
