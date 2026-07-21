"""Tests for conops.roll module."""

from unittest.mock import Mock

import numpy as np
import pytest

from conops import optimum_roll, optimum_roll_sidemount
from conops.common.vector import boresight_axis_permutation
from conops.config.solar_panel import SolarPanel, SolarPanelSet


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

    def test_optimum_roll_sidemount_forwards_boresight_axis(
        self, mock_ephem_sidemount, monkeypatch
    ):
        """optimum_roll_sidemount must forward its boresight_axis kwarg, not drop it."""
        received = {}

        def fake_optimum_roll(ra, dec, utime, ephem, *args, **kwargs):
            received.update(kwargs)
            return 12.5

        monkeypatch.setattr("conops.simulation.roll.optimum_roll", fake_optimum_roll)
        optimum_roll_sidemount(
            10.0, 20.0, 1700000000.0, mock_ephem_sidemount, boresight_axis="+Z"
        )
        assert received.get("boresight_axis") == "+Z"


class TestOptimumRollBoresightAxisInvariance:
    """optimum_roll must give physically equivalent results under axis relabeling.

    A panel physically re-mounted so that its normal, expressed in the
    boresight_axis-labelled user frame, is P @ (normal in the +X-boresight
    baseline frame) must produce the same optimum roll as the +X baseline,
    because it describes the exact same physical hardware.
    """

    @pytest.mark.parametrize("axis", ["+Y", "+Z", "-X", "-Y", "-Z"])
    def test_relabeled_panel_matches_plus_x_baseline(self, mock_ephem, axis):
        mock_ephem.sun_pv.position = [np.array([1.5e8, 2.0e7, 3.0e6])]
        mock_ephem.gcrs_pv.position = [np.array([0.0, 0.0, 6378.0])]
        ra, dec, utime = 30.0, 20.0, 1700000000.0

        baseline_normal = (0.0, 1.0, 0.0)
        baseline_panels = SolarPanelSet(
            panels=[SolarPanel(name="P", normal=baseline_normal, max_power=100.0)]
        )
        baseline_roll = optimum_roll(
            ra, dec, utime, mock_ephem, baseline_panels, boresight_axis="+X"
        )

        p_mat = boresight_axis_permutation(axis)
        relabeled_normal = tuple(p_mat @ np.array(baseline_normal))
        relabeled_panels = SolarPanelSet(
            panels=[SolarPanel(name="P", normal=relabeled_normal, max_power=100.0)]
        )
        relabeled_roll = optimum_roll(
            ra, dec, utime, mock_ephem, relabeled_panels, boresight_axis=axis
        )

        assert relabeled_roll == pytest.approx(baseline_roll, abs=1e-6)
