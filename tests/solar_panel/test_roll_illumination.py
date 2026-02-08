"""
Unit tests for roll angle in solar panel illumination calculations.
Tests the integration of spacecraft roll angle in panel illumination and power calculations.
"""

from unittest.mock import Mock

import numpy as np

from conops.config.solar_panel import SolarPanel, SolarPanelSet


class TestSolarPanelRollIllumination:
    """Test solar panel illumination with roll angle variations."""

    def test_gimbled_panel_always_lit(self) -> None:
        """Test that gimbled panels always have full illumination (not in eclipse)."""
        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        panel = SolarPanel(
            name="GimbledPanel",
            gimbled=True,
            normal=(0.0, 1.0, 0.0),
            max_power=100.0,
        )
        panel._eclipse_constraint = mock_constraint

        illum = panel.panel_illumination_fraction(
            time=1700000000.0,
            ephem=Mock(),
            ra=0.0,
            dec=0.0,
            roll=0.0,
        )

        assert illum == 1.0

    def test_gimbled_panel_zero_in_eclipse(self) -> None:
        """Test that gimbled panels have zero illumination when in eclipse."""
        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=True)  # In eclipse

        panel = SolarPanel(
            name="GimbbledPanel",
            gimbled=True,
            normal=(0.0, 1.0, 0.0),
            max_power=100.0,
        )
        panel._eclipse_constraint = mock_constraint

        illum = panel.panel_illumination_fraction(
            time=1700000000.0,
            ephem=Mock(),
            ra=0.0,
            dec=0.0,
            roll=0.0,
        )

        assert illum == 0.0


class TestSolarPanelSetRoll:
    """Test solar panel set with roll angle."""

    def test_panel_set_illumination_with_zero_roll(
        self, panel_set: SolarPanelSet, mock_ephemeris_with_sun_vectors: Mock
    ) -> None:
        """Test that panel set properly handles roll in illumination calculation."""
        illum = panel_set.panel_illumination_fraction(
            time=1700000000.0,
            ephem=mock_ephemeris_with_sun_vectors,
            ra=0.0,
            dec=0.0,
            roll=0.0,
        )
        assert isinstance(illum, (float, np.floating))
        assert 0 <= illum <= 1

    def test_panel_set_power_with_roll(
        self, panel_set: SolarPanelSet, mock_ephemeris_with_sun_vectors: Mock
    ) -> None:
        """Test panel set power calculation with roll."""
        power = panel_set.power(
            time=1700000000.0,
            ra=0.0,
            dec=0.0,
            ephem=mock_ephemeris_with_sun_vectors,
            roll=10.0,
        )
        assert isinstance(power, (float, np.floating))
        assert power >= 0

    def test_panel_set_illumination_and_power(
        self, mock_ephemeris_with_sun_vectors: Mock
    ) -> None:
        """Test combined illumination and power calculation."""
        from unittest.mock import patch

        # Create panel set with mocked eclipse constraint
        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=False)

        panels = [
            SolarPanel(name="Panel1", normal=(0.0, 1.0, 0.0), max_power=100.0),
            SolarPanel(name="Panel2", normal=(0.0, 1.0, 0.0), max_power=100.0),
            SolarPanel(name="Panel3", normal=(0.0, 0.0, -1.0), max_power=100.0),
        ]
        panel_set = SolarPanelSet(panels=panels)

        # Patch the _get_eclipse_constraint function
        with patch(
            "conops.config.solar_panel._get_eclipse_constraint",
            return_value=mock_constraint,
        ):
            illum, power = panel_set.illumination_and_power(
                time=1700000000.0,
                ephem=mock_ephemeris_with_sun_vectors,
                ra=0.0,
                dec=0.0,
                roll=10.0,
            )

        assert isinstance(illum, (float, np.floating))
        assert 0 <= illum <= 1
        assert isinstance(power, (float, np.floating))
        assert power >= 0
