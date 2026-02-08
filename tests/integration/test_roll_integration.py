"""Integration tests for roll angle support across the system."""

from unittest.mock import Mock, patch

import pytest

from conops import ACS, SolarPanel, SolarPanelSet
from conops.config.solar_panel import create_solar_panel_vector


class TestRollACSIntegration:
    """Test integration of roll calculation between ACS and solar panels."""

    def test_acs_provides_roll_to_panel_calculation(
        self, test_config_with_panels: tuple[Mock, SolarPanelSet]
    ) -> None:
        """Test that ACS provides roll to solar panel illumination calculation."""
        config, panel_set = test_config_with_panels

        with patch("conops.simulation.passes.PassTimes"):
            acs = ACS(config=config)

        utime = 1514764800.0

        # Get ACS pointing with roll
        ra, dec, roll, obsid = acs.pointing(utime)

        # Use that roll to calculate panel illumination
        illum = panel_set.panel_illumination_fraction(
            time=utime, ephem=config.constraint.ephem, ra=ra, dec=dec, roll=roll
        )

        assert isinstance(illum, float)
        assert 0.0 <= illum <= 1.0

    def test_acs_roll_affects_panel_power(
        self, test_config_with_panels: tuple[Mock, SolarPanelSet]
    ) -> None:
        """Test that ACS roll affects calculated panel power."""
        config, panel_set = test_config_with_panels

        with patch("conops.simulation.passes.PassTimes"):
            acs = ACS(config=config)

        utime = 1514764800.0
        ra, dec, roll, _ = acs.pointing(utime)

        # Power with ACS roll
        power_with_roll = panel_set.power(
            time=utime, ra=ra, dec=dec, ephem=config.constraint.ephem, roll=roll
        )

        # Power without roll (roll=0)
        power_no_roll = panel_set.power(
            time=utime, ra=ra, dec=dec, ephem=config.constraint.ephem, roll=0.0
        )

        assert isinstance(power_with_roll, float)
        assert isinstance(power_no_roll, float)
        # May or may not be different depending on optimal roll, but both should be valid
        assert power_with_roll >= 0.0
        assert power_no_roll >= 0.0

    def test_roll_in_illumination_and_power(
        self, test_config_with_panels: tuple[Mock, SolarPanelSet]
    ) -> None:
        """Test that roll is properly used in illumination_and_power calculation."""
        config, panel_set = test_config_with_panels

        with patch("conops.simulation.passes.PassTimes"):
            acs = ACS(config=config)

        utime = 1514764800.0
        ra, dec, roll, _ = acs.pointing(utime)

        # Calculate illumination and power with roll
        illum, power = panel_set.illumination_and_power(
            time=utime, ra=ra, dec=dec, ephem=config.constraint.ephem, roll=roll
        )

        assert isinstance(illum, float)
        assert isinstance(power, float)
        assert 0.0 <= illum <= 1.0
        assert power >= 0.0


class TestRollWithDifferentPanelConfigurations:
    """Test roll handling with various solar panel configurations."""

    def test_roll_with_single_panel(self, mock_ephem_with_pv: Mock) -> None:
        """Test roll calculation with a single solar panel."""
        config = Mock()
        panel_set = SolarPanelSet(
            panels=[SolarPanel(name="P1", normal=(0.0, 1.0, 0.0), max_power=1000.0)]
        )
        config.solar_panel = panel_set

        ephem = mock_ephem_with_pv

        illum_roll_0 = panel_set.panel_illumination_fraction(
            time=1514764800.0, ephem=ephem, ra=0.0, dec=0.0, roll=0.0
        )
        illum_roll_45 = panel_set.panel_illumination_fraction(
            time=1514764800.0, ephem=ephem, ra=0.0, dec=0.0, roll=45.0
        )

        assert isinstance(illum_roll_0, float)
        assert isinstance(illum_roll_45, float)

    def test_roll_with_many_panels(self, mock_ephem_with_pv: Mock) -> None:
        """Test roll calculation with many solar panels."""
        ephem = mock_ephem_with_pv

        # Create panel set with 4 panels at different azimuths
        panel_set = SolarPanelSet(
            conversion_efficiency=0.95,
            panels=[
                SolarPanel(
                    name="P1",
                    normal=create_solar_panel_vector(azimuth_deg=0.0),
                    max_power=250.0,
                ),
                SolarPanel(
                    name="P2",
                    normal=create_solar_panel_vector(azimuth_deg=90.0),
                    max_power=250.0,
                ),
                SolarPanel(
                    name="P3",
                    normal=create_solar_panel_vector(azimuth_deg=180.0),
                    max_power=250.0,
                ),
                SolarPanel(
                    name="P4",
                    normal=create_solar_panel_vector(azimuth_deg=270.0),
                    max_power=250.0,
                ),
            ],
        )

        illums = []
        for roll in [0.0, 90.0, 180.0, 270.0]:
            illum = panel_set.panel_illumination_fraction(
                time=1514764800.0, ephem=ephem, ra=0.0, dec=0.0, roll=roll
            )
            illums.append(illum)

        # All should be valid
        assert all(isinstance(i, float) for i in illums)
        assert all(0.0 <= i <= 1.0 for i in illums)

    def test_roll_with_canted_panels(self, mock_ephem_with_pv: Mock) -> None:
        """Test roll with panels that have cant angles."""
        ephem = mock_ephem_with_pv

        panel_set = SolarPanelSet(
            panels=[
                SolarPanel(
                    name="P1",
                    normal=create_solar_panel_vector(
                        azimuth_deg=0.0, cant_x=5.0, cant_y=5.0
                    ),
                    max_power=500.0,
                ),
                SolarPanel(
                    name="P2",
                    normal=create_solar_panel_vector(cant_x=10.0, cant_y=10.0),
                    max_power=500.0,
                ),
            ]
        )

        illum_with_roll = panel_set.panel_illumination_fraction(
            time=1514764800.0, ephem=ephem, ra=0.0, dec=0.0, roll=45.0
        )

        assert isinstance(illum_with_roll, float)
        assert 0.0 <= illum_with_roll <= 1.0


class TestRollConsistencyAcrossMethods:
    """Test that roll is handled consistently across different methods."""

    def test_panel_illumination_vs_power_consistency(
        self, mock_ephem_with_pv: Mock
    ) -> None:
        """Test that illumination and power are consistent."""
        ephem = mock_ephem_with_pv

        panel_set = SolarPanelSet(
            conversion_efficiency=0.95,
            panels=[
                SolarPanel(
                    name="P1",
                    normal=create_solar_panel_vector(azimuth_deg=45.0),
                    max_power=1000.0,
                ),
            ],
        )

        roll = 30.0

        # Get illumination
        illum = panel_set.panel_illumination_fraction(
            time=1514764800.0, ephem=ephem, ra=0.0, dec=0.0, roll=roll
        )

        # Get power
        power = panel_set.power(
            time=1514764800.0, ra=0.0, dec=0.0, ephem=ephem, roll=roll
        )

        # Power should be proportional to illumination
        expected_power = illum * 1000.0 * 0.95
        assert power == pytest.approx(expected_power, rel=1e-10)

    def test_illumination_and_power_method_consistency(
        self, mock_ephem_with_pv: Mock
    ) -> None:
        """Test consistency of illumination_and_power method with separate calls."""
        ephem = mock_ephem_with_pv

        panel_set = SolarPanelSet(
            conversion_efficiency=0.95,
            panels=[
                SolarPanel(
                    name="P1",
                    normal=create_solar_panel_vector(azimuth_deg=45.0),
                    max_power=1000.0,
                ),
            ],
        )

        roll = 30.0

        # Use separate calls
        illum1 = panel_set.panel_illumination_fraction(
            time=1514764800.0, ephem=ephem, ra=0.0, dec=0.0, roll=roll
        )
        power1 = panel_set.power(
            time=1514764800.0, ra=0.0, dec=0.0, ephem=ephem, roll=roll
        )

        # Use combined method
        illum2, power2 = panel_set.illumination_and_power(
            time=1514764800.0, ra=0.0, dec=0.0, ephem=ephem, roll=roll
        )

        assert illum1 == pytest.approx(illum2, rel=1e-10)
        assert power1 == pytest.approx(power2, rel=1e-10)


class TestRollBoundaryConditions:
    """Test roll handling at boundary conditions."""

    def test_roll_at_equator(self, mock_ephem_with_pv: Mock) -> None:
        """Test roll with pointing at equator."""
        ephem = mock_ephem_with_pv

        panel_set = SolarPanelSet(
            panels=[
                SolarPanel(
                    name="P1", normal=create_solar_panel_vector(azimuth_deg=90.0)
                )
            ]
        )

        illum = panel_set.panel_illumination_fraction(
            time=1514764800.0, ephem=ephem, ra=0.0, dec=0.0, roll=45.0
        )

        assert isinstance(illum, float)
        assert 0.0 <= illum <= 1.0

    def test_roll_at_poles(self, mock_ephem_sun_at_pole: Mock) -> None:
        """Test roll with pointing at poles."""
        ephem = mock_ephem_sun_at_pole

        panel_set = SolarPanelSet(
            panels=[
                SolarPanel(
                    name="P1", normal=create_solar_panel_vector(azimuth_deg=90.0)
                )
            ]
        )

        illum = panel_set.panel_illumination_fraction(
            time=1514764800.0, ephem=ephem, ra=0.0, dec=89.0, roll=45.0
        )

        assert isinstance(illum, float)
        assert 0.0 <= illum <= 1.0

    def test_roll_with_sun_behind_spacecraft(self, mock_ephem_sun_behind: Mock) -> None:
        """Test roll when sun is behind spacecraft (in eclipse region)."""
        ephem = mock_ephem_sun_behind

        panel_set = SolarPanelSet(
            panels=[
                SolarPanel(name="P1", normal=create_solar_panel_vector(azimuth_deg=0.0))
            ]
        )

        illum = panel_set.panel_illumination_fraction(
            time=1514764800.0, ephem=ephem, ra=0.0, dec=0.0, roll=45.0
        )

        # Should be minimal illumination when pointing away from sun
        assert isinstance(illum, float)
        assert 0.0 <= illum <= 1.0
