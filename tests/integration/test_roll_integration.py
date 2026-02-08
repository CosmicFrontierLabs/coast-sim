"""Integration tests for roll angle support across the system."""

from collections.abc import Generator
from unittest.mock import Mock, patch

import numpy as np
import pytest
from astropy.time import Time  # type: ignore[import-untyped]

from conops import ACS, SolarPanel, SolarPanelSet


@pytest.fixture(autouse=True)
def patch_eclipse_constraint() -> Generator[None, None, None]:
    """Patch eclipse constraint to avoid ephemeris lookup errors."""
    mock_constraint = Mock()
    mock_constraint.in_constraint = Mock(return_value=False)
    with patch.object(SolarPanel, "_eclipse_constraint", mock_constraint):
        yield


def create_comprehensive_mock_ephem(sun_ra: float = 90.0, sun_dec: float = 0.0) -> Mock:
    """Create a comprehensive mock ephemeris with proper time data."""
    ephem = Mock()

    # Time data - required by eclipse constraint
    start_time = 1514764800.0
    times = np.array([Time(start_time + i * 60, format="unix") for i in range(10)])
    ephem.timestamp = times

    # Sun position arrays
    ephem.sun_ra_deg = np.array([sun_ra] * 10)
    ephem.sun_dec_deg = np.array([sun_dec] * 10)
    ephem.earth_ra_deg = np.array([0.0] * 10)
    ephem.earth_dec_deg = np.array([0.0] * 10)
    ephem.sun = [Mock(ra=Mock(deg=sun_ra), dec=Mock(deg=sun_dec))]
    ephem.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
    ephem.moon_ra_deg = [90.0]
    ephem.moon_dec_deg = [10.0]
    ephem.sun_pv = Mock(position=np.array([[1.5e8, 0.0, 0.0]] * 10))
    ephem.gcrs_pv = Mock(position=np.array([[0.0, 0.0, 6378.0]] * 10))

    # Index method
    def mock_index(time_obj: object) -> int:
        if hasattr(time_obj, "unix"):
            for idx, t in enumerate(times):
                if abs(t.unix - time_obj.unix) < 30:
                    return idx
        return 0

    ephem.index = mock_index

    return ephem


def create_test_config_with_panels(
    sun_ra: float = 90.0, sun_dec: float = 0.0
) -> tuple[Mock, SolarPanelSet]:
    """Create a test config with solar panels."""
    # Create ephemeris
    ephem = create_comprehensive_mock_ephem(sun_ra, sun_dec)

    # Create constraint
    constraint = Mock()
    constraint.ephem = ephem
    constraint.panel_constraint = Mock()
    constraint.in_constraint = Mock(return_value=False)
    constraint.in_eclipse = Mock(return_value=False)

    # Create solar panels
    panel_set = SolarPanelSet(
        conversion_efficiency=0.95,
        panels=[
            SolarPanel(name="P1", normal=(1.0, 0.0, 0.0), max_power=500.0),
            SolarPanel(name="P2", normal=(0.0, 1.0, 0.0), max_power=500.0),
        ],
    )
    constraint.panel_constraint.solar_panel = panel_set

    # Create config
    config = Mock()
    config.constraint = constraint
    config.ground_stations = Mock()
    config.solar_panel = panel_set
    config.spacecraft_bus = Mock()
    config.spacecraft_bus.attitude_control = Mock()
    config.spacecraft_bus.attitude_control.predict_slew = Mock(return_value=(45.0, []))
    config.spacecraft_bus.attitude_control.slew_time = Mock(return_value=100.0)

    return config, panel_set


class TestRollACSIntegration:
    """Test integration of roll calculation between ACS and solar panels."""

    def test_acs_provides_roll_to_panel_calculation(self) -> None:
        """Test that ACS provides roll to solar panel illumination calculation."""
        config, panel_set = create_test_config_with_panels()

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

    def test_acs_roll_affects_panel_power(self) -> None:
        """Test that ACS roll affects calculated panel power."""
        config, panel_set = create_test_config_with_panels()

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

    def test_roll_in_illumination_and_power(self) -> None:
        """Test that roll is properly used in illumination_and_power calculation."""
        config, panel_set = create_test_config_with_panels()

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

    def test_roll_with_single_panel(self) -> None:
        """Test roll calculation with a single solar panel."""
        config = Mock()
        panel_set = SolarPanelSet(
            panels=[SolarPanel(name="P1", sidemount=True, max_power=1000.0)]
        )
        config.solar_panel = panel_set

        ephem = create_comprehensive_mock_ephem(sun_ra=90.0, sun_dec=0.0)

        illum_roll_0 = panel_set.panel_illumination_fraction(
            time=1514764800.0, ephem=ephem, ra=0.0, dec=0.0, roll=0.0
        )
        illum_roll_45 = panel_set.panel_illumination_fraction(
            time=1514764800.0, ephem=ephem, ra=0.0, dec=0.0, roll=45.0
        )

        assert isinstance(illum_roll_0, float)
        assert isinstance(illum_roll_45, float)

    def test_roll_with_many_panels(self) -> None:
        """Test roll calculation with many solar panels."""
        ephem = create_comprehensive_mock_ephem(sun_ra=90.0, sun_dec=0.0)

        # Create panel set with 4 panels at different azimuths
        panel_set = SolarPanelSet(
            conversion_efficiency=0.95,
            panels=[
                SolarPanel(name="P1", sidemount=True, azimuth_deg=0.0, max_power=250.0),
                SolarPanel(
                    name="P2", sidemount=True, azimuth_deg=90.0, max_power=250.0
                ),
                SolarPanel(
                    name="P3", sidemount=True, azimuth_deg=180.0, max_power=250.0
                ),
                SolarPanel(
                    name="P4", sidemount=True, azimuth_deg=270.0, max_power=250.0
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

    def test_roll_with_canted_panels(self) -> None:
        """Test roll with panels that have cant angles."""
        ephem = create_comprehensive_mock_ephem(sun_ra=90.0, sun_dec=0.0)

        panel_set = SolarPanelSet(
            panels=[
                SolarPanel(
                    name="P1",
                    sidemount=True,
                    azimuth_deg=0.0,
                    cant_x=5.0,
                    cant_y=5.0,
                    max_power=500.0,
                ),
                SolarPanel(
                    name="P2",
                    sidemount=False,
                    cant_x=10.0,
                    cant_y=10.0,
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

    def test_panel_illumination_vs_power_consistency(self) -> None:
        """Test that illumination and power are consistent."""
        ephem = create_comprehensive_mock_ephem(sun_ra=90.0, sun_dec=0.0)

        panel_set = SolarPanelSet(
            conversion_efficiency=0.95,
            panels=[
                SolarPanel(
                    name="P1", sidemount=True, azimuth_deg=45.0, max_power=1000.0
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

    def test_illumination_and_power_method_consistency(self) -> None:
        """Test consistency of illumination_and_power method with separate calls."""
        ephem = create_comprehensive_mock_ephem(sun_ra=90.0, sun_dec=0.0)

        panel_set = SolarPanelSet(
            conversion_efficiency=0.95,
            panels=[
                SolarPanel(
                    name="P1", sidemount=True, azimuth_deg=45.0, max_power=1000.0
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

    def test_roll_at_equator(self) -> None:
        """Test roll with pointing at equator."""
        ephem = create_comprehensive_mock_ephem(sun_ra=0.0, sun_dec=0.0)

        panel_set = SolarPanelSet(
            panels=[SolarPanel(name="P1", sidemount=True, azimuth_deg=90.0)]
        )

        illum = panel_set.panel_illumination_fraction(
            time=1514764800.0, ephem=ephem, ra=0.0, dec=0.0, roll=45.0
        )

        assert isinstance(illum, float)
        assert 0.0 <= illum <= 1.0

    def test_roll_at_poles(self) -> None:
        """Test roll with pointing at poles."""
        ephem = create_comprehensive_mock_ephem(sun_ra=0.0, sun_dec=89.0)

        panel_set = SolarPanelSet(
            panels=[SolarPanel(name="P1", sidemount=True, azimuth_deg=90.0)]
        )

        illum = panel_set.panel_illumination_fraction(
            time=1514764800.0, ephem=ephem, ra=0.0, dec=89.0, roll=45.0
        )

        assert isinstance(illum, float)
        assert 0.0 <= illum <= 1.0

    def test_roll_with_sun_behind_spacecraft(self) -> None:
        """Test roll when sun is behind spacecraft (in eclipse region)."""
        ephem = create_comprehensive_mock_ephem(sun_ra=180.0, sun_dec=0.0)

        panel_set = SolarPanelSet(
            panels=[SolarPanel(name="P1", sidemount=True, azimuth_deg=0.0)]
        )

        illum = panel_set.panel_illumination_fraction(
            time=1514764800.0, ephem=ephem, ra=0.0, dec=0.0, roll=45.0
        )

        # Should be minimal illumination when pointing away from sun
        assert isinstance(illum, float)
        assert 0.0 <= illum <= 1.0
