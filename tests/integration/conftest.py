"""Test fixtures for integration tests."""

from collections.abc import Generator
from unittest.mock import Mock, patch

import numpy as np
import pytest
from astropy.time import Time  # type: ignore[import-untyped]

from conops import SolarPanel, SolarPanelSet


@pytest.fixture(autouse=True)
def patch_eclipse_constraint() -> Generator[None, None, None]:
    """Patch eclipse constraint to avoid ephemeris lookup errors in all integration tests."""
    mock_constraint = Mock()
    mock_constraint.in_constraint = Mock(return_value=False)
    with patch.object(SolarPanel, "_eclipse_constraint", mock_constraint):
        yield


@pytest.fixture
def mock_ephem_with_pv(sun_ra: float = 90.0, sun_dec: float = 0.0) -> Mock:
    """Create a comprehensive mock ephemeris with PV (position-velocity) data.

    This fixture provides a more complete ephemeris mock with sun/earth position vectors
    and all necessary attributes for solar panel and coordinate calculations.

    Args:
        sun_ra: Sun RA in degrees (default 90°)
        sun_dec: Sun Dec in degrees (default 0°)
    """
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

    # Position-velocity vectors for coordinate transformations
    ephem.sun_pv = Mock(position=np.array([[1.5e8, 0.0, 0.0]] * 10))
    ephem.gcrs_pv = Mock(position=np.array([[0.0, 0.0, 6378.0]] * 10))

    # Index method for ephemeris lookups
    def mock_index(time_obj: object) -> int:
        if hasattr(time_obj, "unix"):
            for idx, t in enumerate(times):
                if abs(t.unix - time_obj.unix) < 30:
                    return idx
        return 0

    ephem.index = mock_index

    return ephem


@pytest.fixture
def test_config_with_panels(mock_ephem_with_pv: Mock) -> tuple[Mock, SolarPanelSet]:
    """Create a test configuration with solar panels and mock ephemeris.

    Args:
        mock_ephem_with_pv: Fixture providing mock ephemeris with PV data

    Returns:
        Tuple of (config Mock, SolarPanelSet)
    """
    # Create constraint
    constraint = Mock()
    constraint.ephem = mock_ephem_with_pv
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


@pytest.fixture
def mock_ephem_sun_at_pole() -> Mock:
    """Create mock ephemeris with sun at high declination (north pole)."""
    ephem = Mock()

    start_time = 1514764800.0
    times = np.array([Time(start_time + i * 60, format="unix") for i in range(10)])
    ephem.timestamp = times

    ephem.sun_ra_deg = np.array([0.0] * 10)
    ephem.sun_dec_deg = np.array([89.0] * 10)
    ephem.earth_ra_deg = np.array([0.0] * 10)
    ephem.earth_dec_deg = np.array([0.0] * 10)
    ephem.sun = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=89.0))]
    ephem.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
    ephem.moon_ra_deg = [90.0]
    ephem.moon_dec_deg = [10.0]
    ephem.sun_pv = Mock(position=np.array([[1.5e8, 0.0, 0.0]] * 10))
    ephem.gcrs_pv = Mock(position=np.array([[0.0, 0.0, 6378.0]] * 10))

    def mock_index(time_obj: object) -> int:
        if hasattr(time_obj, "unix"):
            for idx, t in enumerate(times):
                if abs(t.unix - time_obj.unix) < 30:
                    return idx
        return 0

    ephem.index = mock_index
    return ephem


@pytest.fixture
def mock_ephem_sun_behind() -> Mock:
    """Create mock ephemeris with sun behind spacecraft (180° RA)."""
    ephem = Mock()

    start_time = 1514764800.0
    times = np.array([Time(start_time + i * 60, format="unix") for i in range(10)])
    ephem.timestamp = times

    ephem.sun_ra_deg = np.array([180.0] * 10)
    ephem.sun_dec_deg = np.array([0.0] * 10)
    ephem.earth_ra_deg = np.array([0.0] * 10)
    ephem.earth_dec_deg = np.array([0.0] * 10)
    ephem.sun = [Mock(ra=Mock(deg=180.0), dec=Mock(deg=0.0))]
    ephem.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
    ephem.moon_ra_deg = [90.0]
    ephem.moon_dec_deg = [10.0]
    ephem.sun_pv = Mock(position=np.array([[1.5e8, 0.0, 0.0]] * 10))
    ephem.gcrs_pv = Mock(position=np.array([[0.0, 0.0, 6378.0]] * 10))

    def mock_index(time_obj: object) -> int:
        if hasattr(time_obj, "unix"):
            for idx, t in enumerate(times):
                if abs(t.unix - time_obj.unix) < 30:
                    return idx
        return 0

    ephem.index = mock_index
    return ephem
