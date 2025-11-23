"""Test fixtures for solar_panel subsystem tests."""

from unittest.mock import Mock

import numpy as np
import pytest
from astropy.time import Time  # type: ignore[import-untyped]

from conops import SolarPanel, SolarPanelSet


# Fixtures for mock ephemeris
@pytest.fixture
def mock_ephemeris():
    """Create a comprehensive mock ephemeris object."""
    ephem = Mock()

    # Time data
    start_time = 1514764800.0  # 2018-01-01
    times = np.array([Time(start_time + i * 60, format="unix") for i in range(10)])
    ephem.timestamp = times

    # Sun position (mock SkyCoord objects)
    sun_mocks = []
    for i in range(10):
        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 90.0 + i * 2.0  # Varying RA
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 30.0 - i * 1.0  # Varying Dec
        sun_mock.separation = Mock(return_value=Mock(deg=45.0))
        sun_mocks.append(sun_mock)
    ephem.sun = np.array(sun_mocks)

    # Earth position
    earth_mocks = []
    for i in range(10):
        earth_mock = Mock()
        earth_mock.separation = Mock(return_value=Mock(deg=0.5))
        earth_mocks.append(earth_mock)
    ephem.earth = np.array(earth_mocks)

    # Earth radius angle (angular size of Earth from spacecraft)
    ephem.earth_radius_angle = np.array([Mock(deg=0.3) for _ in range(10)])

    # Mock methods
    def mock_index(time_obj):
        if isinstance(time_obj, Time):
            # Find closest matching time
            for idx, t in enumerate(times):
                if abs(t.unix - time_obj.unix) < 30:
                    return idx
        return 0

    ephem.index = mock_index

    return ephem


@pytest.fixture
def default_panel_set():
    return SolarPanelSet(name="Default Set")


@pytest.fixture
def multi_panel_set():
    return SolarPanelSet(
        name="Array",
        conversion_efficiency=0.95,
        panels=[
            SolarPanel(
                name="P1", sidemount=True, cant_x=5.0, cant_y=0.0, max_power=300.0
            ),
            SolarPanel(
                name="P2", sidemount=False, cant_x=0.0, cant_y=12.0, max_power=700.0
            ),
        ],
    )


@pytest.fixture
def efficiency_fallback_panel_set():
    return SolarPanelSet(
        conversion_efficiency=0.91,
        panels=[
            SolarPanel(name="P1", max_power=100.0, conversion_efficiency=None),
            SolarPanel(name="P2", max_power=100.0, conversion_efficiency=0.88),
        ],
    )
