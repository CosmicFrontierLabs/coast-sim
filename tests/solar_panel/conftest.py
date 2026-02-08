"""Test fixtures for solar_panel subsystem tests."""

from unittest.mock import Mock

import numpy as np
import pytest
from astropy.time import Time  # type: ignore[import-untyped]

from conops import SolarPanel, SolarPanelSet


# Fixtures for mock ephemeris
@pytest.fixture
def mock_ephemeris() -> Mock:
    """Create a comprehensive mock ephemeris object."""
    ephem = Mock()

    # Time data
    start_time = 1514764800.0  # 2018-01-01
    times = np.array([Time(start_time + i * 60, format="unix") for i in range(10)])
    ephem.timestamp = times

    # Sun position (mock SkyCoord objects)
    sun_mocks = []
    sun_ra_deg_list = []
    sun_dec_deg_list = []
    for i in range(10):
        sun_mock = Mock()
        sun_mock.ra = Mock()
        sun_mock.ra.deg = 90.0 + i * 2.0  # Varying RA
        sun_mock.dec = Mock()
        sun_mock.dec.deg = 30.0 - i * 1.0  # Varying Dec
        sun_mock.separation = Mock(return_value=Mock(deg=45.0))
        sun_mocks.append(sun_mock)
        sun_ra_deg_list.append(90.0 + i * 2.0)
        sun_dec_deg_list.append(30.0 - i * 1.0)
    ephem.sun = np.array(sun_mocks)
    # New direct array access (rust-ephem 0.3.0+)
    ephem.sun_ra_deg = np.array(sun_ra_deg_list)
    ephem.sun_dec_deg = np.array(sun_dec_deg_list)

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
    def mock_index(time_obj: Time) -> int:
        if isinstance(time_obj, Time):
            # Find closest matching time
            for idx, t in enumerate(times):
                if abs(t.unix - time_obj.unix) < 30:
                    return idx
        return 0

    ephem.index = mock_index

    return ephem


@pytest.fixture
def mock_eclipse_constraint() -> Mock:
    """Create a mock eclipse constraint."""
    from unittest.mock import Mock

    constraint = Mock()
    constraint.in_constraint.return_value = False  # Default to not in eclipse
    return constraint


@pytest.fixture
def default_solar_panel() -> SolarPanel:
    """Create a default SolarPanel."""
    from conops import SolarPanel

    return SolarPanel()


@pytest.fixture
def standard_solar_panel() -> SolarPanel:
    """Create a standard test SolarPanel with common parameters."""
    from conops import SolarPanel

    return SolarPanel(max_power=500.0, conversion_efficiency=0.9)


@pytest.fixture
def zero_power_solar_panel() -> SolarPanel:
    """Create a SolarPanel with zero power."""
    from conops import SolarPanel

    return SolarPanel(max_power=0.0)


@pytest.fixture
def high_power_solar_panel() -> SolarPanel:
    """Create a SolarPanel with high power."""
    from conops import SolarPanel

    return SolarPanel(max_power=1000.0)


@pytest.fixture
def empty_solar_panel_set() -> SolarPanelSet:
    """Create an empty SolarPanelSet."""
    from conops import SolarPanelSet

    return SolarPanelSet(panels=[])


@pytest.fixture
def single_panel_set(default_solar_panel: SolarPanel) -> SolarPanelSet:
    """Create a SolarPanelSet with a single default panel."""
    from conops import SolarPanelSet

    return SolarPanelSet(panels=[default_solar_panel])


@pytest.fixture
def standard_single_panel_set(standard_solar_panel: SolarPanel) -> SolarPanelSet:
    """Create a SolarPanelSet with a single standard panel."""
    from conops import SolarPanelSet

    return SolarPanelSet(panels=[standard_solar_panel])


@pytest.fixture
def zero_power_panel_set(zero_power_solar_panel: SolarPanel) -> SolarPanelSet:
    """Create a SolarPanelSet with a single zero power panel."""
    from conops import SolarPanelSet

    return SolarPanelSet(panels=[zero_power_solar_panel])


@pytest.fixture
def default_panel_set() -> SolarPanelSet:
    return SolarPanelSet(name="Default Set")


@pytest.fixture
def multi_panel_set() -> SolarPanelSet:
    return SolarPanelSet(
        name="Array",
        conversion_efficiency=0.95,
        panels=[
            SolarPanel(name="P1", normal=(0.0, 1.0, 0.0), max_power=300.0),
            SolarPanel(name="P2", normal=(0.0, 0.0, -1.0), max_power=700.0),
        ],
    )


@pytest.fixture
def solar_panel_y_normal(mock_eclipse_constraint: Mock) -> SolarPanel:
    """Create a side-mounted solar panel (Y normal)."""
    panel = SolarPanel(
        name="TestPanel_Y",
        normal=(0.0, 1.0, 0.0),  # Y-pointing normal
        max_power=100.0,
    )
    panel._eclipse_constraint = mock_eclipse_constraint
    return panel


@pytest.fixture
def mock_ephemeris_with_sun_vectors() -> Mock:
    """Create a mock ephemeris object with sun position vectors."""
    from datetime import datetime, timezone

    ephem = Mock()
    index = 0
    ephem.sun_pv = Mock()
    ephem.sun_pv.position = np.array(
        [
            [1.496e8, 0, 0],  # Sun position in km (scaled)
        ]
    )

    ephem.gcrs_pv = Mock()
    ephem.gcrs_pv.position = np.array(
        [
            [0, 0, 0],  # Spacecraft position (at origin for simplicity)
        ]
    )

    ephem.sun_ra_deg = np.array([0.0])
    ephem.sun_dec_deg = np.array([0.0])
    ephem.index = Mock(return_value=index)
    # Add times array for rust_ephem constraints
    ephem.times = np.array([datetime(2023, 11, 14, 22, 13, 20, tzinfo=timezone.utc)])
    return ephem


@pytest.fixture
def panel_set(mock_eclipse_constraint: Mock) -> SolarPanelSet:
    """Create a test panel set with multiple panels."""
    panels = [
        SolarPanel(name="Panel1", normal=(0.0, 1.0, 0.0), max_power=100.0),
        SolarPanel(name="Panel2", normal=(0.0, 1.0, 0.0), max_power=100.0),
        SolarPanel(name="Panel3", normal=(0.0, 0.0, -1.0), max_power=100.0),
    ]
    # Patch eclipse constraint for all panels
    for panel in panels:
        panel._eclipse_constraint = mock_eclipse_constraint

    return SolarPanelSet(panels=panels)


@pytest.fixture
def efficiency_fallback_panel_set() -> SolarPanelSet:
    return SolarPanelSet(
        conversion_efficiency=0.91,
        panels=[
            SolarPanel(name="P1", max_power=100.0, conversion_efficiency=None),
            SolarPanel(name="P2", max_power=100.0, conversion_efficiency=0.88),
        ],
    )
