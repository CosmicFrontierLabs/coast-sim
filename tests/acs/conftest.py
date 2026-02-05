"""Test fixtures for ACS subsystem tests."""

from collections.abc import Generator
from unittest.mock import Mock, patch

import numpy as np
import pytest

from conops import ACS, AttitudeControlSystem, Constraint, SpacecraftBus
from conops.config.solar_panel import SolarPanel, SolarPanelSet


class DummyEphemeris:
    """Minimal mock ephemeris for testing."""

    def __init__(self) -> None:
        self.step_size = 1.0
        # Mock both earth and sun positions (legacy SkyCoord style)
        self.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
        self.sun = [Mock(ra=Mock(deg=45.0), dec=Mock(deg=23.5))]
        # New direct array access (rust-ephem 0.3.0+)
        self.earth_ra_deg = [0.0]
        self.earth_dec_deg = [0.0]
        self.sun_ra_deg = [45.0]
        self.sun_dec_deg = [23.5]
        self.moon_ra_deg = [90.0]
        self.moon_dec_deg = [10.0]
        # Mock position/velocity data for roll calculation
        self.sun_pv = Mock(position=np.array([[1.5e8, 0.0, 0.0]]))  # Sun position in km
        self.gcrs_pv = Mock(
            position=np.array([[0.0, 0.0, 6378.0]])
        )  # S/C position in km

    def index(self, time: object) -> int:
        return 0


@pytest.fixture
def mock_ephem() -> DummyEphemeris:
    """Create a mock ephemeris object."""
    return DummyEphemeris()


@pytest.fixture
def mock_constraint(mock_ephem: DummyEphemeris) -> Mock:
    """Create a mock constraint."""
    constraint = Mock(spec=Constraint)
    constraint.ephem = mock_ephem
    constraint.panel_constraint = Mock()
    constraint.panel_constraint.solar_panel = None
    constraint.in_constraint = Mock(return_value=False)
    constraint.in_eclipse = Mock(return_value=False)
    return constraint


@pytest.fixture
def mock_config(mock_ephem: DummyEphemeris, mock_constraint: Mock) -> Mock:
    """Create a mock config."""
    config = Mock()
    config.constraint = mock_constraint
    config.ground_stations = Mock()

    panel = SolarPanel(name="Panel", azimuth_deg=0.0, cant_deg=45.0, area_m2=1.0)
    config.solar_panel = SolarPanelSet(panels=[panel])

    config.spacecraft_bus = Mock()
    config.spacecraft_bus.attitude_control = Mock()
    # Mock predict_slew to return distance and path
    config.spacecraft_bus.attitude_control.predict_slew = Mock(return_value=(45.0, []))
    # Mock slew_time to return a reasonable slew duration
    config.spacecraft_bus.attitude_control.slew_time = Mock(return_value=100.0)
    return config


@pytest.fixture
def acs(mock_constraint: Mock, mock_config: Mock) -> Generator[ACS, None, None]:
    """Create an ACS instance with mocked dependencies."""
    with patch("conops.simulation.passes.PassTimes") as mock_passtimes:
        mock_pt = Mock()
        mock_pt.passes = []
        mock_pt.next_pass = Mock(return_value=None)
        mock_pt.__iter__ = Mock(return_value=iter([]))
        mock_passtimes.return_value = mock_pt
        acs_instance = ACS(config=mock_config)
        acs_instance.passrequests = mock_pt
        return acs_instance


@pytest.fixture
def bus() -> SpacecraftBus:
    return SpacecraftBus()


@pytest.fixture
def acs_config() -> AttitudeControlSystem:
    return AttitudeControlSystem(
        slew_acceleration=1.0, max_slew_rate=0.5, settle_time=90.0
    )


@pytest.fixture
def default_acs() -> AttitudeControlSystem:
    return AttitudeControlSystem()
