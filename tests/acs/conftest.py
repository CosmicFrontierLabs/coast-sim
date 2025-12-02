"""Test fixtures for ACS subsystem tests."""

from unittest.mock import Mock, patch

import pytest

from conops import ACS, AttitudeControlSystem, Constraint, SpacecraftBus


class DummyEphemeris:
    """Minimal mock ephemeris for testing."""

    def __init__(self):
        self.step_size = 1.0
        # Mock both earth and sun positions
        self.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
        self.sun = [Mock(ra=Mock(deg=45.0), dec=Mock(deg=23.5))]

    def index(self, time):
        return 0


@pytest.fixture
def mock_ephem():
    """Create a mock ephemeris object."""
    return DummyEphemeris()


@pytest.fixture
def mock_constraint(mock_ephem):
    """Create a mock constraint."""
    constraint = Mock(spec=Constraint)
    constraint.ephem = mock_ephem
    constraint.panel_constraint = Mock()
    constraint.panel_constraint.solar_panel = Mock()
    constraint.in_constraint = Mock(return_value=False)
    constraint.in_eclipse = Mock(return_value=False)
    return constraint


@pytest.fixture
def mock_config(mock_ephem, mock_constraint):
    """Create a mock config."""
    config = Mock()
    config.constraint = mock_constraint
    config.ground_stations = Mock()

    # Create a mock solar panel with the optimal_charging_pointing method
    solar_panel_mock = Mock()

    def mock_optimal_charging_pointing(utime, ephem):
        # Return current sun position
        return (ephem.sun[0].ra.deg, ephem.sun[0].dec.deg)

    solar_panel_mock.optimal_charging_pointing.side_effect = (
        mock_optimal_charging_pointing
    )
    config.solar_panel = solar_panel_mock

    config.spacecraft_bus = Mock()
    config.spacecraft_bus.attitude_control = Mock()
    # Mock predict_slew to return distance and path
    config.spacecraft_bus.attitude_control.predict_slew = Mock(return_value=(45.0, []))
    # Mock slew_time to return a reasonable slew duration
    config.spacecraft_bus.attitude_control.slew_time = Mock(return_value=100.0)
    return config


@pytest.fixture
def acs(mock_constraint, mock_config):
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
def bus():
    return SpacecraftBus()


@pytest.fixture
def acs_config():
    return AttitudeControlSystem(
        slew_acceleration=1.0, max_slew_rate=0.5, settle_time=90.0
    )


@pytest.fixture
def default_acs():
    return AttitudeControlSystem()
