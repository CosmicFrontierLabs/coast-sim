"""Shared pytest fixtures for test suite."""

from unittest.mock import Mock, patch

import pytest

# from across.tools.ephemeris import Ephemeris
from conops.acs import ACS
from conops.constraint import Constraint

# class MockEphemeris(Ephemeris):
#     """Mock ephemeris that inherits from Ephemeris base class."""

#     def __init__(self):
#         self.step_size = 60.0
#         self.timestamp = Mock()
#         self.timestamp.unix = [1514764800.0 + i * 60.0 for i in range(1440)]
#         self.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0)) for _ in range(1440)]
#         self.sun = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]

#     def prepare_data(self):
#         """Required abstract method implementation."""
#         pass

#     def index(self, time):
#         """Return index for given time."""
#         return 0


# @pytest.fixture
# def mock_ephem():
#     """Create mock ephemeris."""
#     return MockEphemeris()


@pytest.fixture
def mock_ephem():
    """Create mock ephemeris."""
    ephem = Mock()
    ephem.step_size = 60.0
    ephem.timestamp = Mock()
    ephem.timestamp.unix = [1514764800.0 + i * 60.0 for i in range(1440)]
    ephem.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0)) for _ in range(1440)]
    ephem.sun = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
    ephem.index.return_value = 0
    return ephem


@pytest.fixture
def mock_constraint(mock_ephem):
    """Create mock constraint."""
    constraint = Mock(spec=Constraint)
    constraint.ephem = mock_ephem
    constraint.panel_constraint = Mock()
    constraint.panel_constraint.solar_panel = Mock()
    constraint.inoccult = Mock(return_value=False)

    # Mock the constraint.evaluate method to return an object with visibility
    mock_result = Mock()
    mock_visibility_item = Mock()
    mock_visibility_item.start_time = Mock()
    mock_visibility_item.start_time.timestamp.return_value = 1514764800.0
    mock_visibility_item.end_time = Mock()
    mock_visibility_item.end_time.timestamp.return_value = 1514764900.0

    mock_result.visibility = [mock_visibility_item]
    constraint.constraint.evaluate.return_value = mock_result

    return constraint


@pytest.fixture
def mock_config():
    """Create mock config."""
    config = Mock()
    config.ground_stations = Mock()
    config.spacecraft_bus = Mock()
    config.spacecraft_bus.attitude_control = Mock()
    config.spacecraft_bus.attitude_control.predict_slew = Mock(
        return_value=(0.0, (Mock(), Mock()))
    )
    config.spacecraft_bus.attitude_control.slew_time = Mock(return_value=10.0)
    config.solar_panel = Mock()
    return config


@pytest.fixture
def acs(mock_constraint, mock_config):
    """Create ACS instance."""
    with patch("conops.acs.PassTimes") as mock_pt_class:
        mock_pt = Mock()
        mock_pt.passes = []
        mock_pt.next_pass = Mock(return_value=None)
        mock_pt.__iter__ = Mock(return_value=iter([]))
        mock_pt_class.return_value = mock_pt
        acs_instance = ACS(constraint=mock_constraint, config=mock_config)
        acs_instance.passrequests = mock_pt
        return acs_instance


@pytest.fixture
def base_constraint():
    """Create a mock base constraint for config."""
    constraint = Mock(spec=Constraint)
    ephem = Mock()
    ephem.step_size = 60
    # Mock earth and sun arrays
    earth_mock = Mock()
    earth_mock.ra = Mock(deg=0.0)
    earth_mock.dec = Mock(deg=0.0)
    ephem.earth = [earth_mock]
    ephem.index = Mock(return_value=0)
    constraint.ephem = ephem
    constraint.in_eclipse = Mock(return_value=False)
    constraint.inoccult = Mock(return_value=False)
    return constraint


@pytest.fixture
def payload_constraint():
    """Create a mock constraint for payload."""
    constraint = Mock(spec=Constraint)
    ephem = Mock()
    ephem.step_size = 60
    # Mock earth and sun arrays
    earth_mock = Mock()
    earth_mock.ra = Mock(deg=0.0)
    earth_mock.dec = Mock(deg=0.0)
    ephem.earth = [earth_mock]
    ephem.index = Mock(return_value=0)
    constraint.ephem = ephem
    constraint.in_eclipse = Mock(return_value=False)
    constraint.inoccult = Mock(return_value=False)
    return constraint


@pytest.fixture
def config_with_payload_constraint(base_constraint, payload_constraint):
    """Create a config with base constraint and payload override."""
    from conops.battery import Battery
    from conops.config import Config
    from conops.groundstation import GroundStationRegistry
    from conops.instrument import Payload
    from conops.solar_panel import SolarPanelSet
    from conops.spacecraft_bus import SpacecraftBus

    spacecraft_bus = Mock(spec=SpacecraftBus)
    spacecraft_bus.attitude_control = Mock()
    spacecraft_bus.attitude_control.predict_slew = Mock(
        return_value=(0.0, (Mock(), Mock()))
    )
    spacecraft_bus.attitude_control.slew_time = Mock(return_value=10.0)

    payload = Mock(spec=Payload)
    payload.constraint = payload_constraint

    solar_panel = Mock(spec=SolarPanelSet)
    solar_panel.optimal_charging_pointing = Mock(return_value=(45.0, 23.5))

    battery = Mock(spec=Battery)
    ground_stations = Mock(spec=GroundStationRegistry)

    config = Config(
        name="Test Config",
        spacecraft_bus=spacecraft_bus,
        solar_panel=solar_panel,
        payload=payload,
        battery=battery,
        constraint=base_constraint,
        ground_stations=ground_stations,
    )
    return config
