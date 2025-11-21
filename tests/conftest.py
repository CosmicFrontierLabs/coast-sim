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
