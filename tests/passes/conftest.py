"""Test fixtures for passes subsystem tests."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import numpy as np
import pytest

from conops import Config, Constraint, Pass


class MockEphemeris:
    """Mock ephemeris for testing."""

    def __init__(self, step_size=60.0, num_points=1440):
        self.step_size = step_size
        self.num_points = num_points

        # Create timestamps for one day
        base_time = datetime(2018, 1, 1, tzinfo=timezone.utc)
        self.timestamp = [
            base_time + timedelta(seconds=i * step_size) for i in range(num_points)
        ]

        # Create mock gcrs_pv and itrs_pv for position data
        self.gcrs_pv = Mock()
        self.itrs_pv = Mock()

        # Create position arrays (simplified orbital positions)
        positions = []
        for i in range(num_points):
            angle = 2 * np.pi * i / 1440  # One orbit per day
            x = 6900 * np.cos(angle)  # Earth radius + 550 km
            y = 6900 * np.sin(angle)
            z = 0
            positions.append([x, y, z])

        self.gcrs_pv.position = np.array(positions)
        self.itrs_pv.position = np.array(positions)

    def index(self, time):
        """Mock index method."""
        return 0


@pytest.fixture
def mock_ephem():
    """Create a mock ephemeris."""
    return MockEphemeris()


@pytest.fixture
def mock_constraint(mock_ephem):
    """Create a mock constraint."""
    constraint = Mock(spec=Constraint)
    constraint.ephem = mock_ephem
    return constraint


@pytest.fixture
def mock_acs_config():
    """Create a mock ACS config."""
    config = Mock()
    config.predict_slew = Mock(return_value=(10.0, np.array([[0, 0, 1]])))
    config.slew_time = Mock(return_value=10.0)  # Add slew_time method
    return config


@pytest.fixture
def mock_config(mock_acs_config):
    """Create a mock Config object."""
    config = Mock(spec=Config)
    config.spacecraft_bus = Mock()
    config.spacecraft_bus.attitude_control = mock_acs_config
    config.ground_stations = None  # Will use default
    return config


@pytest.fixture
def basic_pass(mock_constraint, mock_ephem, mock_acs_config):
    """Create a basic Pass instance."""
    p = Pass(
        constraint=mock_constraint,
        acs_config=mock_acs_config,
        station="SGS",
        begin=1514764800.0,
        length=480.0,
        gsstartra=10.0,
        gsstartdec=20.0,
        gsendra=15.0,
        gsenddec=25.0,
    )
    # Ensure ephem is set (from constraint)
    p.ephem = mock_constraint.ephem
    return p


@pytest.fixture
def mock_ephemeris_100():
    return MockEphemeris(step_size=60.0, num_points=100)
