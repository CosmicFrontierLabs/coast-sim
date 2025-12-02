"""Test fixtures for passes subsystem tests."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import numpy as np
import pytest
from rust_ephem import TLEEphemeris

from conops import Constraint, MissionConfig, Pass
from conops.config import (
    AttitudeControlSystem,
    Battery,
    GroundStationRegistry,
    Payload,
    SolarPanelSet,
    SpacecraftBus,
)


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
def mock_ephem(create_ephem):
    """Create a simple real TLE ephemeris for tests.

    Tests previously used a MockEphemeris that fails Pydantic instance checks; use a
    short TLEEphemeris instead so model validation passes and tests can rely on
    actual ephemeris attributes like .timestamp, .gcrs_pv, etc.
    """
    # Use a short period ephemeris (15 minutes) for speed
    begin = datetime(2025, 8, 15, 0, 0, 0, tzinfo=timezone.utc)
    end = datetime(2025, 8, 15, 0, 15, 0, tzinfo=timezone.utc)
    return create_ephem(begin, end, step_size=60)


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
def mock_config(mock_constraint, mock_acs_config):
    """Create a Config object for testing."""
    # Create minimal required components
    spacecraft_bus = SpacecraftBus(
        attitude_control=AttitudeControlSystem(),
        communications=None,  # Will be set in tests that need it
    )

    # Create minimal payload, battery, solar_panel
    payload = Payload(instruments=[])
    battery = Battery(capacity_wh=1000, max_depth_of_discharge=0.8)
    solar_panel = SolarPanelSet(panels=[])

    config = MissionConfig(
        spacecraft_bus=spacecraft_bus,
        solar_panel=solar_panel,
        payload=payload,
        battery=battery,
        constraint=mock_constraint,
        ground_stations=GroundStationRegistry.default(),
    )
    return config


@pytest.fixture
def basic_pass_mock(mock_config, create_ephem):
    """Create a basic Pass instance."""
    ephem = create_ephem(1514764800.0, 1514764800.0 + 480.0)
    p = Pass(
        config=mock_config,
        ephem=ephem,
        station="SGS",
        begin=1514764800.0,
        length=480.0,
        gsstartra=10.0,
        gsstartdec=20.0,
        gsendra=15.0,
        gsenddec=25.0,
    )
    return p


@pytest.fixture
def mock_ephemeris_100():
    return MockEphemeris(step_size=60.0, num_points=100)


@pytest.fixture
def base_begin():
    return 1514764800.0


@pytest.fixture
def default_length():
    return 480.0


@pytest.fixture
def two_step_utime():
    return [1514764800.0, 1514764900.0]


@pytest.fixture
def three_step_utime():
    return [1514764800.0, 1514764900.0, 1514765000.0]


@pytest.fixture
def small_ra():
    return [10.0, 12.0]


@pytest.fixture
def small_dec():
    return [20.0, 22.0]


@pytest.fixture
def start_ra():
    return [15.0, 16.0]


@pytest.fixture
def start_dec():
    return [25.0, 26.0]


@pytest.fixture
def tle_path():
    return "examples/example.tle"


@pytest.fixture
def create_ephem(tle_path):
    def _create(begin, end, step_size=60):
        # Convert unix timestamp floats to timezone-aware datetimes if needed
        if isinstance(begin, (int, float)):
            begin = datetime.fromtimestamp(begin, tz=timezone.utc)
        if isinstance(end, (int, float)):
            end = datetime.fromtimestamp(end, tz=timezone.utc)
        return TLEEphemeris(tle=tle_path, begin=begin, end=end, step_size=step_size)

    return _create


@pytest.fixture
def acs_mock():
    acs = Mock()
    acs.predict_slew.return_value = (0.1, None)
    acs.slew_time.return_value = 100.0
    return acs


@pytest.fixture
def basic_pass_factory(mock_config, create_ephem):
    def _create(
        station="SGS",
        begin=1514764800.0,
        length=480.0,
    ):
        ephem = create_ephem(begin, begin + length)
        return Pass(
            config=mock_config,
            ephem=ephem,
            station=station,
            begin=begin,
            length=length,
            gsstartra=10.0,
            gsstartdec=20.0,
            gsendra=15.0,
            gsenddec=25.0,
        )

    return _create


@pytest.fixture
def basic_pass(basic_pass_factory):
    return basic_pass_factory()
