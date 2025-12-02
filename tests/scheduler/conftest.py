"""Test fixtures for scheduler subsystem tests."""

from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np
import pytest
from astropy.time import Time  # type: ignore[import-untyped]
from pydantic import ConfigDict

from conops import SAA, DumbScheduler
from conops.config import (
    AttitudeControlSystem,
    Battery,
    GroundStationRegistry,
    MissionConfig,
    Payload,
    SolarPanel,
    SolarPanelSet,
    SpacecraftBus,
)
from conops.config.constraint import Constraint as RealConstraint


class SimpleTarget:
    """Simple target class for testing."""

    def __init__(self, targetid, ra, dec, exptime, name=""):
        self.targetid = targetid
        self.ra = ra
        self.dec = dec
        self.exptime = exptime
        self.name = name or f"Target_{targetid}"
        self.merit = 100
        self.slewtime = 0

    def calc_slewtime(self, ra_from, dec_from):
        """Calculate slew time from prior position."""
        dist = np.sqrt((self.ra - ra_from) ** 2 + (self.dec - dec_from) ** 2)
        self.slewtime = int(dist / 0.25)


@pytest.fixture
def simple_target_factory():
    """Factory fixture for creating SimpleTarget instances."""

    def _factory(targetid, ra, dec, exptime, name=""):
        return SimpleTarget(targetid, ra, dec, exptime, name)

    return _factory


@pytest.fixture
def mock_ephemeris():
    """Create a mock ephemeris object."""
    ephem = Mock()
    # 24 hours of data with 60-second steps
    start_time = 1543276800  # 2018-11-27 00:00:00 UTC
    ephem.utime = np.arange(start_time, start_time + 86400, 60)

    # Create Time objects with timestamp() method
    class TimeWithTimestamp(Time):
        def timestamp(self):
            return float(self.unix)

    ephem.timestamp = [TimeWithTimestamp(t, format="unix") for t in ephem.utime]
    # Provide datetimes list for adapter/datetimes compatibility
    ephem.datetimes = [
        datetime.fromtimestamp(int(t), tz=timezone.utc) for t in ephem.utime
    ]

    # Mock methods
    def mock_index(time_obj):
        if isinstance(time_obj, Time):
            utime = time_obj.unix
        elif isinstance(time_obj, datetime):
            utime = time_obj.timestamp()
        else:
            utime = time_obj
        return int(np.searchsorted(ephem.utime, utime))

    def mock_ephindex(time_obj):
        if isinstance(time_obj, Time):
            utime = time_obj.unix
        else:
            utime = time_obj
        return int(np.searchsorted(ephem.utime, utime))

    def mock_ephtime(utime):
        return Time(utime, format="unix")

    ephem.index = mock_index
    ephem.ephindex = mock_ephindex
    ephem.ephtime = mock_ephtime
    ephem.step_size = 60  # seconds

    return ephem


class MockConstraint(RealConstraint):
    """Test-friendly Constraint subclass that returns array-like results for
    iterable utime values and bool for scalar utime, while remaining a valid
    pydantic Constraint instance for Config validation.
    """

    def __init__(self, ephem):
        # Initialize Constraint with defaults
        super().__init__()
        # Set the ephemeris (other tests rely on this attribute)
        self.ephem = ephem

    # Allow assigning methods/attributes at runtime in tests (monkeypatching)
    model_config = ConfigDict(extra="allow", arbitrary_types_allowed=True)

    def in_constraint(self, ra, dec, utime, hardonly=True):
        # If utime is an iterable (list/np.ndarray or something with __len__),
        # return a numpy boolean array matching its length to mimic the real
        # Constraint behavior.
        if isinstance(utime, (list, np.ndarray)) or hasattr(utime, "__len__"):
            try:
                length = len(utime)
            except Exception:
                # fallback if len() raises for some unusual iterable
                return np.zeros(0, dtype=bool)
            return np.zeros(length, dtype=bool)
        # Otherwise, treat as a scalar time and return a single boolean
        return False


@pytest.fixture
def mock_constraint(mock_ephemeris):
    """Create a mock constraint object."""
    return MockConstraint(mock_ephemeris)


@pytest.fixture
def mock_saa():
    """Create a mock SAA object."""
    saa = Mock(spec=SAA)
    saa.ephem = None  # Will be set by scheduler
    saa.saatimes = []
    saa.insaa = Mock(return_value=False)
    saa.calc = Mock()
    return saa


@pytest.fixture
def mock_config(mock_ephemeris, mock_constraint):
    """Create a mock spacecraft config."""
    # Create minimal required components
    spacecraft_bus = SpacecraftBus(
        attitude_control=AttitudeControlSystem(),
        communications=None,  # Will be set in tests that need it
    )

    # Create minimal payload, battery, solar_panel
    payload = Payload(instruments=[])
    battery = Battery(capacity_wh=1000, max_depth_of_discharge=0.8)
    solar_panel = SolarPanelSet(panels=[SolarPanel(sidemount=False)])

    # Use the mock_constraint fixture so in_constraint returns arrays when
    # utime is an iterable. Assign the ephemeris that was created for
    # the test so other components can rely on it.
    constraint = mock_constraint
    constraint.ephem = mock_ephemeris

    config = MissionConfig(
        spacecraft_bus=spacecraft_bus,
        solar_panel=solar_panel,
        payload=payload,
        battery=battery,
        constraint=constraint,
        ground_stations=GroundStationRegistry.default(),
    )
    return config


@pytest.fixture
def scheduler(mock_config, mock_saa):
    """Create a DumbScheduler instance with mocked dependencies."""
    scheduler = DumbScheduler(config=mock_config, days=1)
    scheduler.saa = mock_saa
    scheduler.config = mock_config  # Set the config for PlanEntry creation
    # Ensure the scheduler uses the mock_constraint that returns arrays
    # from in_constraint so the scheduling logic can iterate over the
    # result without encountering scalar returns.
    scheduler.constraint = mock_config.constraint
    return scheduler


@pytest.fixture
def sample_targets():
    """Create sample target entries."""
    return [
        SimpleTarget(targetid=1, ra=45.0, dec=30.0, exptime=600, name="Galaxy A"),
        SimpleTarget(targetid=2, ra=90.0, dec=-45.0, exptime=480, name="Nebula B"),
        SimpleTarget(targetid=3, ra=180.0, dec=60.0, exptime=720, name="Star C"),
        SimpleTarget(targetid=4, ra=270.0, dec=-30.0, exptime=300, name="Cluster D"),
    ]
