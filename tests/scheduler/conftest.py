"""Test fixtures for scheduler subsystem tests."""

from datetime import datetime, timezone
from unittest.mock import Mock

import numpy as np
import pytest
from astropy.time import Time  # type: ignore[import-untyped]

from conops import DumbScheduler
from conops.constraint import Constraint
from conops.saa import SAA


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
def simple_target():
    return SimpleTarget(targetid=1, ra=45.0, dec=30.0, exptime=600)


@pytest.fixture
def simple_target_with_name():
    return SimpleTarget(targetid=42, ra=45.0, dec=30.0, exptime=600, name="Test")


@pytest.fixture
def two_targets():
    return [
        SimpleTarget(targetid=1, ra=0.0, dec=0.0, exptime=300),
        SimpleTarget(targetid=2, ra=90.0, dec=0.0, exptime=300),
    ]


@pytest.fixture
def target_with_zero_exptime():
    return SimpleTarget(targetid=2, ra=90.0, dec=0.0, exptime=0)


@pytest.fixture
def short_exptime_target():
    return SimpleTarget(targetid=1, ra=45.0, dec=30.0, exptime=60)


@pytest.fixture
def long_exptime_target():
    return SimpleTarget(targetid=1, ra=45.0, dec=30.0, exptime=86400)


@pytest.fixture
def large_target_list():
    return [
        SimpleTarget(
            targetid=i,
            ra=(i * 3.6) % 360,
            dec=(i - 50) % 90 - 45,
            exptime=300,
        )
        for i in range(100)
    ]


@pytest.fixture
def additional_target():
    return SimpleTarget(targetid=2, ra=90.0, dec=0.0, exptime=300)


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

    return ephem


@pytest.fixture
def mock_constraint(mock_ephemeris):
    """Create a mock constraint object."""
    constraint = Mock(spec=Constraint)
    constraint.ephem = mock_ephemeris
    constraint.sun_constraint = Mock()
    constraint.sun_constraint.min_angle = 50.0
    constraint.anti_sun_constraint = Mock()
    constraint.anti_sun_constraint.max_angle = 180.0
    constraint.earth_constraint = Mock()
    constraint.earth_constraint.min_angle = 20.0
    constraint.moon_constraint = Mock()
    constraint.moon_constraint.min_angle = 20.0

    # Mock inoccult to always return False (no occultation)
    def mock_inoccult(ra, dec, utime, hardonly=True):
        if isinstance(utime, (list, np.ndarray)):
            return np.zeros(len(utime), dtype=bool)
        elif hasattr(utime, "__len__"):
            return np.zeros(len(utime), dtype=bool)
        else:
            return np.array([False], dtype=bool)

    constraint.inoccult = mock_inoccult
    return constraint


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
def mock_config():
    """Create a mock spacecraft config."""
    config = Mock()
    config.spacecraft_bus = Mock()
    config.spacecraft_bus.attitude_control = Mock()
    return config


@pytest.fixture
def scheduler(mock_constraint, mock_saa, mock_config):
    """Create a DumbScheduler instance with mocked dependencies."""
    scheduler = DumbScheduler(constraint=mock_constraint, days=1)
    scheduler.saa = mock_saa
    scheduler.config = mock_config
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
