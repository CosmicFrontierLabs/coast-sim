"""Test fixtures for queue subsystem tests."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import numpy as np
import pytest
from astropy.time import Time  # type: ignore[import-untyped]

from conops.constants import DAY_SECONDS
from conops.queue_ditl import QueueDITL
from conops.queue_scheduler import DumbQueueScheduler


class DummyEphemeris:
    """Minimal mock ephemeris for testing."""

    def __init__(self):
        self.step_size = 3600.0
        # Cover 2018 day 331 (Nov 27) for 2 days
        base_time = datetime(2018, 11, 27, tzinfo=timezone.utc)
        # timestamp must be a list of datetime objects for helper functions
        self.timestamp = [
            base_time + timedelta(seconds=i * 3600)
            for i in range(48)  # 2 days worth
        ]

    def index(self, time):
        """Mock index method."""
        return 0

    def in_eclipse(self, utime: float) -> bool:
        """Mock eclipse check - always return False (in sunlight)."""
        return False


@pytest.fixture
def mock_ephem():
    """Create a mock ephemeris object."""
    return DummyEphemeris()


@pytest.fixture
def mock_config():
    """Create a mock config with all required subsystems."""
    config = Mock()

    # Mock constraint
    config.constraint = Mock()
    config.constraint.ephem = DummyEphemeris()
    config.constraint.panel_constraint = Mock()
    config.constraint.panel_constraint.solar_panel = Mock()
    config.constraint.inoccult = Mock(return_value=False)

    # Mock battery
    config.battery = Mock()
    config.battery.battery_level = 0.8
    config.battery.battery_alert = False
    config.battery.drain = Mock()
    config.battery.charge = Mock()

    # Mock spacecraft bus
    config.spacecraft_bus = Mock()
    config.spacecraft_bus.power = Mock(return_value=50.0)
    config.spacecraft_bus.attitude_control = Mock()
    config.spacecraft_bus.attitude_control.predict_slew = Mock(
        return_value=(10.0, [])
    )  # Return (distance, path)
    config.spacecraft_bus.attitude_control.slew_time = Mock(
        return_value=100.0
    )  # Return slew time in seconds

    # Mock payload
    config.payload = Mock()
    config.payload.power = Mock(return_value=30.0)

    # Mock ground stations
    config.ground_stations = Mock()

    return config


@pytest.fixture
def queue_ditl(mock_config, mock_ephem):
    """Create a QueueDITL instance with mocked dependencies."""
    with (
        patch("conops.queue_ditl.Queue") as mock_queue_class,
        patch("conops.ditl_mixin.PassTimes") as mock_passtimes,
        patch("conops.ditl_mixin.ACS") as mock_acs_class,
    ):
        # Mock PassTimes
        mock_pt = Mock()
        mock_pt.passes = []
        mock_pt.get = Mock()
        mock_pt.check_pass_timing = Mock(
            return_value={"start_pass": None, "end_pass": False, "updated_pass": None}
        )
        mock_passtimes.return_value = mock_pt

        # Mock ACS
        mock_acs = Mock()
        mock_acs.ephem = mock_ephem
        mock_acs.slewing = False
        mock_acs.inpass = False
        mock_acs.saa = None
        mock_acs.pointing = Mock(return_value=(0.0, 0.0, 0.0, 0))
        mock_acs.enqueue_command = Mock()
        mock_acs.passrequests = mock_pt
        mock_acs.slew_dists = []
        mock_acs.last_slew = None
        # Mock the helper methods used in _fetch_new_ppt
        mock_target_request = Mock()
        mock_target_request.next_vis = Mock(return_value=1000.0)
        mock_acs._create_target_request = Mock(return_value=mock_target_request)
        mock_acs._initialize_slew_positions = Mock(return_value=True)
        mock_acs._is_slew_valid = Mock(return_value=True)
        mock_acs._calculate_slew_timing = Mock(return_value=1000.0)
        mock_acs_class.return_value = mock_acs

        # Mock solar panel in config (not ACS)
        mock_config.solar_panel.illumination_and_power = Mock(return_value=(0.5, 100.0))

        # Mock Queue
        mock_queue = Mock()
        mock_queue.get = Mock(return_value=None)
        mock_queue_class.return_value = mock_queue

        ditl = QueueDITL(config=mock_config)
        ditl.ephem = mock_ephem
        ditl.acs = mock_acs
        ditl.queue = mock_queue

        return ditl


@pytest.fixture
def mock_ephemeris():
    """Create a mock ephemeris object."""
    ephem = Mock()
    # 24 hours of data starting 2021-01-04
    start_time = Time("2021-01-04 00:00:00", scale="utc").unix
    timestamps = np.arange(start_time, start_time + DAY_SECONDS, 60)
    ephem.timestamp = Mock()
    ephem.timestamp.unix = timestamps
    return ephem


class MockPointing:
    """Mock Pointing class for testing."""

    def __init__(
        self,
        targetid=1,
        ra=45.0,
        dec=30.0,
        merit=100.0,
        ssmin=300,
        ssmax=600,
        name="",
    ):
        self.targetid = targetid
        self.ra = ra
        self.dec = dec
        self.merit = merit
        self.ssmin = ssmin  # Minimum exposure time
        self.ssmax = ssmax  # Maximum exposure time
        self.name = name or f"Target_{targetid}"
        self.done = False
        self.roll = 0.0
        self.slewtime = 0
        self.begin = 0
        self.end = 0

    def calc_slewtime(self, ra_from, dec_from):
        """Calculate slew time from prior position."""
        dist = np.sqrt((self.ra - ra_from) ** 2 + (self.dec - dec_from) ** 2)
        self.slewtime = max(0, int(dist / 0.5))  # Slew rate of 0.5 deg/sec

    def visible(self, start_time, end_time):
        """Check if target is visible during time window."""
        # Mock: always visible unless explicitly marked invisible
        return getattr(self, "_visible", True)


@pytest.fixture
def mock_queue(mock_ephemeris):
    """Create a mock queue with sample targets."""
    queue = Mock()
    queue.ephem = mock_ephemeris
    queue.targets = []

    # Add sample targets
    target1 = MockPointing(targetid=1, ra=45.0, dec=30.0, merit=100, ssmin=300)
    target2 = MockPointing(targetid=2, ra=90.0, dec=-45.0, merit=90, ssmin=300)
    target3 = MockPointing(targetid=3, ra=180.0, dec=60.0, merit=80, ssmin=300)

    queue.targets = [target1, target2, target3]
    queue.__len__ = Mock(return_value=len(queue.targets))
    queue.__getitem__ = Mock(side_effect=lambda i: queue.targets[i])

    def mock_get(ra, dec, utime):
        """Mock get method that returns next available target."""
        for target in queue.targets:
            if not target.done and target.merit > 0:
                target.calc_slewtime(ra, dec)
                if target.visible(utime, utime + target.slewtime + target.ssmax):
                    target.begin = int(utime)
                    target.end = int(utime + target.slewtime + target.ssmax)
                    return target
        return None

    def mock_meritsort(ra, dec):
        """Mock meritsort to sort by merit."""
        queue.targets.sort(key=lambda x: x.merit, reverse=True)

    queue.get = mock_get
    queue.meritsort = mock_meritsort
    queue.reset = Mock()

    return queue


@pytest.fixture
def scheduler(mock_queue, mock_ephemeris):
    """Create a DumbQueueScheduler instance."""
    scheduler = DumbQueueScheduler(queue=mock_queue, year=2021, day=4, length=1)
    scheduler.queue.ephem = mock_ephemeris
    return scheduler


@pytest.fixture
def mock_pointing():
    """Fixture to create a mock pointing object."""

    def _mock_pointing(targetid, ra, dec, merit, ssmin=None):
        pointing = Mock()
        pointing.targetid = targetid
        pointing.ra = ra
        pointing.dec = dec
        pointing.merit = merit
        if ssmin is not None:
            pointing.ssmin = ssmin
        pointing.done = False
        return pointing

    return _mock_pointing
