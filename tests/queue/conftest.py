"""Test fixtures for queue subsystem tests."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import numpy as np
import pytest
from astropy.time import Time  # type: ignore[import-untyped]

from conops import DAY_SECONDS, DumbQueueScheduler, QueueDITL
from conops.targets.plan import Plan


class DummyEphemeris:
    """Minimal mock ephemeris for testing."""

    def __init__(self):
        self.step_size = 3600.0  # Use 1 hour steps for faster tests
        # Cover 2018 day 331 (Nov 27) for 1 day
        base_time = datetime(2018, 11, 27, tzinfo=timezone.utc)
        # timestamp must be a list of datetime objects for helper functions
        # Use 1 hour steps: 24 steps for 1 day
        self.timestamp = [
            base_time + timedelta(seconds=i * 3600)
            for i in range(25)  # 24 hours + 1 for end time
        ]
        # Add earth and sun attributes for ACS initialization
        self.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0)) for _ in range(25)]
        self.sun = [Mock(ra=Mock(deg=45.0), dec=Mock(deg=23.5)) for _ in range(25)]

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
    config.constraint.in_constraint = Mock(return_value=False)

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

    # Mock solar panel
    config.solar_panel = Mock()
    config.solar_panel.optimal_charging_pointing = Mock(return_value=(45.0, 23.5))
    config.solar_panel.illumination_and_power = Mock(return_value=(0.5, 100.0))

    # Mock ground stations
    config.ground_stations = Mock()

    return config


@pytest.fixture
def queue_ditl(mock_config, mock_ephem):
    """Create a QueueDITL instance with mocked dependencies."""
    with (
        patch("conops.Queue") as mock_queue_class,
        patch("conops.PassTimes") as mock_passtimes,
        patch("conops.ACS") as mock_acs_class,
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
        # Set acsmode to a real ACSMode enum value for logging
        from conops import ACSMode

        mock_acs.acsmode = ACSMode.SCIENCE
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

        ditl = QueueDITL(config=mock_config, ephem=mock_ephem, queue=mock_queue)
        ditl.acs = mock_acs

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
        ss_min=300,
        ss_max=600,
        name="",
    ):
        self.targetid = targetid
        self.ra = ra
        self.dec = dec
        self.merit = merit
        self.ss_min = ss_min  # Minimum exposure time
        self.ss_max = ss_max  # Maximum exposure time
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
    target1 = MockPointing(targetid=1, ra=45.0, dec=30.0, merit=100, ss_min=300)
    target2 = MockPointing(targetid=2, ra=90.0, dec=-45.0, merit=90, ss_min=300)
    target3 = MockPointing(targetid=3, ra=180.0, dec=60.0, merit=80, ss_min=300)

    queue.targets = [target1, target2, target3]
    queue.__len__ = Mock(return_value=len(queue.targets))
    queue.__getitem__ = Mock(side_effect=lambda i: queue.targets[i])

    def mock_get(ra, dec, utime):
        """Mock get method that returns next available target."""
        for target in queue.targets:
            if not target.done and target.merit > 0:
                target.calc_slewtime(ra, dec)
                if target.visible(utime, utime + target.slewtime + target.ss_max):
                    target.begin = int(utime)
                    target.end = int(utime + target.slewtime + target.ss_max)
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
    begin = datetime(2021, 1, 4, tzinfo=timezone.utc)
    end = begin + timedelta(days=1)
    scheduler = DumbQueueScheduler(queue=mock_queue, begin=begin, end=end)
    scheduler.queue.ephem = mock_ephemeris
    # Override get to return None for basic tests
    scheduler.queue.get = Mock(return_value=None)
    return scheduler


@pytest.fixture
def mock_pointing():
    """Fixture to create a mock pointing object."""

    def _mock_pointing(targetid, ra, dec, merit, ss_min=None):
        pointing = Mock()
        pointing.targetid = targetid
        pointing.ra = ra
        pointing.dec = dec
        pointing.merit = merit
        if ss_min is not None:
            pointing.ss_min = ss_min
        pointing.done = False
        return pointing

    return _mock_pointing


@pytest.fixture
def make_target(mock_pointing):
    """Return a factory to create targets quickly."""

    def _make(targetid=1, ra=45.0, dec=30.0, merit=100, ss_min=300):
        t = mock_pointing(targetid=targetid, ra=ra, dec=dec, merit=merit, ss_min=ss_min)
        t.done = False
        return t

    return _make


@pytest.fixture
def make_targets(make_target):
    """Return a factory to create a list of targets."""

    def _make_many(count=3, start_ra=45.0):
        targets = []
        for i in range(count):
            ra = (start_ra + i * 45) % 360
            dec = -45 + i * 30
            targets.append(
                make_target(targetid=i + 1, ra=ra, dec=dec, merit=100 - i, ss_min=300)
            )
        return targets

    return _make_many


@pytest.fixture
def queue_get_from_list():
    """
    Fixture returning a helper to set scheduler.queue.get to pop entries
    from a provided list. Returns the recorded positions list if tracking
    is enabled.
    """

    def _set_queue_get(scheduler, targets, track_positions=False):
        call_count = {"count": 0}
        positions = []

        def getter(ra, dec, utime):
            if track_positions:
                positions.append((ra, dec))
            if call_count["count"] < len(targets):
                res = targets[call_count["count"]]
                res.done = False
                call_count["count"] += 1
                return res
            return None

        scheduler.queue.get = getter
        return positions

    return _set_queue_get


@pytest.fixture
def scheduler_2022_100_len2(mock_queue):
    plan = Plan()
    begin = datetime(2022, 4, 10, tzinfo=timezone.utc)
    end = begin + timedelta(days=2)
    return DumbQueueScheduler(queue=mock_queue, plan=plan, begin=begin, end=end)


# TOO-related fixtures for reducing test code duplication


@pytest.fixture
def standard_too_params():
    """Standard TOO parameters used in many tests."""
    return {
        "obsid": 1000001,
        "ra": 180.0,
        "dec": 45.0,
        "merit": 10000.0,
        "exptime": 3600,
        "name": "Test TOO",
    }


@pytest.fixture
def submitted_too(queue_ditl, standard_too_params):
    """A QueueDITL instance with a standard TOO already submitted."""
    return queue_ditl.submit_too(**standard_too_params)


@pytest.fixture
def low_merit_current_ppt(queue_ditl):
    """Create a current pointing object with low merit for TOO interrupt tests."""
    from conops import Pointing

    ppt = Pointing(
        config=queue_ditl.config,
        ra=0.0,
        dec=0.0,
        obsid=1,
        name="Current obs",
        merit=100.0,  # Lower merit than typical TOO
        exptime=1800,
    )
    queue_ditl.ppt = ppt
    return ppt


@pytest.fixture
def mock_pointing_visibility():
    """Mock patch for Pointing.visibility method."""
    with patch("conops.targets.pointing.Pointing.visibility") as mock_visibility:
        mock_visibility.return_value = None  # visibility() just populates windows
        yield mock_visibility


@pytest.fixture
def mock_pointing_visible():
    """Mock patch for Pointing.visible method."""
    with patch("conops.targets.pointing.Pointing.visible") as mock_visible:
        mock_visible.return_value = True  # Default to visible
        yield mock_visible


@pytest.fixture
def mock_terminate_ppt():
    """Mock patch for QueueDITL._terminate_ppt method."""
    with patch("conops.QueueDITL._terminate_ppt") as mock_terminate:
        yield mock_terminate


@pytest.fixture
def mock_fetch_new_ppt():
    """Mock patch for QueueDITL._fetch_new_ppt method."""
    with patch("conops.QueueDITL._fetch_new_ppt") as mock_fetch:
        yield mock_fetch


@pytest.fixture
def basic_too_request():
    """Create a basic TOORequest object with standard parameters."""
    from conops.ditl import TOORequest

    return TOORequest(
        obsid=1000001,
        ra=180.0,
        dec=45.0,
        merit=10000.0,
        exptime=3600,
        name="GRB 250101A",
    )


@pytest.fixture
def custom_too_request():
    """Create a TOORequest object with custom submit_time and executed=True."""
    from conops.ditl import TOORequest

    return TOORequest(
        obsid=1000002,
        ra=90.0,
        dec=-30.0,
        merit=5000.0,
        exptime=1800,
        name="Test TOO",
        submit_time=1234567890.0,
        executed=True,
    )


@pytest.fixture
def unix_timestamp_too(queue_ditl):
    """A QueueDITL instance with a TOO submitted using Unix timestamp."""
    submit_time = 1640995200.0  # 2022-01-01 00:00:00 UTC
    return queue_ditl.submit_too(
        obsid=1000002,
        ra=90.0,
        dec=-30.0,
        merit=5000.0,
        exptime=1800,
        name="Scheduled TOO",
        submit_time=submit_time,
    )


@pytest.fixture
def mock_too_interrupt_success(
    mock_pointing_visibility,
    mock_pointing_visible,
    mock_terminate_ppt,
    mock_fetch_new_ppt,
):
    """Combined mock setup for successful TOO interrupt tests."""
    mock_pointing_visibility.return_value = None  # visibility() just populates windows
    mock_pointing_visible.return_value = True  # Target is visible
    return {
        "visibility": mock_pointing_visibility,
        "visible": mock_pointing_visible,
        "terminate": mock_terminate_ppt,
        "fetch": mock_fetch_new_ppt,
    }


@pytest.fixture
def mock_too_interrupt_no_current_obs(
    mock_pointing_visibility, mock_pointing_visible, mock_fetch_new_ppt
):
    """Combined mock setup for TOO interrupt tests with no current observation."""
    mock_pointing_visibility.return_value = None  # visibility() just populates windows
    mock_pointing_visible.return_value = True  # Target is visible
    return {
        "visibility": mock_pointing_visibility,
        "visible": mock_pointing_visible,
        "fetch": mock_fetch_new_ppt,
    }


@pytest.fixture
def queue_ditl_no_queue_log(mock_config, mock_ephem):
    """Create a QueueDITL instance with a queue that has no log initially."""
    with (
        patch("conops.Queue") as mock_queue_class,
        patch("conops.PassTimes") as mock_passtimes,
        patch("conops.ACS") as mock_acs_class,
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
        # Set acsmode to a real ACSMode enum value for logging
        from conops import ACSMode

        mock_acs.acsmode = ACSMode.SCIENCE
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

        # Mock Queue with no log initially
        mock_queue = Mock()
        mock_queue.get = Mock(return_value=None)
        mock_queue.log = None  # Explicitly set log to None
        mock_queue_class.return_value = mock_queue

        ditl = QueueDITL(config=mock_config, ephem=mock_ephem, queue=mock_queue)
        ditl.acs = mock_acs

        return ditl


@pytest.fixture
def queue_ditl_acs_no_ephem(mock_config, mock_ephem):
    """Create a QueueDITL instance where ACS initially has no ephem."""
    with (
        patch("conops.Queue") as mock_queue_class,
        patch("conops.PassTimes") as mock_passtimes,
        patch("conops.ACS") as mock_acs_class,
    ):
        # Mock PassTimes
        mock_pt = Mock()
        mock_pt.passes = []
        mock_pt.get = Mock()
        mock_pt.check_pass_timing = Mock(
            return_value={"start_pass": None, "end_pass": False, "updated_pass": None}
        )
        mock_passtimes.return_value = mock_pt

        # Mock ACS with no ephem initially
        mock_acs = Mock()
        mock_acs.ephem = None  # ACS has no ephem initially
        mock_acs.slewing = False
        mock_acs.inpass = False
        mock_acs.saa = None
        mock_acs.pointing = Mock(return_value=(0.0, 0.0, 0.0, 0))
        mock_acs.enqueue_command = Mock()
        mock_acs.passrequests = mock_pt
        mock_acs.slew_dists = []
        mock_acs.last_slew = None
        # Set acsmode to a real ACSMode enum value for logging
        from conops import ACSMode

        mock_acs.acsmode = ACSMode.SCIENCE
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

        ditl = QueueDITL(config=mock_config, ephem=mock_ephem, queue=mock_queue)
        ditl.acs = mock_acs

        return ditl
