"""Unit tests for QueueDITL class."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import numpy as np
import pytest

from conops.acs import ACSCommandType
from conops.common import ACSMode
from conops.passes import Pass
from conops.queue_ditl import QueueDITL


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

    # Mock instruments
    config.instruments = Mock()
    config.instruments.power = Mock(return_value=30.0)

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


class TestQueueDITLInitialization:
    """Test QueueDITL initialization."""

    def test_initialization(self, mock_config):
        """Test that QueueDITL initializes with correct defaults."""
        with (
            patch("conops.queue_ditl.Queue"),
            patch("conops.ditl_mixin.PassTimes"),
            patch("conops.ditl_mixin.ACS"),
        ):
            ditl = QueueDITL(config=mock_config)

            assert ditl.ppt is None
            assert ditl.charging_ppt is None
            assert ditl.ra == []
            assert ditl.dec == []
            assert ditl.roll == []
            assert ditl.mode == []
            assert ditl.obsid == []
            assert ditl.panel == []
            assert ditl.batterylevel == []
            assert ditl.power == []
            assert ditl.panel_power == []
            assert len(ditl.ppst) == 0
            assert ditl.constraint is mock_config.constraint
            assert ditl.battery is mock_config.battery
            assert ditl.spacecraft_bus is mock_config.spacecraft_bus
            assert ditl.instruments is mock_config.instruments


class TestTimeindex:
    """Test timeindex method."""

    def test_timeindex_basic(self, queue_ditl):
        """Test timeindex returns correct index."""
        queue_ditl.utime = np.array([100.0, 200.0, 300.0, 400.0, 500.0])

        assert queue_ditl.timeindex(150.0) == 0
        assert queue_ditl.timeindex(250.0) == 1
        assert queue_ditl.timeindex(350.0) == 2
        assert queue_ditl.timeindex(500.0) == 4

    def test_timeindex_exact_match(self, queue_ditl):
        """Test timeindex with exact time match."""
        queue_ditl.utime = np.array([100.0, 200.0, 300.0])

        assert queue_ditl.timeindex(200.0) == 1


class TestSetupSimulationTiming:
    """Test _setup_simulation_timing helper method."""

    def test_setup_timing_success(self, queue_ditl):
        """Test successful timing setup."""
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 60

        result = queue_ditl._setup_simulation_timing()

        assert result is True
        assert queue_ditl.ustart > 0
        assert queue_ditl.uend > queue_ditl.ustart
        assert queue_ditl.uend == queue_ditl.ustart + 86400
        assert len(queue_ditl.utime) == 86400 // 60


class TestScheduleGroundstationPasses:
    """Test _schedule_groundstation_passes helper method."""

    def test_schedule_passes_empty(self, queue_ditl, capsys):
        """Test scheduling passes when none exist."""
        queue_ditl.acs.passrequests.passes = []
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1

        queue_ditl._schedule_groundstation_passes()

        queue_ditl.acs.passrequests.get.assert_called_once_with(2018, 331, 1)
        captured = capsys.readouterr()
        assert "Scheduling groundstation passes..." in captured.out

    def test_schedule_passes_already_scheduled(self, queue_ditl):
        """Test when passes are already scheduled."""
        mock_pass = Mock()
        queue_ditl.acs.passrequests.passes = [mock_pass]

        queue_ditl._schedule_groundstation_passes()

        queue_ditl.acs.passrequests.get.assert_not_called()

    def test_schedule_passes_returns_passes(self, queue_ditl, capsys):
        """Test scheduling passes when they are returned."""
        queue_ditl.acs.passrequests.passes = []
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1

        # Create mock passes
        mock_pass1 = Mock()
        mock_pass1.__str__ = Mock(return_value="Pass 1")
        mock_pass2 = Mock()
        mock_pass2.__str__ = Mock(return_value="Pass 2")

        # After calling get, passes should be populated
        def populate_passes(year, day, length):
            queue_ditl.acs.passrequests.passes = [mock_pass1, mock_pass2]

        queue_ditl.acs.passrequests.get.side_effect = populate_passes

        queue_ditl._schedule_groundstation_passes()

        captured = capsys.readouterr()
        assert "Scheduling groundstation passes..." in captured.out
        assert "Scheduled pass: Pass 1" in captured.out
        assert "Scheduled pass: Pass 2" in captured.out


class TestDetermineMode:
    """Test mode determination now handled by ACS.get_mode() - these tests use real ACS instance."""

    def test_determine_mode_slewing(self, mock_config, mock_ephem):
        """Test mode determination when slewing."""
        from conops.acs import ACS
        from conops.constraint import Constraint

        constraint = Constraint(ephem=None)
        constraint.ephem = mock_ephem
        acs = ACS(constraint=constraint, config=mock_config)

        # Set up mock slew to indicate slewing
        mock_slew = Mock()
        mock_slew.is_slewing = Mock(return_value=True)
        mock_slew.obstype = "PPT"
        acs.current_slew = mock_slew

        mode = acs.get_mode(1000.0)

        assert mode == ACSMode.SLEWING

    def test_determine_mode_pass(self, mock_config, mock_ephem):
        """Test mode determination during ground station pass."""
        from conops.acs import ACS
        from conops.constraint import Constraint
        from conops.passes import Pass

        constraint = Constraint(ephem=None)
        constraint.ephem = mock_ephem
        acs = ACS(constraint=constraint, config=mock_config)

        # Set up real Pass in dwell phase
        mock_pass = Mock(spec=Pass)
        mock_pass.is_slewing = Mock(return_value=False)
        mock_pass.obstype = "GSP"
        mock_pass.slewend = 900.0
        mock_pass.begin = 950.0
        mock_pass.length = 200.0
        acs.current_slew = mock_pass

        mode = acs.get_mode(1000.0)

        assert mode == ACSMode.PASS

    def test_determine_mode_saa(self, mock_config, mock_ephem):
        """Test mode determination in SAA."""
        from conops.acs import ACS
        from conops.constraint import Constraint

        constraint = Constraint(ephem=None)
        constraint.ephem = mock_ephem
        acs = ACS(constraint=constraint, config=mock_config)

        acs.current_slew = None
        acs.saa = Mock()
        acs.saa.insaa = Mock(return_value=True)

        mode = acs.get_mode(1000.0)

        assert mode == ACSMode.SAA

    def test_determine_mode_charging(self, mock_config, mock_ephem, monkeypatch):
        """Test mode determination during emergency charging."""
        from conops.acs import ACS
        from conops.constraint import Constraint

        constraint = Mock(spec=Constraint)
        constraint.ephem = mock_ephem

        acs = ACS(constraint=constraint, config=mock_config)

        # Patch constraint.in_eclipse to avoid eclipse checking
        monkeypatch.setattr(acs.constraint, "in_eclipse", lambda ra, dec, time: False)

        # Create a charging slew
        charging_slew = Mock()
        charging_slew.obstype = "CHARGE"
        charging_slew.is_slewing = Mock(
            return_value=False
        )  # Dwelling, not actively slewing

        acs.current_slew = None  # Not actively slewing
        acs.last_slew = charging_slew  # But last slew was to charge pointing
        acs.saa = None

        mode = acs.get_mode(1000.0)

        assert mode == ACSMode.CHARGING

    def test_determine_mode_science(self, mock_config, mock_ephem):
        """Test mode determination in science mode."""
        from conops.acs import ACS
        from conops.constraint import Constraint

        constraint = Constraint(ephem=None)
        constraint.ephem = mock_ephem
        acs = ACS(constraint=constraint, config=mock_config)

        acs.current_slew = None
        acs.saa = None
        acs.battery_alert = False
        acs.in_emergency_charging = False

        mode = acs.get_mode(1000.0)

        assert mode == ACSMode.SCIENCE


class TestHandlePassMode:
    """Test _handle_pass_mode helper method."""

    def test_handle_pass_terminates_ppt(self, queue_ditl):
        """Test that pass mode terminates active PPT."""
        mock_ppt = Mock()
        mock_ppt.end = 0
        mock_ppt.done = False
        queue_ditl.ppt = mock_ppt

        queue_ditl._handle_pass_mode(1000.0)

        assert mock_ppt.end == 1000.0
        assert mock_ppt.done is True
        assert queue_ditl.ppt is None

    def test_handle_pass_terminates_charging_ppt(self, queue_ditl):
        """Test that pass mode terminates charging PPT."""
        mock_charging = Mock()
        mock_charging.end = 0
        mock_charging.done = False
        queue_ditl.charging_ppt = mock_charging

        queue_ditl._handle_pass_mode(1000.0)

        assert mock_charging.end == 1000.0
        assert mock_charging.done is True
        assert queue_ditl.charging_ppt is None

    def test_handle_pass_no_ppt(self, queue_ditl):
        """Test pass mode when no PPT is active."""
        queue_ditl.ppt = None
        queue_ditl.charging_ppt = None

        # Should not raise any errors
        queue_ditl._handle_pass_mode(1000.0)


class TestHandleChargingMode:
    """Test _handle_charging_mode helper method."""

    def test_charging_ends_when_battery_recharged(self, queue_ditl, capsys):
        """Test charging terminates when battery is recharged."""
        queue_ditl.battery.battery_alert = False
        queue_ditl.battery.battery_level = 0.85
        mock_charging = Mock()
        mock_charging.end = 0
        mock_charging.done = False
        queue_ditl.charging_ppt = mock_charging

        queue_ditl._handle_charging_mode(1000.0)

        assert mock_charging.end == 1000.0
        assert mock_charging.done is True
        assert queue_ditl.charging_ppt is None
        captured = capsys.readouterr()
        assert "Battery recharged" in captured.out

    def test_charging_ends_when_constrained(self, queue_ditl, capsys):
        """Test charging terminates when pointing becomes constrained."""
        queue_ditl.battery.battery_alert = True
        mock_charging = Mock()
        mock_charging.ra = 10.0
        mock_charging.dec = 20.0
        mock_charging.end = 0
        mock_charging.done = False
        queue_ditl.charging_ppt = mock_charging
        queue_ditl.constraint.inoccult = Mock(return_value=True)

        queue_ditl._handle_charging_mode(1000.0)

        assert mock_charging.end == 1000.0
        assert mock_charging.done is True
        assert queue_ditl.charging_ppt is None
        captured = capsys.readouterr()
        assert "Charging pointing constrained" in captured.out

    def test_charging_ends_in_eclipse(self, queue_ditl, capsys):
        """Test charging terminates when entering eclipse."""
        queue_ditl.battery.battery_alert = True
        mock_charging = Mock()
        mock_charging.ra = 10.0
        mock_charging.dec = 20.0
        queue_ditl.charging_ppt = mock_charging
        queue_ditl.emergency_charging._is_in_sunlight = Mock(return_value=False)

        queue_ditl._handle_charging_mode(1000.0)

        assert queue_ditl.charging_ppt is None
        captured = capsys.readouterr()
        assert "Entered eclipse" in captured.out

    def test_charging_continues(self, queue_ditl):
        """Test charging continues when conditions are still valid."""
        queue_ditl.battery.battery_alert = True
        mock_charging = Mock()
        mock_charging.ra = 10.0
        mock_charging.dec = 20.0
        queue_ditl.charging_ppt = mock_charging
        queue_ditl.emergency_charging._is_in_sunlight = Mock(return_value=True)

        queue_ditl._handle_charging_mode(1000.0)

        # Charging should still be active
        assert queue_ditl.charging_ppt is mock_charging


class TestManagePPTLifecycle:
    """Test _manage_ppt_lifecycle helper method."""

    def test_manage_ppt_science_mode_exposure(self, queue_ditl):
        """Test PPT exposure decrements in science mode."""
        mock_ppt = Mock()
        mock_ppt.exptime = 300.0
        mock_ppt.ra = 10.0
        mock_ppt.dec = 20.0
        mock_ppt.end = 2000.0
        queue_ditl.ppt = mock_ppt
        queue_ditl.charging_ppt = None
        queue_ditl.step_size = 60

        queue_ditl._manage_ppt_lifecycle(1000.0, ACSMode.SCIENCE)

        assert mock_ppt.exptime == 240.0
        assert queue_ditl.ppt is mock_ppt

    def test_manage_ppt_slewing_no_exposure(self, queue_ditl):
        """Test PPT exposure does not decrement when slewing."""
        mock_ppt = Mock()
        mock_ppt.exptime = 300.0
        mock_ppt.ra = 10.0
        mock_ppt.dec = 20.0
        mock_ppt.end = 2000.0
        queue_ditl.ppt = mock_ppt
        queue_ditl.charging_ppt = None

        queue_ditl._manage_ppt_lifecycle(1000.0, ACSMode.SLEWING)

        assert mock_ppt.exptime == 300.0

    def test_manage_ppt_becomes_constrained(self, queue_ditl):
        """Test PPT terminates when constrained."""
        mock_ppt = Mock()
        mock_ppt.exptime = 300.0
        mock_ppt.ra = 10.0
        mock_ppt.dec = 20.0
        mock_ppt.end = 2000.0
        queue_ditl.ppt = mock_ppt
        queue_ditl.charging_ppt = None
        queue_ditl.constraint.inoccult = Mock(return_value=True)

        queue_ditl._manage_ppt_lifecycle(1000.0, ACSMode.SCIENCE)

        assert queue_ditl.ppt is None

    def test_manage_ppt_exposure_complete(self, queue_ditl):
        """Test PPT terminates when exposure complete."""
        mock_ppt = Mock()
        mock_ppt.exptime = 30.0
        mock_ppt.ra = 10.0
        mock_ppt.dec = 20.0
        mock_ppt.end = 2000.0
        mock_ppt.done = False
        queue_ditl.ppt = mock_ppt
        queue_ditl.charging_ppt = None
        queue_ditl.step_size = 60

        queue_ditl._manage_ppt_lifecycle(1000.0, ACSMode.SCIENCE)

        assert queue_ditl.ppt is None
        assert mock_ppt.done is True

    def test_manage_ppt_time_window_elapsed(self, queue_ditl):
        """Test PPT terminates when time window elapsed."""
        mock_ppt = Mock()
        mock_ppt.exptime = 300.0
        mock_ppt.ra = 10.0
        mock_ppt.dec = 20.0
        mock_ppt.end = 500.0
        queue_ditl.ppt = mock_ppt
        queue_ditl.charging_ppt = None

        queue_ditl._manage_ppt_lifecycle(1000.0, ACSMode.SCIENCE)

        assert queue_ditl.ppt is None

    def test_manage_ppt_charging_ppt_ignored(self, queue_ditl):
        """Test that charging PPT is not managed by this method."""
        mock_charging = Mock()
        mock_charging.exptime = 300.0
        queue_ditl.ppt = mock_charging
        queue_ditl.charging_ppt = mock_charging

        queue_ditl._manage_ppt_lifecycle(1000.0, ACSMode.SCIENCE)

        # Exptime should not be decremented
        assert mock_charging.exptime == 300.0


class TestFetchNewPPT:
    """Test _fetch_new_ppt helper method."""

    def test_fetch_ppt_success(self, queue_ditl, capsys):
        """Test successfully fetching a new PPT."""
        mock_ppt = Mock()
        mock_ppt.ra = 45.0
        mock_ppt.dec = 30.0
        mock_ppt.obsid = 1001
        queue_ditl.queue.get = Mock(return_value=mock_ppt)

        # Patch the visibility calculation to avoid constraint evaluation
        with (
            patch("conops.pointing.Pointing.visibility") as mock_vis,
            patch("conops.pointing.Pointing.next_vis") as mock_next_vis,
        ):
            mock_vis.return_value = 1  # Return number of windows found
            mock_next_vis.return_value = 1000.0  # Return valid visibility time

            lastra, lastdec = queue_ditl._fetch_new_ppt(1000.0, 10.0, 20.0)

        assert queue_ditl.ppt is mock_ppt
        assert lastra == 45.0
        assert lastdec == 30.0
        # Verify enqueue_command was called with a SLEW_TO_TARGET command
        queue_ditl.acs.enqueue_command.assert_called_once()

        # Verify the command type is SLEW_TO_TARGET
        call_args = queue_ditl.acs.enqueue_command.call_args
        command = call_args[0][0]
        assert command.command_type == ACSCommandType.SLEW_TO_TARGET
        assert command.slew.endra == 45.0
        assert command.slew.enddec == 30.0
        assert command.slew.obsid == 1001

        captured = capsys.readouterr()
        assert "Fetching new PPT from Queue" in captured.out

    def test_fetch_ppt_none_available(self, queue_ditl, capsys):
        """Test when no PPT is available from queue."""
        queue_ditl.queue.get = Mock(return_value=None)

        lastra, lastdec = queue_ditl._fetch_new_ppt(1000.0, 10.0, 20.0)

        assert queue_ditl.ppt is None
        assert lastra == 10.0
        assert lastdec == 20.0
        captured = capsys.readouterr()
        assert "No targets available from Queue" in captured.out


class TestRecordSpacecraftState:
    """Test _record_spacecraft_state helper method."""

    def test_record_state_mode(self, queue_ditl):
        """Test recorded mode equals ACSMode.SCIENCE."""
        queue_ditl.utime = [1000.0, 1060.0, 1120.0]

        queue_ditl._record_pointing_data(
            ra=45.0,
            dec=30.0,
            roll=15.0,
            obsid=1001,
            mode=ACSMode.SCIENCE,
        )

        assert queue_ditl.mode == [ACSMode.SCIENCE]

    def test_record_state_ra(self, queue_ditl):
        """Test recorded RA equals provided value."""
        queue_ditl.utime = [1000.0, 1060.0, 1120.0]

        queue_ditl._record_pointing_data(
            ra=45.0,
            dec=30.0,
            roll=15.0,
            obsid=1001,
            mode=ACSMode.SCIENCE,
        )

        assert queue_ditl.ra == [45.0]

    def test_record_state_dec(self, queue_ditl):
        """Test recorded Dec equals provided value."""
        queue_ditl.utime = [1000.0, 1060.0, 1120.0]

        queue_ditl._record_pointing_data(
            ra=45.0,
            dec=30.0,
            roll=15.0,
            obsid=1001,
            mode=ACSMode.SCIENCE,
        )

        assert queue_ditl.dec == [30.0]

    def test_record_state_roll(self, queue_ditl):
        """Test recorded roll equals provided value."""
        queue_ditl.utime = [1000.0, 1060.0, 1120.0]

        queue_ditl._record_pointing_data(
            ra=45.0,
            dec=30.0,
            roll=15.0,
            obsid=1001,
            mode=ACSMode.SCIENCE,
        )

        assert queue_ditl.roll == [15.0]

    def test_record_state_obsid(self, queue_ditl):
        """Test recorded obsid equals provided value."""
        queue_ditl.utime = [1000.0, 1060.0, 1120.0]

        queue_ditl._record_pointing_data(
            ra=45.0,
            dec=30.0,
            roll=15.0,
            obsid=1001,
            mode=ACSMode.SCIENCE,
        )

        assert queue_ditl.obsid == [1001]

    def test_record_state_panel_length(self, queue_ditl):
        """Test panel list gets one entry recorded."""
        queue_ditl.utime = [1000.0, 1060.0, 1120.0]

        queue_ditl._record_power_data(
            i=0,
            utime=1000.0,
            ra=45.0,
            dec=30.0,
            mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )

        assert len(queue_ditl.panel) == 1

    def test_record_state_power_length(self, queue_ditl):
        """Test power list gets one entry recorded."""
        queue_ditl.utime = [1000.0, 1060.0, 1120.0]

        queue_ditl._record_power_data(
            i=0,
            utime=1000.0,
            ra=45.0,
            dec=30.0,
            mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )

        assert len(queue_ditl.power) == 1

    def test_record_state_panel_power_length(self, queue_ditl):
        """Test panel_power list gets one entry recorded."""
        queue_ditl.utime = [1000.0, 1060.0, 1120.0]

        queue_ditl._record_power_data(
            i=0,
            utime=1000.0,
            ra=45.0,
            dec=30.0,
            mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )

        assert len(queue_ditl.panel_power) == 1

    def test_record_state_batterylevel_length(self, queue_ditl):
        """Test batterylevel list gets one entry recorded."""
        queue_ditl.utime = [1000.0, 1060.0, 1120.0]

        queue_ditl._record_power_data(
            i=0,
            utime=1000.0,
            ra=45.0,
            dec=30.0,
            mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )

        assert len(queue_ditl.batterylevel) == 1

    def test_record_state_spacecraft_power_call(self, queue_ditl):
        """Spacecraft bus power() called with correct args."""
        queue_ditl.utime = [1000.0]
        queue_ditl.spacecraft_bus.power = Mock(return_value=50.0)
        queue_ditl.instruments.power = Mock(return_value=30.0)
        queue_ditl.acs.solar_panel.power = Mock(return_value=100.0)
        queue_ditl.battery.battery_level = 0.75
        queue_ditl.step_size = 60

        queue_ditl._record_power_data(
            i=0,
            utime=1000.0,
            ra=0.0,
            dec=0.0,
            mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )
        queue_ditl.spacecraft_bus.power.assert_called_once_with(
            mode=ACSMode.SCIENCE, in_eclipse=False
        )

    def test_record_state_instruments_power_call(self, queue_ditl):
        """Instruments power() called with correct args."""
        queue_ditl.utime = [1000.0]
        queue_ditl.spacecraft_bus.power = Mock(return_value=50.0)
        queue_ditl.instruments.power = Mock(return_value=30.0)
        queue_ditl.acs.solar_panel.power = Mock(return_value=100.0)
        queue_ditl.battery.battery_level = 0.75
        queue_ditl.step_size = 60

        queue_ditl._record_power_data(
            i=0,
            utime=1000.0,
            ra=0.0,
            dec=0.0,
            mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )

        queue_ditl.instruments.power.assert_called_once_with(
            mode=ACSMode.SCIENCE, in_eclipse=False
        )

    def test_record_state_power_sum(self, queue_ditl):
        """Total power recorded equals sum of bus and instruments."""
        queue_ditl.utime = [1000.0]
        queue_ditl.spacecraft_bus.power = Mock(return_value=50.0)
        queue_ditl.instruments.power = Mock(return_value=30.0)
        queue_ditl.acs.solar_panel.power = Mock(return_value=100.0)
        queue_ditl.battery.battery_level = 0.75
        queue_ditl.step_size = 60

        queue_ditl._record_power_data(
            i=0,
            utime=1000.0,
            ra=0.0,
            dec=0.0,
            mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )

        assert queue_ditl.power == [80.0]  # 50 + 30

    def test_record_state_battery_drain_called(self, queue_ditl):
        """Battery.drain() called with expected power and timestep."""
        queue_ditl.utime = [1000.0]
        queue_ditl.spacecraft_bus.power = Mock(return_value=50.0)
        queue_ditl.instruments.power = Mock(return_value=30.0)
        queue_ditl.acs.solar_panel.power = Mock(return_value=100.0)
        queue_ditl.battery.battery_level = 0.75
        queue_ditl.step_size = 60

        queue_ditl._record_power_data(
            i=0,
            utime=1000.0,
            ra=0.0,
            dec=0.0,
            mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )

        queue_ditl.battery.drain.assert_called_once_with(80.0, 60)

    def test_record_state_battery_charge_called(self, queue_ditl):
        """Battery.charge() called with expected panel power and timestep."""
        queue_ditl.utime = [1000.0]
        queue_ditl.spacecraft_bus.power = Mock(return_value=50.0)
        queue_ditl.instruments.power = Mock(return_value=30.0)
        queue_ditl.acs.solar_panel.power = Mock(return_value=100.0)
        queue_ditl.battery.battery_level = 0.75
        queue_ditl.step_size = 60

        queue_ditl._record_power_data(
            i=0,
            utime=1000.0,
            ra=0.0,
            dec=0.0,
            mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )
        queue_ditl.battery.charge.assert_called_once_with(100.0, 60)


class TestCalcMethod:
    """Test main calc method integration."""

    def test_calc_requires_ephemeris(self, queue_ditl):
        """Test calc raises assertion if ephemeris not set."""
        queue_ditl.ephem = None

        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            queue_ditl.calc()

    def test_calc_basic_success(self, queue_ditl):
        """Test successful calc execution."""
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600  # 1 hour steps for faster test

        result = queue_ditl.calc()

        assert result is True
        assert len(queue_ditl.mode) == 24  # 24 hours
        assert len(queue_ditl.ra) == 24
        assert len(queue_ditl.dec) == 24

    def test_calc_sets_acs_ephemeris(self, queue_ditl):
        """Test that calc sets ACS ephemeris if not already set."""
        queue_ditl.acs.ephem = None
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600

        queue_ditl.calc()

        assert queue_ditl.acs.ephem is queue_ditl.ephem

    def test_calc_tracks_ppt_in_ppst(self, queue_ditl):
        """Test that calc tracks PPTs in ppst plan."""
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600

        # Create a mock PPT that will be returned
        mock_ppt = Mock()
        mock_ppt.ra = 45.0
        mock_ppt.dec = 30.0
        mock_ppt.obsid = 1001
        mock_ppt.exptime = 7200.0
        mock_ppt.begin = 1543622400
        mock_ppt.end = 1543629600
        mock_ppt.done = False

        # Queue returns PPT on first call, None thereafter
        queue_ditl.queue.get = Mock(side_effect=[mock_ppt] + [None] * 100)

        # Patch visibility calculation to avoid constraint evaluation
        with patch("conops.pointing.Pointing.visibility") as mock_vis:
            mock_vis.return_value = 1
            queue_ditl.calc()

        # PPT should be tracked in ppst
        assert len(queue_ditl.ppst) > 0

    def test_calc_handles_pass_mode(self, queue_ditl):
        """Test calc properly handles ground station pass mode."""
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600

        # Mock ACS to return PASS mode from get_mode
        queue_ditl.acs.get_mode = Mock(return_value=ACSMode.PASS)

        result = queue_ditl.calc()

        assert result is True
        # Mode should include PASS
        assert ACSMode.PASS in queue_ditl.mode

    def test_calc_handles_emergency_charging(self, queue_ditl):
        """Test calc initiates emergency charging when battery alert."""
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600

        # Set up battery alert
        queue_ditl.battery.battery_alert = True

        # Create charging PPT
        mock_charging = Mock()
        mock_charging.ra = 100.0
        mock_charging.dec = 50.0
        mock_charging.obsid = 999001

        # Mock should_initiate_charging to return True
        queue_ditl.emergency_charging.should_initiate_charging = Mock(return_value=True)
        # Mock emergency_charging.initiate_emergency_charging to return charging PPT
        queue_ditl.emergency_charging.initiate_emergency_charging = Mock(
            return_value=mock_charging
        )
        # Mock enqueue_command so we can verify it was called
        queue_ditl.acs.enqueue_command = Mock()

        result = queue_ditl.calc()

        assert result is True
        # Should have initiated emergency charging via emergency_charging module
        assert queue_ditl.emergency_charging.initiate_emergency_charging.called
        # Should have sent battery charge command to ACS via enqueue_command
        assert queue_ditl.acs.enqueue_command.called
        # Verify START_BATTERY_CHARGE command was sent
        command_types = [
            call[0][0].command_type.name
            for call in queue_ditl.acs.enqueue_command.call_args_list
        ]
        assert "START_BATTERY_CHARGE" in command_types

    def test_calc_closes_final_ppt(self, queue_ditl):
        """Test that calc closes the final PPT at end of day."""
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600

        # Create a PPT that lasts the whole day
        mock_ppt = Mock()
        mock_ppt.ra = 45.0
        mock_ppt.dec = 30.0
        mock_ppt.obsid = 1001
        mock_ppt.exptime = 86400.0
        mock_ppt.begin = 1543622400
        mock_ppt.end = 1543708800
        mock_ppt.done = False

        queue_ditl.queue.get = Mock(return_value=mock_ppt)

        # Patch visibility calculation to avoid constraint evaluation
        with patch("conops.pointing.Pointing.visibility") as mock_vis:
            mock_vis.return_value = 1
            queue_ditl.calc()

        # Final PPT should have end time set
        if queue_ditl.ppst:
            assert queue_ditl.ppst[-1].end > 0


class TestGetConstraintName:
    """Tests for _get_constraint_name method."""

    def test_get_constraint_name_earth(self, queue_ditl):
        """Returns 'Earth Limb' when in_earth is True."""
        ra, dec, utime = 10.0, 20.0, 1000.0
        queue_ditl.constraint.in_earth = Mock(return_value=True)
        queue_ditl.constraint.in_moon = Mock(return_value=False)
        queue_ditl.constraint.in_sun = Mock(return_value=False)
        queue_ditl.constraint.in_panel = Mock(return_value=False)

        name = queue_ditl._get_constraint_name(ra, dec, utime)

        assert name == "Earth Limb"
        queue_ditl.constraint.in_earth.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_moon.assert_not_called()
        queue_ditl.constraint.in_sun.assert_not_called()
        queue_ditl.constraint.in_panel.assert_not_called()

    def test_get_constraint_name_moon(self, queue_ditl):
        """Returns 'Moon' when in_moon is True and in_earth is False."""
        ra, dec, utime = 11.0, 21.0, 2000.0
        queue_ditl.constraint.in_earth = Mock(return_value=False)
        queue_ditl.constraint.in_moon = Mock(return_value=True)
        queue_ditl.constraint.in_sun = Mock(return_value=False)
        queue_ditl.constraint.in_panel = Mock(return_value=False)

        name = queue_ditl._get_constraint_name(ra, dec, utime)

        assert name == "Moon"
        queue_ditl.constraint.in_earth.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_moon.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_sun.assert_not_called()
        queue_ditl.constraint.in_panel.assert_not_called()

    def test_get_constraint_name_sun(self, queue_ditl):
        """Returns 'Sun' when in_sun is True and earlier checks are False."""
        ra, dec, utime = 12.0, 22.0, 3000.0
        queue_ditl.constraint.in_earth = Mock(return_value=False)
        queue_ditl.constraint.in_moon = Mock(return_value=False)
        queue_ditl.constraint.in_sun = Mock(return_value=True)
        queue_ditl.constraint.in_panel = Mock(return_value=False)

        name = queue_ditl._get_constraint_name(ra, dec, utime)

        assert name == "Sun"
        queue_ditl.constraint.in_earth.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_moon.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_sun.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_panel.assert_not_called()

    def test_get_constraint_name_panel(self, queue_ditl):
        """Returns 'Panel' when in_panel is True and earlier checks are False."""
        ra, dec, utime = 13.0, 23.0, 4000.0
        queue_ditl.constraint.in_earth = Mock(return_value=False)
        queue_ditl.constraint.in_moon = Mock(return_value=False)
        queue_ditl.constraint.in_sun = Mock(return_value=False)
        queue_ditl.constraint.in_panel = Mock(return_value=True)

        name = queue_ditl._get_constraint_name(ra, dec, utime)

        assert name == "Panel"
        queue_ditl.constraint.in_earth.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_moon.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_sun.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_panel.assert_called_once_with(ra, dec, utime)

    def test_get_constraint_name_unknown(self, queue_ditl):
        """Returns 'Unknown' when no constraints are triggered."""
        ra, dec, utime = 14.0, 24.0, 5000.0
        queue_ditl.constraint.in_earth = Mock(return_value=False)
        queue_ditl.constraint.in_moon = Mock(return_value=False)
        queue_ditl.constraint.in_sun = Mock(return_value=False)
        queue_ditl.constraint.in_panel = Mock(return_value=False)

        name = queue_ditl._get_constraint_name(ra, dec, utime)

        assert name == "Unknown"
        queue_ditl.constraint.in_earth.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_moon.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_sun.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_panel.assert_called_once_with(ra, dec, utime)

    def test_get_constraint_name_precedence(self, queue_ditl):
        """Earth Limb should take precedence over other constraints if multiple True."""
        ra, dec, utime = 15.0, 25.0, 6000.0
        queue_ditl.constraint.in_earth = Mock(return_value=True)
        queue_ditl.constraint.in_moon = Mock(return_value=True)
        queue_ditl.constraint.in_sun = Mock(return_value=True)
        queue_ditl.constraint.in_panel = Mock(return_value=True)

        name = queue_ditl._get_constraint_name(ra, dec, utime)

        assert name == "Earth Limb"
        queue_ditl.constraint.in_earth.assert_called_once_with(ra, dec, utime)
        # Ensure other checks are not required to be called but if they are,
        # precedence will still yield Earth Limb result.


class TestCheckAndManagePasses:
    """Tests for _check_and_manage_passes helper method."""

    def test_check_and_manage_passes_end_pass_calls_check_pass_timing(self, queue_ditl):
        """END_PASS should call passrequests.check_pass_timing with expected args."""
        utime = 1000.0
        ra, dec = 10.0, 20.0

        queue_ditl.acs.passrequests.check_pass_timing = Mock(
            return_value={"start_pass": None, "end_pass": True, "updated_pass": None}
        )

        queue_ditl._check_and_manage_passes(utime, ra, dec)

        queue_ditl.acs.passrequests.check_pass_timing.assert_called_once_with(
            utime, ra, dec, queue_ditl.step_size
        )

    def test_check_and_manage_passes_end_pass_enqueues_command(self, queue_ditl):
        """END_PASS should enqueue an ACS command."""
        utime = 1000.0
        ra, dec = 10.0, 20.0

        queue_ditl.acs.passrequests.check_pass_timing = Mock(
            return_value={"start_pass": None, "end_pass": True, "updated_pass": None}
        )

        queue_ditl._check_and_manage_passes(utime, ra, dec)

        queue_ditl.acs.enqueue_command.assert_called_once()

    def test_check_and_manage_passes_end_pass_command_type(self, queue_ditl):
        """Enqueued END_PASS command should have correct type."""
        utime = 1000.0
        ra, dec = 10.0, 20.0

        queue_ditl.acs.passrequests.check_pass_timing = Mock(
            return_value={"start_pass": None, "end_pass": True, "updated_pass": None}
        )

        queue_ditl._check_and_manage_passes(utime, ra, dec)

        cmd = queue_ditl.acs.enqueue_command.call_args[0][0]
        assert cmd.command_type == ACSCommandType.END_PASS

    def test_check_and_manage_passes_end_pass_command_execution_time(self, queue_ditl):
        """END_PASS command execution_time should match current utime."""
        utime = 1000.0
        ra, dec = 10.0, 20.0

        queue_ditl.acs.passrequests.check_pass_timing = Mock(
            return_value={"start_pass": None, "end_pass": True, "updated_pass": None}
        )

        queue_ditl._check_and_manage_passes(utime, ra, dec)

        cmd = queue_ditl.acs.enqueue_command.call_args[0][0]
        assert cmd.execution_time == utime

    def test_check_and_manage_passes_start_pass_calls_check_pass_timing(
        self, queue_ditl
    ):
        """START_PASS should call passrequests.check_pass_timing with expected args."""
        utime = 1000.0
        ra, dec = 10.0, 20.0

        pass_obj = Pass(station="GS_STATION", begin=950.0, slewrequired=900.0)

        queue_ditl.acs.passrequests.check_pass_timing = Mock(
            return_value={
                "start_pass": pass_obj,
                "end_pass": False,
                "updated_pass": None,
            }
        )

        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        queue_ditl.acs.last_ppt = Mock(obsid=1234)

        queue_ditl._check_and_manage_passes(utime, ra, dec)

        queue_ditl.acs.passrequests.check_pass_timing.assert_called_once_with(
            utime, ra, dec, queue_ditl.step_size
        )

    def test_check_and_manage_passes_start_pass_enqueues_command(self, queue_ditl):
        """When START_PASS is scheduled in SCIENCE mode, an enqueue should occur."""
        utime = 1000.0
        ra, dec = 10.0, 20.0

        pass_obj = Pass(station="GS_STATION", begin=950.0, slewrequired=900.0)

        queue_ditl.acs.passrequests.check_pass_timing = Mock(
            return_value={
                "start_pass": pass_obj,
                "end_pass": False,
                "updated_pass": None,
            }
        )

        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        queue_ditl.acs.last_ppt = Mock(obsid=1234)

        queue_ditl._check_and_manage_passes(utime, ra, dec)

        queue_ditl.acs.enqueue_command.assert_called_once()

    def test_check_and_manage_passes_start_pass_command_type_and_exec_time(
        self, queue_ditl
    ):
        """START_PASS command should have correct type and execution_time."""
        utime = 1000.0
        ra, dec = 10.0, 20.0

        pass_obj = Pass(station="GS_STATION", begin=950.0, slewrequired=900.0)

        queue_ditl.acs.passrequests.check_pass_timing = Mock(
            return_value={
                "start_pass": pass_obj,
                "end_pass": False,
                "updated_pass": None,
            }
        )

        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        queue_ditl.acs.last_ppt = Mock(obsid=1234)

        queue_ditl._check_and_manage_passes(utime, ra, dec)

        cmd = queue_ditl.acs.enqueue_command.call_args[0][0]
        assert cmd.command_type == ACSCommandType.START_PASS
        assert cmd.execution_time == pass_obj.slewrequired

    def test_check_and_manage_passes_start_pass_sets_obsid(self, queue_ditl):
        """START_PASS should set pass.obsid from last_ppt.obsid."""
        utime = 1000.0
        ra, dec = 10.0, 20.0

        pass_obj = Pass(station="GS_STATION", begin=950.0, slewrequired=900.0)

        queue_ditl.acs.passrequests.check_pass_timing = Mock(
            return_value={
                "start_pass": pass_obj,
                "end_pass": False,
                "updated_pass": None,
            }
        )

        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        queue_ditl.acs.last_ppt = Mock(obsid=1234)

        queue_ditl._check_and_manage_passes(utime, ra, dec)

        assert pass_obj.obsid == 1234

    def test_check_and_manage_passes_start_pass_command_slew_station(self, queue_ditl):
        """The START_PASS command should include a copied slew/pass with station preserved."""
        utime = 1000.0
        ra, dec = 10.0, 20.0

        pass_obj = Pass(station="GS_STATION", begin=950.0, slewrequired=900.0)

        queue_ditl.acs.passrequests.check_pass_timing = Mock(
            return_value={
                "start_pass": pass_obj,
                "end_pass": False,
                "updated_pass": None,
            }
        )

        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        queue_ditl.acs.last_ppt = Mock(obsid=1234)

        queue_ditl._check_and_manage_passes(utime, ra, dec)

        cmd = queue_ditl.acs.enqueue_command.call_args[0][0]
        assert getattr(cmd.slew, "station", None) == pass_obj.station

    def test_check_and_manage_passes_start_pass_not_enqueued_when_not_science_calls_check(
        self, queue_ditl
    ):
        """If ACS mode is not SCIENCE, check_pass_timing still called but no command enqueued."""
        utime = 2000.0
        ra, dec = 30.0, 40.0

        pass_obj = Pass(station="GS2", begin=1850.0, slewrequired=1800.0)

        queue_ditl.acs.passrequests.check_pass_timing = Mock(
            return_value={
                "start_pass": pass_obj,
                "end_pass": False,
                "updated_pass": None,
            }
        )

        queue_ditl.acs.acsmode = ACSMode.SAA

        queue_ditl._check_and_manage_passes(utime, ra, dec)

        queue_ditl.acs.passrequests.check_pass_timing.assert_called_once_with(
            utime, ra, dec, queue_ditl.step_size
        )
        queue_ditl.acs.enqueue_command.assert_not_called()

    def test_check_and_manage_passes_both_end_and_start_calls_check_pass_timing(
        self, queue_ditl
    ):
        """Both end and start should still call check_pass_timing."""
        utime = 3000.0
        ra, dec = 0.0, 0.0

        pass_obj = Pass(station="GS_ORDER", begin=2950.0, slewrequired=2900.0)

        queue_ditl.acs.passrequests.check_pass_timing = Mock(
            return_value={
                "start_pass": pass_obj,
                "end_pass": True,
                "updated_pass": None,
            }
        )
        queue_ditl.acs.acsmode = ACSMode.SLEWING
        queue_ditl.acs.last_ppt = Mock(obsid=0xBEEF)

        queue_ditl._check_and_manage_passes(utime, ra, dec)

        queue_ditl.acs.passrequests.check_pass_timing.assert_called_once_with(
            utime, ra, dec, queue_ditl.step_size
        )

    def test_check_and_manage_passes_both_end_and_start_enqueues_two_commands(
        self, queue_ditl
    ):
        """When both end and start requested, two commands should be enqueued."""
        utime = 3000.0
        ra, dec = 0.0, 0.0

        pass_obj = Pass(station="GS_ORDER", begin=2950.0, slewrequired=2900.0)

        queue_ditl.acs.passrequests.check_pass_timing = Mock(
            return_value={
                "start_pass": pass_obj,
                "end_pass": True,
                "updated_pass": None,
            }
        )
        queue_ditl.acs.acsmode = ACSMode.SLEWING
        queue_ditl.acs.last_ppt = Mock(obsid=0xBEEF)

        queue_ditl._check_and_manage_passes(utime, ra, dec)

        assert len(queue_ditl.acs.enqueue_command.call_args_list) == 2

    def test_check_and_manage_passes_both_end_and_start_command_order(self, queue_ditl):
        """The order should be END_PASS first then START_PASS."""
        utime = 3000.0
        ra, dec = 0.0, 0.0

        pass_obj = Pass(station="GS_ORDER", begin=2950.0, slewrequired=2900.0)

        queue_ditl.acs.passrequests.check_pass_timing = Mock(
            return_value={
                "start_pass": pass_obj,
                "end_pass": True,
                "updated_pass": None,
            }
        )
        queue_ditl.acs.acsmode = ACSMode.SLEWING
        queue_ditl.acs.last_ppt = Mock(obsid=0xBEEF)

        queue_ditl._check_and_manage_passes(utime, ra, dec)

        types = [
            call_args[0][0].command_type.name
            for call_args in queue_ditl.acs.enqueue_command.call_args_list
        ]
        assert types == ["END_PASS", "START_PASS"]

    def test_check_and_manage_passes_both_end_and_start_start_command_exec_time_and_slew(
        self, queue_ditl
    ):
        """START_PASS from both-end-start case should have correct execution_time and station."""
        utime = 3000.0
        ra, dec = 0.0, 0.0

        pass_obj = Pass(station="GS_ORDER", begin=2950.0, slewrequired=2900.0)

        queue_ditl.acs.passrequests.check_pass_timing = Mock(
            return_value={
                "start_pass": pass_obj,
                "end_pass": True,
                "updated_pass": None,
            }
        )
        queue_ditl.acs.acsmode = ACSMode.SLEWING
        queue_ditl.acs.last_ppt = Mock(obsid=0xBEEF)

        queue_ditl._check_and_manage_passes(utime, ra, dec)

        cmd_start = queue_ditl.acs.enqueue_command.call_args_list[1][0][0]
        assert cmd_start.command_type == ACSCommandType.START_PASS
        assert cmd_start.execution_time == pass_obj.slewrequired
        assert getattr(cmd_start.slew, "station", None) == pass_obj.station

    def test_check_and_manage_passes_both_end_and_start_sets_obsid_from_last_ppt(
        self, queue_ditl
    ):
        """When both end and start requested, the new pass obsid should be set from last_ppt."""
        utime = 3000.0
        ra, dec = 0.0, 0.0

        pass_obj = Pass(station="GS_ORDER", begin=2950.0, slewrequired=2900.0)

        queue_ditl.acs.passrequests.check_pass_timing = Mock(
            return_value={
                "start_pass": pass_obj,
                "end_pass": True,
                "updated_pass": None,
            }
        )
        queue_ditl.acs.acsmode = ACSMode.SLEWING
        queue_ditl.acs.last_ppt = Mock(obsid=0xBEEF)

        queue_ditl._check_and_manage_passes(utime, ra, dec)

        assert pass_obj.obsid == 0xBEEF
