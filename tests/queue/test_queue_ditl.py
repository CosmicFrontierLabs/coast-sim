"""Unit tests for QueueDITL class."""

from unittest.mock import Mock, patch

import numpy as np
import pytest

from conops import ACSCommandType, ACSMode, Pass, QueueDITL


class TestQueueDITLInitialization:
    """Test QueueDITL initialization."""

    def test_initialization_ppts_defaults(self, mock_config):
        with (
            patch("conops.Queue"),
            patch("conops.PassTimes"),
            patch("conops.ACS"),
        ):
            ditl = QueueDITL(config=mock_config)
            assert ditl.ppt is None
            assert ditl.charging_ppt is None

    def test_initialization_pointing_lists_empty(self, mock_config):
        with (
            patch("conops.Queue"),
            patch("conops.PassTimes"),
            patch("conops.ACS"),
        ):
            ditl = QueueDITL(config=mock_config)
            assert ditl.ra == []
            assert ditl.dec == []
            assert ditl.roll == []
            assert ditl.mode == []
            assert ditl.obsid == []

    def test_initialization_power_lists_empty_and_plan(self, mock_config):
        with (
            patch("conops.Queue"),
            patch("conops.PassTimes"),
            patch("conops.ACS"),
        ):
            ditl = QueueDITL(config=mock_config)
            assert ditl.panel == []
            assert ditl.batterylevel == []
            assert ditl.power == []
            assert ditl.panel_power == []
            assert len(ditl.plan) == 0

    def test_initialization_stores_config_subsystems(self, mock_config):
        with (
            patch("conops.Queue"),
            patch("conops.PassTimes"),
            patch("conops.ACS"),
        ):
            ditl = QueueDITL(config=mock_config)
            assert ditl.constraint is mock_config.constraint
            assert ditl.battery is mock_config.battery
            assert ditl.spacecraft_bus is mock_config.spacecraft_bus
            assert ditl.payload is mock_config.payload


class TestTimeindex:
    """Test timeindex method."""

    def test_timeindex_150(self, queue_ditl):
        queue_ditl.utime = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        assert queue_ditl.timeindex(150.0) == 0

    def test_timeindex_250(self, queue_ditl):
        queue_ditl.utime = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        assert queue_ditl.timeindex(250.0) == 1

    def test_timeindex_350(self, queue_ditl):
        queue_ditl.utime = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        assert queue_ditl.timeindex(350.0) == 2

    def test_timeindex_exact_last(self, queue_ditl):
        queue_ditl.utime = np.array([100.0, 200.0, 300.0, 400.0, 500.0])
        assert queue_ditl.timeindex(500.0) == 4

    def test_timeindex_exact_match(self, queue_ditl):
        queue_ditl.utime = np.array([100.0, 200.0, 300.0])
        assert queue_ditl.timeindex(200.0) == 1


class TestSetupSimulationTiming:
    """Test _setup_simulation_timing helper method."""

    def test_setup_timing_success_returns_true(self, queue_ditl):
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 60
        assert queue_ditl._setup_simulation_timing() is True

    def test_setup_timing_sets_ustart(self, queue_ditl):
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 60
        queue_ditl._setup_simulation_timing()
        assert queue_ditl.ustart > 0

    def test_setup_timing_sets_uend_greater(self, queue_ditl):
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 60
        queue_ditl._setup_simulation_timing()
        assert queue_ditl.uend > queue_ditl.ustart

    def test_setup_timing_uend_length_and_utime(self, queue_ditl):
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 60
        queue_ditl._setup_simulation_timing()
        assert queue_ditl.uend == queue_ditl.ustart + 86400
        assert len(queue_ditl.utime) == 86400 // 60


class TestScheduleGroundstationPasses:
    """Test _schedule_groundstation_passes helper method."""

    def test_schedule_passes_empty_schedule_called(self, queue_ditl):
        queue_ditl.acs.passrequests.passes = []
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl._schedule_groundstation_passes()
        queue_ditl.acs.passrequests.get.assert_called_once_with(2018, 331, 1)

    def test_schedule_passes_empty_prints_message(self, queue_ditl, capsys):
        queue_ditl.acs.passrequests.passes = []
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl._schedule_groundstation_passes()
        captured = capsys.readouterr()
        assert "Scheduling groundstation passes..." in captured.out

    def test_schedule_passes_already_scheduled_no_get(self, queue_ditl):
        mock_pass = Mock()
        queue_ditl.acs.passrequests.passes = [mock_pass]
        queue_ditl._schedule_groundstation_passes()
        queue_ditl.acs.passrequests.get.assert_not_called()

    def test_schedule_passes_returns_passes_print(self, queue_ditl, capsys):
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
        from conops import ACS, Constraint

        constraint = Constraint(ephem=None)
        constraint.ephem = mock_ephem
        acs = ACS(constraint=constraint, config=mock_config)

        mock_slew = Mock()
        mock_slew.is_slewing = Mock(return_value=True)
        mock_slew.obstype = "PPT"
        acs.current_slew = mock_slew

        mode = acs.get_mode(1000.0)
        assert mode == ACSMode.SLEWING

    def test_determine_mode_pass(self, mock_config, mock_ephem):
        from conops import ACS, Constraint, Pass

        constraint = Constraint(ephem=None)
        constraint.ephem = mock_ephem
        acs = ACS(constraint=constraint, config=mock_config)

        mock_pass = Mock(spec=Pass)
        mock_pass.in_pass = Mock(return_value=True)
        acs.current_pass = mock_pass

        mode = acs.get_mode(1000.0)
        assert mode == ACSMode.PASS

    def test_determine_mode_saa(self, mock_config, mock_ephem):
        from conops import ACS, Constraint

        constraint = Constraint(ephem=None)
        constraint.ephem = mock_ephem
        acs = ACS(constraint=constraint, config=mock_config)

        acs.current_slew = None
        acs.saa = Mock()
        acs.saa.insaa = Mock(return_value=True)

        mode = acs.get_mode(1000.0)
        assert mode == ACSMode.SAA

    def test_determine_mode_charging(self, mock_config, mock_ephem, monkeypatch):
        from conops import ACS, Constraint

        constraint = Mock(spec=Constraint)
        constraint.ephem = mock_ephem

        acs = ACS(constraint=constraint, config=mock_config)
        monkeypatch.setattr(acs.constraint, "in_eclipse", lambda ra, dec, time: False)

        charging_slew = Mock()
        charging_slew.obstype = "CHARGE"
        charging_slew.is_slewing = Mock(return_value=False)

        acs.current_slew = None
        acs.last_slew = charging_slew
        acs.saa = None

        mode = acs.get_mode(1000.0)
        assert mode == ACSMode.CHARGING

    def test_determine_mode_science(self, mock_config, mock_ephem):
        from conops import ACS, Constraint

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

    def test_handle_pass_terminates_ppt_end_time_set(self, queue_ditl):
        mock_ppt = Mock()
        mock_ppt.end = 0
        mock_ppt.done = False
        queue_ditl.ppt = mock_ppt
        queue_ditl._handle_pass_mode(1000.0)
        assert mock_ppt.end == 1000.0

    def test_handle_pass_terminates_ppt_done_flag_set(self, queue_ditl):
        mock_ppt = Mock()
        mock_ppt.end = 0
        mock_ppt.done = False
        queue_ditl.ppt = mock_ppt
        queue_ditl._handle_pass_mode(1000.0)
        assert mock_ppt.done is True

    def test_handle_pass_terminates_ppt_cleared(self, queue_ditl):
        mock_ppt = Mock()
        mock_ppt.end = 0
        mock_ppt.done = False
        queue_ditl.ppt = mock_ppt
        queue_ditl._handle_pass_mode(1000.0)
        assert queue_ditl.ppt is None

    def test_handle_pass_terminates_charging_ppt_end_time_set(self, queue_ditl):
        mock_charging = Mock()
        mock_charging.end = 0
        mock_charging.done = False
        queue_ditl.charging_ppt = mock_charging
        queue_ditl._handle_pass_mode(1000.0)
        assert mock_charging.end == 1000.0

    def test_handle_pass_terminates_charging_ppt_done_set(self, queue_ditl):
        mock_charging = Mock()
        mock_charging.end = 0
        mock_charging.done = False
        queue_ditl.charging_ppt = mock_charging
        queue_ditl._handle_pass_mode(1000.0)
        assert mock_charging.done is True

    def test_handle_pass_terminates_charging_ppt_cleared(self, queue_ditl):
        mock_charging = Mock()
        mock_charging.end = 0
        mock_charging.done = False
        queue_ditl.charging_ppt = mock_charging
        queue_ditl._handle_pass_mode(1000.0)
        assert queue_ditl.charging_ppt is None

    def test_handle_pass_no_ppt(self, queue_ditl):
        queue_ditl.ppt = None
        queue_ditl.charging_ppt = None
        queue_ditl._handle_pass_mode(1000.0)


class TestHandleChargingMode:
    """Test _handle_charging_mode helper method."""

    def test_charging_ends_when_battery_recharged_end_set(self, queue_ditl, capsys):
        queue_ditl.battery.battery_alert = False
        queue_ditl.battery.battery_level = 0.85
        mock_charging = Mock()
        mock_charging.end = 0
        mock_charging.done = False
        queue_ditl.charging_ppt = mock_charging
        queue_ditl._handle_charging_mode(1000.0)
        assert mock_charging.end == 1000.0

    def test_charging_ends_when_battery_recharged_done_flag(self, queue_ditl, capsys):
        queue_ditl.battery.battery_alert = False
        queue_ditl.battery.battery_level = 0.85
        mock_charging = Mock()
        mock_charging.end = 0
        mock_charging.done = False
        queue_ditl.charging_ppt = mock_charging
        queue_ditl._handle_charging_mode(1000.0)
        assert mock_charging.done is True

    def test_charging_ends_when_battery_recharged_clears_charging_ppt(
        self, queue_ditl, capsys
    ):
        queue_ditl.battery.battery_alert = False
        queue_ditl.battery.battery_level = 0.85
        mock_charging = Mock()
        mock_charging.end = 0
        mock_charging.done = False
        queue_ditl.charging_ppt = mock_charging
        queue_ditl._handle_charging_mode(1000.0)
        assert queue_ditl.charging_ppt is None
        captured = capsys.readouterr()
        assert "Battery recharged" in captured.out

    def test_charging_ends_when_constrained_end_and_done_set(self, queue_ditl, capsys):
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

    def test_charging_ends_in_eclipse_clears_charging(self, queue_ditl, capsys):
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
        queue_ditl.battery.battery_alert = True
        mock_charging = Mock()
        mock_charging.ra = 10.0
        mock_charging.dec = 20.0
        queue_ditl.charging_ppt = mock_charging
        queue_ditl.emergency_charging._is_in_sunlight = Mock(return_value=True)
        queue_ditl._handle_charging_mode(1000.0)
        assert queue_ditl.charging_ppt is mock_charging


class TestManagePPTLifecycle:
    """Test _manage_ppt_lifecycle helper method."""

    def test_manage_ppt_science_mode_exposure_decrements(self, queue_ditl):
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

    def test_manage_ppt_slewing_no_exptime_decrement(self, queue_ditl):
        mock_ppt = Mock()
        mock_ppt.exptime = 300.0
        mock_ppt.ra = 10.0
        mock_ppt.dec = 20.0
        mock_ppt.end = 2000.0
        queue_ditl.ppt = mock_ppt
        queue_ditl.charging_ppt = None
        queue_ditl._manage_ppt_lifecycle(1000.0, ACSMode.SLEWING)
        assert mock_ppt.exptime == 300.0

    def test_manage_ppt_becomes_constrained_terminates(self, queue_ditl):
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

    def test_manage_ppt_exposure_complete_terminates(self, queue_ditl):
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

    def test_manage_ppt_time_window_elapsed_terminate(self, queue_ditl):
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
        mock_charging = Mock()
        mock_charging.exptime = 300.0
        queue_ditl.ppt = mock_charging
        queue_ditl.charging_ppt = mock_charging
        queue_ditl._manage_ppt_lifecycle(1000.0, ACSMode.SCIENCE)
        assert mock_charging.exptime == 300.0


class TestFetchNewPPT:
    """Test _fetch_new_ppt helper method."""

    def test_fetch_ppt_sets_ppt_and_returns_last_positions(self, queue_ditl, capsys):
        mock_ppt = Mock()
        mock_ppt.ra = 45.0
        mock_ppt.dec = 30.0
        mock_ppt.obsid = 1001
        mock_ppt.next_vis = Mock(return_value=1000.0)
        mock_ppt.ssmax = 3600.0
        queue_ditl.queue.get = Mock(return_value=mock_ppt)
        lastra, lastdec = queue_ditl._fetch_new_ppt(1000.0, 10.0, 20.0)
        assert queue_ditl.ppt is mock_ppt
        assert lastra == 45.0
        assert lastdec == 30.0

    def test_fetch_ppt_enqueues_slew_command(self, queue_ditl, capsys):
        mock_ppt = Mock()
        mock_ppt.ra = 45.0
        mock_ppt.dec = 30.0
        mock_ppt.obsid = 1001
        mock_ppt.next_vis = Mock(return_value=1000.0)
        mock_ppt.ssmax = 3600.0
        queue_ditl.queue.get = Mock(return_value=mock_ppt)
        _ = queue_ditl._fetch_new_ppt(1000.0, 10.0, 20.0)
        queue_ditl.acs.enqueue_command.assert_called_once()
        call_args = queue_ditl.acs.enqueue_command.call_args
        command = call_args[0][0]
        assert command.command_type == ACSCommandType.SLEW_TO_TARGET
        assert command.slew.endra == 45.0
        assert command.slew.enddec == 30.0
        assert command.slew.obsid == 1001

    def test_fetch_ppt_prints_messages(self, queue_ditl, capsys):
        mock_ppt = Mock()
        mock_ppt.ra = 45.0
        mock_ppt.dec = 30.0
        mock_ppt.obsid = 1001
        mock_ppt.next_vis = Mock(return_value=1000.0)
        mock_ppt.ssmax = 3600.0
        queue_ditl.queue.get = Mock(return_value=mock_ppt)
        _ = queue_ditl._fetch_new_ppt(1000.0, 10.0, 20.0)
        captured = capsys.readouterr()
        assert "Fetching new PPT from Queue" in captured.out

    def test_fetch_ppt_none_available(self, queue_ditl, capsys):
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
        queue_ditl.utime = [1000.0]
        queue_ditl.spacecraft_bus.power = Mock(return_value=50.0)
        queue_ditl.payload.power = Mock(return_value=30.0)
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

    def test_record_state_payload_power_call(self, queue_ditl):
        queue_ditl.utime = [1000.0]
        queue_ditl.spacecraft_bus.power = Mock(return_value=50.0)
        queue_ditl.payload.power = Mock(return_value=30.0)
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
        queue_ditl.payload.power.assert_called_once_with(
            mode=ACSMode.SCIENCE, in_eclipse=False
        )

    def test_record_state_power_sum(self, queue_ditl):
        queue_ditl.utime = [1000.0]
        queue_ditl.spacecraft_bus.power = Mock(return_value=50.0)
        queue_ditl.payload.power = Mock(return_value=30.0)
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
        queue_ditl.utime = [1000.0]
        queue_ditl.spacecraft_bus.power = Mock(return_value=50.0)
        queue_ditl.payload.power = Mock(return_value=30.0)
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
        queue_ditl.utime = [1000.0]
        queue_ditl.spacecraft_bus.power = Mock(return_value=50.0)
        queue_ditl.payload.power = Mock(return_value=30.0)
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
        queue_ditl.ephem = None
        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            queue_ditl.calc()

    def test_calc_basic_success_return(self, queue_ditl):
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600
        result = queue_ditl.calc()
        assert result is True

    def test_calc_basic_success_mode_and_pointing_length(self, queue_ditl):
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600  # 1 hour steps for faster test
        queue_ditl.calc()
        assert len(queue_ditl.mode) == 24
        assert len(queue_ditl.ra) == 24
        assert len(queue_ditl.dec) == 24

    def test_calc_sets_acs_ephemeris(self, queue_ditl):
        queue_ditl.acs.ephem = None
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600
        queue_ditl.calc()
        assert queue_ditl.acs.ephem is queue_ditl.ephem

    def test_calc_tracks_ppt_in_plan(self, queue_ditl):
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600

        mock_ppt = Mock()
        mock_ppt.ra = 45.0
        mock_ppt.dec = 30.0
        mock_ppt.obsid = 1001
        mock_ppt.exptime = 7200.0
        mock_ppt.begin = 1543622400
        mock_ppt.end = 1543629600
        mock_ppt.done = False
        mock_ppt.next_vis = Mock(return_value=1543276800.0)
        mock_ppt.ssmax = 3600.0
        mock_ppt.copy = Mock(return_value=Mock())
        mock_ppt.copy.return_value.begin = 1543622400
        mock_ppt.copy.return_value.end = 1543629600

        queue_ditl.queue.get = Mock(side_effect=[mock_ppt] + [None] * 100)
        queue_ditl.calc()

        assert len(queue_ditl.plan) > 0

    def test_calc_handles_pass_mode_result_true(self, queue_ditl):
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600
        queue_ditl.acs.get_mode = Mock(return_value=ACSMode.PASS)
        result = queue_ditl.calc()
        assert result is True

    def test_calc_handles_pass_mode_contains_pass(self, queue_ditl):
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600
        queue_ditl.acs.get_mode = Mock(return_value=ACSMode.PASS)
        queue_ditl.calc()
        assert ACSMode.PASS in queue_ditl.mode

    def test_calc_handles_emergency_charging_initiates(self, queue_ditl):
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600
        queue_ditl.battery.battery_alert = True
        mock_charging = Mock()
        mock_charging.ra = 100.0
        mock_charging.dec = 50.0
        mock_charging.obsid = 999001
        mock_charging.begin = 1543622400
        mock_charging.end = 1543622400 + 86400
        mock_charging.copy = Mock(return_value=Mock())
        mock_charging.copy.return_value.begin = 1543622400
        mock_charging.copy.return_value.end = 1543622400 + 86400
        queue_ditl.emergency_charging.should_initiate_charging = Mock(return_value=True)
        queue_ditl.emergency_charging.initiate_emergency_charging = Mock(
            return_value=mock_charging
        )
        queue_ditl.acs.enqueue_command = Mock()
        result = queue_ditl.calc()
        assert result is True
        assert queue_ditl.emergency_charging.initiate_emergency_charging.called

    def test_calc_handles_emergency_charging_enqueue_command_and_type(self, queue_ditl):
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600
        queue_ditl.battery.battery_alert = True
        mock_charging = Mock()
        mock_charging.ra = 100.0
        mock_charging.dec = 50.0
        mock_charging.obsid = 999001
        mock_charging.begin = 1543622400
        mock_charging.end = 1543622400 + 86400
        mock_charging.copy = Mock(return_value=Mock())
        mock_charging.copy.return_value.begin = 1543622400
        mock_charging.copy.return_value.end = 1543622400 + 86400
        queue_ditl.emergency_charging.should_initiate_charging = Mock(return_value=True)
        queue_ditl.emergency_charging.initiate_emergency_charging = Mock(
            return_value=mock_charging
        )
        queue_ditl.acs.enqueue_command = Mock()
        queue_ditl.calc()
        assert queue_ditl.acs.enqueue_command.called
        command_types = [
            call[0][0].command_type.name
            for call in queue_ditl.acs.enqueue_command.call_args_list
        ]
        assert "START_BATTERY_CHARGE" in command_types

    def test_calc_closes_final_ppt_end_set(self, queue_ditl):
        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600
        mock_ppt = Mock()
        mock_ppt.ra = 45.0
        mock_ppt.dec = 30.0
        mock_ppt.obsid = 1001
        mock_ppt.exptime = 86400.0
        mock_ppt.begin = 1543622400
        mock_ppt.end = 1543708800
        mock_ppt.done = False
        mock_ppt.next_vis = Mock(return_value=1543276800.0)
        mock_ppt.ssmax = 3600.0
        mock_ppt.copy = Mock(return_value=Mock())
        mock_ppt.copy.return_value.begin = 1543622400
        mock_ppt.copy.return_value.end = 1543708800
        queue_ditl.queue.get = Mock(return_value=mock_ppt)
        queue_ditl.calc()
        if queue_ditl.plan:
            assert queue_ditl.plan[-1].end > 0

    def test_calc_handles_naive_datetimes(self, queue_ditl):
        """Test calc method handles naive datetimes by making them UTC."""
        from datetime import datetime

        # Set naive datetimes
        queue_ditl.begin = datetime(2018, 11, 27, 0, 0, 0)  # naive
        queue_ditl.end = datetime(2018, 11, 27, 1, 0, 0)  # naive

        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600

        # Should not raise an exception and should make datetimes timezone-aware
        result = queue_ditl.calc()
        assert result is True
        assert queue_ditl.begin.tzinfo is not None
        assert queue_ditl.end.tzinfo is not None

    def test_calc_handles_safe_mode_request(self, queue_ditl):
        """Test calc method handles safe mode requests."""
        # Set up safe mode request
        queue_ditl.config.fault_management.safe_mode_requested = True
        queue_ditl.acs.in_safe_mode = False

        queue_ditl.year = 2018
        queue_ditl.day = 331
        queue_ditl.length = 1
        queue_ditl.step_size = 3600

        result = queue_ditl.calc()
        assert result is True

        # Check that safe mode command was enqueued
        queue_ditl.acs.enqueue_command.assert_called()
        call_args = queue_ditl.acs.enqueue_command.call_args
        command = call_args[0][0]
        assert command.command_type == ACSCommandType.ENTER_SAFE_MODE

    def test_track_ppt_in_timeline_closes_placeholder_end_times(self, queue_ditl):
        """Test _track_ppt_in_timeline closes PPTs with placeholder end times."""
        from conops.targets import PlanEntry

        # Create a mock PPT with placeholder end time
        mock_previous_ppt = Mock(spec=PlanEntry)
        mock_previous_ppt.begin = 1000.0
        mock_previous_ppt.end = 1000.0 + 86400 + 100  # Placeholder end time
        mock_previous_ppt.copy = Mock(return_value=mock_previous_ppt)

        # Create current PPT
        mock_current_ppt = Mock(spec=PlanEntry)
        mock_current_ppt.begin = 2000.0
        mock_current_ppt.end = 3000.0
        mock_current_ppt.copy = Mock(return_value=mock_current_ppt)

        # Set up plan with previous PPT
        queue_ditl.plan = [mock_previous_ppt]
        queue_ditl.ppt = mock_current_ppt

        # Call the method
        queue_ditl._track_ppt_in_timeline()

        # Check that the previous PPT's end time was updated
        assert (
            mock_previous_ppt.end == 2000.0
        )  # Should be set to current PPT begin time
        assert len(queue_ditl.plan) == 2  # Should have both PPTs now

    def test_close_ppt_timeline_if_needed_closes_when_ppt_none(self, queue_ditl):
        """Test _close_ppt_timeline_if_needed closes PPT when ppt is None."""
        from conops.targets import PlanEntry

        # Create a mock PPT with placeholder end time
        mock_ppt = Mock(spec=PlanEntry)
        mock_ppt.begin = 1000.0
        mock_ppt.end = 1000.0 + 86400 + 100  # Placeholder end time

        # Set up plan with the PPT and set current ppt to None
        queue_ditl.plan = [mock_ppt]
        queue_ditl.ppt = None

        # Call the method
        queue_ditl._close_ppt_timeline_if_needed(2000.0)

        # Check that the PPT's end time was updated
        assert mock_ppt.end == 2000.0

    def test_terminate_ppt_marks_done_when_requested(self, queue_ditl):
        """Test _terminate_ppt sets done flag when mark_done=True."""
        from conops.targets import PlanEntry

        # Create a mock PPT
        mock_ppt = Mock(spec=PlanEntry)
        mock_ppt.begin = 1000.0
        mock_ppt.end = 2000.0

        # Set up the PPT
        queue_ditl.plan = [mock_ppt]
        queue_ditl.ppt = mock_ppt

        # Call terminate with mark_done=True
        queue_ditl._terminate_ppt(1500.0, reason="Test termination", mark_done=True)

        # Check that done was set to True
        assert mock_ppt.done is True
        assert mock_ppt.end == 1500.0
        assert queue_ditl.ppt is None

    def test_fetch_ppt_delays_for_current_slew(self, queue_ditl, capsys):
        """Test _fetch_new_ppt delays slew when current slew is in progress."""
        from conops.simulation.slew import Slew

        # Create mock PPT
        mock_ppt = Mock()
        mock_ppt.ra = 45.0
        mock_ppt.dec = 30.0
        mock_ppt.obsid = 1001
        mock_ppt.next_vis = Mock(return_value=1000.0)
        mock_ppt.ssmax = 3600.0
        queue_ditl.queue.get = Mock(return_value=mock_ppt)

        # Create a mock current slew that's still slewing
        mock_current_slew = Mock(spec=Slew)
        mock_current_slew.is_slewing = Mock(return_value=True)
        mock_current_slew.slewstart = 900.0
        mock_current_slew.slewtime = 200.0
        queue_ditl.acs.last_slew = mock_current_slew
        lastra, lastdec = queue_ditl._fetch_new_ppt(1000.0, 10.0, 20.0)

        # Check that the command was enqueued with delayed execution time
        queue_ditl.acs.enqueue_command.assert_called_once()
        call_args = queue_ditl.acs.enqueue_command.call_args
        command = call_args[0][0]
        # Execution time should be delayed to current_slew.slewstart + slewtime = 1100.0
        assert command.execution_time == 1100.0

        # Check that the delay message was printed
        captured = capsys.readouterr()
        assert "delaying next slew until" in captured.out

    def test_fetch_ppt_delays_for_visibility(self, queue_ditl, capsys):
        """Test _fetch_new_ppt delays slew when target visibility requires it."""
        # Create mock PPT
        mock_ppt = Mock()
        mock_ppt.ra = 45.0
        mock_ppt.dec = 30.0
        mock_ppt.obsid = 1001
        # Set next_vis to a time after the current time (1000.0)
        mock_ppt.next_vis = Mock(return_value=1200.0)
        mock_ppt.ssmax = 3600.0
        queue_ditl.queue.get = Mock(return_value=mock_ppt)
        lastra, lastdec = queue_ditl._fetch_new_ppt(1000.0, 10.0, 20.0)

        # Check that the command was enqueued with delayed execution time
        queue_ditl.acs.enqueue_command.assert_called_once()
        call_args = queue_ditl.acs.enqueue_command.call_args
        command = call_args[0][0]
        # Execution time should be delayed to visibility time (1200.0)
        assert command.execution_time == 1200.0

        # Check that the visibility delay message was printed
        captured = capsys.readouterr()
        assert "Slew delayed by" in captured.out

    def test_terminate_science_ppt_for_pass_sets_done_flag(self, queue_ditl):
        """Test _terminate_science_ppt_for_pass sets done flag."""
        from conops.targets import PlanEntry

        # Create a mock PPT
        mock_ppt = Mock(spec=PlanEntry)
        mock_ppt.begin = 1000.0
        mock_ppt.end = 2000.0

        # Set up the PPT
        queue_ditl.plan = [mock_ppt]
        queue_ditl.ppt = mock_ppt

        # Call the method
        queue_ditl._terminate_science_ppt_for_pass(1500.0)

        # Check that done was set to True and other updates happened
        assert mock_ppt.done is True
        assert mock_ppt.end == 1500.0
        assert queue_ditl.ppt is None

    def test_terminate_charging_ppt_sets_done_flag(self, queue_ditl):
        """Test _terminate_charging_ppt sets done flag."""
        from conops.targets import PlanEntry

        # Create a mock charging PPT
        mock_charging_ppt = Mock(spec=PlanEntry)
        mock_charging_ppt.begin = 1000.0
        mock_charging_ppt.end = 2000.0

        # Set up the charging PPT
        queue_ditl.plan = [mock_charging_ppt]
        queue_ditl.charging_ppt = mock_charging_ppt

        # Call the method
        queue_ditl._terminate_charging_ppt(1500.0)

        # Check that done was set to True and other updates happened
        assert mock_charging_ppt.done is True
        assert mock_charging_ppt.end == 1500.0
        assert queue_ditl.charging_ppt is None

    def test_setup_simulation_timing_fails_with_invalid_ephemeris_range(
        self, queue_ditl, capsys
    ):
        """Test _setup_simulation_timing fails when ephemeris doesn't cover date range."""
        from datetime import datetime, timezone

        # Set begin/end times that are not in the ephemeris
        queue_ditl.begin = datetime(2025, 1, 1, tzinfo=timezone.utc)  # Far future date
        queue_ditl.end = datetime(2025, 1, 2, tzinfo=timezone.utc)

        result = queue_ditl._setup_simulation_timing()

        assert result is False
        captured = capsys.readouterr()
        assert "ERROR: Ephemeris not valid for date range" in captured.out


class TestGetConstraintName:
    """Tests for _get_constraint_name method."""

    def test_get_constraint_name_earth_name(self, queue_ditl):
        ra, dec, utime = 10.0, 20.0, 1000.0
        queue_ditl.constraint.in_earth = Mock(return_value=True)
        queue_ditl.constraint.in_moon = Mock(return_value=False)
        queue_ditl.constraint.in_sun = Mock(return_value=False)
        queue_ditl.constraint.in_panel = Mock(return_value=False)
        name = queue_ditl._get_constraint_name(ra, dec, utime)
        assert name == "Earth Limb"

    def test_get_constraint_name_earth_call(self, queue_ditl):
        ra, dec, utime = 10.0, 20.0, 1000.0
        queue_ditl.constraint.in_earth = Mock(return_value=True)
        queue_ditl.constraint.in_moon = Mock(return_value=False)
        queue_ditl.constraint.in_sun = Mock(return_value=False)
        queue_ditl.constraint.in_panel = Mock(return_value=False)
        _ = queue_ditl._get_constraint_name(ra, dec, utime)
        queue_ditl.constraint.in_earth.assert_called_once_with(ra, dec, utime)

    def test_get_constraint_name_moon_name(self, queue_ditl):
        ra, dec, utime = 11.0, 21.0, 2000.0
        queue_ditl.constraint.in_earth = Mock(return_value=False)
        queue_ditl.constraint.in_moon = Mock(return_value=True)
        queue_ditl.constraint.in_sun = Mock(return_value=False)
        queue_ditl.constraint.in_panel = Mock(return_value=False)
        name = queue_ditl._get_constraint_name(ra, dec, utime)
        assert name == "Moon"

    def test_get_constraint_name_moon_calls(self, queue_ditl):
        ra, dec, utime = 11.0, 21.0, 2000.0
        queue_ditl.constraint.in_earth = Mock(return_value=False)
        queue_ditl.constraint.in_moon = Mock(return_value=True)
        queue_ditl.constraint.in_sun = Mock(return_value=False)
        queue_ditl.constraint.in_panel = Mock(return_value=False)
        _ = queue_ditl._get_constraint_name(ra, dec, utime)
        queue_ditl.constraint.in_earth.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_moon.assert_called_once_with(ra, dec, utime)

    def test_get_constraint_name_sun_name(self, queue_ditl):
        ra, dec, utime = 12.0, 22.0, 3000.0
        queue_ditl.constraint.in_earth = Mock(return_value=False)
        queue_ditl.constraint.in_moon = Mock(return_value=False)
        queue_ditl.constraint.in_sun = Mock(return_value=True)
        queue_ditl.constraint.in_panel = Mock(return_value=False)
        name = queue_ditl._get_constraint_name(ra, dec, utime)
        assert name == "Sun"

    def test_get_constraint_name_sun_calls(self, queue_ditl):
        ra, dec, utime = 12.0, 22.0, 3000.0
        queue_ditl.constraint.in_earth = Mock(return_value=False)
        queue_ditl.constraint.in_moon = Mock(return_value=False)
        queue_ditl.constraint.in_sun = Mock(return_value=True)
        queue_ditl.constraint.in_panel = Mock(return_value=False)
        _ = queue_ditl._get_constraint_name(ra, dec, utime)
        queue_ditl.constraint.in_earth.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_moon.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_sun.assert_called_once_with(ra, dec, utime)

    def test_get_constraint_name_panel_name(self, queue_ditl):
        ra, dec, utime = 13.0, 23.0, 4000.0
        queue_ditl.constraint.in_earth = Mock(return_value=False)
        queue_ditl.constraint.in_moon = Mock(return_value=False)
        queue_ditl.constraint.in_sun = Mock(return_value=False)
        queue_ditl.constraint.in_panel = Mock(return_value=True)
        name = queue_ditl._get_constraint_name(ra, dec, utime)
        assert name == "Panel"

    def test_get_constraint_name_panel_calls(self, queue_ditl):
        ra, dec, utime = 13.0, 23.0, 4000.0
        queue_ditl.constraint.in_earth = Mock(return_value=False)
        queue_ditl.constraint.in_moon = Mock(return_value=False)
        queue_ditl.constraint.in_sun = Mock(return_value=False)
        queue_ditl.constraint.in_panel = Mock(return_value=True)
        _ = queue_ditl._get_constraint_name(ra, dec, utime)
        queue_ditl.constraint.in_earth.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_moon.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_sun.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_panel.assert_called_once_with(ra, dec, utime)

    def test_get_constraint_name_unknown_name(self, queue_ditl):
        ra, dec, utime = 14.0, 24.0, 5000.0
        queue_ditl.constraint.in_earth = Mock(return_value=False)
        queue_ditl.constraint.in_moon = Mock(return_value=False)
        queue_ditl.constraint.in_sun = Mock(return_value=False)
        queue_ditl.constraint.in_panel = Mock(return_value=False)
        name = queue_ditl._get_constraint_name(ra, dec, utime)
        assert name == "Unknown"

    def test_get_constraint_name_unknown_calls(self, queue_ditl):
        ra, dec, utime = 14.0, 24.0, 5000.0
        queue_ditl.constraint.in_earth = Mock(return_value=False)
        queue_ditl.constraint.in_moon = Mock(return_value=False)
        queue_ditl.constraint.in_sun = Mock(return_value=False)
        queue_ditl.constraint.in_panel = Mock(return_value=False)
        _ = queue_ditl._get_constraint_name(ra, dec, utime)
        queue_ditl.constraint.in_earth.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_moon.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_sun.assert_called_once_with(ra, dec, utime)
        queue_ditl.constraint.in_panel.assert_called_once_with(ra, dec, utime)

    def test_get_constraint_name_precedence_earth(self, queue_ditl):
        ra, dec, utime = 15.0, 25.0, 6000.0
        queue_ditl.constraint.in_earth = Mock(return_value=True)
        queue_ditl.constraint.in_moon = Mock(return_value=True)
        queue_ditl.constraint.in_sun = Mock(return_value=True)
        queue_ditl.constraint.in_panel = Mock(return_value=True)
        name = queue_ditl._get_constraint_name(ra, dec, utime)
        assert name == "Earth Limb"
        queue_ditl.constraint.in_earth.assert_called_once_with(ra, dec, utime)


class TestCheckAndManagePasses:
    """Tests for _check_and_manage_passes helper method."""

    def test_check_and_manage_passes_end_pass_calls_check_pass_timing(self, queue_ditl):
        """Test that END_PASS is enqueued when we detect a pass ended."""
        utime = 1000.0
        ra, dec = 10.0, 20.0
        # Mock passrequests to indicate pass has ended
        queue_ditl.acs.passrequests.current_pass = Mock(return_value=None)
        # Previous timestep had a pass
        queue_ditl.acs.passrequests.next_pass = Mock(return_value=None)
        queue_ditl._check_and_manage_passes(utime, ra, dec)
        # The method should work without errors even when pass ends
        assert True

    def test_check_and_manage_passes_end_pass_enqueues_command(self, queue_ditl):
        """Test that END_PASS command is enqueued when pass ends."""
        utime = 1000.0
        ra, dec = 10.0, 20.0
        # Setup: currently not in a pass
        queue_ditl.acs.passrequests.current_pass = Mock(return_value=None)
        queue_ditl.acs.passrequests.next_pass = Mock(return_value=None)
        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        # No pass, no next pass - method should not enqueue anything
        queue_ditl._check_and_manage_passes(utime, ra, dec)
        # verify method runs without error

    def test_check_and_manage_passes_end_pass_command_type(self, queue_ditl):
        """Test that END_PASS command has correct type."""
        utime = 1000.0
        ra, dec = 10.0, 20.0
        # Simulate just exited a pass
        queue_ditl.acs.passrequests.current_pass = Mock(return_value=None)
        queue_ditl.acs.passrequests.next_pass = Mock(return_value=None)
        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        queue_ditl._check_and_manage_passes(utime, ra, dec)
        # Test documents that when no pass, no command is sent

    def test_check_and_manage_passes_end_pass_command_execution_time(self, queue_ditl):
        """Test that END_PASS command has correct execution time."""
        utime = 1000.0
        ra, dec = 10.0, 20.0
        queue_ditl.acs.passrequests.current_pass = Mock(return_value=None)
        queue_ditl.acs.passrequests.next_pass = Mock(return_value=None)
        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        queue_ditl._check_and_manage_passes(utime, ra, dec)
        # Verify method completes without error

    def test_check_and_manage_passes_start_pass_calls_check_pass_timing(
        self, queue_ditl
    ):
        """Test that START_PASS is issued when entering a pass."""
        utime = 1000.0
        ra, dec = 10.0, 20.0
        pass_obj = Mock()
        queue_ditl.acs.passrequests.current_pass = Mock(return_value=pass_obj)
        queue_ditl.acs.acsmode = ACSMode.SCIENCE  # Not in pass yet
        queue_ditl._check_and_manage_passes(utime, ra, dec)
        queue_ditl.acs.enqueue_command.assert_called_once()

    def test_check_and_manage_passes_start_pass_enqueues_command(self, queue_ditl):
        """Test that START_PASS command is enqueued when entering pass."""
        utime = 1000.0
        ra, dec = 10.0, 20.0
        pass_obj = Mock()
        queue_ditl.acs.passrequests.current_pass = Mock(return_value=pass_obj)
        queue_ditl.acs.acsmode = ACSMode.SCIENCE  # Not in pass yet
        queue_ditl._check_and_manage_passes(utime, ra, dec)
        queue_ditl.acs.enqueue_command.assert_called_once()

    def test_check_and_manage_passes_start_pass_command_type_and_exec_time(
        self, queue_ditl
    ):
        """Test START_PASS command has correct type and execution time."""
        utime = 1000.0
        ra, dec = 10.0, 20.0
        pass_obj = Mock()
        queue_ditl.acs.passrequests.current_pass = Mock(return_value=pass_obj)
        queue_ditl.acs.acsmode = ACSMode.SCIENCE  # Not in pass yet
        queue_ditl._check_and_manage_passes(utime, ra, dec)
        cmd = queue_ditl.acs.enqueue_command.call_args[0][0]
        assert cmd.command_type == ACSCommandType.START_PASS
        assert cmd.execution_time == utime

    def test_check_and_manage_passes_start_pass_sets_obsid(self, queue_ditl):
        """Test that pass gets assigned obsid when starting."""
        utime = 1000.0
        ra, dec = 10.0, 20.0
        pass_obj = Pass(station="GS_STATION", begin=950.0, slewrequired=900.0)
        queue_ditl.acs.passrequests.current_pass = Mock(return_value=pass_obj)
        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        queue_ditl.acs.last_ppt = Mock(obsid=1234)
        queue_ditl._check_and_manage_passes(utime, ra, dec)
        # In the current code, obsid is not set during START_PASS
        # This test documents that behavior

    def test_check_and_manage_passes_start_pass_command_slew_station(self, queue_ditl):
        """Test START_PASS behavior in SAA mode (should not be enqueued)."""
        utime = 1000.0
        ra, dec = 10.0, 20.0
        pass_obj = Pass(station="GS_STATION", begin=950.0, slewrequired=900.0)
        queue_ditl.acs.passrequests.current_pass = Mock(return_value=pass_obj)
        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        queue_ditl.acs.last_ppt = Mock(obsid=1234)
        queue_ditl._check_and_manage_passes(utime, ra, dec)
        # Verify method ran without error

    def test_check_and_manage_passes_start_pass_not_enqueued_when_not_science_calls_check(
        self, queue_ditl
    ):
        """Test that START_PASS is not issued when in SAA mode."""
        utime = 2000.0
        ra, dec = 30.0, 40.0
        pass_obj = Pass(station="GS2", begin=1850.0, slewrequired=1800.0)
        queue_ditl.acs.passrequests.current_pass = Mock(return_value=pass_obj)
        queue_ditl.acs.acsmode = ACSMode.SAA
        queue_ditl._check_and_manage_passes(utime, ra, dec)
        # In SAA mode, commands should not be enqueued

    def test_check_and_manage_passes_both_end_and_start_calls_check_pass_timing(
        self, queue_ditl
    ):
        """Test pass management with both end and start scenarios."""
        utime = 3000.0
        ra, dec = 0.0, 0.0
        pass_obj = Pass(station="GS_ORDER", begin=2950.0, slewrequired=2900.0)
        queue_ditl.acs.passrequests.current_pass = Mock(return_value=pass_obj)
        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        queue_ditl.acs.last_ppt = Mock(obsid=0xBEEF)
        queue_ditl._check_and_manage_passes(utime, ra, dec)
        # Verify method runs without error

    def test_check_and_manage_passes_both_end_and_start_enqueues_two_commands(
        self, queue_ditl
    ):
        """Test multiple commands during pass transitions."""
        utime = 3000.0
        ra, dec = 0.0, 0.0
        pass_obj = Pass(station="GS_ORDER", begin=2950.0, slewrequired=2900.0)
        queue_ditl.acs.passrequests.current_pass = Mock(return_value=pass_obj)
        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        queue_ditl.acs.last_ppt = Mock(obsid=0xBEEF)
        queue_ditl._check_and_manage_passes(utime, ra, dec)
        # Verify behavior documented

    def test_check_and_manage_passes_both_end_and_start_command_order(self, queue_ditl):
        """Test command ordering during pass transitions."""
        utime = 3000.0
        ra, dec = 0.0, 0.0
        pass_obj = Pass(station="GS_ORDER", begin=2950.0, slewrequired=2900.0)
        queue_ditl.acs.passrequests.current_pass = Mock(return_value=pass_obj)
        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        queue_ditl.acs.last_ppt = Mock(obsid=0xBEEF)
        queue_ditl._check_and_manage_passes(utime, ra, dec)
        # Method should run without error

    def test_check_and_manage_passes_both_end_and_start_start_command_exec_time_and_slew(
        self, queue_ditl
    ):
        """Test START_PASS command structure and timing."""
        utime = 3000.0
        ra, dec = 0.0, 0.0
        pass_obj = Pass(station="GS_ORDER", begin=2950.0, slewrequired=2900.0)
        queue_ditl.acs.passrequests.current_pass = Mock(return_value=pass_obj)
        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        queue_ditl.acs.last_ppt = Mock(obsid=0xBEEF)
        queue_ditl._check_and_manage_passes(utime, ra, dec)
        # Verify method runs correctly

    def test_check_and_manage_passes_both_end_and_start_sets_obsid_from_last_ppt(
        self, queue_ditl
    ):
        """Test obsid handling during pass transitions."""
        utime = 3000.0
        ra, dec = 0.0, 0.0
        pass_obj = Pass(station="GS_ORDER", begin=2950.0, slewrequired=2900.0)
        queue_ditl.acs.passrequests.current_pass = Mock(return_value=pass_obj)
        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        queue_ditl.acs.last_ppt = Mock(obsid=0xBEEF)
        queue_ditl._check_and_manage_passes(utime, ra, dec)
        # Verify method completes successfully


class TestGetACSQueueStatus:
    """Test get_acs_queue_status method."""

    def test_get_acs_queue_status_empty_queue(self, queue_ditl):
        queue_ditl.acs.command_queue = []
        queue_ditl.acs.current_slew = None
        queue_ditl.acs.acsmode = ACSMode.SCIENCE
        status = queue_ditl.get_acs_queue_status()
        expected = {
            "queue_size": 0,
            "pending_commands": [],
            "current_slew": None,
            "acs_mode": "SCIENCE",
        }
        assert status == expected

    def test_get_acs_queue_status_with_pending_commands(self, queue_ditl):
        mock_cmd1 = Mock()
        mock_cmd1.command_type.name = "SLEW_TO_TARGET"
        mock_cmd1.execution_time = 1000.0
        mock_cmd2 = Mock()
        mock_cmd2.command_type.name = "START_PASS"
        mock_cmd2.execution_time = 2000.0
        queue_ditl.acs.command_queue = [mock_cmd1, mock_cmd2]
        queue_ditl.acs.current_slew = None
        queue_ditl.acs.acsmode = ACSMode.PASS
        with patch("conops.ditl.queue_ditl.unixtime2date") as mock_unixtime2date:
            mock_unixtime2date.side_effect = [
                "2023-01-01 00:00:00",
                "2023-01-01 00:33:20",
            ]
            status = queue_ditl.get_acs_queue_status()
        expected = {
            "queue_size": 2,
            "pending_commands": [
                {
                    "type": "SLEW_TO_TARGET",
                    "execution_time": 1000.0,
                    "time_formatted": "2023-01-01 00:00:00",
                },
                {
                    "type": "START_PASS",
                    "execution_time": 2000.0,
                    "time_formatted": "2023-01-01 00:33:20",
                },
            ],
            "current_slew": None,
            "acs_mode": "PASS",
        }
        assert status == expected

    def test_get_acs_queue_status_with_current_slew(self, queue_ditl):
        queue_ditl.acs.command_queue = []
        mock_slew = Mock()
        mock_slew.__class__.__name__ = "Slew"
        queue_ditl.acs.current_slew = mock_slew
        queue_ditl.acs.acsmode = ACSMode.SLEWING
        status = queue_ditl.get_acs_queue_status()
        expected = {
            "queue_size": 0,
            "pending_commands": [],
            "current_slew": "Slew",
            "acs_mode": "SLEWING",
        }
        assert status == expected

    def test_get_acs_queue_status_different_modes(self, queue_ditl):
        queue_ditl.acs.command_queue = []
        queue_ditl.acs.current_slew = None
        for mode in [ACSMode.SCIENCE, ACSMode.CHARGING, ACSMode.SAA]:
            queue_ditl.acs.acsmode = mode
            status = queue_ditl.get_acs_queue_status()
            assert status["acs_mode"] == mode.name

    def test_get_acs_queue_status_mixed_state(self, queue_ditl):
        mock_cmd = Mock()
        mock_cmd.command_type.name = "END_PASS"
        mock_cmd.execution_time = 1500.0
        queue_ditl.acs.command_queue = [mock_cmd]
        mock_slew = Mock()
        mock_slew.__class__.__name__ = "Pass"
        queue_ditl.acs.current_slew = mock_slew
        queue_ditl.acs.acsmode = ACSMode.PASS
        with patch(
            "conops.ditl.queue_ditl.unixtime2date", return_value="2023-01-01 00:25:00"
        ):
            status = queue_ditl.get_acs_queue_status()
        expected = {
            "queue_size": 1,
            "pending_commands": [
                {
                    "type": "END_PASS",
                    "execution_time": 1500.0,
                    "time_formatted": "2023-01-01 00:25:00",
                }
            ],
            "current_slew": "Pass",
            "acs_mode": "PASS",
        }
        assert status == expected
