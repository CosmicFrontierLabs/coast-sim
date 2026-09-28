"""Additional tests to achieve 100% coverage for ACS class."""

from unittest.mock import Mock, patch

import pytest

from conops import ACSCommand, ACSCommandType, ACSMode, DITLLog, Pass, Slew
from conops.common import ObsType


class TestExecuteCommandCoverage:
    """Test command execution handler methods."""

    def test_end_pass_adds_slew_with_last_ppt(self, acs) -> None:
        """END_PASS should NOT call enqueue_command for last_ppt in queue-driven mode."""
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        # Directly call _end_pass
        acs._end_pass(1514764800.0)
        # Verify currentpass is cleared and mode is set to IDLE
        assert acs.current_pass is None
        assert acs.acsmode == ACSMode.IDLE

    def test_end_pass_clears_currentpass(self, acs) -> None:
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert acs.current_pass is None
        assert acs.acsmode == ACSMode.IDLE

    def test_end_pass_sets_mode_idle(self, acs) -> None:
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert acs.acsmode == ACSMode.IDLE

    def test_end_pass_no_last_ppt_clears_currentpass(self, acs) -> None:
        acs.last_ppt = None
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert acs.current_pass is None

    def test_end_pass_no_last_ppt_sets_mode_idle(self, acs) -> None:
        acs.last_ppt = None
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert acs.acsmode == ACSMode.IDLE

    def test_execute_null_slew_does_not_start(self, acs) -> None:
        command = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764800.0,
            slew=None,
        )

        with patch.object(acs, "_start_slew") as mock_start_slew:
            acs._handle_slew_command(command, 1514764800.0)
            mock_start_slew.assert_not_called()

    def test_execute_slew_to_target_none_slew_does_not_start(self, acs) -> None:
        command = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764800.0,
            slew=None,
        )

        with patch.object(acs, "_start_slew") as mock_start_slew:
            acs._handle_slew_command(command, 1514764800.0)
            mock_start_slew.assert_not_called()

    def test_execute_start_pass_none_slew_does_not_start(self, acs) -> None:
        command = ACSCommand(
            command_type=ACSCommandType.START_PASS,
            execution_time=1514764800.0,
            slew=None,
        )

        with patch.object(acs, "_start_slew") as mock_start_slew:
            acs._start_pass(command, 1514764800.0)
            mock_start_slew.assert_not_called()

    def test_execute_start_pass_non_pass_slew_does_not_start(self, acs) -> None:
        mock_slew = Mock(spec=Slew)
        command = ACSCommand(
            command_type=ACSCommandType.START_PASS,
            execution_time=1514764800.0,
            slew=mock_slew,
        )

        with patch.object(acs, "_start_slew") as mock_start_slew:
            acs._start_pass(command, 1514764800.0)
            mock_start_slew.assert_not_called()


class TestEnqueueCommandQueueManagement:
    """Test command-queue replacement behavior."""

    def _make_slew_command(self, execution_time: float, obsid: int) -> ACSCommand:
        slew = Mock(spec=Slew)
        slew.obsid = obsid
        slew.obstype = "PPT"
        return ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=execution_time,
            slew=slew,
        )

    def test_new_slew_cancels_pending_slew_command(self, acs) -> None:
        acs.log = DITLLog()
        stale_return = self._make_slew_command(1514767440.0, 10019)
        end_pass = ACSCommand(
            command_type=ACSCommandType.END_PASS,
            execution_time=1514767200.0,
        )
        replacement = self._make_slew_command(1514767260.0, 10053)
        acs.command_queue = [stale_return, end_pass]

        acs.enqueue_command(replacement)

        assert all(command is not stale_return for command in acs.command_queue)
        assert any(command is end_pass for command in acs.command_queue)
        assert any(command is replacement for command in acs.command_queue)
        assert [command.execution_time for command in acs.command_queue] == [
            1514767200.0,
            1514767260.0,
        ]

        log_text = "\n".join(event.description for event in acs.log.events)
        assert "Canceled pending SLEW_TO_TARGET" in log_text
        assert "obsid=10019" in log_text
        assert "obsid=10053" in log_text

    def test_non_slew_command_preserves_pending_slew_command(self, acs) -> None:
        pending_slew = self._make_slew_command(1514767440.0, 10019)
        end_pass = ACSCommand(
            command_type=ACSCommandType.END_PASS,
            execution_time=1514767200.0,
        )
        acs.command_queue = [pending_slew]

        acs.enqueue_command(end_pass)

        assert any(command is pending_slew for command in acs.command_queue)
        assert any(command is end_pass for command in acs.command_queue)
        assert [command.execution_time for command in acs.command_queue] == [
            1514767200.0,
            1514767440.0,
        ]

    def test_start_pass_cancels_pending_slew_command(self, acs) -> None:
        acs.log = DITLLog()
        stale_return = self._make_slew_command(1514767440.0, 10019)
        start_pass = ACSCommand(
            command_type=ACSCommandType.START_PASS,
            execution_time=1514767200.0,
        )
        acs.command_queue = [stale_return]

        acs.enqueue_command(start_pass)

        assert all(command is not stale_return for command in acs.command_queue)
        assert acs.command_queue == [start_pass]

        log_text = "\n".join(event.description for event in acs.log.events)
        assert "Canceled pending SLEW_TO_TARGET" in log_text
        assert "obsid=10019" in log_text
        assert "superseded by START_PASS" in log_text

    def test_cancel_pending_battery_charge_removes_unexecuted_start(self, acs) -> None:
        acs.log = DITLLog()
        start_charge = ACSCommand(
            command_type=ACSCommandType.START_BATTERY_CHARGE,
            execution_time=1514767200.0,
        )
        pending_slew = self._make_slew_command(1514767260.0, 10019)
        acs.command_queue = [start_charge, pending_slew]

        canceled = acs.cancel_pending_battery_charge(1514767200.0)

        assert canceled is True
        assert all(
            command.command_type != ACSCommandType.START_BATTERY_CHARGE
            for command in acs.command_queue
        )
        # the unrelated pending slew is left intact
        assert any(command is pending_slew for command in acs.command_queue)

    def test_cancel_pending_battery_charge_noop_when_none_pending(self, acs) -> None:
        pending_slew = self._make_slew_command(1514767260.0, 10019)
        acs.command_queue = [pending_slew]

        canceled = acs.cancel_pending_battery_charge(1514767200.0)

        assert canceled is False
        assert acs.command_queue == [pending_slew]

    def test_cancel_pending_battery_charge_asserts_on_duplicate(self, acs) -> None:
        """Two START_BATTERY_CHARGE commands violate the QueueDITL invariant."""
        charge1 = ACSCommand(
            command_type=ACSCommandType.START_BATTERY_CHARGE,
            execution_time=1514767200.0,
        )
        charge2 = ACSCommand(
            command_type=ACSCommandType.START_BATTERY_CHARGE,
            execution_time=1514767260.0,
        )
        acs.command_queue = [charge1, charge2]

        with pytest.raises(AssertionError, match="Invariant violated"):
            acs.cancel_pending_battery_charge(1514767200.0)


class TestStartSlewCoverage:
    """Slew execution uses current physical state, not precomputed start metadata."""

    @pytest.mark.parametrize("obstype", [ObsType.PPT, ObsType.GSP])
    @pytest.mark.parametrize("initial", [(0, 0, 0), (10, 20, 30)])
    def test_start_slew_replans_from_executed_state(self, acs, obstype, initial):
        acs.ra, acs.dec, acs.roll = initial
        slew = Slew(config=acs.config, endra=45, enddec=30, obstype=obstype)
        slew.startra, slew.startdec, slew.startroll = 90, -40, 50
        slew.slewstart = 999
        acs._start_slew(slew, 1514764800)
        assert (slew.startra, slew.startdec, slew.startroll) == pytest.approx(initial)
        assert slew.slewstart == 1514764800
        assert slew.slewtime > 0
        assert slew.slewend == slew.slewstart + slew.slewtime
        assert acs.current_slew is slew
        assert acs.last_ppt is (slew if obstype == ObsType.PPT else None)


class TestEndPassCoverage:
    """Test _end_pass method."""

    def test_end_pass_adds_slew_returning_to_last_ppt(self, acs) -> None:
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt

        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        with patch.object(
            acs, "enqueue_command", return_value=True
        ) as mock_enqueue_command:
            acs._end_pass(1514764800.0)
            mock_enqueue_command.assert_not_called()

    def test_end_pass_clears_currentpass(self, acs) -> None:
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt

        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        with patch.object(acs, "enqueue_command", return_value=True):
            acs._end_pass(1514764800.0)
            assert acs.current_pass is None

    def test_end_pass_sets_mode_idle_on_end(self, acs) -> None:
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt

        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        with patch.object(acs, "enqueue_command", return_value=True):
            acs._end_pass(1514764800.0)
            assert acs.acsmode == ACSMode.IDLE

    def test_end_pass_with_no_last_ppt_clears_currentpass(self, acs) -> None:
        acs.last_ppt = None
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert acs.current_pass is None

    def test_end_pass_with_no_last_ppt_sets_mode_idle(self, acs) -> None:
        acs.last_ppt = None
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert acs.acsmode == ACSMode.IDLE

    def test_end_pass_no_last_ppt_does_not_enqueue_slew(self, acs) -> None:
        acs.last_ppt = None
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert len(acs.command_queue) == 0

    def test_end_pass_replaces_stale_slew_with_attitude_hold(self, acs) -> None:
        """A stale pass-ingress slew is replaced with the final tracked attitude."""
        from conops.simulation.slew import Slew

        # Create a slew that started in the past (stale)
        mock_slew = Mock(spec=Slew)
        mock_slew.slewstart = 1514764700.0  # Started 100 seconds ago
        acs.last_slew = mock_slew

        # Set up pass state
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS
        acs.ra = 45.0
        acs.dec = -30.0
        acs.roll = 210.0

        # End the pass
        acs._end_pass(1514764800.0)

        assert acs.last_slew is not None
        assert acs.last_slew.obstype == ObsType.IDLE
        assert (
            acs.last_slew.endra,
            acs.last_slew.enddec,
            acs.last_slew.endroll,
        ) == pytest.approx(
            (
                45.0,
                -30.0,
                210.0,
            )
        )
        # Pass should be cleared
        assert acs.current_pass is None
        # Mode should be IDLE
        assert acs.acsmode == ACSMode.IDLE

    def test_end_pass_replaces_precomputed_slew_with_attitude_hold(self, acs) -> None:
        """A future command must not change physical attitude before it executes."""
        from conops.simulation.slew import Slew

        # Create a slew that starts in the future (fresh)
        mock_slew = Mock(spec=Slew)
        mock_slew.slewstart = 1514764900.0  # Starts 100 seconds from now
        acs.last_slew = mock_slew

        # Set up pass state
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS
        acs.ra = 45.0
        acs.dec = -30.0
        acs.roll = 210.0

        # End the pass
        acs._end_pass(1514764800.0)

        assert acs.last_slew is not mock_slew
        assert acs.last_slew is not None
        assert acs.last_slew.obstype == ObsType.IDLE
        assert acs.pointing(1514764800.0)[2] == pytest.approx(210.0)
        # Pass should still be cleared
        assert acs.current_pass is None
        # Mode should be IDLE
        assert acs.acsmode == ACSMode.IDLE

    def test_end_pass_handles_no_slew(self, acs) -> None:
        """Test that _end_pass works correctly when there's no last_slew."""
        # No slew at all
        acs.last_slew = None

        # Set up pass state
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        # End the pass (should not error)
        acs._end_pass(1514764800.0)

        # The current attitude becomes an explicit IDLE hold.
        assert acs.last_slew is not None
        assert acs.last_slew.obstype == ObsType.IDLE
        assert acs.current_pass is None
        assert acs.acsmode == ACSMode.IDLE

    def test_end_pass_replaces_slew_starting_at_current_time(self, acs) -> None:
        """END_PASS owns the boundary until a queued slew command executes."""
        from conops.simulation.slew import Slew

        # Create a slew that starts exactly at current time
        mock_slew = Mock(spec=Slew)
        mock_slew.slewstart = 1514764800.0  # Starts at exact current time
        acs.last_slew = mock_slew

        # Set up pass state
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        # End the pass
        acs._end_pass(1514764800.0)

        assert acs.last_slew is not mock_slew
        assert acs.last_slew is not None
        assert acs.last_slew.obstype == ObsType.IDLE


class TestProcessCommandsCoverage:
    """Command dispatch preserves queue order and leaves future work pending."""

    def test_process_due_commands(self, acs):
        commands = [
            ACSCommand(
                command_type=ACSCommandType.SLEW_TO_TARGET,
                execution_time=time,
                slew=Slew(config=acs.config, endra=ra, enddec=30),
            )
            for time, ra in [(1514764800, 45), (1514764810, 90), (1514764900, 120)]
        ]
        acs.command_queue = list(commands)
        acs._process_commands(1514764815)
        assert acs.executed_commands == commands[:2]
        assert acs.command_queue == commands[2:]
        assert acs.current_slew is commands[1].slew


class TestExecuteCommandLogging:
    """Test command handler logging."""

    def test_handle_slew_command_executes(self, acs) -> None:
        mock_slew = Slew(config=acs.config)
        mock_slew.startra = 0.0
        mock_slew.startdec = 0.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"
        mock_slew.slewstart = 1514764800.0
        mock_slew.slewtime = 60.0
        mock_slew.slewdist = 45.0
        command1 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764800.0,
            slew=mock_slew,
        )

        # Test that command executes without error
        acs._handle_slew_command(command1, 1514764800.0)
        assert acs.current_slew == mock_slew

    def test_start_pass_executes(self, acs) -> None:
        mock_slew = Slew(config=acs.config)
        mock_slew.startra = 0.0
        mock_slew.startdec = 0.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "GSP"
        mock_slew.slewstart = 1514764800.0
        mock_slew.slewtime = 60.0
        mock_slew.slewdist = 45.0
        command2 = ACSCommand(
            command_type=ACSCommandType.START_PASS,
            execution_time=1514764800.0,
            slew=mock_slew,
        )

        gspass = Mock()
        gspass.utime = [1514764800.0, 1514764860.0]
        gspass.ra = [acs.ra, acs.ra]
        gspass.dec = [acs.dec, acs.dec]
        gspass.roll = [0.0, 0.0]
        acs.roll = 0.0
        acs.passrequests.current_pass.return_value = gspass
        acs._start_pass(command2, 1514764800.0)
        # _start_pass sets current_pass, not current_slew
        assert acs.acsmode == ACSMode.PASS


class TestGetModeCharging:
    """Test get_mode for CHARGING mode."""

    def test_get_mode_returns_slewing_for_active_charge_slew(self, acs) -> None:
        mock_slew = Mock(spec=Slew)
        mock_slew.obstype = ObsType.CHARGE
        mock_slew.is_slewing = Mock(return_value=True)
        acs.current_slew = mock_slew

        mode = acs.get_mode(1514764800.0)
        assert mode == ACSMode.SLEWING

    def test_get_mode_charging_in_dwell_when_not_slewing(
        self, acs, monkeypatch
    ) -> None:
        mock_slew = Mock(spec=Slew)
        mock_slew.obstype = ObsType.CHARGE
        mock_slew.is_slewing = Mock(return_value=False)
        acs.last_slew = mock_slew
        acs.current_slew = mock_slew

        mock_ephem = Mock()
        from datetime import datetime, timezone

        mock_ephem.timestamp = [datetime.fromtimestamp(1514764800.0, tz=timezone.utc)]
        mock_ephem.sun = [Mock()]
        mock_ephem.earth = [Mock()]
        mock_ephem.earth_radius_angle = [1.0]
        mock_ephem.in_eclipse = Mock(return_value=False)
        acs.ephem = mock_ephem

        # Mock constraint.in_eclipse to return False (in sunlight)
        monkeypatch.setattr(acs.constraint, "in_eclipse", lambda ra, dec, time: False)

        mode = acs.get_mode(1514764800.0)
        assert mode == ACSMode.CHARGING


class TestIsInChargingMode:
    """Test _is_in_charging_mode method."""

    def test_is_in_charging_mode_returns_true_when_ephem_lacks_in_eclipse(
        self, acs, monkeypatch
    ):
        mock_slew = Mock(spec=Slew)
        mock_slew.obstype = ObsType.CHARGE
        mock_slew.is_slewing = Mock(return_value=False)
        acs.last_slew = mock_slew
        acs.current_slew = mock_slew

        mock_ephem = Mock(spec=["other_method"])
        from datetime import datetime, timezone

        mock_ephem.timestamp = [datetime.fromtimestamp(1514764800.0, tz=timezone.utc)]
        mock_ephem.sun = [Mock()]
        mock_ephem.earth = [Mock()]
        mock_ephem.earth_radius_angle = [1.0]
        acs.ephem = mock_ephem

        # Mock constraint.in_eclipse to return False (in sunlight)
        monkeypatch.setattr(acs.constraint, "in_eclipse", lambda ra, dec, time: False)

        assert acs._is_in_charging_mode(1514764800.0) is True


class TestBatteryChargingMethods:
    """Test battery charging request and execution methods."""

    def test_request_battery_charge_enqueues_command(self, acs) -> None:
        ra, dec, roll, obsid = 45.0, 30.0, 15.0, 0xBEEF
        utime = 1514764800.0

        acs.request_battery_charge(utime, ra, dec, roll, obsid)
        assert len(acs.command_queue) == 1

    def test_request_battery_charge_creates_correct_command_type(self, acs) -> None:
        ra, dec, roll, obsid = 45.0, 30.0, 15.0, 0xBEEF
        utime = 1514764800.0

        acs.request_battery_charge(utime, ra, dec, roll, obsid)
        cmd = acs.command_queue[0]
        assert cmd.command_type == ACSCommandType.START_BATTERY_CHARGE

    def test_request_battery_charge_sets_execution_time(self, acs) -> None:
        ra, dec, roll, obsid = 45.0, 30.0, 15.0, 0xBEEF
        utime = 1514764800.0

        acs.request_battery_charge(utime, ra, dec, roll, obsid)
        cmd = acs.command_queue[0]
        assert cmd.execution_time == utime

    def test_request_battery_charge_sets_ra_dec_obsid(self, acs) -> None:
        ra, dec, roll, obsid = 45.0, 30.0, 15.0, 0xBEEF
        utime = 1514764800.0

        acs.request_battery_charge(utime, ra, dec, roll, obsid)
        cmd = acs.command_queue[0]
        assert cmd.ra == ra and cmd.dec == dec and cmd.obsid == obsid

    def test_request_battery_charge_logs_info(self, acs) -> None:
        ra, dec, roll, obsid = 45.0, 30.0, 15.0, 0xBEEF
        utime = 1514764800.0

        acs.request_battery_charge(utime, ra, dec, roll, obsid)
        # Test passes if no exception is raised - logging is tested via print statements

    def test_request_end_battery_charge_enqueues_command(self, acs) -> None:
        utime = 1514764800.0

        acs.request_end_battery_charge(utime)
        assert len(acs.command_queue) == 1

    def test_request_end_battery_charge_creates_correct_command_type(self, acs) -> None:
        utime = 1514764800.0

        acs.request_end_battery_charge(utime)
        cmd = acs.command_queue[0]
        assert cmd.command_type == ACSCommandType.END_BATTERY_CHARGE

    def test_request_end_battery_charge_sets_execution_time(self, acs) -> None:
        utime = 1514764800.0

        acs.request_end_battery_charge(utime)
        cmd = acs.command_queue[0]
        assert cmd.execution_time == utime

    def test_request_end_battery_charge_logs_info(self, acs) -> None:
        utime = 1514764800.0

        acs.request_end_battery_charge(utime)
        # Test passes if no exception is raised - logging is tested via print statements

    def test_initiate_emergency_charging_calls_emergency_module(self, acs) -> None:
        utime = 1514764800.0
        lastra, lastdec = 10.0, 20.0
        current_ppt = Mock()

        mock_emergency_charging = Mock()
        mock_charging_ppt = Mock()
        mock_charging_ppt.ra = 45.0
        mock_charging_ppt.dec = 30.0
        mock_charging_ppt.obsid = 0xC4A6
        mock_emergency_charging.initiate_emergency_charging = Mock(
            return_value=mock_charging_ppt
        )

        mock_ephem = Mock()

        with patch.object(acs, "request_battery_charge"):
            acs.initiate_emergency_charging(
                utime, mock_ephem, mock_emergency_charging, lastra, lastdec, current_ppt
            )
            mock_emergency_charging.initiate_emergency_charging.assert_called_once_with(
                utime, mock_ephem, lastra, lastdec, current_ppt
            )

    def test_initiate_emergency_charging_requests_charge(self, acs) -> None:
        utime = 1514764800.0
        lastra, lastdec = 10.0, 20.0
        current_ppt = Mock()

        mock_emergency_charging = Mock()
        mock_charging_ppt = Mock()
        mock_charging_ppt.ra = 45.0
        mock_charging_ppt.dec = 30.0
        mock_charging_ppt.roll = 15.0
        mock_charging_ppt.obsid = 0xC4A6
        mock_emergency_charging.initiate_emergency_charging = Mock(
            return_value=mock_charging_ppt
        )

        mock_ephem = Mock()

        with patch.object(acs, "request_battery_charge") as mock_request:
            acs.initiate_emergency_charging(
                utime, mock_ephem, mock_emergency_charging, lastra, lastdec, current_ppt
            )
            mock_request.assert_called_once_with(utime, 45.0, 30.0, 15.0, 0xC4A6)

    def test_initiate_emergency_charging_returns_updated_ra_dec_ppt(self, acs) -> None:
        utime = 1514764800.0
        lastra, lastdec = 10.0, 20.0
        current_ppt = Mock()

        mock_emergency_charging = Mock()
        mock_charging_ppt = Mock()
        mock_charging_ppt.ra = 45.0
        mock_charging_ppt.dec = 30.0
        mock_charging_ppt.roll = 15.0
        mock_charging_ppt.obsid = 0xC4A6
        mock_emergency_charging.initiate_emergency_charging = Mock(
            return_value=mock_charging_ppt
        )

        mock_ephem = Mock()

        with patch.object(acs, "request_battery_charge"):
            ra, dec, ppt = acs.initiate_emergency_charging(
                utime, mock_ephem, mock_emergency_charging, lastra, lastdec, current_ppt
            )
            assert ra == 45.0

    def test_initiate_emergency_charging_returns_correct_dec(self, acs) -> None:
        utime = 1514764800.0
        lastra, lastdec = 10.0, 20.0
        current_ppt = Mock()

        mock_emergency_charging = Mock()
        mock_charging_ppt = Mock()
        mock_charging_ppt.ra = 45.0
        mock_charging_ppt.dec = 30.0
        mock_charging_ppt.roll = 15.0
        mock_charging_ppt.obsid = 0xC4A6
        mock_emergency_charging.initiate_emergency_charging = Mock(
            return_value=mock_charging_ppt
        )

        mock_ephem = Mock()

        with patch.object(acs, "request_battery_charge"):
            ra, dec, ppt = acs.initiate_emergency_charging(
                utime, mock_ephem, mock_emergency_charging, lastra, lastdec, current_ppt
            )
            assert dec == 30.0

    def test_initiate_emergency_charging_returns_ppt(self, acs) -> None:
        utime = 1514764800.0
        lastra, lastdec = 10.0, 20.0
        current_ppt = Mock()

        mock_emergency_charging = Mock()
        mock_charging_ppt = Mock()
        mock_charging_ppt.ra = 45.0
        mock_charging_ppt.dec = 30.0
        mock_charging_ppt.roll = 15.0
        mock_charging_ppt.obsid = 0xC4A6
        mock_emergency_charging.initiate_emergency_charging = Mock(
            return_value=mock_charging_ppt
        )

        mock_ephem = Mock()

        with patch.object(acs, "request_battery_charge"):
            ra, dec, ppt = acs.initiate_emergency_charging(
                utime, mock_ephem, mock_emergency_charging, lastra, lastdec, current_ppt
            )
            assert ppt == mock_charging_ppt

    def test_initiate_emergency_charging_failure_does_not_request(self, acs) -> None:
        utime = 1514764800.0
        lastra, lastdec = 10.0, 20.0
        current_ppt = Mock()

        mock_emergency_charging = Mock()
        mock_emergency_charging.initiate_emergency_charging = Mock(return_value=None)

        mock_ephem = Mock()

        with patch.object(acs, "request_battery_charge") as mock_request:
            acs.initiate_emergency_charging(
                utime, mock_ephem, mock_emergency_charging, lastra, lastdec, current_ppt
            )
            mock_request.assert_not_called()

    def test_initiate_emergency_charging_failure_returns_lastra(self, acs) -> None:
        utime = 1514764800.0
        lastra, lastdec = 10.0, 20.0
        current_ppt = Mock()

        mock_emergency_charging = Mock()
        mock_emergency_charging.initiate_emergency_charging = Mock(return_value=None)

        mock_ephem = Mock()

        with patch.object(acs, "request_battery_charge"):
            ra, dec, ppt = acs.initiate_emergency_charging(
                utime, mock_ephem, mock_emergency_charging, lastra, lastdec, current_ppt
            )
            assert ra == lastra

    def test_initiate_emergency_charging_failure_returns_lastdec(self, acs) -> None:
        utime = 1514764800.0
        lastra, lastdec = 10.0, 20.0
        current_ppt = Mock()

        mock_emergency_charging = Mock()
        mock_emergency_charging.initiate_emergency_charging = Mock(return_value=None)

        mock_ephem = Mock()

        with patch.object(acs, "request_battery_charge"):
            ra, dec, ppt = acs.initiate_emergency_charging(
                utime, mock_ephem, mock_emergency_charging, lastra, lastdec, current_ppt
            )
            assert dec == lastdec

    def test_initiate_emergency_charging_failure_returns_none_ppt(self, acs) -> None:
        utime = 1514764800.0
        lastra, lastdec = 10.0, 20.0
        current_ppt = Mock()

        mock_emergency_charging = Mock()
        mock_emergency_charging.initiate_emergency_charging = Mock(return_value=None)

        mock_ephem = Mock()

        with patch.object(acs, "request_battery_charge"):
            ra, dec, ppt = acs.initiate_emergency_charging(
                utime, mock_ephem, mock_emergency_charging, lastra, lastdec, current_ppt
            )
            assert ppt is None

    def test_start_battery_charge_executes_enqueue_command(self, acs) -> None:
        command = ACSCommand(
            command_type=ACSCommandType.START_BATTERY_CHARGE,
            execution_time=1514764800.0,
            ra=45.0,
            dec=30.0,
            obsid=0xBEEF,
        )

        with (
            patch.object(acs, "enqueue_command") as mock_enqueue_command,
            patch(
                "conops.Pointing.visibility",
                new=lambda self, *args, **kwargs: (
                    setattr(self, "windows", [[1514764800.0, 1514764900.0]]),
                    0,
                )[-1],
            ),
            patch("conops.Pointing.next_vis", return_value=1514764800.0),
        ):
            acs._start_battery_charge(command, 1514764800.0)
            # Check that enqueue_command was called (which will enqueue a SLEW_TO_TARGET command)
            assert mock_enqueue_command.call_count == 1
            enqueued_command = mock_enqueue_command.call_args[0][0]
            assert enqueued_command.command_type == ACSCommandType.SLEW_TO_TARGET
            assert enqueued_command.slew.endra == 45.0
            assert enqueued_command.slew.enddec == 30.0
            assert enqueued_command.slew.obsid == 0xBEEF
            assert enqueued_command.slew.obstype == ObsType.CHARGE

    def test_start_battery_charge_logs(self, acs) -> None:
        command = ACSCommand(
            command_type=ACSCommandType.START_BATTERY_CHARGE,
            execution_time=1514764800.0,
            ra=45.0,
            dec=30.0,
            obsid=0xBEEF,
        )

        with (
            patch.object(acs, "enqueue_command"),
            patch(
                "conops.Pointing.visibility",
                new=lambda self, *args, **kwargs: (
                    setattr(self, "windows", [[1514764800.0, 1514764900.0]]),
                    0,
                )[-1],
            ),
            patch("conops.Pointing.next_vis", return_value=1514764800.0),
        ):
            acs._start_battery_charge(command, 1514764800.0)
            # Test passes if no exception is raised - logging is tested via print statements

    def test_start_battery_charge_missing_params_does_not_enqueue_command(
        self, acs
    ) -> None:
        command = ACSCommand(
            command_type=ACSCommandType.START_BATTERY_CHARGE,
            execution_time=1514764800.0,
            ra=None,
            dec=None,
            obsid=None,
        )

        with patch.object(acs, "enqueue_command") as mock_enqueue_command:
            acs._start_battery_charge(command, 1514764800.0)
            mock_enqueue_command.assert_not_called()

    def test_end_battery_charge_does_not_return_to_last_ppt(self, acs) -> None:
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt

        with patch.object(acs, "_enqueue_slew") as mock_enqueue_slew:
            acs._end_battery_charge(1514764800.0)

        mock_enqueue_slew.assert_not_called()

    def test_end_battery_charge_keeps_pending_target_slew(self, acs) -> None:
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt
        pending_slew = Mock(spec=Slew)
        pending_command = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764860.0,
            slew=pending_slew,
        )
        acs.command_queue = [pending_command]

        with patch.object(acs, "_enqueue_slew") as mock_enqueue_slew:
            acs._end_battery_charge(1514764800.0)

        mock_enqueue_slew.assert_not_called()
        assert acs.command_queue == [pending_command]

    def test_end_battery_charge_no_last_ppt_does_not_enqueue_command(self, acs) -> None:
        acs.last_ppt = None

        with patch.object(acs, "enqueue_command") as mock_enqueue_command:
            acs._end_battery_charge(1514764800.0)
            mock_enqueue_command.assert_not_called()

    def test_end_battery_charge_no_last_ppt_logs(self, acs) -> None:
        acs.last_ppt = None

        # Test passes if no exception is raised - logging is tested via print statements
        acs._end_battery_charge(1514764800.0)

    def test_process_commands_calls_start_battery_charge(self, acs) -> None:
        command = ACSCommand(
            command_type=ACSCommandType.START_BATTERY_CHARGE,
            execution_time=1514764800.0,
            ra=45.0,
            dec=30.0,
            obsid=0xBEEF,
        )
        acs.command_queue = [command]

        with patch.object(acs, "_start_battery_charge") as mock_start:
            acs._process_commands(1514764800.0)
            mock_start.assert_called_once_with(command, 1514764800.0)

    def test_process_commands_calls_end_battery_charge(self, acs) -> None:
        command = ACSCommand(
            command_type=ACSCommandType.END_BATTERY_CHARGE,
            execution_time=1514764800.0,
        )
        acs.command_queue = [command]

        with patch.object(acs, "_end_battery_charge") as mock_end:
            acs._process_commands(1514764800.0)
            mock_end.assert_called_once_with(1514764800.0)
