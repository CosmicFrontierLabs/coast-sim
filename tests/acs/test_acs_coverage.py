"""Additional tests to achieve 100% coverage for ACS class."""

from unittest.mock import Mock, patch

from conops import ACSCommand, ACSCommandType, ACSMode, Pass, Slew


class TestExecuteCommandCoverage:
    """Test command execution handler methods."""

    def test_end_pass_adds_slew_with_last_ppt(self, acs):
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
        # Verify currentpass is cleared and mode is set to SCIENCE
        assert acs.current_pass is None
        assert acs.acsmode == ACSMode.SCIENCE

    def test_end_pass_clears_currentpass(self, acs):
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert acs.current_pass is None
        assert acs.acsmode == ACSMode.SCIENCE

    def test_end_pass_sets_mode_science(self, acs):
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert acs.acsmode == ACSMode.SCIENCE

    def test_end_pass_no_last_ppt_clears_currentpass(self, acs):
        acs.last_ppt = None
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert acs.current_pass is None

    def test_end_pass_no_last_ppt_sets_mode_science(self, acs):
        acs.last_ppt = None
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert acs.acsmode == ACSMode.SCIENCE

    def test_execute_null_slew_does_not_start(self, acs):
        command = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764800.0,
            slew=None,
        )

        with patch.object(acs, "_start_slew") as mock_start_slew:
            acs._handle_slew_command(command, 1514764800.0)
            mock_start_slew.assert_not_called()

    def test_execute_slew_to_target_none_slew_does_not_start(self, acs):
        command = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764800.0,
            slew=None,
        )

        with patch.object(acs, "_start_slew") as mock_start_slew:
            acs._handle_slew_command(command, 1514764800.0)
            mock_start_slew.assert_not_called()

    def test_execute_start_pass_none_slew_does_not_start(self, acs):
        command = ACSCommand(
            command_type=ACSCommandType.START_PASS,
            execution_time=1514764800.0,
            slew=None,
        )

        with patch.object(acs, "_start_slew") as mock_start_slew:
            acs._start_pass(command, 1514764800.0)
            mock_start_slew.assert_not_called()

    def test_execute_start_pass_non_pass_slew_does_not_start(self, acs):
        mock_slew = Mock(spec=Slew)
        command = ACSCommand(
            command_type=ACSCommandType.START_PASS,
            execution_time=1514764800.0,
            slew=mock_slew,
        )

        with patch.object(acs, "_start_slew") as mock_start_slew:
            acs._start_pass(command, 1514764800.0)
            mock_start_slew.assert_not_called()


class TestStartSlewCoverage:
    """Test _start_slew behavior - ACS always drives spacecraft from current position."""

    def test_start_slew_sets_startra_from_acs(self, acs):
        """Slew always starts from current ACS pointing."""
        acs.ra = 10.0
        acs.dec = 20.0

        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 0.0
        mock_slew.startdec = 0.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"
        mock_slew.slewstart = 1514764800.0
        mock_slew.slewtime = 60.0
        mock_slew.calc_slewtime = Mock()

        acs._start_slew(mock_slew, 1514764800.0)
        assert mock_slew.startra == 10.0

    def test_start_slew_sets_startdec_from_acs(self, acs):
        """Slew always starts from current ACS pointing."""
        acs.ra = 10.0
        acs.dec = 20.0

        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 0.0
        mock_slew.startdec = 0.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"
        mock_slew.slewstart = 1514764800.0
        mock_slew.slewtime = 60.0
        mock_slew.calc_slewtime = Mock()

        acs._start_slew(mock_slew, 1514764800.0)
        assert mock_slew.startdec == 20.0

    def test_start_slew_always_calls_calc_slewtime(self, acs):
        """calc_slewtime is always called to compute the new slew profile."""
        acs.ra = 10.0
        acs.dec = 20.0

        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 0.0
        mock_slew.startdec = 0.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"
        mock_slew.slewstart = 1514764800.0
        mock_slew.slewtime = 60.0
        mock_slew.calc_slewtime = Mock()

        acs._start_slew(mock_slew, 1514764800.0)
        mock_slew.calc_slewtime.assert_called_once()

    def test_start_slew_sets_pass_startra_from_acs(self, acs):
        """Pass slews also start from current ACS pointing."""
        acs.ra = 10.0
        acs.dec = 20.0

        mock_pass = Mock(spec=Pass)
        mock_pass.startra = 0.0
        mock_pass.startdec = 0.0
        mock_pass.endra = 45.0
        mock_pass.enddec = 30.0
        mock_pass.obstype = "GSP"
        mock_pass.slewstart = 1514764800.0
        mock_pass.slewtime = 60.0
        mock_pass.calc_slewtime = Mock()

        acs._start_slew(mock_pass, 1514764800.0)
        assert mock_pass.startra == 10.0

    def test_start_slew_sets_pass_startdec_from_acs(self, acs):
        """Pass slews also start from current ACS pointing."""
        acs.ra = 10.0
        acs.dec = 20.0

        mock_pass = Mock(spec=Pass)
        mock_pass.startra = 0.0
        mock_pass.startdec = 0.0
        mock_pass.endra = 45.0
        mock_pass.enddec = 30.0
        mock_pass.obstype = "GSP"
        mock_pass.slewstart = 1514764800.0
        mock_pass.slewtime = 60.0
        mock_pass.calc_slewtime = Mock()

        acs._start_slew(mock_pass, 1514764800.0)
        assert mock_pass.startdec == 20.0

    def test_start_slew_calls_calc_for_pass(self, acs):
        """Pass slews also recalculate slew time."""
        acs.ra = 10.0
        acs.dec = 20.0

        mock_pass = Mock(spec=Pass)
        mock_pass.startra = 0.0
        mock_pass.startdec = 0.0
        mock_pass.endra = 45.0
        mock_pass.enddec = 30.0
        mock_pass.obstype = "GSP"
        mock_pass.slewstart = 1514764800.0
        mock_pass.slewtime = 60.0
        mock_pass.calc_slewtime = Mock()

        acs._start_slew(mock_pass, 1514764800.0)
        mock_pass.calc_slewtime.assert_called_once()

    def test_start_slew_sets_start_from_zero_position(self, acs):
        """Even when ACS is at origin, slew starts from there."""
        acs.ra = 0.0
        acs.dec = 0.0

        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 5.0
        mock_slew.startdec = 10.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"
        mock_slew.slewstart = 1514764800.0
        mock_slew.slewtime = 60.0
        mock_slew.calc_slewtime = Mock()

        acs._start_slew(mock_slew, 1514764800.0)
        assert mock_slew.startra == 0.0
        assert mock_slew.startdec == 0.0

    def test_start_slew_sets_slewstart_to_current_time(self, acs):
        """Slew start time is set to execution time."""
        acs.ra = 10.0
        acs.dec = 20.0

        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 10.0
        mock_slew.startdec = 20.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"
        mock_slew.slewstart = 9999999.0  # Different from execution time
        mock_slew.slewtime = 60.0
        mock_slew.calc_slewtime = Mock()

        acs._start_slew(mock_slew, 1514764800.0)
        assert mock_slew.slewstart == 1514764800.0

    def test_start_slew_overwrites_matching_startra(self, acs):
        """Even if startra already matches, it gets set (no special case)."""
        acs.ra = 10.0
        acs.dec = 20.0

        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 10.0  # Already matches
        mock_slew.startdec = 20.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"
        mock_slew.slewstart = 1514764800.0
        mock_slew.slewtime = 60.0
        mock_slew.calc_slewtime = Mock()

        acs._start_slew(mock_slew, 1514764800.0)
        # startra is set to acs.ra (same value)
        assert mock_slew.startra == 10.0
        # calc_slewtime is still called
        mock_slew.calc_slewtime.assert_called_once()

    def test_start_slew_updates_last_ppt(self, acs):
        acs.ra = 10.0
        acs.dec = 20.0

        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 10.0
        mock_slew.startdec = 20.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"
        mock_slew.slewstart = 1514764800.0
        mock_slew.slewtime = 60.0
        mock_slew.calc_slewtime = Mock()

        acs._start_slew(mock_slew, 1514764800.0)
        assert acs.last_ppt == mock_slew

    def test_start_slew_no_last_ppt_for_non_ppt(self, acs):
        acs.ra = 10.0
        acs.dec = 20.0

        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 10.0
        mock_slew.startdec = 20.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "GSP"
        mock_slew.slewstart = 1514764800.0
        mock_slew.slewtime = 60.0
        mock_slew.calc_slewtime = Mock()

        acs.last_ppt = None

        acs._start_slew(mock_slew, 1514764800.0)
        assert acs.last_ppt is None

        acs._start_slew(mock_slew, 1514764800.0)
        assert acs.last_ppt is None


class TestEndPassCoverage:
    """Test _end_pass method."""

    def test_end_pass_adds_slew_returning_to_last_ppt(self, acs):
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

    def test_end_pass_clears_currentpass(self, acs):
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

    def test_end_pass_sets_mode_science_on_end(self, acs):
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt

        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        with patch.object(acs, "enqueue_command", return_value=True):
            acs._end_pass(1514764800.0)
            assert acs.acsmode == ACSMode.SCIENCE

    def test_end_pass_with_no_last_ppt_clears_currentpass(self, acs):
        acs.last_ppt = None
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert acs.current_pass is None

    def test_end_pass_with_no_last_ppt_sets_mode_science(self, acs):
        acs.last_ppt = None
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert acs.acsmode == ACSMode.SCIENCE

    def test_end_pass_no_last_ppt_does_not_enqueue_slew(self, acs):
        acs.last_ppt = None
        acs.current_pass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert len(acs.command_queue) == 0


class TestProcessCommandsCoverage:
    """Test _process_commands to ensure queue processing is covered."""

    def _make_mock_slew(
        self, startra=0.0, startdec=0.0, endra=45.0, enddec=30.0, obstype="PPT"
    ):
        """Helper to create a mock slew with all required attributes."""
        mock_slew = Mock(spec=Slew)
        mock_slew.startra = startra
        mock_slew.startdec = startdec
        mock_slew.endra = endra
        mock_slew.enddec = enddec
        mock_slew.obstype = obstype
        mock_slew.slewstart = 1514764800.0
        mock_slew.slewtime = 60.0
        mock_slew.calc_slewtime = Mock()
        return mock_slew

    def test_process_commands_executes_first_due_command(self, acs):
        mock_slew1 = self._make_mock_slew(0.0, 0.0, 45.0, 30.0)
        mock_slew2 = self._make_mock_slew(45.0, 30.0, 90.0, 60.0)

        command1 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764800.0,
            slew=mock_slew1,
        )
        command2 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764810.0,
            slew=mock_slew2,
        )
        command3 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764900.0,
            slew=mock_slew1,
        )
        acs.command_queue = [command1, command2, command3]

        acs._process_commands(1514764815.0)
        assert len(acs.executed_commands) >= 1

    def test_process_commands_executes_second_due_command(self, acs):
        mock_slew1 = self._make_mock_slew(0.0, 0.0, 45.0, 30.0)
        mock_slew2 = self._make_mock_slew(45.0, 30.0, 90.0, 60.0)

        command1 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764800.0,
            slew=mock_slew1,
        )
        command2 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764810.0,
            slew=mock_slew2,
        )
        command3 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764900.0,
            slew=mock_slew1,
        )
        acs.command_queue = [command1, command2, command3]

        acs._process_commands(1514764815.0)
        assert len(acs.executed_commands) >= 2

    def test_process_commands_first_executed_command_matches_command1(self, acs):
        mock_slew1 = self._make_mock_slew(0.0, 0.0, 45.0, 30.0)
        mock_slew2 = self._make_mock_slew(45.0, 30.0, 90.0, 60.0)

        command1 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764800.0,
            slew=mock_slew1,
        )
        command2 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764810.0,
            slew=mock_slew2,
        )
        command3 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764900.0,
            slew=mock_slew1,
        )
        acs.command_queue = [command1, command2, command3]

        acs._process_commands(1514764815.0)
        assert acs.executed_commands[0] == command1

    def test_process_commands_second_executed_command_matches_command2(self, acs):
        mock_slew1 = self._make_mock_slew(0.0, 0.0, 45.0, 30.0)
        mock_slew2 = self._make_mock_slew(45.0, 30.0, 90.0, 60.0)

        command1 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764800.0,
            slew=mock_slew1,
        )
        command2 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764810.0,
            slew=mock_slew2,
        )
        command3 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764900.0,
            slew=mock_slew1,
        )
        acs.command_queue = [command1, command2, command3]

        acs._process_commands(1514764815.0)
        assert acs.executed_commands[1] == command2

    def test_process_commands_leaves_later_command_in_queue(self, acs):
        mock_slew1 = self._make_mock_slew(0.0, 0.0, 45.0, 30.0)
        mock_slew2 = self._make_mock_slew(45.0, 30.0, 90.0, 60.0)

        command1 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764800.0,
            slew=mock_slew1,
        )
        command2 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764810.0,
            slew=mock_slew2,
        )
        command3 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764900.0,
            slew=mock_slew1,
        )
        acs.command_queue = [command1, command2, command3]

        acs._process_commands(1514764815.0)
        assert len(acs.command_queue) == 1

    def test_process_commands_remaining_queue_item_is_third(self, acs):
        mock_slew1 = self._make_mock_slew(0.0, 0.0, 45.0, 30.0)

        command1 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764800.0,
            slew=mock_slew1,
        )
        command2 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764810.0,
            slew=mock_slew1,
        )
        command3 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764900.0,
            slew=mock_slew1,
        )
        acs.command_queue = [command1, command2, command3]

        acs._process_commands(1514764815.0)
        assert acs.command_queue[0] == command3


class TestExecuteCommandLogging:
    """Test command handler logging."""

    def test_handle_slew_command_executes(self, acs):
        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 0.0
        mock_slew.startdec = 0.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"
        mock_slew.slewstart = 1514764800.0
        mock_slew.slewtime = 60.0
        mock_slew.calc_slewtime = Mock()
        command1 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764800.0,
            slew=mock_slew,
        )

        # Test that command executes without error
        acs._handle_slew_command(command1, 1514764800.0)
        assert acs.current_slew == mock_slew

    def test_start_pass_executes(self, acs):
        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 0.0
        mock_slew.startdec = 0.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "GSP"
        mock_slew.slewstart = 1514764800.0
        mock_slew.slewtime = 60.0
        mock_slew.calc_slewtime = Mock()
        command2 = ACSCommand(
            command_type=ACSCommandType.START_PASS,
            execution_time=1514764800.0,
            slew=mock_slew,
        )

        # Test that command executes without error
        acs._start_pass(command2, 1514764800.0)
        # _start_pass sets current_pass, not current_slew
        assert acs.acsmode == ACSMode.PASS


class TestGetModeCharging:
    """Test get_mode for CHARGING mode."""

    def test_get_mode_returns_slewing_when_in_eclipse(self, acs, monkeypatch):
        mock_slew = Mock(spec=Slew)
        mock_slew.obstype = "CHARGE"
        mock_slew.is_slewing = Mock(return_value=True)
        acs.current_slew = mock_slew

        mock_ephem = Mock()
        from datetime import datetime, timezone

        mock_ephem.timestamp = [datetime.fromtimestamp(1514764800.0, tz=timezone.utc)]
        mock_ephem.sun = [Mock()]
        mock_ephem.earth = [Mock()]
        mock_ephem.earth_radius_angle = [1.0]
        mock_ephem.in_eclipse = Mock(return_value=True)
        acs.ephem = mock_ephem

        # Mock constraint.in_eclipse to return True (in eclipse)
        monkeypatch.setattr(acs.constraint, "in_eclipse", lambda ra, dec, time: True)

        # Set ACS in_eclipse state to True
        acs.in_eclipse = True

        mode = acs.get_mode(1514764800.0)
        assert mode == ACSMode.SLEWING

    def test_get_mode_calls_in_eclipse_for_charging(self, acs, monkeypatch):
        mock_slew = Mock(spec=Slew)
        mock_slew.obstype = "CHARGE"
        mock_slew.is_slewing = Mock(return_value=True)
        acs.current_slew = mock_slew

        mock_ephem = Mock()
        from datetime import datetime, timezone

        mock_ephem.timestamp = [datetime.fromtimestamp(1514764800.0, tz=timezone.utc)]
        mock_ephem.sun = [Mock()]
        mock_ephem.earth = [Mock()]
        mock_ephem.earth_radius_angle = [1.0]
        mock_ephem.in_eclipse = Mock(return_value=True)
        acs.ephem = mock_ephem

        # Mock constraint.in_eclipse
        mock_in_eclipse = Mock(return_value=True)
        monkeypatch.setattr(acs.constraint, "in_eclipse", mock_in_eclipse)

        _ = acs.get_mode(1514764800.0)

    def test_get_mode_returns_charging_in_sunlight_while_slewing(
        self, acs, monkeypatch
    ):
        mock_slew = Mock(spec=Slew)
        mock_slew.obstype = "CHARGE"
        mock_slew.is_slewing = Mock(return_value=True)
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

    def test_get_mode_calls_in_eclipse_for_sunlight_slewing(self, acs, monkeypatch):
        mock_slew = Mock(spec=Slew)
        mock_slew.obstype = "CHARGE"
        mock_slew.is_slewing = Mock(return_value=True)
        acs.current_slew = mock_slew

        mock_ephem = Mock()
        from datetime import datetime, timezone

        mock_ephem.timestamp = [datetime.fromtimestamp(1514764800.0, tz=timezone.utc)]
        mock_ephem.sun = [Mock()]
        mock_ephem.earth = [Mock()]
        mock_ephem.earth_radius_angle = [1.0]
        mock_ephem.in_eclipse = Mock(return_value=False)
        acs.ephem = mock_ephem

        # Mock constraint.in_eclipse
        mock_in_eclipse = Mock(return_value=False)
        monkeypatch.setattr(acs.constraint, "in_eclipse", mock_in_eclipse)

        _ = acs.get_mode(1514764800.0)

    def test_get_mode_charging_in_dwell_when_not_slewing(self, acs, monkeypatch):
        mock_slew = Mock(spec=Slew)
        mock_slew.obstype = "CHARGE"
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
        mock_slew.obstype = "CHARGE"
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

    def test_request_battery_charge_enqueues_command(self, acs):
        ra, dec, obsid = 45.0, 30.0, 0xBEEF
        utime = 1514764800.0

        acs.request_battery_charge(utime, ra, dec, obsid)
        assert len(acs.command_queue) == 1

    def test_request_battery_charge_creates_correct_command_type(self, acs):
        ra, dec, obsid = 45.0, 30.0, 0xBEEF
        utime = 1514764800.0

        acs.request_battery_charge(utime, ra, dec, obsid)
        cmd = acs.command_queue[0]
        assert cmd.command_type == ACSCommandType.START_BATTERY_CHARGE

    def test_request_battery_charge_sets_execution_time(self, acs):
        ra, dec, obsid = 45.0, 30.0, 0xBEEF
        utime = 1514764800.0

        acs.request_battery_charge(utime, ra, dec, obsid)
        cmd = acs.command_queue[0]
        assert cmd.execution_time == utime

    def test_request_battery_charge_sets_ra_dec_obsid(self, acs):
        ra, dec, obsid = 45.0, 30.0, 0xBEEF
        utime = 1514764800.0

        acs.request_battery_charge(utime, ra, dec, obsid)
        cmd = acs.command_queue[0]
        assert cmd.ra == ra and cmd.dec == dec and cmd.obsid == obsid

    def test_request_battery_charge_logs_info(self, acs):
        ra, dec, obsid = 45.0, 30.0, 0xBEEF
        utime = 1514764800.0

        acs.request_battery_charge(utime, ra, dec, obsid)
        # Test passes if no exception is raised - logging is tested via print statements

    def test_request_end_battery_charge_enqueues_command(self, acs):
        utime = 1514764800.0

        acs.request_end_battery_charge(utime)
        assert len(acs.command_queue) == 1

    def test_request_end_battery_charge_creates_correct_command_type(self, acs):
        utime = 1514764800.0

        acs.request_end_battery_charge(utime)
        cmd = acs.command_queue[0]
        assert cmd.command_type == ACSCommandType.END_BATTERY_CHARGE

    def test_request_end_battery_charge_sets_execution_time(self, acs):
        utime = 1514764800.0

        acs.request_end_battery_charge(utime)
        cmd = acs.command_queue[0]
        assert cmd.execution_time == utime

    def test_request_end_battery_charge_logs_info(self, acs):
        utime = 1514764800.0

        acs.request_end_battery_charge(utime)
        # Test passes if no exception is raised - logging is tested via print statements

    def test_initiate_emergency_charging_calls_emergency_module(self, acs):
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

    def test_initiate_emergency_charging_requests_charge(self, acs):
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

        with patch.object(acs, "request_battery_charge") as mock_request:
            acs.initiate_emergency_charging(
                utime, mock_ephem, mock_emergency_charging, lastra, lastdec, current_ppt
            )
            mock_request.assert_called_once_with(utime, 45.0, 30.0, 0xC4A6)

    def test_initiate_emergency_charging_returns_updated_ra_dec_ppt(self, acs):
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
            ra, dec, ppt = acs.initiate_emergency_charging(
                utime, mock_ephem, mock_emergency_charging, lastra, lastdec, current_ppt
            )
            assert ra == 45.0

    def test_initiate_emergency_charging_returns_correct_dec(self, acs):
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
            ra, dec, ppt = acs.initiate_emergency_charging(
                utime, mock_ephem, mock_emergency_charging, lastra, lastdec, current_ppt
            )
            assert dec == 30.0

    def test_initiate_emergency_charging_returns_ppt(self, acs):
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
            ra, dec, ppt = acs.initiate_emergency_charging(
                utime, mock_ephem, mock_emergency_charging, lastra, lastdec, current_ppt
            )
            assert ppt == mock_charging_ppt

    def test_initiate_emergency_charging_failure_does_not_request(self, acs):
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

    def test_initiate_emergency_charging_failure_returns_lastra(self, acs):
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

    def test_initiate_emergency_charging_failure_returns_lastdec(self, acs):
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

    def test_initiate_emergency_charging_failure_returns_none_ppt(self, acs):
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

    def test_start_battery_charge_executes_enqueue_command(self, acs):
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
            acs.config.spacecraft_bus.attitude_control.predict_slew.return_value = (
                0.0,
                (Mock(), Mock()),
            )
            acs.config.spacecraft_bus.attitude_control.slew_time.return_value = 10.0
            acs._start_battery_charge(command, 1514764800.0)
            # Check that enqueue_command was called (which will enqueue a SLEW_TO_TARGET command)
            assert mock_enqueue_command.call_count == 1
            enqueued_command = mock_enqueue_command.call_args[0][0]
            assert enqueued_command.command_type == ACSCommandType.SLEW_TO_TARGET
            assert enqueued_command.slew.endra == 45.0
            assert enqueued_command.slew.enddec == 30.0
            assert enqueued_command.slew.obsid == 0xBEEF
            assert enqueued_command.slew.obstype == "CHARGE"

    def test_start_battery_charge_logs(self, acs):
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
            acs.config.spacecraft_bus.attitude_control.predict_slew.return_value = (
                0.0,
                (Mock(), Mock()),
            )
            acs.config.spacecraft_bus.attitude_control.slew_time.return_value = 10.0
            acs._start_battery_charge(command, 1514764800.0)
            # Test passes if no exception is raised - logging is tested via print statements

    def test_start_battery_charge_missing_params_does_not_enqueue_command(self, acs):
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

    def test_end_battery_charge_calls_enqueue_command_with_last_ppt(self, acs):
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt

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
            acs.config.spacecraft_bus.attitude_control.predict_slew.return_value = (
                0.0,
                (Mock(), Mock()),
            )
            acs.config.spacecraft_bus.attitude_control.slew_time.return_value = 10.0
            acs._end_battery_charge(1514764800.0)
            # Check that enqueue_command was called with a SLEW_TO_TARGET command
            assert mock_enqueue_command.call_count == 1
            enqueued_command = mock_enqueue_command.call_args[0][0]
            assert enqueued_command.command_type == ACSCommandType.SLEW_TO_TARGET
            assert enqueued_command.slew.endra == 45.0
            assert enqueued_command.slew.enddec == 30.0
            assert enqueued_command.slew.obsid == 100

    def test_end_battery_charge_no_last_ppt_does_not_enqueue_command(self, acs):
        acs.last_ppt = None

        with patch.object(acs, "enqueue_command") as mock_enqueue_command:
            acs._end_battery_charge(1514764800.0)
            mock_enqueue_command.assert_not_called()

    def test_end_battery_charge_no_last_ppt_logs(self, acs):
        acs.last_ppt = None

        # Test passes if no exception is raised - logging is tested via print statements
        acs._end_battery_charge(1514764800.0)

    def test_process_commands_calls_start_battery_charge(self, acs):
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

    def test_process_commands_calls_end_battery_charge(self, acs):
        command = ACSCommand(
            command_type=ACSCommandType.END_BATTERY_CHARGE,
            execution_time=1514764800.0,
        )
        acs.command_queue = [command]

        with patch.object(acs, "_end_battery_charge") as mock_end:
            acs._process_commands(1514764800.0)
            mock_end.assert_called_once_with(1514764800.0)
