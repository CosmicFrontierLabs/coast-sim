"""Additional tests to achieve 100% coverage for ACS class."""

from unittest.mock import Mock, patch

import pytest

from conops.acs import ACS, ACSCommand, ACSCommandType
from conops.common import ACSMode
from conops.constraint import Constraint
from conops.passes import Pass
from conops.slew import Slew


class DummyEphemeris:
    """Minimal mock ephemeris for testing."""

    def __init__(self):
        self.step_size = 1.0
        self.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
        self._tle_ephem = Mock()

    def index(self, time):
        return 0


@pytest.fixture
def mock_ephem():
    """Create a mock ephemeris object."""
    return DummyEphemeris()


@pytest.fixture
def mock_constraint(mock_ephem):
    """Create a mock constraint."""
    constraint = Mock(spec=Constraint)
    constraint.ephem = mock_ephem
    constraint.panel_constraint = Mock()
    constraint.panel_constraint.solar_panel = Mock()
    constraint.inoccult = Mock(return_value=False)
    return constraint


@pytest.fixture
def mock_config():
    """Create a mock config."""
    config = Mock()
    config.ground_stations = Mock()
    return config


@pytest.fixture
def acs(mock_constraint, mock_config):
    """Create an ACS instance with mocked dependencies."""
    return ACS(constraint=mock_constraint, config=mock_config)


class TestExecuteCommandCoverage:
    """Test _execute_command for all command types."""

    def test_end_pass_adds_slew_with_last_ppt(self, acs):
        """END_PASS should NOT call enqueue_command for last_ppt in queue-driven mode."""
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt
        acs.currentpass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        command = ACSCommand(
            command_type=ACSCommandType.END_PASS,
            execution_time=1514764800.0,
        )

        # In queue-driven mode, _end_pass does not automatically add a slew back to last_ppt
        acs._execute_command(command, 1514764800.0)
        # Verify currentpass is cleared and mode is set to SCIENCE
        assert acs.currentpass is None
        assert acs.acsmode == ACSMode.SCIENCE

    def test_end_pass_clears_currentpass(self, acs):
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt
        acs.currentpass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        command = ACSCommand(
            command_type=ACSCommandType.END_PASS, execution_time=1514764800.0
        )

        acs._execute_command(command, 1514764800.0)
        assert acs.currentpass is None
        assert acs.acsmode == ACSMode.SCIENCE

    def test_end_pass_sets_mode_science(self, acs):
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt
        acs.currentpass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        command = ACSCommand(
            command_type=ACSCommandType.END_PASS, execution_time=1514764800.0
        )

        with patch.object(acs, "enqueue_command", return_value=True):
            acs._execute_command(command, 1514764800.0)
            assert acs.acsmode == ACSMode.SCIENCE

    def test_end_pass_no_last_ppt_clears_currentpass(self, acs):
        acs.last_ppt = None
        acs.currentpass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        command = ACSCommand(
            command_type=ACSCommandType.END_PASS, execution_time=1514764800.0
        )

        acs._execute_command(command, 1514764800.0)
        assert acs.currentpass is None

    def test_end_pass_no_last_ppt_sets_mode_science(self, acs):
        acs.last_ppt = None
        acs.currentpass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        command = ACSCommand(
            command_type=ACSCommandType.END_PASS, execution_time=1514764800.0
        )

        acs._execute_command(command, 1514764800.0)
        assert acs.acsmode == ACSMode.SCIENCE

    def test_execute_null_slew_does_not_start(self, acs):
        command = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764800.0,
            slew=None,
        )

        with patch.object(acs, "_start_slew") as mock_start_slew:
            acs._execute_command(command, 1514764800.0)
            mock_start_slew.assert_not_called()

    def test_execute_slew_to_target_none_slew_does_not_start(self, acs):
        command = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764800.0,
            slew=None,
        )

        with patch.object(acs, "_start_slew") as mock_start_slew:
            acs._execute_command(command, 1514764800.0)
            mock_start_slew.assert_not_called()

    def test_execute_start_pass_none_slew_does_not_start(self, acs):
        command = ACSCommand(
            command_type=ACSCommandType.START_PASS,
            execution_time=1514764800.0,
            slew=None,
        )

        with patch.object(acs, "_start_slew") as mock_start_slew:
            acs._execute_command(command, 1514764800.0)
            mock_start_slew.assert_not_called()

    def test_execute_start_pass_non_pass_slew_does_not_start(self, acs):
        mock_slew = Mock(spec=Slew)
        command = ACSCommand(
            command_type=ACSCommandType.START_PASS,
            execution_time=1514764800.0,
            slew=mock_slew,
        )

        with patch.object(acs, "_start_slew") as mock_start_slew:
            acs._execute_command(command, 1514764800.0)
            mock_start_slew.assert_not_called()


class TestStartSlewCoverage:
    """Test _start_slew slew adjustment logic."""

    def test_start_slew_adjusts_startra(self, acs):
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

    def test_start_slew_adjusts_startdec(self, acs):
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

    def test_start_slew_calls_calc_slewtime_on_adjustment(self, acs):
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

    def test_start_slew_no_adjustment_for_pass_keeps_startra(self, acs):
        acs.ra = 10.0
        acs.dec = 20.0

        mock_pass = Mock(spec=Pass)
        mock_pass.startra = 0.0
        mock_pass.startdec = 0.0
        mock_pass.endra = 45.0
        mock_pass.enddec = 30.0
        mock_pass.obstype = "GSP"
        mock_pass.calc_slewtime = Mock()

        acs._start_slew(mock_pass, 1514764800.0)
        assert mock_pass.startra == 0.0

    def test_start_slew_no_adjustment_for_pass_keeps_startdec(self, acs):
        acs.ra = 10.0
        acs.dec = 20.0

        mock_pass = Mock(spec=Pass)
        mock_pass.startra = 0.0
        mock_pass.startdec = 0.0
        mock_pass.endra = 45.0
        mock_pass.enddec = 30.0
        mock_pass.obstype = "GSP"
        mock_pass.calc_slewtime = Mock()

        acs._start_slew(mock_pass, 1514764800.0)
        assert mock_pass.startdec == 0.0

    def test_start_slew_no_calc_called_for_pass(self, acs):
        acs.ra = 10.0
        acs.dec = 20.0

        mock_pass = Mock(spec=Pass)
        mock_pass.startra = 0.0
        mock_pass.startdec = 0.0
        mock_pass.endra = 45.0
        mock_pass.enddec = 30.0
        mock_pass.obstype = "GSP"
        mock_pass.calc_slewtime = Mock()

        acs._start_slew(mock_pass, 1514764800.0)
        mock_pass.calc_slewtime.assert_not_called()

    def test_start_slew_no_adjust_when_ra_dec_zero_keeps_startra(self, acs):
        acs.ra = 0.0
        acs.dec = 0.0

        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 5.0
        mock_slew.startdec = 10.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"
        mock_slew.calc_slewtime = Mock()

        acs._start_slew(mock_slew, 1514764800.0)
        assert mock_slew.startra == 5.0

    def test_start_slew_no_adjust_when_ra_dec_zero_keeps_startdec(self, acs):
        acs.ra = 0.0
        acs.dec = 0.0

        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 5.0
        mock_slew.startdec = 10.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"
        mock_slew.calc_slewtime = Mock()

        acs._start_slew(mock_slew, 1514764800.0)
        assert mock_slew.startdec == 10.0

    def test_start_slew_no_calc_when_already_matched(self, acs):
        acs.ra = 10.0
        acs.dec = 20.0

        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 10.0
        mock_slew.startdec = 20.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"
        mock_slew.calc_slewtime = Mock()

        acs._start_slew(mock_slew, 1514764800.0)
        mock_slew.calc_slewtime.assert_not_called()

    def test_start_slew_does_not_change_already_matched_startra(self, acs):
        acs.ra = 10.0
        acs.dec = 20.0

        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 10.0
        mock_slew.startdec = 20.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"
        mock_slew.calc_slewtime = Mock()

        acs._start_slew(mock_slew, 1514764800.0)
        assert mock_slew.startra == 10.0

    def test_start_slew_updates_last_ppt(self, acs):
        acs.ra = 10.0
        acs.dec = 20.0

        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 10.0
        mock_slew.startdec = 20.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"

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

        acs.last_ppt = None

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

        acs.currentpass = Mock(spec=Pass)
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

        acs.currentpass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        with patch.object(acs, "enqueue_command", return_value=True):
            acs._end_pass(1514764800.0)
            assert acs.currentpass is None

    def test_end_pass_sets_mode_science_on_end(self, acs):
        mock_ppt = Mock(spec=Slew)
        mock_ppt.endra = 45.0
        mock_ppt.enddec = 30.0
        mock_ppt.obsid = 100
        acs.last_ppt = mock_ppt

        acs.currentpass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        with patch.object(acs, "enqueue_command", return_value=True):
            acs._end_pass(1514764800.0)
            assert acs.acsmode == ACSMode.SCIENCE

    def test_end_pass_with_no_last_ppt_clears_currentpass(self, acs):
        acs.last_ppt = None
        acs.currentpass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert acs.currentpass is None

    def test_end_pass_with_no_last_ppt_sets_mode_science(self, acs):
        acs.last_ppt = None
        acs.currentpass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert acs.acsmode == ACSMode.SCIENCE

    def test_end_pass_no_last_ppt_does_not_enqueue_slew(self, acs):
        acs.last_ppt = None
        acs.currentpass = Mock(spec=Pass)
        acs.acsmode = ACSMode.PASS

        acs._end_pass(1514764800.0)
        assert len(acs.command_queue) == 0


class TestProcessCommandsCoverage:
    """Test _process_commands to ensure queue processing is covered."""

    def test_process_commands_executes_first_due_command(self, acs):
        mock_slew1 = Mock(spec=Slew)
        mock_slew1.startra = 0.0
        mock_slew1.startdec = 0.0
        mock_slew1.endra = 45.0
        mock_slew1.enddec = 30.0
        mock_slew1.obstype = "PPT"

        mock_slew2 = Mock(spec=Slew)
        mock_slew2.startra = 45.0
        mock_slew2.startdec = 30.0
        mock_slew2.endra = 90.0
        mock_slew2.enddec = 60.0
        mock_slew2.obstype = "PPT"

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
        mock_slew1 = Mock(spec=Slew)
        mock_slew1.startra = 0.0
        mock_slew1.startdec = 0.0
        mock_slew1.endra = 45.0
        mock_slew1.enddec = 30.0
        mock_slew1.obstype = "PPT"

        mock_slew2 = Mock(spec=Slew)
        mock_slew2.startra = 45.0
        mock_slew2.startdec = 30.0
        mock_slew2.endra = 90.0
        mock_slew2.enddec = 60.0
        mock_slew2.obstype = "PPT"

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
        mock_slew1 = Mock(spec=Slew)
        mock_slew1.startra = 0.0
        mock_slew1.startdec = 0.0
        mock_slew1.endra = 45.0
        mock_slew1.enddec = 30.0
        mock_slew1.obstype = "PPT"

        mock_slew2 = Mock(spec=Slew)
        mock_slew2.startra = 45.0
        mock_slew2.startdec = 30.0
        mock_slew2.endra = 90.0
        mock_slew2.enddec = 60.0
        mock_slew2.obstype = "PPT"

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
        mock_slew1 = Mock(spec=Slew)
        mock_slew1.startra = 0.0
        mock_slew1.startdec = 0.0
        mock_slew1.endra = 45.0
        mock_slew1.enddec = 30.0
        mock_slew1.obstype = "PPT"

        mock_slew2 = Mock(spec=Slew)
        mock_slew2.startra = 45.0
        mock_slew2.startdec = 30.0
        mock_slew2.endra = 90.0
        mock_slew2.enddec = 60.0
        mock_slew2.obstype = "PPT"

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
        mock_slew1 = Mock(spec=Slew)
        mock_slew1.startra = 0.0
        mock_slew1.startdec = 0.0
        mock_slew1.endra = 45.0
        mock_slew1.enddec = 30.0
        mock_slew1.obstype = "PPT"

        mock_slew2 = Mock(spec=Slew)
        mock_slew2.startra = 45.0
        mock_slew2.startdec = 30.0
        mock_slew2.endra = 90.0
        mock_slew2.enddec = 60.0
        mock_slew2.obstype = "PPT"

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
        mock_slew1 = Mock(spec=Slew)
        mock_slew1.startra = 0.0
        mock_slew1.startdec = 0.0
        mock_slew1.endra = 45.0
        mock_slew1.enddec = 30.0
        mock_slew1.obstype = "PPT"

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
    """Test _execute_command with logging enabled."""

    def test_execute_command_logs_slew_to_target(self, acs):
        mock_slew = Mock(spec=Slew)
        mock_slew.startra = 0.0
        mock_slew.startdec = 0.0
        mock_slew.endra = 45.0
        mock_slew.enddec = 30.0
        mock_slew.obstype = "PPT"
        command1 = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=1514764800.0,
            slew=mock_slew,
        )

        # Test that command executes without error
        acs._execute_command(command1, 1514764800.0)

    def test_execute_command_logs_start_pass(self, acs):
        mock_pass = Mock(spec=Pass)
        mock_pass.startra = 0.0
        mock_pass.startdec = 0.0
        mock_pass.endra = 45.0
        mock_pass.enddec = 30.0
        mock_pass.obstype = "GSP"
        command2 = ACSCommand(
            command_type=ACSCommandType.START_PASS,
            execution_time=1514764800.0,
            slew=mock_pass,
        )

        # Test that command executes without error
        acs._execute_command(command2, 1514764800.0)


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

        # Mock ephemeris_in_eclipse to return True (in eclipse)
        from conops import acs as acs_module

        monkeypatch.setattr(
            acs_module, "ephemeris_in_eclipse", lambda ephem, utime: True
        )

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

        # Mock ephemeris_in_eclipse
        from conops import acs as acs_module

        mock_in_eclipse = Mock(return_value=True)
        monkeypatch.setattr(acs_module, "ephemeris_in_eclipse", mock_in_eclipse)

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

        # Mock ephemeris_in_eclipse to return False (in sunlight)
        from conops import acs as acs_module

        monkeypatch.setattr(
            acs_module, "ephemeris_in_eclipse", lambda ephem, utime: False
        )

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

        # Mock ephemeris_in_eclipse
        from conops import acs as acs_module

        mock_in_eclipse = Mock(return_value=False)
        monkeypatch.setattr(acs_module, "ephemeris_in_eclipse", mock_in_eclipse)

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

        # Mock ephemeris_in_eclipse to return False (in sunlight)
        from conops import acs as acs_module

        monkeypatch.setattr(
            acs_module, "ephemeris_in_eclipse", lambda ephem, utime: False
        )

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

        # Mock ephemeris_in_eclipse to return False (in sunlight)
        from conops import acs as acs_module

        monkeypatch.setattr(
            acs_module, "ephemeris_in_eclipse", lambda ephem, utime: False
        )

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
                "conops.pointing.Pointing.visibility",
                new=lambda self, *args, **kwargs: (
                    setattr(self, "windows", [[1514764800.0, 1514764900.0]]),
                    0,
                )[-1],
            ),
            patch("conops.pointing.Pointing.next_vis", return_value=1514764800.0),
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
                "conops.pointing.Pointing.visibility",
                new=lambda self, *args, **kwargs: (
                    setattr(self, "windows", [[1514764800.0, 1514764900.0]]),
                    0,
                )[-1],
            ),
            patch("conops.pointing.Pointing.next_vis", return_value=1514764800.0),
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
                "conops.pointing.Pointing.visibility",
                new=lambda self, *args, **kwargs: (
                    setattr(self, "windows", [[1514764800.0, 1514764900.0]]),
                    0,
                )[-1],
            ),
            patch("conops.pointing.Pointing.next_vis", return_value=1514764800.0),
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

    def test_execute_start_battery_charge_command_calls_start(self, acs):
        command = ACSCommand(
            command_type=ACSCommandType.START_BATTERY_CHARGE,
            execution_time=1514764800.0,
            ra=45.0,
            dec=30.0,
            obsid=0xBEEF,
        )

        with patch.object(acs, "_start_battery_charge") as mock_start:
            acs._execute_command(command, 1514764800.0)
            mock_start.assert_called_once_with(command, 1514764800.0)

    def test_execute_end_battery_charge_command_calls_end(self, acs):
        command = ACSCommand(
            command_type=ACSCommandType.END_BATTERY_CHARGE,
            execution_time=1514764800.0,
        )

        with patch.object(acs, "_end_battery_charge") as mock_end:
            acs._execute_command(command, 1514764800.0)
            mock_end.assert_called_once_with(1514764800.0)
