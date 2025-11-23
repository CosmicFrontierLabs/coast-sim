"""Unit tests for ACS Safe Mode functionality."""

from unittest.mock import Mock

from conops import ACSCommand, ACSCommandType, ACSMode


class TestSafeModeInitialization:
    """Test safe mode initialization."""

    def test_acs_initializes_not_in_safe_mode(self, acs):
        """Test that ACS initializes with safe mode flag set to False."""
        assert acs.in_safe_mode is False
        assert acs.acsmode == ACSMode.SCIENCE

    def test_safe_mode_enum_exists(self):
        """Test that SAFE mode exists in ACSMode enum."""
        assert hasattr(ACSMode, "SAFE")
        assert ACSMode.SAFE == 5

    def test_enter_safe_mode_command_exists(self):
        """Test that ENTER_SAFE_MODE command type exists."""
        assert hasattr(ACSCommandType, "ENTER_SAFE_MODE")


class TestSafeModeRequest:
    """Test requesting safe mode entry."""

    def test_request_safe_mode_enqueues_command(self, acs):
        """Test that request_safe_mode enqueues the correct command."""
        utime = 1000.0
        acs.request_safe_mode(utime)

        assert len(acs.command_queue) == 1
        command = acs.command_queue[0]
        assert command.command_type == ACSCommandType.ENTER_SAFE_MODE
        assert command.execution_time == utime

    def test_request_safe_mode_before_entering(self, acs):
        """Test that requesting safe mode doesn't immediately enter it."""
        utime = 1000.0
        acs.request_safe_mode(utime)

        # Should still be False until command is processed
        assert acs.in_safe_mode is False
        assert acs.get_mode(utime - 1) != ACSMode.SAFE


class TestSafeModeEntry:
    """Test safe mode entry behavior."""

    def test_safe_mode_command_execution(self, acs):
        """Test that safe mode command sets the flag."""
        utime = 1000.0
        acs.request_safe_mode(utime)

        # Process the command
        acs.pointing(utime)

        # Safe mode flag should now be True
        assert acs.in_safe_mode is True

    def test_safe_mode_clears_command_queue(self, acs):
        """Test that entering safe mode clears all queued commands."""
        utime = 1000.0

        # Enqueue multiple commands
        acs.request_safe_mode(utime)
        acs.request_battery_charge(utime + 100, 10.0, 20.0, 123)
        acs.request_end_battery_charge(utime + 200)

        assert len(acs.command_queue) == 3

        # Execute safe mode command
        acs.pointing(utime)

        # Queue should be cleared except for executed commands
        assert len(acs.command_queue) == 0
        assert acs.in_safe_mode is True


class TestSafeModeIrreversibility:
    """Test that safe mode cannot be exited."""

    def test_safe_mode_cannot_be_exited(self, acs):
        """Test that once in safe mode, spacecraft stays in safe mode."""
        utime = 1000.0

        # Enter safe mode
        acs.request_safe_mode(utime)
        acs.pointing(utime)
        assert acs.in_safe_mode is True

        # Try to execute other commands - they shouldn't work
        acs.request_battery_charge(utime + 100, 10.0, 20.0, 123)
        acs.pointing(utime + 100)

        # Should still be in safe mode
        assert acs.in_safe_mode is True
        assert acs.get_mode(utime + 100) == ACSMode.SAFE

    def test_safe_mode_persists_across_time(self, acs):
        """Test that safe mode persists across multiple time steps."""
        utime = 1000.0

        # Enter safe mode
        acs.request_safe_mode(utime)
        acs.pointing(utime)

        # Check multiple future times
        for t in [utime + 100, utime + 1000, utime + 10000]:
            acs.pointing(t)
            assert acs.in_safe_mode is True
            assert acs.get_mode(t) == ACSMode.SAFE


class TestSafeModePointing:
    """Test safe mode pointing behavior."""

    def test_safe_mode_points_at_sun(self, acs, mock_ephem):
        """Test that safe mode points spacecraft at the Sun."""
        utime = 1000.0

        # Enter safe mode
        acs.request_safe_mode(utime)
        acs.pointing(utime)

        # After slew completes, pointing should be at Sun's RA/Dec
        # The mock slew_time returns 100.0 seconds, so advance beyond that
        acs.pointing(utime + 200)

        assert acs.ra == mock_ephem.sun[0].ra.deg
        assert acs.dec == mock_ephem.sun[0].dec.deg
        assert acs.ra == 45.0
        assert acs.dec == 23.5

    def test_safe_mode_pointing_updates_with_sun(self, acs, mock_ephem):
        """Test that safe mode pointing tracks the Sun over time."""
        utime = 1000.0

        # Enter safe mode
        acs.request_safe_mode(utime)
        acs.pointing(utime)

        initial_ra = acs.ra
        initial_dec = acs.dec

        # Update sun position
        mock_ephem.sun[0].ra.deg = 90.0
        mock_ephem.sun[0].dec.deg = 45.0

        # Update pointing
        acs.pointing(utime + 1000)

        # Pointing should update to new Sun position
        assert acs.ra == 90.0
        assert acs.dec == 45.0
        assert acs.ra != initial_ra
        assert acs.dec != initial_dec


class TestSafeModeOverridesPriority:
    """Test that safe mode takes priority over all other modes."""

    def test_safe_mode_overrides_slewing(self, acs):
        """Test that safe mode takes priority over slewing mode."""
        utime = 1000.0

        # Start a slew (mock)
        acs.current_slew = Mock()
        acs.current_slew.is_slewing = Mock(return_value=True)
        acs.current_slew.obstype = "PPT"

        # Enter safe mode
        acs.in_safe_mode = True

        # Mode should be SAFE, not SLEWING
        assert acs.get_mode(utime) == ACSMode.SAFE

    def test_safe_mode_overrides_charging(self, acs):
        """Test that safe mode takes priority over charging mode."""
        utime = 1000.0

        # Set up charging mode conditions
        slew_mock = Mock()
        slew_mock.is_slewing = Mock(return_value=False)
        slew_mock.obstype = "CHARGE"
        acs.last_slew = slew_mock
        acs.in_eclipse = False

        # Enter safe mode
        acs.in_safe_mode = True

        # Mode should be SAFE, not CHARGING
        assert acs.get_mode(utime) == ACSMode.SAFE

    def test_safe_mode_overrides_pass(self, acs):
        """Test that safe mode takes priority over pass mode."""
        utime = 1000.0

        # Set up pass mode conditions
        pass_mock = Mock()
        pass_mock.is_slewing = Mock(return_value=False)
        pass_mock.obstype = "GSP"
        pass_mock.slewend = utime - 100
        pass_mock.begin = utime - 100
        pass_mock.length = 200
        acs.current_slew = pass_mock

        # Enter safe mode
        acs.in_safe_mode = True

        # Mode should be SAFE, not PASS
        assert acs.get_mode(utime) == ACSMode.SAFE

    def test_safe_mode_overrides_saa(self, acs):
        """Test that safe mode takes priority over SAA mode."""
        utime = 1000.0

        # Set up SAA conditions
        acs.saa = Mock()
        acs.saa.insaa = Mock(return_value=True)

        # Enter safe mode
        acs.in_safe_mode = True

        # Mode should be SAFE, not SAA
        assert acs.get_mode(utime) == ACSMode.SAFE


class TestSafeModeCommandInterface:
    """Test safe mode command interface."""

    def test_acs_command_with_safe_mode_type(self):
        """Test creating an ACSCommand with ENTER_SAFE_MODE type."""
        command = ACSCommand(
            command_type=ACSCommandType.ENTER_SAFE_MODE,
            execution_time=1000.0,
        )

        assert command.command_type == ACSCommandType.ENTER_SAFE_MODE
        assert command.execution_time == 1000.0

    def test_safe_mode_command_in_executed_commands(self, acs):
        """Test that safe mode command appears in executed commands."""
        utime = 1000.0
        acs.request_safe_mode(utime)
        acs.pointing(utime)

        # Both ENTER_SAFE_MODE and SAFE slew should be executed
        assert len(acs.executed_commands) == 2
        assert acs.executed_commands[0].command_type == ACSCommandType.ENTER_SAFE_MODE
        assert acs.executed_commands[1].command_type == ACSCommandType.SLEW_TO_TARGET
        assert acs.executed_commands[1].slew.obstype == "SAFE"
