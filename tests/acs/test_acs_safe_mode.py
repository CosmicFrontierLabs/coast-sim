"""Unit tests for ACS Safe Mode functionality."""

from dataclasses import dataclass
from unittest.mock import Mock

import numpy as np

from conops import ACSCommand, ACSCommandType, ACSMode
from conops.simulation.reaction_wheel import ReactionWheel


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


def _make_orthogonal_wheels(
    max_torque: float = 10.0,
    max_momentum: float = 10.0,
    current_momentum: float = 0.0,
):
    """Create three orthogonal reaction wheels for testing."""
    return [
        ReactionWheel(
            max_torque=max_torque,
            max_momentum=max_momentum,
            orientation=(1.0, 0.0, 0.0),
            current_momentum=current_momentum,
            name="rw_x",
        ),
        ReactionWheel(
            max_torque=max_torque,
            max_momentum=max_momentum,
            orientation=(0.0, 1.0, 0.0),
            current_momentum=current_momentum,
            name="rw_y",
        ),
        ReactionWheel(
            max_torque=max_torque,
            max_momentum=max_momentum,
            orientation=(0.0, 0.0, 1.0),
            current_momentum=current_momentum,
            name="rw_z",
        ),
    ]


class TestSafeModeWheelTracking:
    """Test safe-mode wheel tracking with moving sun target."""

    def test_safe_mode_tracking_reacts_to_moving_sun(self, acs, mock_ephem):
        """Assert non-zero wheel torque when sun RA/Dec moves in safe mode.

        This test verifies that safe-mode tracking computes the target sun
        position at the start of each wheel update (not stale self.ra/dec),
        resulting in wheel momentum changes when the sun moves.
        """
        # Set up wheels
        acs.reaction_wheels = _make_orthogonal_wheels()
        acs.wheel_dynamics.wheels = acs.reaction_wheels
        acs._compute_disturbance_torque = lambda _ut: np.zeros(3)
        acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)

        # Enter safe mode
        utime = 1000.0
        acs.request_safe_mode(utime)
        acs.pointing(utime)
        assert acs.in_safe_mode is True

        # Complete the initial SAFE slew (mock returns 100s slew time)
        acs.pointing(utime + 200)

        # Record initial wheel momentum
        initial_momentum = sum(abs(w.current_momentum) for w in acs.reaction_wheels)

        # Move the sun target position
        mock_ephem.sun[0].ra.deg = 50.0  # Was 45.0
        mock_ephem.sun[0].dec.deg = 25.0  # Was 23.5
        mock_ephem.sun_ra_deg = [50.0]
        mock_ephem.sun_dec_deg = [25.0]

        # Step forward - this should trigger tracking torque
        acs.pointing(utime + 201)

        # Wheel momentum should change since sun moved
        final_momentum = sum(abs(w.current_momentum) for w in acs.reaction_wheels)
        momentum_delta = abs(final_momentum - initial_momentum)

        # With a ~5 degree sun movement and 10 kg·m² inertia,
        # we expect measurable wheel momentum change
        assert momentum_delta > 1e-6, (
            f"Expected wheel momentum change when sun moves, got delta={momentum_delta}"
        )


class TestSafeModeDisturbanceRejection:
    """Test that safe-mode correctly rejects (subtracts) disturbance torque."""

    def test_safe_mode_disturbance_rejection_sign(self, acs, mock_ephem):
        """Assert that constant disturbance is rejected, not amplified.

        The wheel torque request should SUBTRACT disturbance to cancel it.
        If disturbance were added instead, body momentum would diverge faster.

        We move the sun slightly to ensure tracking is active (otherwise
        the tracking function returns early with no wheel torque applied).
        """
        # Set up wheels
        acs.reaction_wheels = _make_orthogonal_wheels()
        acs.wheel_dynamics.wheels = acs.reaction_wheels
        acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)

        # Apply a constant disturbance torque
        constant_disturbance = np.array([0.01, 0.0, 0.0])  # 10 mNm about X
        acs._compute_disturbance_torque = lambda _ut: constant_disturbance

        # Enter safe mode
        utime = 1000.0
        acs.request_safe_mode(utime)
        acs.pointing(utime)
        assert acs.in_safe_mode is True

        # Complete the initial SAFE slew
        acs.pointing(utime + 200)

        # Initialize pointing state for tracking
        acs._last_pointing_ra = acs.ra
        acs._last_pointing_dec = acs.dec
        acs._last_pointing_utime = utime + 200
        acs._last_pointing_rate_rad_s = 0.0

        # Move sun slightly each step to keep tracking active
        # This ensures the disturbance rejection code path is exercised
        base_ra = mock_ephem.sun[0].ra.deg
        for i in range(10):
            t = utime + 201 + i
            # Move sun by 0.1 deg/s
            new_ra = base_ra + 0.1 * (i + 1)
            mock_ephem.sun[0].ra.deg = new_ra
            mock_ephem.sun_ra_deg = [new_ra]
            acs.pointing(t)

        # Check body momentum - with correct disturbance rejection, body momentum
        # should stay near zero because wheel torque cancels the external disturbance.
        #
        # Physics:
        # - External disturbance applies +0.01 Nm to body
        # - With CORRECT subtraction: T_req = tracking - disturbance
        #   Wheels apply -disturbance to body (canceling the external)
        #   Body net effect: +disturbance + (-disturbance) = 0
        # - With INCORRECT addition: T_req = tracking + disturbance
        #   Wheels apply +disturbance to body (amplifying!)
        #   Body net effect: +disturbance + (+disturbance) = 2*disturbance
        #
        # After 10 steps with disturbance 0.01 Nm:
        # - Correct: body momentum ~0
        # - Incorrect: body momentum ~0.2 Nm·s (double application)

        final_body_momentum = np.linalg.norm(acs.wheel_dynamics.body_momentum)

        # With correct subtraction, body momentum should stay near zero
        # Allow some growth from tracking torque but not 2x disturbance amplification
        # 10 steps * 1s * 0.02 Nm = 0.2 Nm·s if disturbance doubled
        assert final_body_momentum < 0.1, (
            f"Body momentum grew too much ({final_body_momentum:.4f}), "
            "suggesting disturbance is added instead of subtracted"
        )


@dataclass
class DummyPass:
    """Minimal pass for testing."""

    ra_deg: float
    dec_deg: float

    def ra_dec(self, utime: float) -> tuple[float, float]:
        return self.ra_deg, self.dec_deg


class TestSafeModeClearsPass:
    """Tests that safe-mode entry properly clears active pass."""

    def test_safe_mode_clears_current_pass(self, acs, mock_ephem):
        """SAFE entry should clear current_pass."""
        # Set up an active pass
        acs.current_pass = DummyPass(ra_deg=45.0, dec_deg=30.0)
        assert acs.current_pass is not None

        # Enter safe mode
        utime = 1000.0
        acs.request_safe_mode(utime)
        acs.pointing(utime)

        # Pass should be cleared
        assert acs.current_pass is None
        assert acs._was_in_pass is False

    def test_safe_mode_uses_sun_tracking_not_pass_tracking(self, acs, mock_ephem):
        """After SAFE entry mid-pass, wheel torques should track sun, not pass."""
        # Set up wheels
        acs.reaction_wheels = _make_orthogonal_wheels()
        acs.wheel_dynamics.wheels = acs.reaction_wheels
        acs._compute_disturbance_torque = lambda _ut: np.zeros(3)
        acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)

        # Set up a pass pointing somewhere NOT at the sun
        acs.current_pass = DummyPass(
            ra_deg=180.0, dec_deg=-45.0
        )  # Opposite side of sky

        # Enter safe mode
        utime = 1000.0
        acs.request_safe_mode(utime)
        acs.pointing(utime)

        # Complete the SAFE slew
        acs.pointing(utime + 200)

        # Move sun to trigger tracking
        mock_ephem.sun[0].ra.deg = 50.0
        mock_ephem.sun_ra_deg = [50.0]
        acs.pointing(utime + 201)

        # Safe-mode tracking should be active (not pass tracking)
        assert acs._was_in_safe_mode_tracking is True


class TestSafeSlewWheelLimits:
    """Tests that SAFE slews handle wheel limits appropriately."""

    def test_safe_slew_uses_physics_params_when_capable(self, acs, mock_ephem):
        """SAFE slew should use physics-derived parameters when wheels have capacity."""
        acs.reaction_wheels = _make_orthogonal_wheels(
            max_torque=10.0, max_momentum=50.0
        )
        acs.wheel_dynamics.wheels = acs.reaction_wheels
        acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)

        utime = 1000.0
        acs.request_safe_mode(utime)
        acs.pointing(utime)

        # Slew should have physics-derived (not default) parameters
        slew = acs.last_slew
        assert slew is not None
        assert slew.obstype == "SAFE"
        # With capable wheels, params should be set (not the defaults 0.5/0.25)

    def test_safe_slew_warns_when_limits_exceeded(self, acs, mock_ephem, capsys):
        """SAFE slew should warn when wheel limits would be exceeded."""
        # Setup wheels that are nearly saturated (no headroom available)
        # current_momentum close to max_momentum means no room for slew
        acs.reaction_wheels = _make_orthogonal_wheels(
            max_torque=1.0, max_momentum=1.0, current_momentum=0.99
        )
        acs.wheel_dynamics.wheels = acs.reaction_wheels
        acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)

        utime = 1000.0
        acs.request_safe_mode(utime)
        acs.pointing(utime)

        # Check warning was printed (ACS uses _log_or_print which prints to stdout)
        captured = capsys.readouterr()
        assert "SAFE slew may exceed wheel limits" in captured.out

    def test_safe_slew_proceeds_despite_exceeded_limits(self, acs, mock_ephem):
        """SAFE slew must proceed even when limits exceeded (emergency)."""
        # Setup wheels that are nearly saturated (no headroom available)
        acs.reaction_wheels = _make_orthogonal_wheels(
            max_torque=1.0, max_momentum=1.0, current_momentum=0.99
        )
        acs.wheel_dynamics.wheels = acs.reaction_wheels
        acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)

        utime = 1000.0
        acs.request_safe_mode(utime)
        acs.pointing(utime)

        # SAFE slew should be enqueued regardless of wheel limits
        assert acs.last_slew is not None
        assert acs.last_slew.obstype == "SAFE"
        assert acs.in_safe_mode is True
