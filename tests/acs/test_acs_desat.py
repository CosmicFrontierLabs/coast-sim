"""Tests for reaction wheel desaturation and headroom handling."""

from unittest.mock import Mock

import pytest

from conops import ACSCommand, ACSCommandType, ACSMode, ReactionWheel


class TestDesatCommand:
    """Ensure DESAT command updates state correctly."""

    def test_desat_command_activates_desat_mode(self, acs):
        """DESAT command activates desat state and mode."""
        # Ensure we have a wheel with stored momentum
        acs.reaction_wheels = [ReactionWheel(max_torque=0.1, max_momentum=0.2)]
        acs.reaction_wheels[0].current_momentum = 0.2
        # Force ACS into an active slew state so DESAT is not deferred
        acs.current_slew = Mock(obstype="PPT")
        acs.current_slew.is_slewing = Mock(side_effect=lambda t: t < 50)

        cmd = ACSCommand(
            command_type=ACSCommandType.DESAT,
            execution_time=0.0,
            duration=120.0,
        )
        acs.enqueue_command(cmd)

        # Process command and verify state
        acs._process_commands(0.0)
        assert acs._desat_active is True
        assert acs._desat_end == pytest.approx(120.0)

        # During desat, mode is DESAT (holding position)
        assert acs.get_mode(10.0) == ACSMode.DESAT
        # After desat window, mode returns to science when idle
        assert acs.get_mode(200.0) == ACSMode.SCIENCE

    def test_desat_without_mtq_preserves_momentum(self, acs):
        """Without magnetorquers, DESAT cannot reduce momentum (conservation)."""
        # Ensure we have a wheel with stored momentum and no MTQs
        acs.reaction_wheels = [ReactionWheel(max_torque=0.1, max_momentum=0.2)]
        acs.reaction_wheels[0].current_momentum = 0.2
        acs.magnetorquers = []  # No MTQs available

        cmd = ACSCommand(
            command_type=ACSCommandType.DESAT,
            execution_time=0.0,
            duration=120.0,
        )
        acs.enqueue_command(cmd)

        # Process command
        acs._process_commands(0.0)
        assert acs._desat_active is True
        assert acs._desat_use_mtq is False

        # Momentum must be preserved - no external torque available
        assert acs.reaction_wheels[0].current_momentum == pytest.approx(0.2)


class TestCheckCurrentLoad:
    """Tests for _check_current_load() momentum threshold check."""

    def _setup_wheels(self, acs, wheels):
        """Helper to sync reaction_wheels and wheel_dynamics.wheels."""
        acs.reaction_wheels = wheels
        acs.wheel_dynamics.wheels = wheels

    def test_no_wheels_returns_true(self, acs):
        """Without reaction wheels, load check always passes."""
        self._setup_wheels(acs, [])
        assert acs._check_current_load(utime=0.0) is True

    def test_below_threshold_returns_true(self, acs):
        """Wheels below 60% capacity should pass the load check."""
        wheel = ReactionWheel(max_torque=0.1, max_momentum=1.0)
        wheel.current_momentum = 0.59  # 59% < 60%
        self._setup_wheels(acs, [wheel])
        assert acs._check_current_load(utime=0.0) is True

    def test_at_threshold_returns_false_and_requests_desat(self, acs):
        """Wheels at 60% capacity should fail and request desat."""
        wheel = ReactionWheel(max_torque=0.1, max_momentum=1.0)
        wheel.current_momentum = 0.60  # exactly 60%
        self._setup_wheels(acs, [wheel])
        # Clear magnetorquers to test non-MTQ desat path
        acs.magnetorquers = []
        # Use utime past the desat cooldown period (1800s)
        utime = 2000.0
        initial_desat_requests = acs.desat_requests

        result = acs._check_current_load(utime=utime)

        assert result is False
        # Verify desat was requested
        assert acs.desat_requests == initial_desat_requests + 1

    def test_above_threshold_returns_false(self, acs):
        """Wheels above 60% capacity should fail."""
        wheel = ReactionWheel(max_torque=0.1, max_momentum=1.0)
        wheel.current_momentum = 0.80  # 80% > 60%
        self._setup_wheels(acs, [wheel])

        result = acs._check_current_load(utime=0.0)

        assert result is False


class TestHeadroomGate:
    """Slew enqueue should request desat when wheels lack headroom."""

    def test_enqueue_slew_triggers_desat_when_headroom_insufficient(
        self, acs, monkeypatch
    ):
        # Add a wheel with no capability
        acs.reaction_wheels = [ReactionWheel(max_torque=0.0, max_momentum=0.0)]

        # Patch helpers to avoid full pointing/visibility logic
        monkeypatch.setattr(
            acs, "_create_target_request", Mock(next_vis=Mock(return_value=0))
        )
        monkeypatch.setattr(acs, "_is_slew_valid", lambda *args, **kwargs: True)
        monkeypatch.setattr(acs, "_calculate_slew_timing", lambda *args, **kwargs: 0.0)

        # Attempt to enqueue a slew; should be rejected and desat requested
        result = acs._enqueue_slew(ra=10.0, dec=5.0, obsid=1, utime=0.0, obstype="PPT")
        assert result is False
        assert not acs.command_queue  # desat request is not enqueued when MTQs bleed
