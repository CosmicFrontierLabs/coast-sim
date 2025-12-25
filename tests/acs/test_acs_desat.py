"""Tests for reaction wheel desaturation and headroom handling."""

from unittest.mock import Mock

import pytest

from conops import ACSCommand, ACSCommandType, ACSMode, ReactionWheel


class TestDesatCommand:
    """Ensure DESAT command updates state and wheel momentum."""

    def test_desat_command_resets_momentum_and_mode(self, acs):
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
        assert all(w.current_momentum == 0.0 for w in acs.reaction_wheels)

        # During desat, mode is treated as slewing/maintenance
        assert acs.get_mode(10.0) == ACSMode.SLEWING
        # After desat window, mode returns to science when idle
        assert acs.get_mode(200.0) == ACSMode.SCIENCE


class TestHeadroomGate:
    """Slew enqueue should request desat when wheels lack headroom."""

    def test_enqueue_slew_triggers_desat_when_headroom_insufficient(self, acs, monkeypatch):
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
        assert acs.command_queue  # desat request enqueued
        assert acs.command_queue[0].command_type == ACSCommandType.DESAT
