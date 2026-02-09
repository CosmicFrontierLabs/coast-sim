"""Tests to achieve 100% coverage for fault_management module."""

from datetime import datetime, timezone
from unittest.mock import Mock, patch

import pytest
import rust_ephem

from conops import ACSMode, FaultManagement
from conops.config.fault_management import FaultEvent
from conops.ditl.telemetry import Housekeeping


class TestFaultEventStringRepresentation:
    """Test FaultEvent.__str__() coverage."""

    def test_fault_event_str_with_valid_timestamp(self) -> None:
        """Test __str__ renders valid timestamp correctly."""
        event = FaultEvent(
            utime=1000.0,
            event_type="threshold_transition",
            name="battery_level",
            cause="Transitioned to yellow",
            metadata={"value": 0.45, "threshold": 0.5},
        )
        result = str(event)
        # Should be able to parse the timestamp
        assert "THRESHOLD_TRANSITION" in result.upper()
        assert "battery_level" in result
        assert "Transitioned to yellow" in result

    def test_fault_event_str_with_no_metadata(self) -> None:
        """Test __str__ handles None metadata."""
        event = FaultEvent(
            utime=1000.0,
            event_type="safe_mode_trigger",
            name="test_fault",
            cause="Test cause",
            metadata=None,
        )
        result = str(event)
        assert "SAFE_MODE_TRIGGER" in result.upper()
        assert "test_fault" in result
        assert "|" not in result  # No metadata section when None

    def test_fault_event_str_with_empty_metadata(self) -> None:
        """Test __str__ handles empty metadata dict."""
        event = FaultEvent(
            utime=1000.0,
            event_type="constraint_violation",
            name="sun_limit",
            cause="In constraint violation",
            metadata={},
        )
        result = str(event)
        assert "sun_limit" in result
        # No pipe separator when metadata is empty
        assert "|" not in result

    def test_fault_event_str_with_long_metadata_values(self) -> None:
        """Test __str__ truncates long metadata values."""
        long_string = "x" * 100
        event = FaultEvent(
            utime=1000.0,
            event_type="threshold_transition",
            name="param",
            cause="Test",
            metadata={"data": long_string},
        )
        result = str(event)
        assert "..." in result  # Should be truncated

    def test_fault_event_str_with_multiple_metadata_fields(self) -> None:
        """Test __str__ shows up to 3 metadata fields."""
        event = FaultEvent(
            utime=1000.0,
            event_type="threshold_transition",
            name="param",
            cause="Test",
            metadata={
                "field1": "value1",
                "field2": "value2",
                "field3": "value3",
                "field4": "value4",  # This one should not appear
            },
        )
        result = str(event)
        # Should show 3 fields and then "..."
        assert result.count("field") == 3
        assert "..." in result

    @patch("conops.config.fault_management.dtutcfromtimestamp")
    def test_fault_event_str_with_invalid_timestamp_exception(
        self, mock_dtutil
    ) -> None:
        """Test __str__ handles exception from dtutcfromtimestamp."""
        mock_dtutil.side_effect = ValueError("Invalid timestamp")
        event = FaultEvent(
            utime=1000.0,
            event_type="test_event",
            name="test",
            cause="Test",
        )
        result = str(event)
        # Should fall back to utime format
        assert "utime=" in result


class TestACSModeFiltering:
    """Test ACS mode-specific threshold checking."""

    def test_threshold_with_acs_modes_filter(self, acs_stub) -> None:
        """Test threshold with acs_modes only triggers in specified modes."""
        fm = FaultManagement()
        fm.add_threshold(
            "parameter",
            yellow=50.0,
            red=60.0,
            direction="above",
            acs_modes=[ACSMode.SCIENCE],
        )

        # Test in SCIENCE mode - should trigger
        acs_stub.acsmode = ACSMode.SCIENCE
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            parameter=65.0,
            acs_mode=ACSMode.SCIENCE,
        )
        fm.check(hk, acs=acs_stub)
        assert fm.statistics()["parameter"]["current"] == "red"

    def test_threshold_skipped_when_acs_mode_not_in_list(self, acs_stub) -> None:
        """Test threshold is skipped when current mode not in acs_modes list."""
        fm = FaultManagement()
        fm.add_threshold(
            "parameter",
            yellow=50.0,
            red=60.0,
            direction="above",
            acs_modes=[ACSMode.SCIENCE],
        )

        # Test in SAFE mode (not in acs_modes)
        acs_stub.acsmode = ACSMode.SAFE
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            parameter=65.0,  # Would be red if checked
            acs_mode=ACSMode.SAFE,
        )
        classifications = fm.check(hk, acs=acs_stub)
        # Parameter should not be in classifications
        assert "parameter" not in classifications

    def test_threshold_with_int_acs_mode_conversion(self, acs_stub) -> None:
        """Test threshold converts int acs_mode to ACSMode enum."""
        fm = FaultManagement()
        fm.add_threshold(
            "parameter",
            yellow=50.0,
            red=60.0,
            direction="above",
            acs_modes=[ACSMode.SCIENCE],
        )

        # Mock housekeeping with int acs_mode (gets converted)
        # ACSMode.SCIENCE = 0, so we use 0
        acs_stub.acsmode = 0
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            parameter=65.0,
            acs_mode=0,  # Pass int 0 which is ACSMode.SCIENCE
        )
        classifications = fm.check(hk, acs=acs_stub)
        # Should be converted to enum and included
        assert "parameter" in classifications
        assert classifications["parameter"] == "red"

    def test_threshold_skipped_when_no_acs_mode_available(self, acs_stub) -> None:
        """Test threshold skipped when acs_mode not available and mode filtering enabled."""
        fm = FaultManagement()
        fm.add_threshold(
            "parameter",
            yellow=50.0,
            red=60.0,
            direction="above",
            acs_modes=[ACSMode.SCIENCE],
        )

        # Both housekeeping and acs acs_mode are None
        acs_stub.acsmode = None
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            parameter=65.0,
            acs_mode=None,
        )
        classifications = fm.check(hk, acs=acs_stub)
        # Should be skipped
        assert "parameter" not in classifications

    def test_threshold_skipped_on_invalid_int_acs_mode(self, acs_stub) -> None:
        """Test threshold skipped when int acs_mode cannot be converted to enum."""
        fm = FaultManagement()
        fm.add_threshold(
            "parameter",
            yellow=50.0,
            red=60.0,
            direction="above",
            acs_modes=[ACSMode.SCIENCE],
        )

        # Invalid acs_mode int that can't convert to ACSMode
        acs_stub.acsmode = 999
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            parameter=65.0,
            acs_mode=999,  # Invalid enum value
        )
        classifications = fm.check(hk, acs=acs_stub)
        # Should be skipped
        assert "parameter" not in classifications

    def test_threshold_with_housekeeping_acs_mode_preferred(self, acs_stub) -> None:
        """Test housekeeping.acs_mode takes precedence over acs.acsmode."""
        fm = FaultManagement()
        fm.add_threshold(
            "parameter",
            yellow=50.0,
            red=60.0,
            direction="above",
            acs_modes=[ACSMode.SCIENCE],
        )

        # Housekeeping has SCIENCE, ACS has SAFE
        acs_stub.acsmode = ACSMode.SAFE
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            parameter=65.0,
            acs_mode=ACSMode.SCIENCE,  # Use this one
        )
        classifications = fm.check(hk, acs=acs_stub)
        # Should be checked because housekeeping has SCIENCE
        assert "parameter" in classifications
        assert classifications["parameter"] == "red"


class TestContinuousViolationRecovery:
    """Test continuous violation time reset on constraint recovery."""

    def test_continuous_violation_accumulates(self, acs_stub) -> None:
        """Test continuous violation time accumulates while in violation."""
        fm = FaultManagement()
        constraint = Mock(spec=rust_ephem.SunConstraint)
        constraint.in_constraint = Mock(return_value=True)
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint,
            time_threshold_seconds=600.0,
        )

        ephem = Mock()
        ephem.step_size = 60.0
        acs_stub.ephem = ephem

        for i in range(3):
            hk = Housekeeping(
                timestamp=datetime.fromtimestamp(1000.0 + i * 60, tz=timezone.utc),
                ra=45.0,
                dec=23.5,
            )
            fm.check(hk, acs=acs_stub)

        stats = fm.statistics()["sun_limit"]
        assert stats["continuous_violation_seconds"] == pytest.approx(180.0)

    def test_continuous_violation_resets_on_recovery(self, acs_stub) -> None:
        """Test continuous violation time resets when constraint is satisfied."""
        fm = FaultManagement()
        constraint = Mock(spec=rust_ephem.SunConstraint)
        constraint.in_constraint = Mock(side_effect=[True, False])
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint,
            time_threshold_seconds=600.0,
        )

        ephem = Mock()
        ephem.step_size = 60.0
        acs_stub.ephem = ephem

        # Cycle 1: In violation
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            ra=45.0,
            dec=23.5,
        )
        fm.check(hk, acs=acs_stub)

        stats = fm.statistics()["sun_limit"]
        assert stats["continuous_violation_seconds"] == pytest.approx(60.0)

        # Cycle 2: Recovered
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1060.0, tz=timezone.utc),
            ra=45.0,
            dec=23.5,
        )
        fm.check(hk, acs=acs_stub)

        stats = fm.statistics()["sun_limit"]
        assert stats["continuous_violation_seconds"] == pytest.approx(0.0)
        # Total red_seconds should remain
        assert stats["red_seconds"] == pytest.approx(60.0)

    def test_constraint_violation_cleared_event_logged(self, acs_stub) -> None:
        """Test event is logged when constraint violation is cleared."""
        fm = FaultManagement()
        constraint = Mock(spec=rust_ephem.SunConstraint)
        constraint.in_constraint = Mock(side_effect=[True, False])
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint,
            time_threshold_seconds=600.0,
            description="Sun angle check",
        )

        ephem = Mock()
        ephem.step_size = 60.0
        acs_stub.ephem = ephem

        # First check: in violation
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            ra=45.0,
            dec=23.5,
        )
        fm.check(hk, acs=acs_stub)

        # Second check: recovered
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1060.0, tz=timezone.utc),
            ra=45.0,
            dec=23.5,
        )
        fm.check(hk, acs=acs_stub)

        # Look for cleared event
        cleared_events = [e for e in fm.events if "Cleared" in e.cause]
        assert len(cleared_events) > 0
        assert "Cleared constraint violation" in cleared_events[0].cause


class TestMissingEphemerisHandling:
    """Test handling of missing ephemeris."""

    def test_check_raises_when_ephemeris_none(self, acs_stub) -> None:
        """Test check raises ValueError when ACS ephemeris is None."""
        fm = FaultManagement()
        fm.add_threshold("parameter", yellow=50.0, red=60.0)

        acs_stub.ephem = None

        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            parameter=40.0,
        )

        with pytest.raises(ValueError, match="ACS ephemeris must be set"):
            fm.check(hk, acs=acs_stub)


class TestMultipleConstraintHandling:
    """Test handling of multiple red limit constraints."""

    def test_no_constraints_checks_thresholds_only(self, acs_stub) -> None:
        """Test without constraints, only thresholds are checked."""
        fm = FaultManagement()
        fm.add_threshold("battery", yellow=0.5, red=0.4, direction="below")

        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            battery=0.45,
        )

        classifications = fm.check(hk, acs=acs_stub)
        assert "battery" in classifications

    def test_early_exit_when_no_ephemeris_for_constraints(self, acs_stub) -> None:
        """Test constraint checking skipped when ephem/ra/dec not available."""
        fm = FaultManagement()
        constraint = rust_ephem.SunConstraint(min_angle=30.0)
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint,
            time_threshold_seconds=300.0,
        )

        # Missing ra/dec
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            ra=None,
            dec=None,
        )

        fm.check(hk, acs=acs_stub)
        # Should not error, just skip constraint checks
        assert len(fm.statistics()) == 0


class TestSafeModeTriggering:
    """Test safe mode trigger conditions."""

    def test_safe_mode_not_retriggered_when_already_in_safe_mode(
        self, acs_stub
    ) -> None:
        """Test safe mode flag not set again if already in safe mode."""
        fm = FaultManagement(safe_mode_on_red=True)
        fm.add_threshold("parameter", yellow=50.0, red=60.0, direction="above")

        acs_stub.in_safe_mode = True

        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            parameter=65.0,
        )

        # Reset the flag to see if it gets set again
        fm.safe_mode_requested = False
        fm.check(hk, acs=acs_stub)

        # Should not set flag since already in safe mode
        assert not fm.safe_mode_requested

    def test_safe_mode_disabled_no_trigger_on_red(self, acs_stub) -> None:
        """Test safe mode not triggered when safe_mode_on_red is False."""
        fm = FaultManagement(safe_mode_on_red=False)
        fm.add_threshold("parameter", yellow=50.0, red=60.0, direction="above")

        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            parameter=65.0,
        )

        fm.check(hk, acs=acs_stub)
        assert not fm.safe_mode_requested
        # Should also not log safe_mode_trigger events
        safe_mode_events = [e for e in fm.events if e.event_type == "safe_mode_trigger"]
        assert len(safe_mode_events) == 0


class TestThresholdTransitionEvents:
    """Test event logging for threshold transitions."""

    def test_transition_event_logged_nominal_to_yellow(self, acs_stub) -> None:
        """Test event logged when transitioning from nominal to yellow."""
        fm = FaultManagement()
        fm.add_threshold("parameter", yellow=50.0, red=60.0, direction="above")

        # First check: nominal
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            parameter=40.0,
        )
        fm.check(hk, acs=acs_stub)

        # Second check: yellow
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1001.0, tz=timezone.utc),
            parameter=55.0,
        )
        fm.check(hk, acs=acs_stub)

        transitions = [e for e in fm.events if "Transitioned" in e.cause]
        assert len(transitions) > 0
        assert "nominal" in transitions[0].cause
        assert "yellow" in transitions[0].cause

    def test_transition_event_contains_threshold_metadata(self, acs_stub) -> None:
        """Test transition event metadata includes thresholds and direction."""
        fm = FaultManagement()
        fm.add_threshold("parameter", yellow=50.0, red=60.0, direction="above")

        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            parameter=55.0,
        )
        fm.check(hk, acs=acs_stub)

        transitions = [e for e in fm.events if "Transitioned" in e.cause]
        assert len(transitions) > 0
        assert transitions[0].metadata is not None
        assert transitions[0].metadata["yellow_threshold"] == 50.0
        assert transitions[0].metadata["red_threshold"] == 60.0
        assert transitions[0].metadata["direction"] == "above"


class TestConstraintViolationTimeThreshold:
    """Test constraint violation exceeding time threshold triggering safe mode."""

    def test_constraint_violation_exceeds_time_threshold_triggers_safe_mode(
        self, acs_stub
    ) -> None:
        """Test safe mode triggered when constraint violation exceeds time threshold."""
        fm = FaultManagement(safe_mode_on_red=True)
        constraint = Mock(spec=rust_ephem.SunConstraint)
        constraint.in_constraint = Mock(return_value=True)
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint,
            time_threshold_seconds=60.0,  # Low threshold to exceed in a few cycles
        )

        ephem = Mock()
        ephem.step_size = 40.0  # 40 seconds per cycle
        acs_stub.ephem = ephem
        acs_stub.in_safe_mode = False

        # Cycle 1: 40 seconds violation (below threshold)
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            ra=45.0,
            dec=23.5,
        )
        fm.check(hk, acs=acs_stub)
        assert not fm.safe_mode_requested

        # Cycle 2: 80 seconds total violation (exceeds 60.0 threshold)
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1040.0, tz=timezone.utc),
            ra=45.0,
            dec=23.5,
        )
        fm.check(hk, acs=acs_stub)

        # Should have triggered safe mode now
        assert fm.safe_mode_requested

        # Check for safe mode trigger event
        safe_mode_events = [e for e in fm.events if e.event_type == "safe_mode_trigger"]
        assert len(safe_mode_events) > 0
        assert "exceeded time threshold" in safe_mode_events[0].cause
