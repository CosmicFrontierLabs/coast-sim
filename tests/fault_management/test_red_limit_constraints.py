"""Tests for spacecraft-level red limit constraints in fault management system."""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest

from conops.config.fault_management import (
    FaultState,
)
from conops.ditl.telemetry import Housekeeping


@pytest.fixture
def acs_with_ephem(ephem):
    acs = Mock()
    acs.ephem = ephem
    acs.in_safe_mode = False
    return acs


class TestFaultConstraint:
    """Test FaultConstraint model."""

    def test_fault_constraint_has_name(self, fault_constraint) -> None:
        assert fault_constraint.name == "test_sun_limit"

    def test_fault_constraint_has_time_threshold(self, fault_constraint) -> None:
        assert fault_constraint.time_threshold_seconds == 300.0

    def test_fault_constraint_has_description(self, fault_constraint) -> None:
        assert fault_constraint.description == "Test sun constraint"

    def test_fault_constraint_has_constraint_object(self, fault_constraint) -> None:
        assert fault_constraint.constraint is not None

    def test_fault_constraint_monitor_has_name(self, fault_monitor_constraint) -> None:
        assert fault_monitor_constraint.name == "test_monitor"

    def test_fault_constraint_monitor_no_threshold(
        self, fault_monitor_constraint
    ) -> None:
        assert fault_monitor_constraint.time_threshold_seconds is None


class TestFaultState:
    """Test FaultState tracking."""

    def test_initial_in_violation_false(self) -> None:
        state = FaultState()
        assert state.in_violation is False

    def test_initial_total_violation_zero(self) -> None:
        state = FaultState()
        assert state.red_seconds == 0.0

    def test_initial_continuous_violation_zero(self) -> None:
        state = FaultState()
        assert state.continuous_violation_seconds == 0.0

    def test_accumulate_violation_time_initial(self) -> None:
        state = FaultState()
        state.in_violation = True
        state.red_seconds += 10.0
        state.continuous_violation_seconds += 10.0
        assert state.red_seconds == 10.0

    def test_accumulate_violation_time_initial_continuous(self) -> None:
        state = FaultState()
        state.in_violation = True
        state.red_seconds += 10.0
        state.continuous_violation_seconds += 10.0
        assert state.continuous_violation_seconds == 10.0

    def test_accumulate_violation_time_additional(self) -> None:
        state = FaultState()
        # Start with 10 seconds then add 5 seconds
        state.in_violation = True
        state.red_seconds += 10.0
        state.continuous_violation_seconds += 10.0
        state.red_seconds += 5.0
        assert state.red_seconds == 15.0

    def test_accumulate_violation_time_additional_continuous(self) -> None:
        state = FaultState()
        # Start with 10 seconds then add 5 seconds
        state.in_violation = True
        state.red_seconds += 10.0
        state.continuous_violation_seconds += 10.0
        state.continuous_violation_seconds += 5.0
        assert state.continuous_violation_seconds == 15.0

    def test_reset_continuous_on_recovery_total_unchanged(self) -> None:
        state = FaultState()
        state.in_violation = True
        state.red_seconds = 100.0
        state.continuous_violation_seconds = 100.0
        state.in_violation = False
        state.continuous_violation_seconds = 0.0
        assert state.red_seconds == 100.0

    def test_reset_continuous_on_recovery_continuous_zero(self) -> None:
        state = FaultState()
        state.in_violation = True
        state.red_seconds = 100.0
        state.continuous_violation_seconds = 100.0
        state.in_violation = False
        state.continuous_violation_seconds = 0.0
        assert state.continuous_violation_seconds == 0.0


class TestFaultManagementRedLimits:
    """Test FaultManagement integration with red limit constraints."""

    def test_add_red_limit_constraint_entry_exists(self, fm, constraint_sun_30) -> None:
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint_sun_30,
            time_threshold_seconds=300.0,
            description="Sun constraint",
        )
        assert any(c.name == "sun_limit" for c in fm.red_limit_constraints)

    def test_add_red_limit_constraint_name_correct(self, fm, constraint_sun_30) -> None:
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint_sun_30,
            time_threshold_seconds=300.0,
            description="Sun constraint",
        )
        constraint = next(c for c in fm.red_limit_constraints if c.name == "sun_limit")
        assert constraint.name == "sun_limit"

    def test_add_red_limit_constraint_time_threshold(
        self, fm, constraint_sun_30
    ) -> None:
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint_sun_30,
            time_threshold_seconds=300.0,
            description="Sun constraint",
        )
        constraint = next(c for c in fm.red_limit_constraints if c.name == "sun_limit")
        assert constraint.time_threshold_seconds == 300.0

    def test_add_multiple_red_limit_constraints_count(
        self, fm, constraint_sun_30, constraint_earth_10
    ) -> None:
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint_sun_30,
            time_threshold_seconds=300.0,
        )
        fm.add_red_limit_constraint(
            name="earth_limit",
            constraint=constraint_earth_10,
            time_threshold_seconds=600.0,
        )
        assert len(fm.red_limit_constraints) == 2

    def test_add_multiple_red_limit_constraints_contains_sun(
        self, fm, constraint_sun_30, constraint_earth_10
    ) -> None:
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint_sun_30,
            time_threshold_seconds=300.0,
        )
        fm.add_red_limit_constraint(
            name="earth_limit",
            constraint=constraint_earth_10,
            time_threshold_seconds=600.0,
        )
        assert any(c.name == "sun_limit" for c in fm.red_limit_constraints)

    def test_add_multiple_red_limit_constraints_contains_earth(
        self, fm, constraint_sun_30, constraint_earth_10
    ) -> None:
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint_sun_30,
            time_threshold_seconds=300.0,
        )
        fm.add_red_limit_constraint(
            name="earth_limit",
            constraint=constraint_earth_10,
            time_threshold_seconds=600.0,
        )
        assert any(c.name == "earth_limit" for c in fm.red_limit_constraints)

    def test_check_red_limit_constraints_creates_states(
        self, fm, ephem, acs_with_ephem, constraint_sun_30
    ) -> None:
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint_sun_30,
            time_threshold_seconds=300.0,
        )
        hk = Housekeeping(
            timestamp=ephem.timestamp[0],
            ra=180.0,
            dec=0.0,
        )
        fm.check(hk, acs=acs_with_ephem)
        assert "sun_limit" in fm.states

    def test_red_limit_statistics_contains_constraint(
        self, fm, constraint_sun_30
    ) -> None:
        fm.add_red_limit_constraint(
            name="test_constraint",
            constraint=constraint_sun_30,
            time_threshold_seconds=300.0,
        )
        fm.states["test_constraint"] = FaultState(
            in_violation=True,
            red_seconds=150.0,
            continuous_violation_seconds=100.0,
        )
        stats = fm.statistics()
        assert "test_constraint" in stats

    def test_red_limit_statistics_in_violation_true(
        self, fm, constraint_sun_30
    ) -> None:
        fm.add_red_limit_constraint(
            name="test_constraint",
            constraint=constraint_sun_30,
            time_threshold_seconds=300.0,
        )
        fm.states["test_constraint"] = FaultState(
            in_violation=True,
            red_seconds=150.0,
            continuous_violation_seconds=100.0,
        )
        stats = fm.statistics()
        assert stats["test_constraint"]["in_violation"] is True

    def test_red_limit_statistics_red_seconds(self, fm, constraint_sun_30) -> None:
        fm.add_red_limit_constraint(
            name="test_constraint",
            constraint=constraint_sun_30,
            time_threshold_seconds=300.0,
        )
        fm.states["test_constraint"] = FaultState(
            in_violation=True,
            red_seconds=150.0,
            continuous_violation_seconds=100.0,
        )
        stats = fm.statistics()
        assert stats["test_constraint"]["red_seconds"] == 150.0

    def test_red_limit_statistics_continuous_violation_seconds(
        self, fm, constraint_sun_30
    ) -> None:
        fm.add_red_limit_constraint(
            name="test_constraint",
            constraint=constraint_sun_30,
            time_threshold_seconds=300.0,
        )
        fm.states["test_constraint"] = FaultState(
            in_violation=True,
            red_seconds=150.0,
            continuous_violation_seconds=100.0,
        )
        stats = fm.statistics()
        assert stats["test_constraint"]["continuous_violation_seconds"] == 100.0

    def test_safe_mode_not_triggered_below_threshold_state(
        self, fm_safe, ephem, acs_with_ephem, constraint_sun_90
    ) -> None:
        fm = fm_safe
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint_sun_90,
            time_threshold_seconds=300.0,
        )
        for i in range(4):  # 4 * 60 = 240 seconds < 300 second threshold
            hk = Housekeeping(
                timestamp=ephem.timestamp[i],
                ra=0.0,
                dec=0.0,
            )
            fm.check(hk, acs=acs_with_ephem)
        assert fm.safe_mode_requested is False

    def test_continuous_violation_resets_on_recovery_total_stays(self, fm) -> None:
        fm.states["test"] = FaultState(
            in_violation=False,
            red_seconds=500.0,
            continuous_violation_seconds=100.0,
        )
        state = fm.states["test"]
        if not state.in_violation:
            state.continuous_violation_seconds = 0.0
        assert state.red_seconds == 500.0

    def test_continuous_violation_resets_on_recovery_continuous_zero(self, fm) -> None:
        fm.states["test"] = FaultState(
            in_violation=False,
            red_seconds=500.0,
            continuous_violation_seconds=100.0,
        )
        state = fm.states["test"]
        if not state.in_violation:
            state.continuous_violation_seconds = 0.0
        assert state.continuous_violation_seconds == 0.0


class TestFaultConstraintIntegration:
    """Integration tests for red limit constraints in simulation context."""

    def test_constraint_with_no_time_threshold_never_triggers_safe_mode_accumulates(
        self, fm_safe, constraint_sun_90, ephem, acs_with_ephem
    ) -> None:
        fm = fm_safe
        fm.add_red_limit_constraint(
            name="monitor_only",
            constraint=constraint_sun_90,
            time_threshold_seconds=None,
        )
        for i in range(20):
            hk = Housekeeping(
                timestamp=ephem.timestamp[i],
                ra=0.0,
                dec=0.0,
            )
            fm.check(hk, acs=acs_with_ephem)
        assert fm.states["monitor_only"].red_seconds > 0

    def test_constraint_with_no_time_threshold_never_triggers_safe_mode_not_requested(
        self, fm_safe, constraint_sun_90, ephem, acs_with_ephem
    ) -> None:
        fm = fm_safe
        fm.add_red_limit_constraint(
            name="monitor_only",
            constraint=constraint_sun_90,
            time_threshold_seconds=None,
        )
        for i in range(20):
            hk = Housekeeping(
                timestamp=ephem.timestamp[i],
                ra=0.0,
                dec=0.0,
            )
            fm.check(hk, acs=acs_with_ephem)
        assert fm.safe_mode_requested is False

    def test_mixed_regular_fault_classification_is_yellow(
        self, fm, constraint_sun_30, acs_stub
    ) -> None:
        fm.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint_sun_30,
            time_threshold_seconds=300.0,
        )
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            battery_level=0.45,
        )
        classifications = fm.check(hk, acs=acs_stub)
        assert classifications["battery_level"] == "yellow"

    def test_mixed_regular_fault_does_not_trigger_safe_mode_on_yellow(
        self, fm, constraint_sun_30, acs_stub
    ) -> None:
        fm.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint_sun_30,
            time_threshold_seconds=300.0,
        )
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            battery_level=0.45,
        )
        fm.check(hk, acs=acs_stub)
        assert fm.safe_mode_requested is False

    def test_mixed_regular_fault_triggers_safe_mode_on_red(
        self, fm, constraint_sun_30, acs_stub
    ) -> None:
        fm.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")
        fm.add_red_limit_constraint(
            name="sun_limit",
            constraint=constraint_sun_30,
            time_threshold_seconds=300.0,
        )
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1060.0, tz=timezone.utc),
            battery_level=0.35,
        )
        fm.check(hk, acs=acs_stub)
        assert fm.safe_mode_requested is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
