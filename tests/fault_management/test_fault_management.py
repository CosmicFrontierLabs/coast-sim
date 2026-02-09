from datetime import datetime, timezone

import pytest

from conops import ACS, FaultManagement
from conops.config import MissionConfig
from conops.ditl.telemetry import Housekeeping


class TestDefaultConfiguration:
    def test_adds_default_battery_threshold(self, base_config: MissionConfig) -> None:
        assert any(
            t.name == "battery_level" for t in base_config.fault_management.thresholds
        )


class TestYellowState:
    def test_state_is_yellow(
        self, fm_with_yellow_state: tuple[FaultManagement, ACS]
    ) -> None:
        fm, _ = fm_with_yellow_state
        stats = fm.statistics()["battery_level"]
        assert stats["current"] == "yellow"

    def test_accumulates_yellow_seconds(
        self, fm_with_yellow_state: tuple[FaultManagement, ACS]
    ) -> None:
        fm, _ = fm_with_yellow_state
        stats = fm.statistics()["battery_level"]
        assert stats["yellow_seconds"] == pytest.approx(60.0)

    def test_has_zero_red_seconds(
        self, fm_with_yellow_state: tuple[FaultManagement, ACS]
    ) -> None:
        fm, _ = fm_with_yellow_state
        stats = fm.statistics()["battery_level"]
        assert stats["red_seconds"] == 0.0

    def test_does_not_trigger_safe_mode(
        self, fm_with_yellow_state: tuple[FaultManagement, ACS]
    ) -> None:
        _, acs = fm_with_yellow_state
        assert not acs.in_safe_mode


class TestRedState:
    def test_requests_safe_mode(
        self, fm_with_red_state: tuple[FaultManagement, ACS]
    ) -> None:
        fm, _ = fm_with_red_state
        assert fm.safe_mode_requested

    def test_state_is_red(self, fm_with_red_state: tuple[FaultManagement, ACS]) -> None:
        fm, _ = fm_with_red_state
        stats = fm.statistics()["battery_level"]
        assert stats["current"] == "red"

    def test_accumulates_red_seconds(
        self, fm_with_red_state: tuple[FaultManagement, ACS]
    ) -> None:
        fm, _ = fm_with_red_state
        stats = fm.statistics()["battery_level"]
        assert stats["red_seconds"] == pytest.approx(60.0)


class TestMultipleCycles:
    def test_accumulate_yellow_seconds(
        self, fm_with_multiple_cycles: tuple[FaultManagement, ACS]
    ) -> None:
        fm, _ = fm_with_multiple_cycles
        stats = fm.statistics()["battery_level"]
        assert stats["yellow_seconds"] == pytest.approx(120.0)

    def test_have_zero_red_seconds(
        self, fm_with_multiple_cycles: tuple[FaultManagement, ACS]
    ) -> None:
        fm, _ = fm_with_multiple_cycles
        stats = fm.statistics()["battery_level"]
        assert stats["red_seconds"] == 0.0

    def test_do_not_trigger_safe_mode(
        self, fm_with_multiple_cycles: tuple[FaultManagement, ACS]
    ) -> None:
        _, acs = fm_with_multiple_cycles
        assert not acs.in_safe_mode


class TestAboveThreshold:
    def test_classifies_nominal(self, acs_stub) -> None:
        fm = FaultManagement()
        fm.add_threshold("temperature", yellow=50.0, red=60.0, direction="above")
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            temperature=40.0,
        )
        classifications = fm.check(hk, acs=acs_stub)
        assert classifications["temperature"] == "nominal"

    def test_classifies_yellow(self, acs_stub) -> None:
        fm = FaultManagement()
        fm.add_threshold("temperature", yellow=50.0, red=60.0, direction="above")
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1001.0, tz=timezone.utc),
            temperature=55.0,
        )
        classifications = fm.check(hk, acs=acs_stub)
        assert classifications["temperature"] == "yellow"

    def test_classifies_red(self, acs_stub) -> None:
        fm = FaultManagement()
        fm.add_threshold("temperature", yellow=50.0, red=60.0, direction="above")
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1002.0, tz=timezone.utc),
            temperature=65.0,
        )
        classifications = fm.check(hk, acs=acs_stub)
        assert classifications["temperature"] == "red"

    def test_accumulates_yellow_seconds(
        self, fm_with_above_threshold: FaultManagement
    ) -> None:
        stats = fm_with_above_threshold.statistics()["temperature"]
        assert stats["yellow_seconds"] == 1.0

    def test_accumulates_red_seconds(
        self, fm_with_above_threshold: FaultManagement
    ) -> None:
        stats = fm_with_above_threshold.statistics()["temperature"]
        assert stats["red_seconds"] == 1.0

    def test_current_state_is_red(
        self, fm_with_above_threshold: FaultManagement
    ) -> None:
        stats = fm_with_above_threshold.statistics()["temperature"]
        assert stats["current"] == "red"


class TestUnmonitoredParameters:
    def test_includes_battery_level(self, acs_stub) -> None:
        fm = FaultManagement()
        fm.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            battery_level=0.6,
        )
        classifications = fm.check(hk, acs=acs_stub)
        assert "battery_level" in classifications

    def test_excludes_temperature(self, acs_stub) -> None:
        fm = FaultManagement()
        fm.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            battery_level=0.6,
        )
        classifications = fm.check(hk, acs=acs_stub)
        assert "temperature" not in classifications

    def test_classifies_battery_as_nominal(self, acs_stub) -> None:
        fm = FaultManagement()
        fm.add_threshold("battery_level", yellow=0.5, red=0.4, direction="below")
        hk = Housekeeping(
            timestamp=datetime.fromtimestamp(1000.0, tz=timezone.utc),
            battery_level=0.6,
        )
        classifications = fm.check(hk, acs=acs_stub)
        assert classifications["battery_level"] == "nominal"
