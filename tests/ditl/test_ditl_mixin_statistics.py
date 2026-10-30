"""Tests for DITLMixin.print_statistics method, refactored into a test class."""

from typing import Any

import pytest


class TestDITLPrintStatistics:
    """Test class for DITLMixin.print_statistics."""

    # Basic output tests — one assertion per test
    def test_print_statistics_basic_contains_ditl_simulation_statistics(
        self, ditl_basic: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ditl_basic.print_statistics()
        output = capsys.readouterr().out
        assert "DITL SIMULATION STATISTICS" in output

    def test_print_statistics_basic_contains_configuration(
        self, ditl_basic: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ditl_basic.print_statistics()
        output = capsys.readouterr().out
        assert "Configuration: Test Spacecraft" in output

    def test_print_statistics_basic_contains_mode_distribution(
        self, ditl_basic: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ditl_basic.print_statistics()
        output = capsys.readouterr().out
        assert "MODE DISTRIBUTION" in output

    def test_print_statistics_basic_contains_observation_statistics(
        self, ditl_basic: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ditl_basic.print_statistics()
        output = capsys.readouterr().out
        assert "OBSERVATION STATISTICS" in output

    def test_print_statistics_basic_contains_pointing_statistics(
        self, ditl_basic: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ditl_basic.print_statistics()
        output = capsys.readouterr().out
        assert "POINTING STATISTICS" in output

    def test_print_statistics_basic_contains_power_and_battery_statistics(
        self, ditl_basic: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ditl_basic.print_statistics()
        output = capsys.readouterr().out
        assert "POWER AND BATTERY STATISTICS" in output

    def test_print_statistics_basic_contains_battery_capacity(
        self, ditl_basic: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ditl_basic.print_statistics()
        output = capsys.readouterr().out
        assert "Battery Capacity: 100.00 Wh" in output

    def test_print_statistics_basic_contains_science_mode(
        self, ditl_basic: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ditl_basic.print_statistics()
        output = capsys.readouterr().out
        assert "SCIENCE" in output

    def test_print_statistics_basic_contains_slewing_mode(
        self, ditl_basic: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ditl_basic.print_statistics()
        output = capsys.readouterr().out
        assert "SLEWING" in output

    # Queue test remains a single assertion
    def test_print_statistics_with_queue_contains_target_queue_statistics(
        self, ditl_with_queue: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ditl_with_queue.print_statistics()
        output = capsys.readouterr().out
        assert "TARGET QUEUE STATISTICS" in output

    # Empty data tests — one assertion per test
    def test_print_statistics_empty_data_contains_ditl_simulation_statistics(
        self, ditl_empty: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ditl_empty.print_statistics()
        output = capsys.readouterr().out
        assert "DITL SIMULATION STATISTICS" in output

    def test_print_statistics_empty_data_contains_configuration(
        self, ditl_empty: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ditl_empty.print_statistics()
        output = capsys.readouterr().out
        assert "Configuration: Test Spacecraft" in output


class TestInstrumentModeHours:
    def test_mode_hours_are_summed_from_telemetry_and_printed(
        self, ditl_basic: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        from datetime import datetime, timezone

        from conops.ditl.telemetry import Housekeeping, Telemetry

        ditl_basic.step_size = 60
        ditl_basic.telemetry = Telemetry()
        for k, modes in enumerate(
            [{"camera": "SCIENCE"}, {"camera": "SCIENCE"}, {"camera": "IDLE"}]
        ):
            ditl_basic.telemetry.housekeeping.append(
                Housekeeping(
                    timestamp=datetime.fromtimestamp(60.0 * k, tz=timezone.utc),
                    instrument_modes=modes,
                )
            )
        assert ditl_basic.instrument_mode_hours() == {
            "camera": {"SCIENCE": pytest.approx(2 / 60), "IDLE": pytest.approx(1 / 60)}
        }
        ditl_basic.print_statistics()
        output = capsys.readouterr().out
        assert "INSTRUMENT MODE HOURS" in output
        assert "camera" in output

    def test_no_table_without_modes(
        self, ditl_basic: Any, capsys: pytest.CaptureFixture[str]
    ) -> None:
        ditl_basic.print_statistics()
        assert "INSTRUMENT MODE HOURS" not in capsys.readouterr().out
