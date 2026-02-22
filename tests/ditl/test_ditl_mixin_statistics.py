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
