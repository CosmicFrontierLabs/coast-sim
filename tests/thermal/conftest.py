"""Test fixtures for thermal subsystem tests."""

import pytest

from conops import Heater, PowerDraw

"""Unit tests for conops.thermal.Heater"""


@pytest.fixture
def simple_power_draw():
    """Simple PowerDraw with only nominal power."""
    return PowerDraw(nominal_power=10.0, peak_power=15.0)


@pytest.fixture
def mode_based_power_draw():
    """PowerDraw with mode-specific power values."""
    return PowerDraw(
        nominal_power=10.0,
        peak_power=15.0,
        power_mode={0: 5.0, 1: 10.0, 2: 12.0, 3: 20.0},
    )


@pytest.fixture
def simple_heater(simple_power_draw):
    """Heater with simple power draw."""
    return Heater(name="Test Heater", power_draw=simple_power_draw)


@pytest.fixture
def mode_heater(mode_based_power_draw):
    """Heater with mode-based power draw."""
    return Heater(name="Mode Heater", power_draw=mode_based_power_draw)
