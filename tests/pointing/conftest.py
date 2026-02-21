"""Test fixtures for pointing subsystem tests."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from conops import Pointing
from conops.config import MissionConfig


class DummyConstraint:
    def __init__(
        self,
        in_constraint_val=False,
        in_sun_val=False,
        in_earth_val=False,
        in_moon_val=False,
        in_panel_val=False,
        step_size=1,
    ):
        self._in_constraint = in_constraint_val
        self._in_sun = in_sun_val
        self._in_earth = in_earth_val
        self._in_moon = in_moon_val
        self._in_panel = in_panel_val
        self.ephem = SimpleNamespace(step_size=step_size)

    def in_constraint(self, ra, dec, utime, hardonly=False):
        return self._in_constraint

    def in_sun(self, ra, dec, utime):
        return self._in_sun

    def in_earth(self, ra, dec, utime):
        return self._in_earth

    def in_moon(self, ra, dec, utime):
        return self._in_moon

    def in_panel(self, ra, dec, utime):
        return self._in_panel


@pytest.fixture
def constraint():
    return DummyConstraint()


class TestMissionConfig(MissionConfig):
    """Minimal MissionConfig for testing."""

    pass


@pytest.fixture
def mock_config(constraint):
    """Create a test config."""
    config = TestMissionConfig()
    config.constraint = constraint
    config.spacecraft_bus = Mock()
    config.spacecraft_bus.attitude_control = Mock()
    return config


@pytest.fixture
def pointing(mock_config):
    return Pointing(config=mock_config)


@pytest.fixture
def dummy_constraint():
    """Fixture providing a DummyConstraint with common test values."""
    return DummyConstraint(
        in_constraint_val=False,
        in_sun_val=True,
        in_earth_val=True,
        in_moon_val=False,
        in_panel_val=True,
    )
