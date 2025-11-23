"""Test fixtures for pointing subsystem tests."""

from types import SimpleNamespace

import pytest

from conops.pointing import Pointing


class DummyACSConfig:
    """Dummy ACS configuration for testing."""

    pass


class DummyConstraint:
    def __init__(
        self,
        inoccult_val=False,
        in_sun_val=False,
        in_earth_val=False,
        in_moon_val=False,
        in_panel_val=False,
        step_size=1,
    ):
        self._inoccult = inoccult_val
        self._in_sun = in_sun_val
        self._in_earth = in_earth_val
        self._in_moon = in_moon_val
        self._in_panel = in_panel_val
        self.ephem = SimpleNamespace(step_size=step_size)

    def inoccult(self, ra, dec, utime, hardonly=False):
        return self._inoccult

    def in_sun(self, ra, dec, utime):
        return self._in_sun

    def in_earth(self, ra, dec, utime):
        return self._in_earth

    def in_moon(self, ra, dec, utime):
        return self._in_moon

    def in_panel(self, ra, dec, utime):
        return self._in_panel


class DummySAA:
    def __init__(self, value=0):
        self._value = value

    def insaa(self, t):
        # Return same value for any time step
        return self._value


@pytest.fixture
def acs_config():
    return DummyACSConfig()


@pytest.fixture
def constraint():
    return DummyConstraint()


@pytest.fixture
def pointing(constraint, acs_config):
    return Pointing(constraint=constraint, acs_config=acs_config)


@pytest.fixture
def dummy_constraint():
    """Fixture providing a DummyConstraint with common test values."""
    return DummyConstraint(
        inoccult_val=False,
        in_sun_val=True,
        in_earth_val=True,
        in_moon_val=False,
        in_panel_val=True,
    )
