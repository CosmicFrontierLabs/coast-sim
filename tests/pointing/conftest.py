"""Test fixtures for pointing subsystem tests."""

from unittest.mock import Mock

import pytest
import rust_ephem
from pydantic import PrivateAttr

from conops import (
    AttitudeControlSystem,
    Constraint,
    MissionConfig,
    Pointing,
)


class DummyConstraint(Constraint):
    """Stub Constraint for testing.

    Subclasses the real Constraint (a Pydantic model) rather than being a bare
    stand-in object, since pydantic-core validates model-typed fields against
    the actual class/MRO and does not honor duck-typing or ABC registration.
    """

    _in_constraint: bool = PrivateAttr(default=False)
    _in_sun: bool = PrivateAttr(default=False)
    _in_earth: bool = PrivateAttr(default=False)
    _in_moon: bool = PrivateAttr(default=False)
    _in_panel: bool = PrivateAttr(default=False)

    def __init__(
        self,
        in_constraint_val=False,
        in_sun_val=False,
        in_earth_val=False,
        in_moon_val=False,
        in_panel_val=False,
        step_size=1,
    ):
        super().__init__()
        self._in_constraint = in_constraint_val
        self._in_sun = in_sun_val
        self._in_earth = in_earth_val
        self._in_moon = in_moon_val
        self._in_panel = in_panel_val
        fake_ephem = Mock()
        fake_ephem.__class__ = rust_ephem.Ephemeris
        fake_ephem.step_size = step_size
        self.ephem = fake_ephem

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


@pytest.fixture
def mock_config(constraint):
    """Create a mock config."""
    config = Mock()
    config.__class__ = MissionConfig
    # MissionConfig's init_fault_management_defaults model_validator re-runs
    # whenever this config is embedded as a nested field elsewhere (e.g. on
    # PlanEntry.config) and needs battery/recorder threshold fields this
    # fixture doesn't populate; None short-circuits it via the validator's
    # own early-return guard.
    config.fault_management = None
    config.constraint = constraint
    config.spacecraft_bus.attitude_control.__class__ = AttitudeControlSystem
    return config


@pytest.fixture
def pointing(mock_config):
    return Pointing.from_config(config=mock_config, exptime=None)


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
