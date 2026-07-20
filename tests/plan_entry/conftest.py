"""Test fixtures for plan_entry subsystem tests."""

from unittest.mock import Mock

import numpy as np
import pytest
import rust_ephem

from conops import (
    AttitudeControlSystem,
    Constraint,
    MissionConfig,
    PlanEntry,
)


class MockEphemeris:
    """Mock ephemeris for testing."""

    def __init__(self):
        self.utime = np.array([100, 200, 300, 400])
        self.long = np.array([0.0, 10.0, 20.0, 30.0])
        self.lat = np.array([0.0, 5.0, 10.0, 15.0])
        # Add _tle_ephem attribute for visibility calculations
        self._tle_ephem = Mock()


# Register as a virtual subclass so isinstance checks (e.g. PlanEntry's
# pydantic field) pass without implementing every abstract Ephemeris member.
rust_ephem.Ephemeris.register(MockEphemeris)


class MockConstraint(Constraint):
    """Mock constraint for testing.

    Subclasses the real Constraint (a Pydantic model) rather than being a bare
    stand-in object, since pydantic-core validates model-typed fields against
    the actual class/MRO and does not honor ABC virtual-subclass registration.
    """

    def __init__(self):
        super().__init__(ignore_roll=False)
        self.ephem = MockEphemeris()
        # Add a mock constraint object with evaluate method (overrides the
        # real `constraint` cached_property via plain assignment).
        mock_constraint_obj = Mock()
        mock_result = Mock()
        mock_result.visibility = [
            Mock(
                start_time=Mock(timestamp=lambda: 100),
                end_time=Mock(timestamp=lambda: 200),
            ),
            Mock(
                start_time=Mock(timestamp=lambda: 300),
                end_time=Mock(timestamp=lambda: 400),
            ),
        ]
        mock_result.constraint_array = np.array([False, True, False, True])
        mock_constraint_obj.evaluate = Mock(return_value=mock_result)
        self.constraint = mock_constraint_obj


class MockACS(AttitudeControlSystem):
    """Mock ACS configuration for testing.

    Subclasses the real AttitudeControlSystem (a Pydantic model) for the same
    reason as MockConstraint above.
    """

    def __init__(self):
        super().__init__(slew_acceleration=0.5, max_slew_rate=0.25)

    def slew_time(self, distance):
        """Mock slew time calculation."""
        return distance / 0.25  # Simple linear relationship for testing

    def predict_slew(self, ra1, dec1, ra2, dec2, steps=20):
        """Mock predict slew."""
        # Calculate simple angular distance
        distance = abs(ra2 - ra1) + abs(dec2 - dec1)
        # Return mock slew path
        path = (
            np.linspace(ra1, ra2, steps),
            np.linspace(dec1, dec2, steps),
        )
        return distance, path


@pytest.fixture
def mock_constraint():
    """Fixture for mock constraint."""
    return MockConstraint()


@pytest.fixture
def mock_acs():
    """Fixture for mock ACS."""
    return MockACS()


@pytest.fixture
def mock_config(mock_constraint, mock_acs):
    """Fixture for mock config."""
    config = Mock()
    config.__class__ = MissionConfig
    # MissionConfig's init_fault_management_defaults model_validator re-runs
    # whenever this config is embedded as a nested field elsewhere (e.g. on
    # PlanEntry.config) and needs battery/recorder threshold fields this
    # fixture doesn't populate; None short-circuits it via the validator's
    # own early-return guard.
    config.fault_management = None
    config.constraint = mock_constraint
    config.spacecraft_bus = Mock()
    config.spacecraft_bus.attitude_control = mock_acs
    return config


@pytest.fixture
def plan_entry(mock_config):
    """Fixture for PlanEntry with mocks."""
    return PlanEntry(config=mock_config)
