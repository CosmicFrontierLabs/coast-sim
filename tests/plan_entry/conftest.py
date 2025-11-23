"""Test fixtures for plan_entry subsystem tests."""

from unittest.mock import Mock

import numpy as np
import pytest

from conops.plan_entry import PlanEntry


class MockEphemeris:
    """Mock ephemeris for testing."""

    def __init__(self):
        self.utime = np.array([100, 200, 300, 400])
        self.long = np.array([0.0, 10.0, 20.0, 30.0])
        self.lat = np.array([0.0, 5.0, 10.0, 15.0])
        # Add _tle_ephem attribute for visibility calculations
        self._tle_ephem = Mock()


class MockConstraint:
    """Mock constraint for testing."""

    def __init__(self):
        self.ephem = MockEphemeris()
        # Add a mock constraint object with evaluate method
        self.constraint = Mock()
        # Mock the evaluate return value
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
        self.constraint.evaluate = Mock(return_value=mock_result)


class MockACS:
    """Mock ACS configuration for testing."""

    def __init__(self):
        self.slew_acceleration = 0.5
        self.max_slew_rate = 0.25

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
def plan_entry(mock_constraint, mock_acs):
    """Fixture for PlanEntry with mocks."""
    return PlanEntry(constraint=mock_constraint, acs_config=mock_acs)
