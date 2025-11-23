"""Test fixtures for constraint subsystem tests."""

from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from astropy.time import Time  # type: ignore[import-untyped]

from conops import Constraint


@pytest.fixture
def constraint():
    """Fixture for a basic Constraint instance."""
    return Constraint()


@pytest.fixture
def constraint_with_ephem():
    """Fixture for a Constraint instance with mocked ephem."""
    c = Constraint()
    c.ephem = Mock()
    c.ephem._tle_ephem = Mock()
    return c


@pytest.fixture
def time_list():
    """Fixture for a list of datetime objects."""
    return [
        datetime.fromtimestamp(1700000000.0, tz=timezone.utc),
        datetime.fromtimestamp(1700000100.0, tz=timezone.utc),
    ]


@pytest.fixture
def time_astropy():
    """Fixture for an astropy Time object."""
    return Time([1700000000.0, 1700000100.0], format="unix")
