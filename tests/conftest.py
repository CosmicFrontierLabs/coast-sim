"""Shared pytest fixtures for test suite."""

from unittest.mock import Mock

import pytest


@pytest.fixture
def mock_ephem():
    """Create mock ephemeris."""
    ephem = Mock()
    ephem.step_size = 60.0
    ephem.timestamp = Mock()
    ephem.timestamp.unix = [1514764800.0 + i * 60.0 for i in range(1440)]
    ephem.earth = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0)) for _ in range(1440)]
    ephem.sun = [Mock(ra=Mock(deg=0.0), dec=Mock(deg=0.0))]
    ephem.index.return_value = 0
    return ephem
