"""Test fixtures for ephemeris subsystem tests."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
import rust_ephem


@pytest.fixture
def mock_ephemeris():
    """Create a mock TLEEphemeris object."""
    mock = Mock(spec=rust_ephem.TLEEphemeris)
    base_time = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
    mock.timestamp = [
        base_time,
        base_time + timedelta(seconds=60),
        base_time + timedelta(seconds=120),
    ]
    mock.index = Mock(return_value=1)
    return mock
