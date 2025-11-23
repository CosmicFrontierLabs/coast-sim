"""Tests for conops.ephemeris module to achieve 100% coverage."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
import rust_ephem

from conops.ephemeris import (
    compute_tle_ephemeris,
)


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


class TestComputeTleEphemeris:
    """Tests for compute_tle_ephemeris function."""

    def test_compute_with_naive_datetimes(self, monkeypatch):
        """Test that compute_tle_ephemeris converts naive datetimes to UTC."""
        mock_tleephem_class = Mock()
        mock_instance = Mock()
        mock_tleephem_class.return_value = mock_instance
        monkeypatch.setattr(
            "conops.ephemeris.rust_ephem.TLEEphemeris", mock_tleephem_class
        )

        # Use naive datetimes
        begin = datetime(2024, 1, 1, 0, 0, 0)
        end = datetime(2024, 1, 1, 1, 0, 0)
        tle_string = "1 25544U\n2 25544"

        result = compute_tle_ephemeris(tle_string, begin, end, step_size=60)

        # Verify TLEEphemeris was called
        assert mock_tleephem_class.called

        # Verify datetimes were made timezone-aware
        call_kwargs = mock_tleephem_class.call_args[1]
        assert call_kwargs["begin"].tzinfo is not None
        assert call_kwargs["end"].tzinfo is not None

        # Verify result
        assert result is mock_instance

    def test_compute_with_aware_datetimes(self, monkeypatch):
        """Test that compute_tle_ephemeris handles timezone-aware datetimes."""
        mock_tleephem_class = Mock()
        mock_instance = Mock()
        mock_tleephem_class.return_value = mock_instance
        monkeypatch.setattr(
            "conops.ephemeris.rust_ephem.TLEEphemeris", mock_tleephem_class
        )

        # Use timezone-aware datetimes
        begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        tle_string = "1 25544U\n2 25544"

        result = compute_tle_ephemeris(tle_string, begin, end, step_size=120)

        # Verify TLEEphemeris was called with correct parameters
        call_kwargs = mock_tleephem_class.call_args[1]
        assert call_kwargs["tle"] == tle_string
        assert call_kwargs["begin"] == begin
        assert call_kwargs["end"] == end
        assert call_kwargs["step_size"] == 120

        # Verify result
        assert result is mock_instance

    def test_compute_accepts_kwargs(self, monkeypatch):
        """Test that compute_tle_ephemeris accepts additional kwargs."""
        mock_tleephem_class = Mock()
        mock_instance = Mock()
        mock_tleephem_class.return_value = mock_instance
        monkeypatch.setattr(
            "conops.ephemeris.rust_ephem.TLEEphemeris", mock_tleephem_class
        )

        begin = datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        end = datetime(2024, 1, 1, 1, 0, 0, tzinfo=timezone.utc)
        tle_string = "1 25544U\n2 25544"

        # Should not raise even with extra kwargs
        result = compute_tle_ephemeris(
            tle_string, begin, end, step_size=60, extra_param="ignored"
        )

        # Verify TLEEphemeris was called
        assert mock_tleephem_class.called
        assert result is mock_instance
