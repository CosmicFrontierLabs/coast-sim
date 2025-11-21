"""Tests for conops.ephemeris module to achieve 100% coverage."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest
import rust_ephem

from conops.ephemeris import (
    compute_tle_ephemeris,
    ephemeris_in_eclipse,
    ephemeris_index,
    ephemeris_utime,
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


class TestEphemerisInEclipse:
    """Tests for ephemeris_in_eclipse function."""

    def test_in_eclipse_calls_eclipse_constraint(self, monkeypatch):
        """Test that ephemeris_in_eclipse calls rust_ephem.EclipseConstraint."""
        mock_ephem = Mock()
        mock_constraint = Mock()
        mock_constraint.in_constraint = Mock(return_value=True)

        # Mock the EclipseConstraint class
        mock_constraint_class = Mock(return_value=mock_constraint)
        monkeypatch.setattr(
            "conops.ephemeris.rust_ephem.EclipseConstraint", mock_constraint_class
        )

        utime = 1704067200.0  # 2024-01-01 00:00:00 UTC
        result = ephemeris_in_eclipse(mock_ephem, utime)

        # Verify EclipseConstraint was created
        mock_constraint_class.assert_called_once()

        # Verify in_constraint was called with correct datetime
        assert mock_constraint.in_constraint.called
        call_args = mock_constraint.in_constraint.call_args
        dt_arg = call_args[0][0]
        assert isinstance(dt_arg, datetime)
        assert dt_arg.tzinfo is not None

        # Verify result
        assert result is True


class TestEphemerisIndex:
    """Tests for ephemeris_index function."""

    def test_index_returns_correct_value(self, mock_ephemeris):
        """Test that ephemeris_index calls ephem.index with correct datetime."""
        utime = 1704067200.0  # 2024-01-01 00:00:00 UTC
        result = ephemeris_index(mock_ephemeris, utime)

        # Verify index method was called
        assert mock_ephemeris.index.called

        # Verify datetime argument was timezone-aware
        call_args = mock_ephemeris.index.call_args
        dt_arg = call_args[0][0]
        assert isinstance(dt_arg, datetime)
        assert dt_arg.tzinfo is not None

        # Verify result
        assert result == 1


class TestEphemerisUtime:
    """Tests for ephemeris_utime function."""

    def test_utime_converts_timestamps_to_floats(self, mock_ephemeris):
        """Test that ephemeris_utime converts timestamp list to Unix timestamps."""
        result = ephemeris_utime(mock_ephemeris)

        # Verify result is a list of floats
        assert isinstance(result, list)
        assert len(result) == 3
        assert all(isinstance(x, float) for x in result)

        # Verify timestamps are correct
        assert result[0] == 1704067200.0  # 2024-01-01 00:00:00 UTC
        assert result[1] == 1704067260.0  # 2024-01-01 00:01:00 UTC
        assert result[2] == 1704067320.0  # 2024-01-01 00:02:00 UTC


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
