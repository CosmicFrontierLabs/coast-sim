"""Tests for conops.ephemeris module to achieve 100% coverage."""

from datetime import datetime, timezone
from unittest.mock import Mock

from conops import (
    compute_tle_ephemeris,
)


class TestComputeTleEphemeris:
    """Tests for compute_tle_ephemeris function."""

    def test_compute_with_naive_datetimes(self, monkeypatch):
        """Test that compute_tle_ephemeris converts naive datetimes to UTC."""
        mock_tleephem_class = Mock()
        mock_instance = Mock()
        mock_tleephem_class.return_value = mock_instance
        monkeypatch.setattr(
            "conops.config.ephemeris.rust_ephem.TLEEphemeris", mock_tleephem_class
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
            "conops.config.ephemeris.rust_ephem.TLEEphemeris", mock_tleephem_class
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
            "conops.config.ephemeris.rust_ephem.TLEEphemeris", mock_tleephem_class
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
