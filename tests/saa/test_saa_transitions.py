"""Tests for SAA transition detection and timing."""

import numpy as np
import pytest

from conops import SAA


class DummyEphem:
    """Mock ephemeris for SAA testing."""

    def __init__(self, utime, longs, lats):
        from datetime import datetime, timezone

        self.utime = np.array(utime)
        self.longitude_deg = np.array(longs)
        self.latitude_deg = np.array(lats)
        self.timestamp = [datetime.fromtimestamp(t, tz=timezone.utc) for t in utime]

    def index(self, dt):
        utime = dt.timestamp()
        idx = np.searchsorted(self.utime, utime)
        return max(0, min(idx, len(self.utime) - 1))


class FakePoly:
    """Mock polygon that checks containment against a set of coordinates."""

    def __init__(self, inside_coords):
        self._inside = {(float(x), float(y)) for x, y in inside_coords}

    def contains(self, point):
        return (float(point.x), float(point.y)) in self._inside


def make_saa(utime, longs, lats, inside_coords):
    """Create SAA instance with mock ephemeris."""
    s = SAA(year=2020, day=1)
    s.ephem = DummyEphem(utime, longs, lats)
    s.saapoly = FakePoly(inside_coords)
    return s


class TestGetNextSAATime:
    """Tests for get_next_saa_time() method."""

    def test_get_next_saa_time_returns_first_interval(self):
        """Returns first SAA interval when query is before all intervals."""
        s = make_saa(
            utime=[10, 20, 30, 40],
            longs=[0.0, -60.0, -60.0, 0.0],
            lats=[0.0, -11.0, -11.0, 0.0],
            inside_coords={(-60.0, -11.0)},
        )
        s.calc()

        result = s.get_next_saa_time(5.0)
        assert result == (20, 30)

    def test_get_next_saa_time_returns_second_interval(self):
        """Returns second interval when query is after first."""
        s = make_saa(
            utime=[1, 2, 3, 4, 5, 6, 7],
            longs=[0.0, -60.0, -60.0, 0.0, -60.0, -60.0, 0.0],
            lats=[0.0, -11.0, -11.0, 0.0, -11.0, -11.0, 0.0],
            inside_coords={(-60.0, -11.0)},
        )
        s.calc()

        result = s.get_next_saa_time(3.5)
        assert result == (5, 6)

    def test_get_next_saa_time_returns_none_after_all(self):
        """Returns None when query is after all SAA intervals."""
        s = make_saa(
            utime=[10, 20, 30, 40],
            longs=[0.0, -60.0, -60.0, 0.0],
            lats=[0.0, -11.0, -11.0, 0.0],
            inside_coords={(-60.0, -11.0)},
        )
        s.calc()

        result = s.get_next_saa_time(50.0)
        assert result is None

    def test_get_next_saa_time_triggers_calc(self):
        """get_next_saa_time calls calc() if not already calculated."""
        s = make_saa(
            utime=[10, 20, 30, 40],
            longs=[0.0, -60.0, -60.0, 0.0],
            lats=[0.0, -11.0, -11.0, 0.0],
            inside_coords={(-60.0, -11.0)},
        )
        assert not s.calculated

        result = s.get_next_saa_time(5.0)

        assert s.calculated
        assert result is not None


class TestSAAPolygonBoundary:
    """Tests for SAA polygon boundary detection."""

    def test_point_clearly_inside_saa(self):
        """Point well inside SAA returns True."""
        s = SAA()
        # (-50, -15) is clearly inside the SAA polygon
        s.ephem = DummyEphem([100], [-50.0], [-15.0])

        result = s.insaa_calc(100)
        assert result is True

    def test_point_clearly_outside_saa(self):
        """Point well outside SAA returns False."""
        s = SAA()
        # (0, 0) is far from the SAA
        s.ephem = DummyEphem([100], [0.0], [0.0])

        result = s.insaa_calc(100)
        assert result is False

    def test_point_outside_saa_north(self):
        """Point north of SAA is outside."""
        s = SAA()
        # (-50, 10) is north of the SAA (SAA is in southern hemisphere)
        s.ephem = DummyEphem([100], [-50.0], [10.0])

        result = s.insaa_calc(100)
        assert result is False

    def test_point_outside_saa_east(self):
        """Point east of SAA is outside."""
        s = SAA()
        # (20, -15) is east of the SAA (SAA is roughly -85 to 0 longitude)
        s.ephem = DummyEphem([100], [20.0], [-15.0])

        result = s.insaa_calc(100)
        assert result is False


class TestSAANoIntervals:
    """Tests when no SAA intervals exist."""

    def test_calc_with_no_saa_crossings(self):
        """Ephemeris with no SAA crossings yields empty intervals."""
        s = make_saa(
            utime=[10, 20, 30, 40],
            longs=[0.0, 0.0, 0.0, 0.0],  # All outside SAA
            lats=[0.0, 0.0, 0.0, 0.0],
            inside_coords={(-60.0, -11.0)},  # Different coords
        )
        s.calc()

        assert len(s.saatimes) == 0

    def test_insaa_returns_zero_with_no_intervals(self):
        """insaa() returns 0 when no SAA intervals exist."""
        s = make_saa(
            utime=[10, 20, 30, 40],
            longs=[0.0, 0.0, 0.0, 0.0],
            lats=[0.0, 0.0, 0.0, 0.0],
            inside_coords={(-60.0, -11.0)},
        )
        s.calc()

        assert s.insaa(25) == 0

    def test_get_next_saa_returns_none_with_no_intervals(self):
        """get_next_saa_time() returns None when no SAA intervals exist."""
        s = make_saa(
            utime=[10, 20, 30, 40],
            longs=[0.0, 0.0, 0.0, 0.0],
            lats=[0.0, 0.0, 0.0, 0.0],
            inside_coords={(-60.0, -11.0)},
        )
        s.calc()

        assert s.get_next_saa_time(15) is None


class TestSAAMultiplePasses:
    """Tests for multiple SAA passes per ephemeris."""

    def test_three_saa_passes_detected(self):
        """Three separate SAA crossings are detected."""
        # Simulate orbit with 3 SAA passes
        s = make_saa(
            utime=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            longs=[0, -60, -60, 0, 0, -60, -60, 0, 0, -60, 0],
            lats=[0, -11, -11, 0, 0, -11, -11, 0, 0, -11, 0],
            inside_coords={(-60.0, -11.0)},
        )
        s.calc()

        assert len(s.saatimes) == 3

    def test_consecutive_saa_passes_have_gaps(self):
        """SAA passes don't merge when there's a gap between them."""
        s = make_saa(
            utime=[1, 2, 3, 4, 5, 6, 7],
            longs=[0, -60, -60, 0, -60, -60, 0],
            lats=[0, -11, -11, 0, -11, -11, 0],
            inside_coords={(-60.0, -11.0)},
        )
        s.calc()

        # Should be 2 passes, not 1 merged one
        assert len(s.saatimes) == 2
        # First pass ends at t=3, second starts at t=5
        assert s.saatimes[0][1] == 3
        assert s.saatimes[1][0] == 5


class TestSAAStateTracking:
    """Tests for SAA state variables."""

    def test_lat_long_set_after_insaa_calc(self):
        """insaa_calc() sets lat/long attributes."""
        s = make_saa(
            utime=[100],
            longs=[-45.0],
            lats=[-12.0],
            inside_coords={},
        )

        s.insaa_calc(100)

        assert s.long == -45.0
        assert s.lat == -12.0

    def test_get_saa_times_triggers_calc(self):
        """get_saa_times() triggers calc() if not calculated."""
        s = make_saa(
            utime=[10, 20, 30, 40],
            longs=[0.0, -60.0, -60.0, 0.0],
            lats=[0.0, -11.0, -11.0, 0.0],
            inside_coords={(-60.0, -11.0)},
        )
        assert not s.calculated

        times = s.get_saa_times()

        assert s.calculated
        assert len(times) > 0


class TestSAAMissingEphemeris:
    """Tests for error handling with missing ephemeris."""

    def test_insaa_calc_raises_without_ephem(self):
        """insaa_calc() raises ValueError without ephemeris."""
        s = SAA()
        s.ephem = None

        with pytest.raises(ValueError, match="Ephemeris must be set"):
            s.insaa_calc(100)

    def test_calc_raises_without_ephem(self):
        """calc() raises ValueError without ephemeris."""
        s = SAA()
        s.ephem = None

        with pytest.raises(ValueError, match="Ephemeris must be set"):
            s.calc()
