import numpy as np

from conops.saa import SAA


class DummyEphem:
    """Minimal ephem stub matching SAA expectations."""

    def __init__(self, utime, longs, lats):
        from datetime import datetime, timezone

        self.utime = np.array(utime)
        self.long = np.array(longs)
        self.lat = np.array(lats)
        # Add timestamp as list of datetime objects for helper functions
        self.timestamp = [datetime.fromtimestamp(t, tz=timezone.utc) for t in utime]

    def ephindex(self, ut):
        idx = np.where(self.utime == ut)[0]
        if len(idx) == 0:
            raise ValueError("time not found in ephem.utime")
        return int(idx[0])

    def index(self, dt):
        """Find index for datetime object."""
        utime = dt.timestamp()
        return self.ephindex(utime)


class FakePoly:
    """Fake polygon that returns containments for a set of (lon, lat) pairs."""

    def __init__(self, inside_coords):
        # store as floats to match shapely Point.x/y
        self._inside = {(float(x), float(y)) for x, y in inside_coords}

    def contains(self, point):
        return (float(point.x), float(point.y)) in self._inside


def build_saa_with_ephem(utime, longs, lats, inside_coords):
    s = SAA(year=2020, day=1)
    s.ephem = DummyEphem(utime, longs, lats)
    s.saapoly = FakePoly(inside_coords)
    return s


class TestSAAInsaaCalc:
    def test_insaa_calc_true_when_point_inside(self):
        s = build_saa_with_ephem(
            utime=[100],
            longs=[-60.0],
            lats=[-11.0],
            inside_coords={(-60.0, -11.0)},
        )
        assert s.insaa_calc(100) is True

    def test_insaa_calc_false_when_point_outside(self):
        s = build_saa_with_ephem(
            utime=[100],
            longs=[-30.0],
            lats=[10.0],
            inside_coords={(-60.0, -11.0)},
        )
        assert s.insaa_calc(100) is False


class TestSAACalcAndInsaa:
    # Single interval tests ----------------------------------------------------------------------------
    def test_calc_sets_calculated_true(self):
        utime = [10, 20, 30, 40]
        longs = [0.0, -60.0, -60.0, 0.0]
        lats = [0.0, -11.0, -11.0, 0.0]
        inside_coords = {(-60.0, -11.0)}

        s = build_saa_with_ephem(utime, longs, lats, inside_coords)
        s.calc()
        assert s.calculated is True

    def test_calc_sets_single_interval(self):
        utime = [10, 20, 30, 40]
        longs = [0.0, -60.0, -60.0, 0.0]
        lats = [0.0, -11.0, -11.0, 0.0]
        inside_coords = {(-60.0, -11.0)}

        s = build_saa_with_ephem(utime, longs, lats, inside_coords)
        s.calc()
        np.testing.assert_array_equal(s.saatimes, [[20, 30]])

    def test_insaa_returns_one_for_start_of_interval(self):
        utime = [10, 20, 30, 40]
        longs = [0.0, -60.0, -60.0, 0.0]
        lats = [0.0, -11.0, -11.0, 0.0]
        inside_coords = {(-60.0, -11.0)}

        s = build_saa_with_ephem(utime, longs, lats, inside_coords)
        s.calc()
        assert s.insaa(20) == 1

    def test_insaa_returns_one_for_middle_of_interval(self):
        utime = [10, 20, 30, 40]
        longs = [0.0, -60.0, -60.0, 0.0]
        lats = [0.0, -11.0, -11.0, 0.0]
        inside_coords = {(-60.0, -11.0)}

        s = build_saa_with_ephem(utime, longs, lats, inside_coords)
        s.calc()
        assert s.insaa(30) == 1

    def test_insaa_returns_one_for_end_of_interval_inclusive(self):
        utime = [10, 20, 30, 40]
        longs = [0.0, -60.0, -60.0, 0.0]
        lats = [0.0, -11.0, -11.0, 0.0]
        inside_coords = {(-60.0, -11.0)}

        s = build_saa_with_ephem(utime, longs, lats, inside_coords)
        s.calc()
        assert s.insaa(30) == 1

    def test_insaa_returns_zero_before_interval(self):
        utime = [10, 20, 30, 40]
        longs = [0.0, -60.0, -60.0, 0.0]
        lats = [0.0, -11.0, -11.0, 0.0]
        inside_coords = {(-60.0, -11.0)}

        s = build_saa_with_ephem(utime, longs, lats, inside_coords)
        s.calc()
        assert s.insaa(10) == 0

    def test_insaa_returns_zero_after_interval(self):
        utime = [10, 20, 30, 40]
        longs = [0.0, -60.0, -60.0, 0.0]
        lats = [0.0, -11.0, -11.0, 0.0]
        inside_coords = {(-60.0, -11.0)}

        s = build_saa_with_ephem(utime, longs, lats, inside_coords)
        s.calc()
        assert s.insaa(41) == 0

    # Multiple interval tests --------------------------------------------------------------------------
    def test_calc_detects_multiple_intervals(self):
        utime = [1, 2, 3, 4, 5, 6, 7]
        longs = [0.0, -60.0, -60.0, 0.0, -60.0, -60.0, 0.0]
        lats = [0.0, -11.0, -11.0, 0.0, -11.0, -11.0, 0.0]
        inside_coords = {(-60.0, -11.0)}

        s = build_saa_with_ephem(utime, longs, lats, inside_coords)
        s.calc()
        np.testing.assert_array_equal(s.saatimes, [[2, 3], [5, 6]])

    def test_insaa_returns_one_for_first_interval_start(self):
        utime = [1, 2, 3, 4, 5, 6, 7]
        longs = [0.0, -60.0, -60.0, 0.0, -60.0, -60.0, 0.0]
        lats = [0.0, -11.0, -11.0, 0.0, -11.0, -11.0, 0.0]
        inside_coords = {(-60.0, -11.0)}

        s = build_saa_with_ephem(utime, longs, lats, inside_coords)
        s.calc()
        assert s.insaa(2) == 1

    def test_insaa_returns_one_for_first_interval_middle(self):
        utime = [1, 2, 3, 4, 5, 6, 7]
        longs = [0.0, -60.0, -60.0, 0.0, -60.0, -60.0, 0.0]
        lats = [0.0, -11.0, -11.0, 0.0, -11.0, -11.0, 0.0]
        inside_coords = {(-60.0, -11.0)}

        s = build_saa_with_ephem(utime, longs, lats, inside_coords)
        s.calc()
        assert s.insaa(3) == 1

    def test_insaa_returns_one_for_first_interval_end_inclusive(self):
        utime = [1, 2, 3, 4, 5, 6, 7]
        longs = [0.0, -60.0, -60.0, 0.0, -60.0, -60.0, 0.0]
        lats = [0.0, -11.0, -11.0, 0.0, -11.0, -11.0, 0.0]
        inside_coords = {(-60.0, -11.0)}

        s = build_saa_with_ephem(utime, longs, lats, inside_coords)
        s.calc()
        assert s.insaa(3) == 1

    def test_insaa_returns_one_for_second_interval_start(self):
        utime = [1, 2, 3, 4, 5, 6, 7]
        longs = [0.0, -60.0, -60.0, 0.0, -60.0, -60.0, 0.0]
        lats = [0.0, -11.0, -11.0, 0.0, -11.0, -11.0, 0.0]
        inside_coords = {(-60.0, -11.0)}

        s = build_saa_with_ephem(utime, longs, lats, inside_coords)
        s.calc()
        assert s.insaa(5) == 1

    def test_insaa_returns_one_for_second_interval_middle(self):
        utime = [1, 2, 3, 4, 5, 6, 7]
        longs = [0.0, -60.0, -60.0, 0.0, -60.0, -60.0, 0.0]
        lats = [0.0, -11.0, -11.0, 0.0, -11.0, -11.0, 0.0]
        inside_coords = {(-60.0, -11.0)}

        s = build_saa_with_ephem(utime, longs, lats, inside_coords)
        s.calc()
        assert s.insaa(6) == 1

    def test_insaa_returns_one_for_second_interval_end_inclusive(self):
        utime = [1, 2, 3, 4, 5, 6, 7]
        longs = [0.0, -60.0, -60.0, 0.0, -60.0, -60.0, 0.0]
        lats = [0.0, -11.0, -11.0, 0.0, -11.0, -11.0, 0.0]
        inside_coords = {(-60.0, -11.0)}

        s = build_saa_with_ephem(utime, longs, lats, inside_coords)
        s.calc()
        assert s.insaa(6) == 1

    def test_insaa_returns_zero_between_intervals(self):
        utime = [1, 2, 3, 4, 5, 6, 7]
        longs = [0.0, -60.0, -60.0, 0.0, -60.0, -60.0, 0.0]
        lats = [0.0, -11.0, -11.0, 0.0, -11.0, -11.0, 0.0]
        inside_coords = {(-60.0, -11.0)}

        s = build_saa_with_ephem(utime, longs, lats, inside_coords)
        s.calc()
        assert s.insaa(4.5) == 0
