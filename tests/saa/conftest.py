"""Test fixtures for saa subsystem tests."""

import numpy as np
import pytest

from conops import SAA


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


@pytest.fixture
def build_saa_with_ephem_fixture():
    def _build(utime, longs, lats, inside_coords):
        s = SAA(year=2020, day=1)
        s.ephem = DummyEphem(utime, longs, lats)
        s.saapoly = FakePoly(inside_coords)
        return s

    return _build


@pytest.fixture
def saa_single(build_saa_with_ephem_fixture):
    return build_saa_with_ephem_fixture(
        utime=[10, 20, 30, 40],
        longs=[0.0, -60.0, -60.0, 0.0],
        lats=[0.0, -11.0, -11.0, 0.0],
        inside_coords={(-60.0, -11.0)},
    )


@pytest.fixture
def saa_multiple(build_saa_with_ephem_fixture):
    return build_saa_with_ephem_fixture(
        utime=[1, 2, 3, 4, 5, 6, 7],
        longs=[0.0, -60.0, -60.0, 0.0, -60.0, -60.0, 0.0],
        lats=[0.0, -11.0, -11.0, 0.0, -11.0, -11.0, 0.0],
        inside_coords={(-60.0, -11.0)},
    )


@pytest.fixture
def saa_inside(build_saa_with_ephem_fixture):
    return build_saa_with_ephem_fixture(
        utime=[100],
        longs=[-60.0],
        lats=[-11.0],
        inside_coords={(-60.0, -11.0)},
    )


@pytest.fixture
def saa_outside(build_saa_with_ephem_fixture):
    return build_saa_with_ephem_fixture(
        utime=[100],
        longs=[-30.0],
        lats=[10.0],
        inside_coords={(-60.0, -11.0)},
    )
