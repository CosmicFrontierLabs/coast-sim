"""``visibility_in`` equals the whole-ephemeris windows clipped to the interval."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import rust_ephem

from conops.config import Constraint, MissionConfig
from conops.targets import MovingPointing, Pointing, track_from_table

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
T0 = BEGIN.timestamp()
HOUR = 3600


@pytest.fixture(scope="module")
def ephem():
    return rust_ephem.TLEEphemeris(
        tle="examples/example.tle",
        begin=BEGIN,
        end=BEGIN + timedelta(hours=6),
        step_size=60,
    )


def target(ephem, kind):
    constraint = Constraint(
        sun_constraint=rust_ephem.SunConstraint(min_angle=45),
        earth_constraint=rust_ephem.EarthLimbConstraint(min_angle=10),
    )
    if kind == "roll":
        constraint = Constraint(
            ignore_roll=False,
            science_constraint=rust_ephem.SunConstraint(min_angle=45).boresight_offset(
                yaw_deg=30
            ),
        )
    if kind == "empty":
        constraint = Constraint(sun_constraint=None, earth_constraint=None)
    constraint.ephem = ephem
    config = MissionConfig(constraint=constraint)
    sun_ra, sun_dec = float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0])
    if kind == "moving":
        return MovingPointing(
            config=config,
            obsid=1,
            track=track_from_table(
                [T0, T0 + 6 * HOUR],
                [(sun_ra + 70) % 360, (sun_ra + 10) % 360],
                [sun_dec] * 2,
            ),
        )
    return Pointing(
        config=config, obsid=1, ra=(sun_ra + 60) % 360, dec=sun_dec, roll=15
    )


def clipped(windows, begin, end):
    return [[max(a, begin), min(b, end)] for a, b in windows if a <= end and b >= begin]


@pytest.mark.parametrize("kind", ["fixed", "moving", "roll", "empty"])
@pytest.mark.parametrize(
    "bounds", [(0, 360), (61, 1799), (3601, 17039), (21599, 21600), (4800, 4800)]
)
def test_bounded_result_matches_clipping_whole_ephemeris(ephem, kind, bounds):
    t = target(ephem, kind)
    t.visibility()
    whole = [list(w) for w in t.windows]
    begin, end = (T0 + b for b in bounds)
    np.testing.assert_allclose(
        np.asarray(t.visibility_in(begin, end)),
        np.asarray(clipped(whole, begin, end)),
        atol=1e-6,
    )
    assert t.windows == whole  # A bounded query never replaces stored windows.


def test_bounded_query_evaluates_only_the_interval(ephem, monkeypatch):
    t = target(ephem, "fixed")
    seen = []
    original = type(t)._visibility_windows
    monkeypatch.setattr(
        type(t),
        "_visibility_windows",
        lambda self, times=None: (seen.append(times), original(self, times))[1],
    )
    t.visibility_in(T0 + HOUR, T0 + 2 * HOUR)
    assert [s.timestamp() for s in (seen[0][0], seen[0][-1])] == [
        T0 + HOUR - 60,
        T0 + 2 * HOUR + 60,
    ]


@pytest.mark.parametrize("bounds", [(100, 0), (float("nan"), 100), (0, float("inf"))])
def test_invalid_bounds_are_rejected(ephem, bounds):
    with pytest.raises(ValueError, match="Visibility bounds"):
        target(ephem, "fixed").visibility_in(T0 + bounds[0], T0 + bounds[1])
