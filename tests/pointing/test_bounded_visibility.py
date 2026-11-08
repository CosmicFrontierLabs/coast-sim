"""Bounded visibility keeps whole-grid semantics without evaluating every row."""

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
    whole, bounded = target(ephem, kind), target(ephem, kind)
    whole.visibility()
    begin, end = (T0 + t for t in bounds)
    assert bounded.visibility(begin=begin, end=end) == 0
    np.testing.assert_allclose(
        np.asarray(bounded.windows),
        np.asarray(clipped(whole.windows, begin, end)),
        atol=1e-6,
    )


def test_queries_accumulate_extend_and_reuse_coverage(ephem, monkeypatch):
    t = target(ephem, "moving")
    calls = []
    original = type(t)._evaluate_visibility

    def evaluate(self, times=None):
        calls.append(times)
        return original(self, times)

    monkeypatch.setattr(type(t), "_evaluate_visibility", evaluate)
    t.visibility(begin=T0 + HOUR, end=T0 + 2 * HOUR)
    assert len(calls) == 1 and len(calls[0]) < len(ephem.timestamp) / 2
    first = list(t.windows)
    t.visibility(begin=T0 + HOUR + 120, end=T0 + 2 * HOUR - 120)
    assert len(calls) == 1 and t.windows == first
    t.visibility(begin=T0 + 4 * HOUR, end=T0 + 5 * HOUR)
    assert len(calls) == 2
    t.visibility(begin=T0, end=T0 + 6 * HOUR)
    whole = target(ephem, "moving")
    original(whole)
    assert t.windows == whole.windows
    assert len(calls) == 5  # Only the three gaps around the two known intervals.
    t.visibility()  # The legacy default always restores whole-ephemeris visibility.
    assert t.windows == whole.windows


@pytest.mark.parametrize("kind", ["fixed", "moving", "roll", "empty"])
def test_wider_query_evaluates_only_missing_intervals(ephem, kind, monkeypatch):
    bounded, whole = target(ephem, kind), target(ephem, kind)
    whole.visibility()
    calls = []
    original = type(bounded)._evaluate_visibility

    def evaluate(self, times=None):
        calls.append([t.timestamp() for t in times])
        return original(self, times)

    monkeypatch.setattr(type(bounded), "_evaluate_visibility", evaluate)
    bounded.visibility(begin=T0 + HOUR, end=T0 + 2 * HOUR)
    bounded.visibility(begin=T0, end=T0 + 3 * HOUR)
    assert len(calls) == 3
    assert max(calls[1]) == T0 + HOUR + 60
    assert min(calls[2]) == T0 + 2 * HOUR - 60
    np.testing.assert_allclose(
        np.asarray(bounded.windows),
        np.asarray(clipped(whole.windows, T0, T0 + 3 * HOUR)),
        atol=1e-6,
    )
    bounded.visibility(begin=T0 + 30, end=T0 + 3 * HOUR - 30)
    assert len(calls) == 3


def test_query_bridging_known_intervals_evaluates_each_gap(ephem, monkeypatch):
    bounded, whole = target(ephem, "moving"), target(ephem, "moving")
    whole.visibility()
    calls = []
    original = type(bounded)._evaluate_visibility

    def evaluate(self, times=None):
        calls.append([t.timestamp() for t in times])
        return original(self, times)

    monkeypatch.setattr(type(bounded), "_evaluate_visibility", evaluate)
    bounded.visibility(begin=T0 + HOUR, end=T0 + 2 * HOUR)
    bounded.visibility(begin=T0 + 4 * HOUR, end=T0 + 5 * HOUR)
    bounded.visibility(begin=T0, end=T0 + 6 * HOUR)
    assert len(calls) == 5
    assert (min(calls[2]), max(calls[2])) == (T0, T0 + HOUR + 60)
    assert (min(calls[3]), max(calls[3])) == (
        T0 + 2 * HOUR - 60,
        T0 + 4 * HOUR + 60,
    )
    assert (min(calls[4]), max(calls[4])) == (
        T0 + 5 * HOUR - 60,
        T0 + 6 * HOUR,
    )
    assert bounded.windows == whole.windows


@pytest.mark.parametrize(
    "changed", ["roll", "track", "constraint", "ephemeris", "pointing"]
)
def test_changed_inputs_invalidate_bounded_cache(ephem, changed):
    t = target(ephem, "moving" if changed == "track" else "fixed")
    t.visibility(begin=T0, end=T0 + 6 * HOUR)
    previous = t._visibility_key
    if changed == "roll":
        t.roll = 90
    if changed == "track":
        t.track = track_from_table([T0, T0 + 6 * HOUR], [30, 90], [0, 0])
    if changed == "constraint":
        t.config.constraint = Constraint(
            sun_constraint=rust_ephem.SunConstraint(min_angle=80)
        )
        t.config.constraint.ephem = ephem
        t.constraint = t.config.constraint
    if changed == "ephemeris":
        t.ephem = rust_ephem.TLEEphemeris(
            tle="examples/example.tle",
            begin=BEGIN,
            end=BEGIN + timedelta(hours=6),
            step_size=60,
        )
    if changed == "pointing":
        t.ra = (t.ra + 150) % 360
    t.visibility(begin=T0, end=T0 + 6 * HOUR)
    assert t._visibility_key != previous
    bounded = [list(w) for w in t.windows]
    t.visibility()
    assert t.windows == bounded


@pytest.mark.parametrize("bounds", [(100, 0), (float("nan"), 100), (0, float("inf"))])
def test_invalid_bounds_are_rejected(ephem, bounds):
    t = target(ephem, "fixed")
    with pytest.raises(ValueError, match="Visibility bounds"):
        t.visibility(begin=T0 + bounds[0], end=T0 + bounds[1])


@pytest.mark.parametrize("replace", [False, True])
def test_changed_constraint_components_rebuild_cached_backend_geometry(ephem, replace):
    t = target(ephem, "fixed")
    t.visibility()  # Populate the backend's own combined and Rust caches first.
    t.visibility(begin=T0, end=T0 + HOUR)
    if replace:
        t.constraint.sun_constraint = rust_ephem.SunConstraint(min_angle=90)
    else:
        t.constraint.sun_constraint.min_angle = 90
    t.visibility(begin=T0, end=T0 + HOUR)
    fresh = target(ephem, "fixed")
    fresh.constraint.sun_constraint = rust_ephem.SunConstraint(min_angle=90)
    fresh.visibility()
    assert t.windows == clipped(fresh.windows, T0, T0 + HOUR)
