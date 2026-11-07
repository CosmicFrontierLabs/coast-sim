"""The planning prefilter follows a track through constraint crossings."""

from datetime import timedelta

import pytest

from conops.schedulers.context import SchedulingContext
from conops.targets import MovingPointing, track_from_table
from conops.targets.track import TrackPoint

from .planning_scenario import BEGIN, T0, make_config


def _scenario(start: float, end: float):
    config = make_config(1)
    config.constraint.earth_constraint = None
    config.constraint.ignore_roll = True
    target = MovingPointing(
        config=config,
        obsid=701,
        ra=start,
        dec=10,
        track=track_from_table([T0, T0 + 3600], [start, end], [10, 10]),
    )
    ctx = SchedulingContext(config, BEGIN, BEGIN + timedelta(hours=1))
    return config, target, ctx


@pytest.mark.parametrize("start,end", [(105, 230), (230, 105)])
def test_scheduler_windows_follow_the_track_across_a_sun_cone(
    start: float, end: float
) -> None:
    _, target, ctx = _scenario(start, end)
    target.visibility()
    expected = [tuple(window) for window in target.windows]
    assert expected and expected != [(T0, T0 + 3600)]
    assert ctx.visibility_windows(target) == expected


def test_replacing_a_track_invalidates_its_planning_windows() -> None:
    _, target, ctx = _scenario(105, 105)
    assert ctx.visibility_windows(target) == [(T0, T0 + 3600)]
    target.track = track_from_table([T0, T0 + 3600], [105, 230], [10, 10])
    target.visibility()
    expected = [tuple(window) for window in target.windows]
    assert expected != [(T0, T0 + 3600)]
    assert ctx.visibility_windows(target) == expected


def test_body_track_is_loaded_once_before_the_scheduler_queries_windows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, expected_target, ctx = _scenario(105, 230)
    expected_target.visibility()
    expected = [tuple(window) for window in expected_target.windows]
    calls = []

    def fetch(target: MovingPointing) -> list[TrackPoint]:
        calls.append(target.obsid)
        target.track = list(expected_target.track)
        target.ra, target.dec = target.track[0].ra, target.track[0].dec
        return target.track

    monkeypatch.setattr(MovingPointing, "fetch_track", fetch)
    target = MovingPointing(config=config, obsid=702, body="501")
    windows = ctx.visibility_windows(target)
    assert calls == [702]
    assert windows == expected
    windows.clear()
    assert ctx.visibility_windows(target) == expected
    assert calls == [702]
