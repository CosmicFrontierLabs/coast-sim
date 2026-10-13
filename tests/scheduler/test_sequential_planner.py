"""SequentialPlanner: horizon, reruns and reporting, and its DumbScheduler shim."""

from datetime import timedelta

import pytest

from conops import DITLLog, DumbScheduler, SequentialPlanner

from .test_priority_planner import BEGIN, HOUR, _config, _target


def _targets(config, count=12):
    return [
        _target(config, obsid, (obsid * 37.0) % 360, (obsid * 23.0) % 140 - 70)
        for obsid in range(1, count + 1)
    ]


def test_plans_within_horizon() -> None:
    config = _config(6)
    begin, end = BEGIN + timedelta(hours=1), BEGIN + timedelta(hours=3)
    planner = SequentialPlanner(config, _targets(config), begin, end)

    plan = planner.schedule()

    assert plan is planner.plan
    assert len(plan) > 0
    assert all(begin.timestamp() <= e.begin < end.timestamp() for e in plan)
    assert all(e.end <= end.timestamp() for e in plan)
    assert plan.metadata == {"planner": "sequential"}


def test_rerun_starts_fresh() -> None:
    config = _config(6)
    planner = SequentialPlanner(
        config, _targets(config), BEGIN, BEGIN + timedelta(hours=2)
    )
    first = planner.schedule()

    second = planner.schedule()

    assert second is not first
    assert planner.scheduled == [e.obsid for e in second]


def test_reports_unplaced_targets() -> None:
    config = _config(6)
    targets = _targets(config)
    planner = SequentialPlanner(config, targets, BEGIN, BEGIN + timedelta(hours=1))

    planner.schedule()

    assert planner.unplaced
    assert planner.unplaced == [t for t in targets if t.exptime]


def test_starts_on_the_first_step_at_or_after_begin() -> None:
    config = _config(6)
    begin = BEGIN + timedelta(seconds=1)
    planner = SequentialPlanner(
        config, _targets(config), begin, BEGIN + timedelta(hours=2)
    )

    plan = planner.schedule()

    assert plan[0].begin == (BEGIN + timedelta(minutes=1)).timestamp()


def test_partially_observed_target_is_unplaced() -> None:
    config = _config(6)
    target = _target(config, 3, 111.0, -1.0, minutes=600)
    planner = SequentialPlanner(config, [target], BEGIN, BEGIN + timedelta(hours=2))

    planner.schedule()

    assert planner.scheduled == [3]
    assert 0 < target.exptime < 600 * 60
    assert planner.unplaced == [target]


def test_logs_instead_of_printing(capsys: pytest.CaptureFixture[str]) -> None:
    config = _config(6)
    log = DITLLog()
    SequentialPlanner(
        config, _targets(config), BEGIN, BEGIN + timedelta(hours=1), log=log
    ).schedule()

    assert capsys.readouterr().out == ""
    assert any("Scheduled" in event.description for event in log.events)


class TestDumbSchedulerShim:
    def test_is_a_sequential_planner(self) -> None:
        assert issubclass(DumbScheduler, SequentialPlanner)

    def test_matches_sequential_planner_over_whole_days(self) -> None:
        config = _config(24)
        planner = SequentialPlanner(
            config, _targets(config), BEGIN, BEGIN + timedelta(days=1)
        )
        legacy_config = _config(24)
        legacy = DumbScheduler(legacy_config, days=1, log=DITLLog())
        for target in _targets(legacy_config):
            legacy.targlist.add_target(target)

        expected = planner.schedule()
        actual = legacy.schedule()

        assert [(e.obsid, e.begin, e.end, e.exposure) for e in actual] == [
            (e.obsid, e.begin, e.end, e.exposure) for e in expected
        ]

    def test_mintime_is_min_window(self) -> None:
        legacy = DumbScheduler(_config(1))
        legacy.mintime = 600
        assert legacy.min_window == 600

    def test_horizon_follows_days(self) -> None:
        legacy = DumbScheduler(_config(48), days=1)
        legacy.days = 2
        ustart, uend = legacy._horizon()
        assert uend - ustart == 48 * HOUR
