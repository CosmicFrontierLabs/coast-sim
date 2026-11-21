"""A remainder shorter than ``ss_min`` that the priority-first plan put earlier
in time than the rest of its request's exposure survives time-ordered decoding,
and the solver starts from a whole solution."""

import math
from datetime import timedelta

import pytest

from conops.benchmark.scenarios import baseline
from conops.schedulers import LocalSearchScheduler
from conops.schedulers.local_search_scheduler import _Snapshot
from conops.schedulers.priority_scheduler import _collection_seconds

from .planning_scenario import BEGIN, MIN, PATCH, T0, make_config, make_target

OBSID = 100


def _scheduler() -> LocalSearchScheduler:
    """One request of 15 minutes taken in 10-minute snapshots: a full
    snapshot and a 5-minute remainder."""
    config = make_config(4)
    target = make_target(config, OBSID, *PATCH[0], minutes=15, snapshot=10)
    scheduler = LocalSearchScheduler(
        config, [target], BEGIN, BEGIN + timedelta(hours=4), time_limit=0
    )
    scheduler._start_from_priority_plan()
    return scheduler


def _science(scheduler: LocalSearchScheduler, sequence: list[_Snapshot]) -> list[float]:
    final = scheduler._decode(sequence).final
    return [_collection_seconds(b) for b in final.timeline if not b.is_fixed]


def test_a_remainder_before_its_full_snapshot_is_decoded() -> None:
    scheduler = _scheduler()
    sequence = [
        _Snapshot(obsid=OBSID, seconds=5 * MIN, release=T0, remainder=True),
        _Snapshot(obsid=OBSID, seconds=10 * MIN),
    ]
    decoded = scheduler._decode(sequence)
    assert _science(scheduler, sequence) == [5 * MIN, 10 * MIN]
    assert decoded.states[1].owed == {OBSID}
    assert decoded.final.owed == frozenset()
    assert scheduler._objective(decoded.final) == decoded.final.score > 0


def test_a_remainder_without_its_full_snapshot_is_not_a_plan() -> None:
    scheduler = _scheduler()
    decoded = scheduler._decode(
        [_Snapshot(obsid=OBSID, seconds=5 * MIN, release=T0, remainder=True)]
    )
    assert decoded.final.owed == {OBSID}
    assert scheduler._objective(decoded.final) == -math.inf


def test_a_short_snapshot_that_is_not_a_remainder_is_left_out() -> None:
    scheduler = _scheduler()
    assert _science(scheduler, [_Snapshot(obsid=OBSID, seconds=5 * MIN)]) == []


def test_the_priority_plan_decodes_to_itself() -> None:
    """The baseline's priority-first plan has a remainder ahead of its full
    snapshot; decoding in time order used to drop it."""
    scenario = baseline("examples/example.tle")
    config = scenario.make_config()
    scheduler = LocalSearchScheduler(
        config,
        scenario.make_targets(config),
        scenario.begin,
        scenario.end,
        time_limit=0,
    )
    start = scheduler._start_from_priority_plan()
    assert any(snapshot.remainder for snapshot in start.sequence)
    planned = [b for b in scheduler._greedy_timeline if not b.is_fixed]
    decoded = [b for b in start.final.timeline if not b.is_fixed]
    assert len(decoded) == len(planned)
    assert scheduler.start_score == pytest.approx(scheduler.initial_score)


def test_the_solver_is_hinted_every_variable(monkeypatch: pytest.MonkeyPatch) -> None:
    pytest.importorskip("ortools")
    from conops.schedulers import CpSatScheduler
    from conops.schedulers.cpsat_scheduler import _ChunkModel

    hinted: list[tuple[int, int]] = []

    def record(chunk: _ChunkModel, *args: object, **kwargs: object) -> object:
        proto = chunk.model.proto
        hinted.append((len(proto.solution_hint.vars), len(proto.variables)))
        return "UNKNOWN", "solver", None

    monkeypatch.setattr(_ChunkModel, "solve", record)
    scenario = baseline("examples/example.tle", hours=6)
    config = scenario.make_config()
    CpSatScheduler(
        config,
        scenario.make_targets(config),
        scenario.begin,
        scenario.end,
        solver_time_limit=1,
    ).schedule()
    assert hinted
    assert all(given == variables for given, variables in hinted)
