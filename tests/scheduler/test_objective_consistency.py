"""Timeline and decoded scores must use each observation's own visit bounds."""

from datetime import timedelta

import pytest

from conops.config import ObservationGroup, ObservationGroups
from conops.schedulers.local_search_scheduler import _Snapshot
from conops.targets import (
    ObservationSequence,
    Plan,
    Pointing,
    SequenceStep,
    VisitWindow,
)

from .planning_scenario import (
    BEGIN,
    HOUR,
    MIN,
    T0,
    collected_seconds,
    make_config,
    make_target,
    planner,
    replay,
)


def scheduler(*, group=False, sequence=False):
    c = make_config(3)
    c.constraint.sun_constraint = None
    c.constraint.earth_constraint = None
    t = Pointing(
        config=c,
        obsid=100,
        ra=105,
        dec=10,
        merit=100,
        ss_min=10 * MIN,
        ss_max=10 * MIN,
        sequence=ObservationSequence(
            steps=[
                SequenceStep(
                    name="calibration", duration_seconds=2 * MIN, science=False
                ),
                SequenceStep(name="science A", duration_seconds=5 * MIN),
                SequenceStep(name="science B", duration_seconds=5 * MIN),
            ]
        )
        if sequence
        else None,
        visit_windows=[
            VisitWindow(start=T0 + 5 * MIN, deadline=T0 + 65 * MIN),
            VisitWindow(start=T0 + 100 * MIN, deadline=T0 + 160 * MIN),
        ],
    )
    if group:
        c.observation_groups = ObservationGroups(
            groups=[
                ObservationGroup(
                    name="Campaign",
                    members=[100],
                    window_start=T0 + 10 * MIN,
                    window_deadline=T0 + 145 * MIN,
                )
            ]
        )
    other = make_target(c, 101, 110, 10, merit=1000, minutes=40, snapshot=40, ss_min=40)
    cls, options = planner("local_search")
    return cls(
        c,
        [t, other],
        BEGIN,
        BEGIN + timedelta(hours=3),
        include_passes=False,
        earliness_weight=0.5,
        **options,
    )


@pytest.mark.parametrize("group", [False, True])
def test_rescoring_a_timeline_does_not_depend_on_request_visit_cursor(group):
    s = scheduler(group=group)
    start = s._start_from_priority_plan()
    request = s._by_obsid[100]
    blocks = [b for b in start.final.timeline if not b.is_fixed]
    scores = []
    for visit in (None, 0, 1, None):
        request.select_visit(visit, s.ctx)
        scores.append(s._timeline_score(blocks))
        assert request.visit == visit  # Scoring must not mutate admission state.
    assert scores == pytest.approx([start.final.score] * len(scores))


@pytest.mark.parametrize("group", [False, True])
def test_each_visit_uses_its_own_deadline_and_group_intersection(group):
    s = scheduler(group=group)
    start = s._start_from_priority_plan()
    request = s._by_obsid[100]
    blocks = [b for b in start.final.timeline if b.entry.obsid == 100]
    assert [b.entry.visit for b in blocks] == [0, 1]
    request.select_visit(None, s.ctx)
    for block in blocks:
        entry = block.entry
        window = request.target.visit_windows[entry.visit]
        opens, deadline = window.start, window.deadline
        if group:
            opens = max(opens, T0 + 10 * MIN)
            deadline = min(deadline, T0 + 145 * MIN)
        delay = max(0, float(entry.collection_begin) - opens)
        fraction = min(1, delay / (deadline - opens))
        expected = (
            100
            * (float(entry.collection_end) - float(entry.collection_begin))
            * (1 - 0.5 * fraction)
        )
        assert s._contribution(request, block, 100) == pytest.approx(expected)
        assert request.visit is None


def test_an_improvement_is_not_rejected_against_an_undiscounted_baseline():
    s = scheduler()
    start = s._start_from_priority_plan()
    assert [v.obsid for v in start.sequence] == [101, 100, 100]
    candidate = s._decode(
        [
            _Snapshot(obsid=100, seconds=10 * MIN),
            _Snapshot(obsid=101, seconds=40 * MIN),
            _Snapshot(obsid=100, seconds=10 * MIN),
        ]
    )
    assert candidate.final.score > start.final.score
    assert candidate.final.score > s.initial_score
    plan = s._finish_best(candidate)
    science = [e for e in plan if e.collection_begin is not None]
    assert science[0].obsid == 100
    assert s.score == candidate.final.score
    run = replay(s.config, plan, BEGIN, BEGIN + timedelta(hours=3))
    assert collected_seconds(run) == HOUR


@pytest.mark.parametrize("group", [False, True])
def test_sequence_blocks_and_saved_steps_retain_the_scored_visit(group, tmp_path):
    s = scheduler(group=group, sequence=True)
    start = s._start_from_priority_plan()
    blocks = [b for b in start.final.timeline if b.entry.obsid == 100]
    assert [b.entry.visit for b in blocks] == [0, 1]
    for b in blocks:
        assert all(e.visit == b.entry.visit for e in b.steps)
    request = s._by_obsid[100]
    request.select_visit(None, s.ctx)
    assert s._timeline_score(
        [b for b in start.final.timeline if not b.is_fixed]
    ) == pytest.approx(start.final.score)
    plan = s._finish_best(start)
    saved = Plan.load(plan.save(tmp_path / "sequence.json"))
    assert [e.visit for e in saved if e.obsid == 100] == [0, 0, 0, 1, 1, 1]
    run = replay(s.config, saved, BEGIN, BEGIN + timedelta(hours=3))
    assert collected_seconds(run, 100, science=True) == 20 * MIN
