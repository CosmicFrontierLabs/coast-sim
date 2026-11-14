"""Sequence overhead must not make a completed visit look like a later window."""

from datetime import timedelta

import pytest

from conops import create_ditl
from conops.config import ObservationTiming
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.targets import ObservationSequence, Pointing, SequenceStep, VisitWindow

from .planning_scenario import BEGIN, HOUR, MIN, T0, make_config


def _case():
    config = make_config(3)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.payload.observation_timing = ObservationTiming()
    target = Pointing(
        config=config,
        obsid=700,
        ra=105,
        dec=10,
        sequence=ObservationSequence(
            steps=[
                SequenceStep(name="science A", duration_seconds=300),
                SequenceStep(name="calibration", duration_seconds=1800, science=False),
                SequenceStep(name="science B", duration_seconds=300),
            ]
        ),
        visit_windows=[
            VisitWindow(start=T0 + HOUR, deadline=T0 + HOUR + 5 * MIN),
            VisitWindow(start=T0 + 1.5 * HOUR, deadline=T0 + 2.5 * HOUR),
        ],
    )
    return config, target


@pytest.mark.parametrize(
    "planner",
    ["dispatch", "priority", "local", "cp_sat", "local_decode", "cp_sat_decode"],
)
def test_long_calibration_does_not_skip_the_next_admission_window(planner):
    if planner.startswith("cp_sat"):
        pytest.importorskip("ortools")
    config, target = _case()
    args = (config, [target], BEGIN, BEGIN + timedelta(hours=3))
    if planner == "dispatch":
        run = create_ditl(*args)
        assert run.calc()
        assert not run.validate_plan_matches_execution()
        plan = run.plan
    elif planner == "priority":
        plan = PriorityScheduler(*args, include_passes=False).schedule()
    elif planner.startswith("local"):
        scheduler = LocalSearchScheduler(
            *args, include_passes=False, max_iterations=30, time_limit=60, seed=2029
        )
        if planner.endswith("_decode"):
            decoded = scheduler._start_from_priority_plan()
            plan = [e for b in decoded.final.timeline for e in b.steps or [b.entry]]
        else:
            plan = scheduler.schedule()
    else:
        scheduler = CpSatScheduler(
            *args,
            include_passes=False,
            solver_deterministic_time=1,
            solver_time_limit=60,
            workers=1,
            seed=2029,
        )
        if planner.endswith("_decode"):
            decoded = scheduler._start_from_priority_plan()
            plan = [e for b in decoded.final.timeline for e in b.steps or [b.entry]]
        else:
            plan = scheduler.schedule()
    entries = [e for e in plan if e.obsid == 700]
    assert [e.visit for e in entries] == [0, 0, 0, 1, 1, 1]
    assert [e.step for e in entries] == ["science A", "calibration", "science B"] * 2
    assert all(e.exposure == (1800 if e.calibration else 300) for e in entries)
    assert float(entries[2].end) > target.visit_windows[1].start
    assert float(entries[3].collection_begin) <= target.visit_windows[1].deadline


@pytest.mark.parametrize("planner", ["priority", "local", "cp_sat"])
def test_committed_sequence_keeps_its_actual_window_across_a_replan(planner):
    if planner == "cp_sat":
        pytest.importorskip("ortools")
    config, target = _case()
    end = BEGIN + timedelta(hours=3)
    first = PriorityScheduler(
        config, [target], BEGIN, end, include_passes=False
    ).schedule()
    locked = [e for e in first if e.obsid == 700 and e.visit == 0]
    assert len(locked) == 3
    # Replan during calibration, after only the first science phase was flown.
    target.record_collection(T0 + 65 * MIN, 300, visit=0)
    args = (config, [target], BEGIN + timedelta(minutes=80), end)
    options = dict(include_passes=False, locked=locked)
    if planner == "priority":
        scheduler = PriorityScheduler(*args, **options)
    elif planner == "local":
        scheduler = LocalSearchScheduler(
            *args, **options, max_iterations=30, time_limit=60, seed=2029
        )
    else:
        scheduler = CpSatScheduler(
            *args,
            **options,
            solver_deterministic_time=1,
            solver_time_limit=60,
            workers=1,
            seed=2029,
        )
    plan = scheduler.schedule()
    later = [e for e in plan if e.obsid == 700 and e.visit == 1]
    assert [e.step for e in later] == ["science A", "calibration", "science B"]
    assert [e.exposure for e in later] == [300, 1800, 300]
    assert target.visit_windows[1].start <= float(later[0].collection_begin)
    assert float(later[0].collection_begin) <= target.visit_windows[1].deadline
