"""A future committed visit must coexist with earlier collected history."""

from datetime import timedelta

import pytest

from conops.ditl import RollingHorizonDITL
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.targets import (
    ObservationSequence,
    PlanEntry,
    Pointing,
    SequenceStep,
    VisitWindow,
)

from .planning_scenario import BEGIN, HOUR, MIN, T0, make_config


def _planner(kind, config, target, **options):
    args = (config, [target], BEGIN + timedelta(minutes=80), BEGIN + timedelta(hours=4))
    options = dict(include_passes=False, **options)
    if kind == "priority":
        return PriorityScheduler(*args, **options)
    if kind == "local":
        return LocalSearchScheduler(*args, max_iterations=10, time_limit=30, **options)
    pytest.importorskip("ortools")
    return CpSatScheduler(
        *args,
        workers=1,
        seed=2029,
        solver_deterministic_time=1,
        solver_time_limit=30,
        **options,
    )


def _case():
    config = make_config(4)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    target = Pointing(
        config=config,
        obsid=700,
        ra=105,
        dec=10,
        ss_min=10 * MIN,
        ss_max=10 * MIN,
        visit_windows=[
            VisitWindow(start=T0 + HOUR, deadline=T0 + 70 * MIN),
            VisitWindow(start=T0 + 90 * MIN, deadline=T0 + 150 * MIN),
            VisitWindow(start=T0 + 3 * HOUR, deadline=T0 + 210 * MIN),
        ],
    )
    target.record_collection(T0 + 70 * MIN, 10 * MIN, visit=0)
    return config, target


@pytest.mark.parametrize("kind", ["priority", "local", "cp_sat"])
def test_committed_later_window_is_not_placed_again_after_completed_history(kind):
    config, target = _case()
    # Window 0 is already collected. Window 1 is commanded outside the new
    # horizon but still collecting after its start. Only window 2 is available.
    scheduler = _planner(
        kind,
        config,
        target,
        reserved_seconds={700: 10 * MIN},
        reserved_visits={700: T0 + 100 * MIN},
    )
    plan = scheduler.schedule()
    assert [e.visit for e in plan if e.obsid == 700] == [2]
    assert target.visits_done == {0}


@pytest.mark.parametrize("kind", ["priority", "local", "cp_sat"])
def test_recorded_committed_window_survives_sequence_overhead(kind):
    config, target = _case()
    target.sequence = ObservationSequence(
        steps=[
            SequenceStep(name="science A", duration_seconds=5 * MIN),
            SequenceStep(name="calibration", duration_seconds=50 * MIN, science=False),
            SequenceStep(name="science B", duration_seconds=5 * MIN),
        ]
    )
    # The committed window-1 visit ends at 190 minutes. Subtracting its ten
    # science minutes incorrectly identifies window 2; use its recorded index.
    scheduler = _planner(
        kind,
        config,
        target,
        reserved_seconds={700: 10 * MIN},
        reserved_visits={700: T0 + 190 * MIN},
        reserved_visit_windows={700: 1},
    )
    plan = scheduler.schedule()
    assert [e.visit for e in plan if e.obsid == 700] == [2, 2, 2]


@pytest.mark.parametrize("kind", ["priority", "local", "cp_sat"])
def test_rolling_replan_keeps_one_whole_visit_per_window(kind):
    config, target = _case()
    target.reset()
    scheduler = _planner(kind, config, target)
    options = {}
    if kind == "local":
        options = dict(max_iterations=10, time_limit=30)
    elif kind == "cp_sat":
        options = dict(
            workers=1, seed=2029, solver_deterministic_time=1, solver_time_limit=30
        )
    run = RollingHorizonDITL(
        config,
        [target],
        begin=BEGIN,
        end=BEGIN + timedelta(hours=4),
        horizon=timedelta(hours=3),
        replan_interval=timedelta(minutes=80),
        commit_lead_time=timedelta(minutes=30),
        scheduler=type(scheduler),
        scheduler_options=options,
        include_passes=False,
    )
    run.step_size = 60
    assert run.calc()
    assert not run.validate_plan_matches_execution()
    entries = [e for e in run.plan if e.obsid == 700 and e.exposure > 0]
    assert [e.visit for e in entries] == [0, 1, 2]
    assert sum(h.collection_seconds for h in run.telemetry.housekeeping) == 30 * MIN


def test_rolling_reservation_uses_only_future_science_and_recorded_indices():
    config, target = _case()
    run = RollingHorizonDITL(
        config, [target], begin=BEGIN, end=BEGIN + timedelta(hours=4)
    )

    def entry(visit, stop, *, calibration=False):
        return PlanEntry(
            obsid=700,
            visit=visit,
            begin=T0,
            end=stop,
            collection_begin=T0,
            collection_end=stop,
            calibration=calibration,
        )

    now = T0 + 80 * MIN
    entries = [
        entry(0, now),
        entry(1, now + MIN),
        entry(2, now + 2 * MIN, calibration=True),
        entry(None, now + 3 * MIN),
    ]
    assert run._reserved_visit_windows(entries, now) == {700: 1}
    assert run._reserved_visit_windows(entries, now + MIN) == {}


@pytest.mark.parametrize("kind", ["priority", "local", "cp_sat"])
def test_older_reservation_does_not_erase_newer_completed_history(kind):
    config, target = _case()
    target.record_collection(T0 + 100 * MIN, 10 * MIN, visit=1)
    scheduler = _planner(
        kind,
        config,
        target,
        reserved_visits={700: T0 + 70 * MIN},
        reserved_visit_windows={700: 0},
    )
    assert [e.visit for e in scheduler.schedule() if e.obsid == 700] == [2]


@pytest.mark.parametrize("kind", ["priority", "local", "cp_sat"])
def test_a_future_cut_attempt_does_not_reserve_its_window_as_complete(kind):
    config, target = _case()
    run = RollingHorizonDITL(
        config, [target], begin=BEGIN, end=BEGIN + timedelta(hours=4)
    )
    cut = PlanEntry(
        obsid=700,
        visit=1,
        begin=T0 + 99 * MIN,
        end=T0 + 106 * MIN,
        collection_begin=T0 + 100 * MIN,
        collection_end=T0 + 105 * MIN,
    )
    windows = run._reserved_visit_windows([cut], T0 + 80 * MIN)
    assert windows == {}
    scheduler = _planner(
        kind,
        config,
        target,
        reserved_seconds={700: 5 * MIN},
        reserved_visits={700: T0 + 105 * MIN},
        reserved_visit_windows=windows,
    )
    # Explicitly no whole window is reserved: the pending cut attempt will
    # collect only five of the ten required minutes. A whole retry is allowed.
    assert [e.visit for e in scheduler.schedule() if e.obsid == 700] == [1, 2]


@pytest.mark.parametrize("last_seconds", [0, 150, 300])
def test_sequence_reservation_counts_its_science_steps_without_calibration(
    last_seconds,
):
    config, target = _case()
    target.sequence = ObservationSequence(
        steps=[
            SequenceStep(name="science A", duration_seconds=300),
            SequenceStep(name="calibration", duration_seconds=50 * MIN, science=False),
            SequenceStep(name="science B", duration_seconds=300),
        ]
    )
    run = RollingHorizonDITL(
        config, [target], begin=BEGIN, end=BEGIN + timedelta(hours=4)
    )

    def entry(name, start, duration, *, calibration=False):
        return PlanEntry(
            obsid=700,
            visit=1,
            step=name,
            begin=start,
            end=start + duration,
            collection_begin=start,
            collection_end=start + duration,
            calibration=calibration,
        )

    entries = [
        entry("science A", T0 + 90 * MIN, 300),
        entry("calibration", T0 + 95 * MIN, 50 * MIN, calibration=True),
        entry("science B", T0 + 145 * MIN, last_seconds),
    ]
    expected = {700: 1} if last_seconds == 300 else {}
    assert run._reserved_visit_windows(entries, T0 + 120 * MIN) == expected
