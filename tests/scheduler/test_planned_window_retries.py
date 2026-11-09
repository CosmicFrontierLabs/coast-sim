"""A commanded ToO cut leaves a whole windowed visit to retry."""

from datetime import timedelta

import pytest

from conops.config import ObservationTiming
from conops.ditl import RollingHorizonDITL
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.targets import PlanEntry, Pointing, VisitWindow

from .planning_scenario import BEGIN, MIN, T0, make_config


def _scheduler(kind):
    if kind == "priority":
        return PriorityScheduler, {}
    if kind == "local":
        return LocalSearchScheduler, dict(max_iterations=10, time_limit=30)
    pytest.importorskip("ortools")
    return CpSatScheduler, dict(
        workers=1, seed=2029, solver_deterministic_time=1, solver_time_limit=30
    )


def _case(kind, lead_minutes, *, possible=True, cleanup_seconds=0, replan_minutes=240):
    config = make_config(4)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.payload.observation_timing = ObservationTiming(
        cleanup_seconds=cleanup_seconds
    )
    target = Pointing(
        config=config,
        obsid=700,
        ra=105,
        dec=10,
        merit=10,
        ss_min=600,
        ss_max=600,
        visit_windows=[VisitWindow(start=T0 + 60 * MIN, deadline=T0 + 90 * MIN)],
    )
    scheduler, options = _scheduler(kind)
    run = RollingHorizonDITL(
        config,
        [target],
        begin=BEGIN,
        end=BEGIN + timedelta(hours=4),
        horizon=timedelta(hours=4),
        replan_interval=timedelta(minutes=replan_minutes),
        commit_lead_time=timedelta(minutes=lead_minutes),
        include_passes=False,
        scheduler=scheduler,
        scheduler_options=options,
    )
    run.step_size = 60
    trigger = T0 + (65 - lead_minutes) * MIN
    run.submit_too(
        obsid=701,
        ra=107,
        dec=10,
        merit=1000,
        exptime=120,
        name="Urgent",
        submit_time=trigger,
        deadline=T0 + 67 * MIN if possible else trigger + 5,
    )
    return run, target


@pytest.mark.parametrize("kind", ["priority", "local", "cp_sat"])
@pytest.mark.parametrize("lead_minutes", [0, 2])
def test_actual_cut_is_recorded_and_retried_whole_in_the_same_plan(kind, lead_minutes):
    run, target = _case(kind, lead_minutes)
    assert run.calc()
    assert not run.validate_plan_matches_execution()
    entries = [e for e in run.plan if e.obsid == 700]
    assert [e.exposure for e in entries] == [300, 600]
    assert [e.visit for e in entries] == [0, 0]
    assert target.collected_seconds == 900
    assert target.visits_done == {0}
    assert target.interruptions == (T0 + 65 * MIN,)
    assert entries[0].interruptions == target.interruptions
    assert entries[1].interruptions == ()
    assert any(r.interrupted_obsid == 700 for r in run.replans)


@pytest.mark.parametrize("kind", ["priority", "local", "cp_sat"])
def test_rejected_cut_does_not_change_actual_request_history(kind):
    run, target = _case(kind, 0, possible=False)
    assert run.calc()
    assert not run.validate_plan_matches_execution()
    entries = [e for e in run.plan if e.obsid == 700]
    assert [e.exposure for e in entries] == [600]
    assert target.collected_seconds == 600
    assert target.interruptions == ()
    assert entries[0].interruptions == ()
    assert not any(r.interrupted_obsid == 700 for r in run.replans)


@pytest.mark.parametrize("kind", ["priority", "local", "cp_sat"])
def test_explicit_no_completed_windows_overrides_elapsed_time_inference(kind):
    config = make_config(4)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    target = Pointing(
        config=config,
        obsid=700,
        ra=105,
        dec=10,
        ss_min=600,
        ss_max=600,
        visit_windows=[VisitWindow(start=T0 + 30 * MIN, deadline=T0 + 180 * MIN)],
    )
    scheduler, options = _scheduler(kind)
    planner = scheduler(
        config,
        [target],
        BEGIN + timedelta(minutes=80),
        BEGIN + timedelta(hours=4),
        include_passes=False,
        reserved_seconds={700: 300},
        reserved_visits={700: T0 + 85 * MIN},
        reserved_visit_windows={},
        **options,
    )
    assert [e.visit for e in planner.schedule() if e.obsid == 700] == [0]


@pytest.mark.parametrize("kind", ["priority", "local", "cp_sat"])
def test_cut_credits_actual_science_then_finishes_cleanup_before_retry(kind):
    run, target = _case(kind, 2, cleanup_seconds=60)
    assert run.calc()
    assert not run.validate_plan_matches_execution()
    entries = [e for e in run.plan if e.obsid == 700]
    assert [e.exposure for e in entries] == [240, 600]
    assert float(entries[0].end) - float(entries[0].collection_end) == 60
    assert target.collected_seconds == 840
    assert target.interruptions == (T0 + 64 * MIN,)


@pytest.mark.parametrize("kind", ["priority", "local", "cp_sat"])
def test_another_replan_before_a_pending_cut_keeps_the_whole_retry(kind):
    run, target = _case(kind, 4, replan_minutes=62)
    assert run.calc()
    assert not run.validate_plan_matches_execution()
    entries = [e for e in run.plan if e.obsid == 700]
    assert [e.exposure for e in entries] == [300, 600]
    assert target.interruptions == (T0 + 65 * MIN,)
    assert any(r.utime == T0 + 62 * MIN for r in run.replans)


def test_projecting_a_cut_copies_history_and_shares_read_only_native_inputs():
    run, target = _case("priority", 2)
    target.record_collection(T0 + 63 * MIN, 180, visit=0)
    cut = PlanEntry(
        obsid=700,
        visit=0,
        begin=T0 + 57 * MIN,
        end=T0 + 65 * MIN,
        collection_begin=T0 + 60 * MIN,
        collection_end=T0 + 65 * MIN,
    )
    (projected,) = run._planning_targets([cut], cut_entry=cut)
    assert projected is not target
    assert projected.visits_done == set()
    assert projected.interruptions == (T0 + 65 * MIN,)
    assert target.visits_done == {0}
    assert target.interruptions == ()
    assert projected.collected_seconds == target.collected_seconds == 180
    assert projected.config is target.config
    assert projected.constraint is target.constraint
    assert projected.ephem is target.ephem
    assert projected.acs_config is target.acs_config


@pytest.mark.parametrize("kind", ["priority", "local", "cp_sat"])
def test_an_ordinary_request_records_its_actual_interruption_once(kind):
    run, target = _case(kind, 2)
    target.visit_windows = []
    target.ss_min = 60
    target.earliest_start = T0 + 60 * MIN
    assert run.calc()
    assert not run.validate_plan_matches_execution()
    assert target.collected_seconds == 600
    assert target.interruptions == (T0 + 65 * MIN,)
