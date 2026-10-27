"""A commanded ToO cut leaves a whole windowed visit to retry."""

from datetime import timedelta

import pytest

from conops.config import ObservationTiming
from conops.ditl import RollingHorizonDITL
from conops.targets import Pointing, VisitWindow

from .planning_scenario import BEGIN, MIN, T0, make_config, planner


def _case(kind, lead_minutes, *, possible=True, cleanup_seconds=0, replan_minutes=240):
    config = make_config(4, open_sky=True)
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
    scheduler, options = planner(kind)
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


@pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
@pytest.mark.parametrize(
    "lead,cleanup,replan,first,cut",
    [
        (0, 0, 240, 300, 65),
        (2, 0, 240, 300, 65),
        (2, 60, 240, 240, 64),  # cleanup finishes before the retry
        (4, 0, 62, 300, 65),  # another replan before the pending cut
    ],
)
def test_actual_cut_is_recorded_and_retried_whole_in_the_same_plan(
    kind, lead, cleanup, replan, first, cut
):
    run, target = _case(kind, lead, cleanup_seconds=cleanup, replan_minutes=replan)
    assert run.calc()
    assert not run.validate_plan_matches_execution()
    entries = [e for e in run.plan if e.obsid == 700]
    assert [e.exposure for e in entries] == [first, 600]
    assert [e.visit for e in entries] == [0, 0]
    assert float(entries[0].end) - float(entries[0].collection_end) == cleanup
    assert target.collected_seconds == first + 600
    assert target.visits_done == {0}
    assert target.interruptions == (T0 + cut * MIN,)
    assert entries[0].interruptions == target.interruptions
    assert entries[1].interruptions == ()
    assert any(r.interrupted_obsid == 700 for r in run.replans)
    assert any(r.utime == T0 + replan * MIN for r in run.replans) == (replan < 240)


@pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
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


@pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
def test_an_ordinary_request_records_its_actual_interruption_once(kind):
    run, target = _case(kind, 2)
    target.visit_windows = []
    target.ss_min = 60
    target.earliest_start = T0 + 60 * MIN
    assert run.calc()
    assert not run.validate_plan_matches_execution()
    assert target.collected_seconds == 600
    assert target.interruptions == (T0 + 65 * MIN,)
