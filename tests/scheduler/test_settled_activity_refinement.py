"""Settling a sliding hold can release time for a previously missed visit."""

from datetime import timedelta

import pytest

from conops import DITL
from conops.config import ObservationTiming, TargetConfig
from conops.schedulers import CpSatScheduler, LocalSearchScheduler
from conops.schedulers.priority_scheduler import StartState
from conops.targets import Plan, Pointing, ScheduledActivity, VisitWindow

from .planning_scenario import BEGIN, MIN, PATCH, T0, make_config, make_target


@pytest.mark.parametrize("scheduler_type", [LocalSearchScheduler, CpSatScheduler])
def test_settled_calibration_releases_time_for_a_missed_visit(
    scheduler_type, monkeypatch
):
    config = make_config(
        2, weights=TargetConfig(urgency_weight=0, completion_deficit_weight=0)
    )
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.battery.planning_level = None
    config.payload.observation_timing = ObservationTiming()
    config.spacecraft_bus.attitude_control.settle_time = 0
    filler = make_target(config, 1, *PATCH[0], minutes=60, snapshot=20, merit=50)
    visit = Pointing(
        config=config,
        obsid=2,
        ra=140,
        dec=10,
        merit=200,
        ss_min=30 * MIN,
        ss_max=30 * MIN,
        visit_windows=[VisitWindow(start=T0 + 35 * MIN, deadline=T0 + 40 * MIN)],
    )
    calibration = ScheduledActivity(
        name="Calibration",
        begin=T0 + 20 * MIN,
        duration_seconds=5 * MIN,
        slide_seconds=15 * MIN,
        obsid=998001,
    )
    dump = ScheduledActivity(
        name="Wheel unload",
        begin=T0 + 70 * MIN,
        duration_seconds=5 * MIN,
        fixed=True,
        desaturation=True,
        obsid=998002,
    )
    options = dict(
        time_limit=0,
        max_iterations=0,
        include_passes=False,
        start_state=StartState(time=T0, attitude=(*PATCH[0], 0.0)),
        activities=[calibration, dump],
    )
    if scheduler_type is CpSatScheduler:
        options.update(solver_time_limit=1.0, solver_deterministic_time=0.01)
    scheduler = scheduler_type(
        config, [filler, visit], BEGIN, BEGIN + timedelta(minutes=75), **options
    )
    settled = []
    build_with_momentum = scheduler._plan_within_momentum

    def capture(build):
        original = build_with_momentum(build)
        held = next(e for e in original if e.obsid == calibration.obsid)
        settled.append((held.activity_hold_begin, held.activity_hold_end))
        return original

    monkeypatch.setattr(scheduler, "_plan_within_momentum", capture)
    plan = scheduler.schedule()
    assert sum(e.exposure for e in plan if e.obsid == visit.obsid) == 30 * MIN
    held = next(e for e in plan if e.obsid == calibration.obsid)
    assert (held.activity_hold_begin, held.activity_hold_end) == settled[0]
    assert held.activity_hold_begin < calibration.latest_start
    unloaded = next(e for e in plan if e.obsid == dump.obsid)
    assert unloaded.activity_hold_begin == dump.begin
    assert unloaded.activity_hold_end == dump.end
    run = DITL(
        config=config,
        ephem=config.constraint.ephem,
        plan=Plan.model_validate_json(plan.model_dump_json()),
        begin=BEGIN,
        end=BEGIN + timedelta(minutes=75),
    )
    run.step_size = 60
    pose = (*PATCH[0], 0.0)
    run.acs.ra, run.acs.dec, run.acs.roll = pose
    run.acs._hold_idle_attitude(*pose, T0)
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
