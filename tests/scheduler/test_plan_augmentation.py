"""Unused science can fit before a hold when its return slew is reserved."""

from datetime import timedelta
from types import SimpleNamespace

import pytest

from conops import DITL
from conops.config import (
    ObservationCategories,
    ObservationCategory,
    ObservationGroup,
    ObservationGroups,
    ObservationTiming,
    ScienceData,
    TargetConfig,
)
from conops.schedulers import CpSatScheduler, LocalSearchScheduler
from conops.schedulers.priority_scheduler import StartState
from conops.targets import Plan, Pointing, ScheduledActivity, VisitWindow

from .planning_scenario import BEGIN, MIN, PATCH, T0, make_config, make_target


def scenario(scheduler_type=LocalSearchScheduler):
    config = make_config(
        2, weights=TargetConfig(urgency_weight=0, completion_deficit_weight=0)
    )
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.battery.planning_level = None
    config.payload.observation_timing = ObservationTiming()
    config.spacecraft_bus.attitude_control.settle_time = 0
    pose = (*PATCH[0], 0.0)
    target = make_target(config, 1, 140, 10, minutes=20)
    locked = make_target(config, 2, *PATCH[0], minutes=10)
    locked.begin, locked.end = T0 + 65 * MIN, T0 + 75 * MIN
    locked.slewtime = 0
    locked.set_collection_window(ObservationTiming())
    activity = ScheduledActivity(
        name="Wheel unload",
        begin=T0 + 60 * MIN,
        duration_seconds=5 * MIN,
        fixed=True,
        desaturation=True,
        obsid=998001,
    )
    options = dict(
        time_limit=0,
        max_iterations=0,
        include_passes=False,
        start_state=StartState(time=T0, attitude=pose),
        locked=[locked],
        activities=[activity],
    )
    if scheduler_type is CpSatScheduler:
        options.update(solver_time_limit=1.0, solver_deterministic_time=0.01)
    scheduler = scheduler_type(
        config, [target, locked], BEGIN, BEGIN + timedelta(minutes=75), **options
    )
    return config, scheduler, locked, activity, pose


@pytest.mark.parametrize("scheduler_type", [LocalSearchScheduler, CpSatScheduler])
def test_unused_science_reserves_a_return_before_inherited_hold(scheduler_type):
    _, scheduler, locked, activity, _ = scenario(scheduler_type)
    plan = scheduler.schedule()
    science = [e for e in plan if e.obsid == 1]
    assert sum(e.exposure for e in science) == 20 * MIN
    hold = next(e for e in plan if e.obsid == activity.obsid)
    following = next(e for e in plan if e.obsid == locked.obsid)
    assert science[0].end <= hold.begin < activity.begin
    assert hold.slewtime > 0
    assert hold.activity_hold_begin == activity.begin
    assert hold.activity_hold_end == activity.end
    assert (following.collection_begin, following.collection_end) == (
        locked.collection_begin,
        locked.collection_end,
    )


def test_augmented_return_and_hold_execute_without_mode_mismatches():
    config, scheduler, _, activity, pose = scenario()
    plan = scheduler.schedule()
    assert any(e.obsid == 1 for e in plan)
    run = DITL(
        config=config,
        ephem=config.constraint.ephem,
        plan=Plan.model_validate_json(plan.model_dump_json()),
        begin=BEGIN,
        end=BEGIN + timedelta(minutes=75),
    )
    run.step_size = 60
    run.acs.ra, run.acs.dec, run.acs.roll = pose
    run.acs._hold_idle_attitude(*pose, T0)
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
    hold = next(e for e in run.plan if e.obsid == activity.obsid)
    assert (
        hold.activity_hold_end - hold.activity_hold_begin == activity.duration_seconds
    )


def test_augmentation_respects_data_reserved_for_future_science():
    config, scheduler, locked, _, _ = scenario()
    config.recorder.capacity_gb = 20
    config.recorder.planning_limit = 0.9
    config.recorder_views(refresh=True)
    scheduler.targets[0].science_data = {
        "Default Instrument": ScienceData(rate_gbps=0.01)
    }
    locked.science_data = {"Default Instrument": ScienceData(rate_gbps=0.02)}
    plan = scheduler.schedule()
    assert not any(e.obsid == 1 for e in plan)
    assert sum(e.exposure for e in plan if e.obsid == 2) == 10 * MIN


@pytest.mark.parametrize("budget", ["category", "group"])
def test_augmentation_counts_future_science_against_program_budgets(budget):
    config, scheduler, _, _, _ = scenario()
    if budget == "category":
        config.observation_categories = ObservationCategories(
            categories=[
                ObservationCategory(
                    name="Capped",
                    obsid_min=1,
                    obsid_max=3,
                    time_budget_seconds=10 * MIN,
                )
            ]
        )
    else:
        config.observation_groups = ObservationGroups(
            groups=[
                ObservationGroup(
                    name="Capped", members=[1, 2], time_budget_seconds=10 * MIN
                )
            ]
        )
    plan = scheduler.schedule()
    assert not any(e.obsid == 1 for e in plan)
    assert sum(e.exposure for e in plan if e.obsid == 2) == 10 * MIN


def test_rejected_momentum_trial_does_not_change_the_incumbent_or_log():
    _, scheduler, _, _, _ = scenario()
    plan = scheduler._plan_within_momentum(scheduler._plan_once)
    before = plan.model_dump_json()
    events = list(scheduler.log.events)
    scheduler._momentum = SimpleNamespace(
        capacity_n_m_s=1.0,
        first_crossing=lambda timeline, *args, **kwargs: (
            T0 if any(b.entry.obsid == 1 for b in timeline) else None
        ),
    )
    assert scheduler._augment_selected_plan(plan) is plan
    assert plan.model_dump_json() == before
    assert scheduler.log.events == events


def test_lower_objective_trial_is_rejected(monkeypatch):
    _, scheduler, _, _, _ = scenario()
    plan = scheduler._plan_within_momentum(scheduler._plan_once)
    before = plan.model_dump_json()
    monkeypatch.setattr(
        scheduler,
        "_timeline_score",
        lambda blocks: -1.0 if any(b.entry.obsid == 1 for b in blocks) else 0.0,
    )
    assert scheduler._augment_selected_plan(plan) is plan
    assert plan.model_dump_json() == before


def test_final_settling_cannot_change_a_selected_hold(monkeypatch):
    _, scheduler, _, activity, _ = scenario()
    plan = scheduler._plan_within_momentum(scheduler._plan_once)
    before = plan.model_dump_json()
    original = scheduler._finish

    def finish(self, timeline):
        result = original.__func__(self, timeline)
        hold = next(b for b in self.timeline if b.entry.obsid == activity.obsid)
        hold.entry.activity_hold_end += MIN
        return result

    monkeypatch.setattr(LocalSearchScheduler, "_finish", finish)
    assert scheduler._augment_selected_plan(plan) is plan
    assert plan.model_dump_json() == before


def test_a_reserved_future_visit_does_not_hide_an_earlier_unused_window():
    config, scheduler, locked, _, _ = scenario()
    target = Pointing(
        config=config,
        obsid=locked.obsid,
        ra=locked.ra,
        dec=locked.dec,
        merit=100,
        ss_min=10 * MIN,
        ss_max=10 * MIN,
        visit_windows=[
            VisitWindow(
                start=T0 + 20 * MIN,
                deadline=T0 + 40 * MIN,
                ra=140,
                dec=10,
            ),
            VisitWindow(start=T0 + 65 * MIN, deadline=T0 + 65 * MIN),
        ],
    )
    locked.visit = 1
    scheduler.targets = [target]
    plan = scheduler.schedule()
    visits = [e for e in plan if e.obsid == target.obsid]
    assert [e.visit for e in visits] == [0, 1]
    assert all(e.exposure == 10 * MIN for e in visits)
    assert visits[1].collection_begin == locked.collection_begin
