"""Momentum repair compares feasible incumbents without sharing hold reservations."""

import pytest

from conops.benchmark.contenders import _dispatch, too_targets
from conops.benchmark.scenarios import program
from conops.common import ObsType
from conops.targets import Plan, assign_obsids

from .planning_scenario import collected_seconds, planner, replay


@pytest.mark.parametrize("winner", ["dispatch", "momentum"])
def test_momentum_contenders_keep_the_winners_activity_reservations(winner):
    scenario = program("examples/example.tle", hours=6)
    dispatched = _dispatch(scenario)
    config = scenario.make_config()
    targets = [*scenario.make_targets(config), *too_targets(config, scenario.toos)]
    scheduler_class, options = planner("local_search")
    settings = dict(
        config=config,
        targets=targets,
        begin=scenario.begin,
        end=scenario.end,
        activities=scenario.activities,
        **options,
    )
    # This contender earns no science but settles holds earlier than the
    # feasible winner. Trying it must not change the selected reservations.
    losing = Plan(
        entries=[
            activity.plan_entry(config, (105.0, 10.0, 0.0))
            for activity in assign_obsids(scenario.activities)
        ]
    )
    builder = scheduler_class(
        **settings, start_plan=dispatched.plan if winner == "dispatch" else losing
    )
    builder._momentum_seed = dispatched.plan if winner == "momentum" else losing
    start = builder._start_from_priority_plan()
    assert builder.start_score > builder.initial_score
    assert builder.seed_score == (builder.start_score if winner == "dispatch" else 0)
    held = {
        entry.obsid: float(entry.begin) + float(entry.slewtime or 0)
        for entry in dispatched.plan
        if entry.obstype == ObsType.ACTIVITY
    }
    assert held
    for block in builder._fixed:
        if block.is_activity:
            assert block.ready == held[block.entry.obsid]
    replayed = builder._decode(list(start.sequence))
    assert replayed.final.score == start.final.score
    plan = builder._finish_best(replayed)
    run = replay(config, plan, scenario.begin, scenario.end)
    assert {
        entry.obsid for entry in run.plan if entry.obstype == ObsType.ACTIVITY
    } == set(held)


def test_momentum_incumbent_is_repaired_against_new_mandatory_blocks():
    from datetime import timedelta

    from conops.targets import ScheduledActivity

    from .planning_scenario import BEGIN, MIN, T0, make_config, make_target

    config = make_config(3, open_sky=True)
    target = make_target(
        config,
        100,
        105,
        10,
        minutes=80,
        snapshot=80,
        ss_min=80,
        deadline=T0 + 20 * MIN,
    )
    scheduler_class, options = planner("local_search")
    settings = dict(
        config=config,
        targets=[target],
        begin=BEGIN,
        end=BEGIN + timedelta(hours=3),
        include_passes=False,
        **options,
    )
    incumbent = scheduler_class(**settings).schedule()
    assert any(entry.obsid == target.obsid for entry in incumbent)
    activity = ScheduledActivity(
        name="Mandatory hold",
        begin=T0 + 60 * MIN,
        duration_seconds=30 * MIN,
        obsid=998000,
        fixed=True,
    )
    builder = scheduler_class(**settings, activities=[activity])
    builder._momentum_seed = incumbent
    start = builder._start_from_priority_plan()
    assert start.final.score == 0
    assert not any(snapshot.obsid == target.obsid for snapshot in start.sequence)
    plan = builder._finish_best(start)
    [hold] = plan.entries
    assert hold.obsid == activity.obsid
    assert (float(hold.begin) + float(hold.slewtime or 0)) == activity.begin
    assert float(hold.end) == activity.end
    assert collected_seconds(replay(config, plan, BEGIN, settings["end"])) == 0


@pytest.mark.parametrize("kind", ["local_search", "cp_sat"])
def test_losing_seeds_preserve_priority_activity_reservations(kind):
    from datetime import timedelta

    from conops.targets import Plan, ScheduledActivity

    from .planning_scenario import BEGIN, MIN, T0, make_config, make_target

    config = make_config(3, open_sky=True)
    activity = ScheduledActivity(
        name="Calibration",
        begin=T0 + 60 * MIN,
        duration_seconds=30 * MIN,
        slide_seconds=60 * MIN,
        obsid=998000,
    )
    target = make_target(
        config,
        100,
        105,
        10,
        minutes=80,
        snapshot=80,
        ss_min=80,
        deadline=T0 + 20 * MIN,
    )
    scheduler_class, options = planner(kind)
    settings = dict(
        config=config,
        targets=[target],
        begin=BEGIN,
        end=BEGIN + timedelta(hours=3),
        activities=[activity],
        include_passes=False,
        **options,
    )
    unseeded = scheduler_class(**settings)
    baseline = unseeded._start_from_priority_plan()
    assert baseline.final.score > 0
    seed = Plan(entries=[activity.plan_entry(config, (105, 10, 0))])
    seeded = scheduler_class(**settings, start_plan=seed)
    seeded._momentum_seed = seed
    start = seeded._start_from_priority_plan()
    assert seeded.seed_score == 0
    assert seeded.start_score == unseeded.start_score
    assert start.final.score == baseline.final.score
    [hold] = [b for b in start.final.timeline if b.is_activity]
    assert hold.ready > activity.begin
    # The chosen fixed reservations must be the ones used when a
    # search move or CP-SAT chunk redecodes from the starting state.
    redecoded = seeded._decode(list(start.sequence))
    assert redecoded.final.score == start.final.score
    run = replay(config, seeded.schedule(), BEGIN, settings["end"])
    assert collected_seconds(run) == 80 * MIN


@pytest.mark.parametrize("kind", ["repair", "priority", "local_search", "cp_sat"])
def test_repaired_activity_data_is_reserved_before_the_contact(monkeypatch, kind):
    from datetime import timedelta
    from unittest.mock import Mock

    from conops.config import HousekeepingData, OnboardRecorder, ScienceData
    from conops.schedulers.recorder import RecorderLedgers
    from conops.simulation import ContactSpan, StationCoverage
    from conops.targets import ScheduledActivity

    from .planning_scenario import BEGIN, MIN, T0, make_config, make_target

    config = make_config(3, open_sky=True)
    config.recorder = OnboardRecorder(capacity_gb=10, planning_limit=1)
    config.spacecraft_bus.data_generation = HousekeepingData(rate_gbps=0)
    activity = ScheduledActivity(
        name="Calibration frames",
        begin=T0 + 50 * MIN,
        duration_seconds=MIN,
        slide_seconds=40 * MIN,
        data_gb=4,
        obsid=998000,
    )
    target = make_target(config, 1, 105, 10, minutes=5, snapshot=5, ss_min=5)
    target.science_data = {
        "Default Instrument": ScienceData(rate_gbps=0, per_observation_gb=9)
    }
    later = make_target(
        config,
        2,
        110,
        10,
        minutes=5,
        snapshot=5,
        ss_min=5,
        merit=40,
        earliest_start=T0 + 100 * MIN,
    )
    later.science_data = {
        "Default Instrument": ScienceData(rate_gbps=0, per_observation_gb=8)
    }
    coverage = Mock(spec=StationCoverage)
    coverage.step_size = 60
    coverage.repoints = []
    coverage.scheduled_passes = []

    def spans(lo, hi, recorder=None):
        begin = max(lo, T0 + 60 * MIN)
        end = min(hi, T0 + 70 * MIN)
        return [ContactSpan(begin, end, "TEST", 100)] if begin < end else []

    coverage.rate_spans.side_effect = spans
    coverage.spans.side_effect = spans
    scheduler_class, options = planner("local_search" if kind == "repair" else kind)
    builder = scheduler_class(
        config,
        [target, later],
        BEGIN,
        BEGIN + timedelta(hours=3),
        activities=[activity],
        include_passes=False,
        coverage=coverage,
        **options,
    )
    monkeypatch.setattr(
        builder,
        "_recorder_ledger",
        lambda: RecorderLedgers(
            config,
            T0,
            T0 + 180 * MIN,
            [],
            step_size=60,
            coverage=coverage,
        ),
    )
    if kind != "repair":
        plan = builder.schedule()
        physical = builder._ledger.initial_volumes(plan.entries)
        assert builder._ledger.fits_all(physical, [])
        assert {int(e.obsid) for e in plan if e.obstype != ObsType.ACTIVITY} == {1, 2}
        assert (
            sum(
                e.collection_seconds_between(e.begin, e.end)
                for e in plan
                if e.obstype != ObsType.ACTIVITY
            )
            == 10 * MIN
        )
        return
    baseline = builder._start_from_priority_plan()
    assert baseline.sequence
    # Nine science Gb precede the link; the priority reservation's four
    # calibration Gb follow it. They fit separately in a ten-Gb recorder.
    early = Plan(entries=[activity.plan_entry(config, (105, 10, 0))])
    builder._settle_from(early)
    from conops.schedulers.local_search_scheduler import _Snapshot

    decoded = builder._decode([_Snapshot(obsid=1, seconds=5 * MIN, release=T0)])
    # Moving the calibration before that link means 9 + 4 Gb coexist.
    # The science must move after the contact or be rejected, never overflow.
    assert builder._ledger.fits_all(decoded.final.volumes, [])
    physical = builder._ledger.initial_volumes(
        [
            entry
            for block in decoded.final.timeline
            for entry in block.steps or [block.entry]
        ]
    )
    assert builder._ledger.fits_all(physical, [])
