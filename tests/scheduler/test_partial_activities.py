"""Partial runs preserve full activity holds without replaying past effects."""

from datetime import timedelta
from unittest.mock import Mock

import pytest

from conops import DITL, QueueDITL, create_ditl
from conops.common import ObsType
from conops.config import StoredMomentumConfig, ThrusterDumpConfig
from conops.config.scheduler import SchedulerMode
from conops.ditl import RollingHorizonDITL
from conops.schedulers import PriorityScheduler
from conops.targets import Plan, Program, ScheduledActivity, assign_obsids

from .planning_scenario import BEGIN, HOUR, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target


def _science(plan: Plan | list) -> list:
    return [e for e in plan if e.obstype not in (ObsType.GSP, ObsType.ACTIVITY)]


def _activities(plan: Plan | list) -> list:
    return [e for e in plan if e.obstype == ObsType.ACTIVITY]


class TestPartialRun:
    @pytest.mark.parametrize("mode", list(SchedulerMode))
    @pytest.mark.parametrize(
        "begin,duration,slide,fixed,hold_begin,hold_end",
        [
            (1.5, 1, 0, True, 1.5, 2.5),
            (1, 1, 0.75, False, 1.75, 2.75),
            (1, 0.25, 2, False, 2, 2.25),
            (2.5, 1, 0, True, 2.5, 3.5),
            (1.5, 2, 0, True, 1.5, 3.5),
        ],
        ids=[
            "resume-fixed",
            "resume-closed-slide",
            "open-slide",
            "cut-end",
            "cut-both",
        ],
    )
    def test_overlapping_activity_reports_its_full_hold(
        self, tmp_path, mode, begin, duration, slide, fixed, hold_begin, hold_end
    ) -> None:
        config = _config(4)
        config.constraint.sun_constraint = None
        config.constraint.earth_constraint = None
        config.scheduler.mode = mode
        config.scheduler.include_passes = False
        config.scheduler.replanning.replan_interval_seconds = 15 * MIN
        activity = ScheduledActivity(
            name="Partial calibration",
            begin=T0 + begin * HOUR,
            duration_seconds=duration * HOUR,
            slide_seconds=slide * HOUR,
            fixed=fixed,
            data_gb=1.5,
        )
        program = Program(
            begin=T0,
            end=T0 + 4 * HOUR,
            activities=[activity],
            requests=[_target(config, 100, *PATCH[0], minutes=40, snapshot=5)],
        )
        program = Program.load(program.save(tmp_path / "program.json"))
        before = program.model_dump(mode="json")
        ditl = create_ditl(
            config,
            program=program,
            begin=BEGIN + timedelta(hours=2),
            end=BEGIN + timedelta(hours=3),
        )
        ditl._add_activity_data = Mock(wraps=ditl._add_activity_data)
        assert ditl.calc()
        [entry] = _activities(ditl.plan)
        assert (float(entry.begin), float(entry.end)) == (
            T0 + max(2, hold_begin) * HOUR,
            T0 + min(3, hold_end) * HOUR,
        )
        assert (entry.activity_hold_begin, entry.activity_hold_end) == (
            T0 + hold_begin * HOUR,
            T0 + hold_end * HOUR,
        )
        assert entry.activity_partial == (hold_begin < 2 or hold_end > 3)
        assert entry.activity_finishes == (hold_end <= 3)
        assert entry.fixed_data_gb == (0 if hold_begin < 2 else activity.data_gb)
        assert ditl._add_activity_data.call_count == (0 if hold_begin < 2 else 1)
        assert ditl.validate_plan_matches_execution() == []
        assert all(
            float(e.end) <= float(entry.begin) or float(e.begin) >= float(entry.end)
            for e in _science(ditl.plan)
        )
        assert [a.model_dump(mode="json") for a in program.activities] == before[
            "activities"
        ]
        events = [e for e in ditl.log if e.event_type == "ACTIVITY"]
        assert len(events) == 2
        assert ("resumes" in events[0].description) == (hold_begin < 2)
        assert ("before completion" in events[1].description) == (hold_end > 3)
        restored = Plan.load(ditl.plan.save(tmp_path / "plan.json"))
        [loaded] = _activities(restored)
        assert loaded.activity_partial == entry.activity_partial
        assert loaded.activity_hold_begin == entry.activity_hold_begin
        assert loaded.activity_hold_end == entry.activity_hold_end

    @pytest.mark.parametrize("mode", list(SchedulerMode))
    @pytest.mark.parametrize("boundary", ["truncated", "exact-end", "resumed"])
    def test_a_dump_unloads_only_when_its_full_hold_ends(self, mode, boundary) -> None:
        truncated = boundary == "truncated"
        config = _config(4)
        config.constraint.sun_constraint = None
        config.constraint.earth_constraint = None
        config.scheduler.mode = mode
        config.scheduler.include_passes = False
        config.spacecraft_bus.inertia_tensor_body_kg_m2 = (
            (1000, 0, 0),
            (0, 1000, 0),
            (0, 0, 1000),
        )
        config.spacecraft_bus.attitude_control.stored_momentum = StoredMomentumConfig(
            gravity_gradient_enabled=True,
            initial_momentum_body_n_m_s=(0.4, 0, 0),
            capacity_n_m_s=10,
            thruster_dump=ThrusterDumpConfig(duration_seconds=HOUR),
        )
        activity = ScheduledActivity(
            name="Boundary dump",
            begin=T0 + (1.5 if boundary == "resumed" else 2.5) * HOUR,
            duration_seconds=HOUR,
            fixed=True,
            desaturation=True,
        )
        stop = T0 + 3 * HOUR if truncated else activity.end
        ditl = create_ditl(
            config,
            begin=BEGIN + timedelta(hours=2),
            end=BEGIN + timedelta(seconds=stop - T0),
            activities=[activity],
        )
        ditl._unload_stored_momentum = Mock(wraps=ditl._unload_stored_momentum)
        assert ditl.calc()
        [entry] = _activities(ditl.plan)
        assert entry.activity_finishes is not truncated
        assert ditl._unload_stored_momentum.call_count == (0 if truncated else 1)
        if not truncated:
            ditl._unload_stored_momentum.assert_called_once_with(stop, entry.obsid)
        assert (
            ditl._stored_momentum_tracker.stored_momentum_norm_n_m_s
            == pytest.approx(0.4 if truncated else 0)
        )
        assert ditl.validate_plan_matches_execution() == []

        if mode is SchedulerMode.PLANNED:
            scheduler = PriorityScheduler(
                config,
                [],
                BEGIN + timedelta(hours=2),
                BEGIN + timedelta(seconds=stop - T0),
                activities=[activity],
                include_passes=False,
            )
            scheduler.schedule()
            _, norms = scheduler._momentum.profile(scheduler.timeline)
            assert norms[-1] == pytest.approx(0.4 if truncated else 0)
            rolling = RollingHorizonDITL(
                config,
                [],
                begin=BEGIN + timedelta(hours=2),
                end=BEGIN + timedelta(seconds=stop - T0),
            )
            rolling._stored_momentum_tracker.start_from_eci((0.4, 0, 0))
            forecast = rolling._momentum_start(T0 + 2 * HOUR, stop, [entry])
            assert forecast == pytest.approx((0.4, 0, 0) if truncated else (0, 0, 0))

    @pytest.mark.parametrize("legacy", [False, True])
    def test_replaying_a_saved_activity_over_a_shorter_interval(self, tmp_path, legacy):
        config = _config(4)
        config.constraint.sun_constraint = None
        config.constraint.earth_constraint = None
        activity = assign_obsids(
            [
                ScheduledActivity(
                    name="Saved hold",
                    begin=T0 + 1.5 * HOUR,
                    duration_seconds=2 * HOUR,
                    fixed=True,
                    desaturation=True,
                    data_gb=2,
                )
            ]
        )[0]
        scheduler = PriorityScheduler(
            config,
            [],
            BEGIN,
            BEGIN + timedelta(hours=4),
            activities=[activity],
            include_passes=False,
        )
        plan = Plan.load(scheduler.schedule().save(tmp_path / "input.json"))
        [entry] = _activities(plan)
        if legacy:
            entry.activity_hold_begin = None
            entry.activity_hold_end = None
        ditl = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=plan,
            begin=BEGIN + timedelta(hours=2),
            end=BEGIN + timedelta(hours=3),
        )
        ditl.acs.ra, ditl.acs.dec, ditl.acs.roll = entry.ra, entry.dec, entry.roll
        ditl._add_activity_data = Mock(wraps=ditl._add_activity_data)
        ditl._unload_stored_momentum = Mock()
        assert ditl.calc()
        assert (float(entry.begin), float(entry.end)) == (T0 + 2 * HOUR, T0 + 3 * HOUR)
        assert entry.activity_hold_begin == activity.begin
        assert entry.activity_hold_end == activity.end
        assert entry.activity_partial and not entry.activity_finishes
        ditl._add_activity_data.assert_not_called()
        ditl._unload_stored_momentum.assert_not_called()
        assert ditl.validate_plan_matches_execution() == []

    @pytest.mark.parametrize("mode", list(SchedulerMode))
    @pytest.mark.parametrize("reload", [False, True])
    def test_expired_activities_are_not_executed_or_renumbered(
        self, tmp_path, mode, reload
    ) -> None:
        config = _config(4)
        config.scheduler.mode = mode
        start, stop = BEGIN + timedelta(hours=2), BEGIN + timedelta(hours=4)
        live = ScheduledActivity(
            name="In run",
            begin=T0 + 3 * HOUR,
            duration_seconds=10 * MIN,
            data_gb=1.5,
        )
        activities = [
            ScheduledActivity(
                name="After run",
                begin=T0 + 4 * HOUR,
                duration_seconds=10 * MIN,
            ),
            live,
            ScheduledActivity(
                name="Expired",
                begin=T0 + HOUR,
                duration_seconds=10 * MIN,
                data_gb=20,
            ),
            ScheduledActivity(
                name="Ends at begin",
                begin=T0 + 90 * MIN,
                duration_seconds=30 * MIN,
            ),
            ScheduledActivity(
                name="Slide ends at begin",
                begin=T0 + 80 * MIN,
                duration_seconds=10 * MIN,
                slide_seconds=30 * MIN,
            ),
        ]
        program = Program(
            begin=T0,
            end=T0 + 4 * HOUR,
            activities=activities,
        )
        if reload:
            program = Program.load(program.save(tmp_path / "program.json"))
        before = program.model_dump(mode="json")
        [expected] = [a for a in assign_obsids(activities) if a.name == live.name]
        ditl = create_ditl(config, program=program, begin=start, end=stop)
        ditl._add_activity_data = Mock(wraps=ditl._add_activity_data)

        assert ditl.calc()

        [entry] = _activities(ditl.plan)
        assert (entry.name, entry.obsid) == (live.name, expected.obsid)
        assert (float(entry.begin), float(entry.end)) == (live.begin, live.end)
        assert ditl.validate_plan_matches_execution() == []
        ditl._add_activity_data.assert_called_once()
        volume, recorded = ditl._add_activity_data.call_args.args
        assert volume == live.data_gb and recorded.obsid == expected.obsid
        assert program.model_dump(mode="json") == before

    def test_direct_dispatch_ignores_an_expired_activity(self) -> None:
        config = _config(4)
        ditl = QueueDITL(
            config=config,
            begin=BEGIN + timedelta(hours=2),
            end=BEGIN + timedelta(hours=4),
            activities=[
                ScheduledActivity(
                    name="Expired",
                    begin=T0 + HOUR,
                    duration_seconds=10 * MIN,
                )
            ],
        )
        assert ditl.calc()
        assert _activities(ditl.plan) == []
        assert not any(e.event_type == "ACTIVITY" for e in ditl.log)
        assert ditl.validate_plan_matches_execution() == []

    def test_an_open_slide_window_is_still_pending(self) -> None:
        config = _config(4)
        ditl = QueueDITL(
            config=config,
            begin=BEGIN + timedelta(hours=2),
            end=BEGIN + timedelta(hours=4),
            activities=[
                ScheduledActivity(
                    name="Still open",
                    begin=T0 + HOUR,
                    duration_seconds=10 * MIN,
                    slide_seconds=2 * HOUR,
                )
            ],
        )
        assert ditl._setup_simulation_timing()
        pending = ditl._activity_pending(ditl.ustart)
        assert pending is not None and pending.name == "Still open"
        assert ditl._next_activity_begin(ditl.ustart) == T0 + 3 * HOUR


def test_partial_dump_across_horizons_retains_coverage_data_and_momentum():
    config = _config(2)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.spacecraft_bus.inertia_tensor_body_kg_m2 = (
        (1000, 0, 0),
        (0, 1000, 0),
        (0, 0, 1000),
    )
    config.spacecraft_bus.attitude_control.stored_momentum = StoredMomentumConfig(
        gravity_gradient_enabled=True,
        initial_momentum_body_n_m_s=(0.4, 0, 0),
        capacity_n_m_s=10,
        thruster_dump=ThrusterDumpConfig(duration_seconds=HOUR),
    )
    activity = ScheduledActivity(
        name="Long dump",
        begin=T0 + 600,
        duration_seconds=HOUR,
        data_gb=1.5,
        desaturation=True,
    )
    run = RollingHorizonDITL(
        config,
        [],
        begin=BEGIN,
        end=BEGIN + timedelta(hours=2),
        horizon=timedelta(minutes=30),
        replan_interval=timedelta(minutes=15),
        activities=[activity],
        include_passes=False,
    )
    run._add_activity_data = Mock(wraps=run._add_activity_data)
    run._unload_stored_momentum = Mock(wraps=run._unload_stored_momentum)
    assert run.calc()
    entries = _activities(run.plan)
    assert len(entries) > 1
    assert float(entries[0].begin) == activity.begin
    assert float(entries[-1].end) == activity.end
    assert all(float(a.end) == float(b.begin) for a, b in zip(entries, entries[1:]))
    assert all(
        (e.activity_hold_begin, e.activity_hold_end) == (activity.begin, activity.end)
        for e in entries
    )
    assert all(not e.activity_finishes for e in entries[:-1])
    assert entries[-1].activity_finishes
    run._add_activity_data.assert_called_once()
    assert run._add_activity_data.call_args.args[0] == activity.data_gb
    run._unload_stored_momentum.assert_called_once_with(activity.end, entries[-1].obsid)
    assert run.validate_plan_matches_execution() == []
