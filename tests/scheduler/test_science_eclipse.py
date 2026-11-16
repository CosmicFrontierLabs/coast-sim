"""One eclipse observing policy across requests, dispatch, planning and replay."""

from datetime import timedelta

import pytest

from conops import DITL, QueueDITL
from conops.common import ACSMode, ObsType
from conops.config import Instrument, MissionConfig, ObservationCategory
from conops.config.constraint import AttitudeConstraintScope
from conops.config.instrument import InstrumentMode
from conops.config.observation_timing import ObservationTiming
from conops.ditl import RollingHorizonDITL
from conops.schedulers import (
    CpSatScheduler,
    LocalSearchScheduler,
    PriorityScheduler,
    SchedulingContext,
)
from conops.targets import (
    MovingPointing,
    ObservationSequence,
    Plan,
    Pointing,
    Program,
    SequenceStep,
    TOORequest,
    track_from_table,
)

from .planning_scenario import BEGIN, HOUR, MIN, T0, make_config, make_target


def _planner(kind):
    if kind == "priority":
        return PriorityScheduler, {}
    if kind == "local_search":
        return LocalSearchScheduler, {"time_limit": 0.0}
    pytest.importorskip("ortools")
    return CpSatScheduler, {
        "time_limit": 0.0,
        "solver_time_limit": 2.0,
        "solver_deterministic_time": 0.05,
        "workers": 1,
        "seed": 1,
    }


def configuration(*, allowed=False, override=None, intervals=None):
    c = make_config(
        3,
        categories=[
            ObservationCategory(
                name="science",
                obsid_min=100,
                obsid_max=200,
                science_in_eclipse=override,
            )
        ],
    )
    c.constraint.sun_constraint = None
    c.constraint.earth_constraint = None
    c.constraint.science_in_eclipse = allowed
    c.constraint.eclipse_intervals = (
        intervals if intervals is not None else [(T0 + 10 * MIN, T0 + 20 * MIN)]
    )
    c.payload.instruments = [
        Instrument(
            name="camera",
            modes=[
                InstrumentMode(name="IDLE", power_w=10),
                InstrumentMode(name="SCIENCE", power_w=200),
            ],
        )
    ]
    return c


def target(c, **kwargs):
    return make_target(c, 100, 105, 10, minutes=100, snapshot=10, ss_min=2, **kwargs)


def collected(run):
    return sum(h.collection_seconds or 0 for h in run.telemetry.housekeeping)


def assert_sunlit(run):
    assert run.validate_plan_matches_execution() == []
    assert collected(run) > 0
    for h in run.telemetry.housekeeping:
        if run.config.constraint.in_eclipse(0, 0, h.timestamp.timestamp()):
            assert not h.collection_seconds
            assert h.instrument_modes == {"camera": "IDLE"}
    for e in run.plan:
        if e.obstype in (ObsType.AT, ObsType.PPT, ObsType.TOO):
            assert (
                e.eclipse_violation(float(e.collection_begin), float(e.collection_end))
                is None
            )


@pytest.mark.parametrize(
    "kind,rolling",
    [
        ("dispatch", False),
        ("priority", False),
        ("local_search", False),
        ("cp_sat", False),
        ("priority", True),
        ("local_search", True),
        ("cp_sat", True),
    ],
)
def test_no_science_in_real_or_injected_eclipse(kind, rolling, tmp_path):
    c = configuration()
    t = target(c)
    end = BEGIN + timedelta(hours=3)
    if kind == "dispatch":
        run = QueueDITL(c, begin=BEGIN, end=end)
        run.queue.add_request(t)
    else:
        planner, options = _planner(kind)
        if rolling:
            run = RollingHorizonDITL(
                c,
                [t],
                begin=BEGIN,
                end=end,
                scheduler=planner,
                scheduler_options=options,
                include_passes=False,
                horizon=timedelta(hours=2),
                replan_interval=timedelta(minutes=45),
                commit_lead_time=timedelta(minutes=10),
            )
        else:
            plan = planner(
                c, [t], BEGIN, end, include_passes=False, **options
            ).schedule()
            run = DITL(
                c,
                plan=Plan.load(plan.save(tmp_path / "plan.json")),
                begin=BEGIN,
                end=end,
            )
    assert run.calc()
    assert_sunlit(run)


@pytest.mark.parametrize(
    "default,override,expected",
    [
        (True, None, True),
        (False, None, False),
        (False, True, True),
        (True, False, False),
    ],
)
def test_category_override_and_legacy_default(default, override, expected):
    c = configuration(allowed=default, override=override)
    run = QueueDITL(c, begin=BEGIN, end=BEGIN + timedelta(hours=3))
    run.queue.add_request(target(c))
    assert c.science_in_eclipse_for(100) == expected
    assert run.calc()
    shadow_science = sum(
        h.collection_seconds or 0
        for h in run.telemetry.housekeeping
        if c.constraint.in_eclipse(0, 0, h.timestamp.timestamp())
    )
    assert (shadow_science > 0) == expected
    if not expected:
        assert_sunlit(run)


def test_injected_boundaries_inside_one_tick_are_removed_exactly():
    c = configuration(intervals=[(T0 + 8 * MIN + 10, T0 + 8 * MIN + 30)])
    t = target(c)
    t.visibility()
    assert t.eclipse_violation(T0 + 8 * MIN, T0 + 9 * MIN) == T0 + 8 * MIN + 10
    assert t.visible(T0 + 8 * MIN, T0 + 8 * MIN + 10)
    assert not t.visible(T0 + 8 * MIN + 10, T0 + 8 * MIN + 30)
    run = QueueDITL(c, begin=BEGIN, end=BEGIN + timedelta(hours=3))
    run.queue.add_request(t)
    assert run.calc()
    assert_sunlit(run)


def test_collection_guard_clips_a_stale_window_at_a_substep_boundary():
    c = configuration(intervals=[(T0 + 8 * MIN + 10, T0 + 8 * MIN + 30)])
    run = QueueDITL(c, begin=BEGIN, end=BEGIN + timedelta(hours=3))
    t = target(c)
    t.begin = T0 + 8 * MIN
    t.end = T0 + 9 * MIN
    t.collection_begin, t.collection_end = float(t.begin), float(t.end)
    run.ppt = t
    run.uend = T0 + 3 * HOUR
    run.acs.last_slew = None
    assert run._collection_interval_for_step(float(t.begin), ACSMode.SCIENCE) == (
        T0 + 8 * MIN,
        T0 + 8 * MIN + 10,
    )


def test_eclipse_policy_does_not_become_a_hardware_or_idle_constraint():
    c = configuration()
    t = target(c)
    ctx = SchedulingContext(c, BEGIN, BEGIN + timedelta(hours=3))
    at = T0 + 15 * MIN
    assert c.constraint.in_science_constraint(105, 10, at)
    assert c.constraint.science_constraint_name(105, 10, at) == "science_eclipse"
    assert c.constraint.in_science_constraint_batch([105], [10], at).tolist() == [True]
    assert ctx.science_violation(t, (105, 10, 0), at) == "science_eclipse"
    assert ctx.attitude_violation((105, 10, 0), at, ACSMode.IDLE) is None
    assert ctx.attitude_violation((105, 10, 0), at, ACSMode.SLEWING) is None
    assert (
        t.attitude_constraint_names(
            [AttitudeConstraintScope.HARDWARE_SAFETY], (105, 10, 0), at, ACSMode.SCIENCE
        )
        == []
    )
    assert t.attitude_constraint_names(
        [AttitudeConstraintScope.IMAGING_QUALITY], (105, 10, 0), at, ACSMode.SCIENCE
    ) == ["science_eclipse"]


def test_policy_and_categories_roundtrip_without_program_execution_state(tmp_path):
    c = configuration(override=True)
    c.to_yaml_file(str(tmp_path / "mission.yaml"))
    restored = MissionConfig.from_yaml_file(str(tmp_path / "mission.yaml"))
    assert restored.constraint.science_in_eclipse is False
    assert restored.science_in_eclipse_for(100) is True
    assert restored.science_in_eclipse_for(500) is False
    p = Program(begin=T0, end=T0 + 3 * HOUR, requests=[target(c)])
    loaded = Program.load(p.save(tmp_path / "program.json"))
    loaded.bind(c)
    assert loaded == p
    assert loaded.requests[0].forbids_eclipse() is False


def test_replay_rejects_an_existing_plan_that_the_new_policy_forbids():
    c = configuration(allowed=True)
    planner, options = _planner("priority")
    plan = planner(
        c,
        [target(c)],
        BEGIN,
        BEGIN + timedelta(hours=3),
        include_passes=False,
        **options,
    ).schedule()
    c.constraint.science_in_eclipse = False
    run = DITL(c, plan=plan, begin=BEGIN, end=BEGIN + timedelta(hours=3))
    with pytest.raises(ValueError, match="science-in-eclipse policy"):
        run.calc()
    assert run.acs.executed_commands == []


@pytest.mark.parametrize("kind", ["dispatch", "priority", "local_search", "cp_sat"])
def test_protected_sequence_waits_until_the_whole_visit_fits(kind):
    c = configuration(intervals=[(T0 + 10 * MIN, T0 + 25 * MIN)])
    sequence = ObservationSequence(
        steps=[
            SequenceStep(name="A", duration_seconds=12 * MIN),
            SequenceStep(name="B", duration_seconds=12 * MIN),
        ]
    )
    t = Pointing(
        config=c,
        obsid=100,
        ra=105,
        dec=10,
        name="Protected visit",
        sequence=sequence,
        visits=1,
    )
    if kind == "dispatch":
        run = QueueDITL(c, begin=BEGIN, end=BEGIN + timedelta(hours=3))
        run.queue.add_request(t)
    else:
        planner, options = _planner(kind)
        plan = planner(
            c, [t], BEGIN, BEGIN + timedelta(hours=3), include_passes=False, **options
        ).schedule()
        run = DITL(c, plan=plan, begin=BEGIN, end=BEGIN + timedelta(hours=3))
    assert run.calc()
    assert_sunlit(run)
    steps = [e for e in run.plan if e.step is not None]
    assert [e.step for e in steps] == ["A", "B"]
    assert min(float(e.collection_begin) for e in steps) >= T0 + 66 * MIN
    assert (
        c.constraint.first_eclipse_overlap(
            float(steps[0].collection_begin), float(steps[-1].end)
        )
        is None
    )


@pytest.mark.parametrize(
    "override,deadline,executed",
    [(None, 19, False), (True, 19, True), (None, 90, True)],
)
@pytest.mark.parametrize("rolling", [False, True])
def test_too_triggered_in_eclipse_obeys_its_category(
    override, deadline, executed, rolling
):
    c = configuration(override=override)
    end = BEGIN + timedelta(hours=3)
    if rolling:
        planner, options = _planner("cp_sat")
        run = RollingHorizonDITL(
            c,
            [],
            begin=BEGIN,
            end=end,
            scheduler=planner,
            scheduler_options=options,
            include_passes=False,
            horizon=timedelta(hours=2),
            replan_interval=timedelta(minutes=45),
            commit_lead_time=timedelta(minutes=1),
        )
    else:
        run = QueueDITL(c, begin=BEGIN, end=end)
    too = run.submit_too(
        obsid=100,
        ra=105,
        dec=10,
        merit=1000,
        exptime=5 * MIN,
        submit_time=T0 + 12 * MIN,
        deadline=T0 + deadline * MIN,
    )
    assert run.calc()
    assert too.executed == executed
    assert run.validate_plan_matches_execution() == []
    assert (collected(run) > 0) == executed
    if executed and override is not True:
        assert_sunlit(run)
        entries = [e for e in run.plan if e.obsid == 100]
        assert float(entries[0].collection_begin) >= T0 + 20 * MIN


@pytest.mark.parametrize("rolling", [False, True])
def test_sequence_too_waits_for_a_whole_sunlit_window(rolling):
    c = configuration(intervals=[(T0 + 10 * MIN, T0 + 25 * MIN)])
    t = Pointing(
        config=c,
        obsid=100,
        ra=105,
        dec=10,
        merit=1000,
        visits=1,
        sequence=ObservationSequence(
            steps=[
                SequenceStep(name="first", duration_seconds=12 * MIN),
                SequenceStep(name="second", duration_seconds=12 * MIN),
            ]
        ),
    )
    end = BEGIN + timedelta(hours=3)
    if rolling:
        planner, options = _planner("cp_sat")
        run = RollingHorizonDITL(
            c,
            [],
            begin=BEGIN,
            end=end,
            scheduler=planner,
            scheduler_options=options,
            include_passes=False,
            horizon=timedelta(hours=2),
            replan_interval=timedelta(minutes=45),
            commit_lead_time=timedelta(minutes=1),
        )
    else:
        run = QueueDITL(c, begin=BEGIN, end=end)
    too = run.submit_too(
        TOORequest(request=t, submit_time=T0 + 12 * MIN, deadline=T0 + 90 * MIN)
    )
    assert run.calc()
    assert too.executed
    assert_sunlit(run)
    steps = [e for e in run.plan if e.step is not None]
    assert [e.step for e in steps] == ["first", "second"]
    assert float(steps[0].collection_begin) >= T0 + 66 * MIN


@pytest.mark.parametrize("kind", ["dispatch", "priority", "local_search", "cp_sat"])
def test_setup_cleanup_and_moving_track_stay_within_sunlit_windows(kind):
    c = configuration()
    c.payload.observation_timing = ObservationTiming(
        setup_seconds=90, cleanup_seconds=90
    )
    t = MovingPointing(
        config=c,
        obsid=100,
        ra=105,
        dec=10,
        track=track_from_table([T0, T0 + 3 * HOUR], [105, 106], [10, 10]),
        exptime=60 * MIN,
        ss_min=2 * MIN,
        ss_max=10 * MIN,
    )
    t.visibility()
    assert not t.visible(T0 + 10 * MIN, T0 + 20 * MIN)
    end = BEGIN + timedelta(hours=3)
    if kind == "dispatch":
        run = QueueDITL(c, begin=BEGIN, end=end)
        run.queue.add_request(t)
    else:
        planner, options = _planner(kind)
        plan = planner(c, [t], BEGIN, end, include_passes=False, **options).schedule()
        run = DITL(c, plan=plan, begin=BEGIN, end=end)
    assert run.calc()
    assert_sunlit(run)
    entries = [e for e in run.plan if e.obsid == 100]
    assert entries and all(e.moves for e in entries)
    for e in entries:
        assert float(e.collection_begin) - (
            float(e.begin) + e.slewtime
        ) == pytest.approx(90)
        assert float(e.end) - float(e.collection_end) == pytest.approx(90)
        assert e.eclipse_violation(float(e.begin) + e.slewtime, float(e.end)) is None


@pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
@pytest.mark.parametrize("default,override", [(False, True), (True, False)])
def test_planners_resolve_category_override_in_both_directions(kind, default, override):
    c = configuration(allowed=default, override=override)
    planner, options = _planner(kind)
    end = BEGIN + timedelta(hours=3)
    plan = planner(
        c, [target(c)], BEGIN, end, include_passes=False, **options
    ).schedule()
    run = DITL(c, plan=plan, begin=BEGIN, end=end)
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
    shadow = sum(
        h.collection_seconds or 0
        for h in run.telemetry.housekeeping
        if c.constraint.in_eclipse(0, 0, h.timestamp.timestamp())
    )
    assert (shadow > 0) == override


def test_eclipse_windows_cache_only_the_natural_ephemeris_result():
    from unittest.mock import Mock

    c = configuration(allowed=True)
    evaluator = Mock(wraps=c.constraint._eclipse_constraint)
    c.constraint._eclipse_constraint = evaluator
    t = target(c)
    t.visibility()
    evaluator.evaluate.assert_not_called()
    c.constraint.science_in_eclipse = False
    assert t.eclipse_violation(T0 + 12 * MIN, T0 + 13 * MIN) == T0 + 12 * MIN
    assert t.eclipse_violation(T0 + 22 * MIN, T0 + 23 * MIN) is None
    c.constraint.eclipse_intervals = [(T0 + 21 * MIN, T0 + 24 * MIN)]
    assert t.eclipse_violation(T0 + 12 * MIN, T0 + 13 * MIN) is None
    assert t.eclipse_violation(T0 + 22 * MIN, T0 + 23 * MIN) == T0 + 22 * MIN
    assert evaluator.evaluate.call_count == 1
    c.constraint.ephem = configuration().constraint.ephem
    c.constraint.sunlit_windows()
    assert evaluator.evaluate.call_count == 2


def test_first_science_violation_keeps_an_earlier_geometric_failure(monkeypatch):
    c = configuration()
    ctx = SchedulingContext(c, BEGIN, BEGIN + timedelta(hours=3))
    t = target(c)
    monkeypatch.setattr(
        ctx,
        "science_violation",
        lambda entry, attitude, utime: "geometry" if utime >= T0 + MIN else None,
    )
    assert ctx.first_science_violation(t, (105, 10, 0), T0, T0 + 25 * MIN) == T0 + MIN


@pytest.mark.parametrize("planned", [False, True])
def test_eclipse_policy_applies_in_the_mounted_telescope_frame(planned):
    from conops.config import Telescope

    c = configuration()
    c.payload.instruments = [
        Telescope(
            name="camera", boresight=(0, 1, 0), modes=c.payload.instruments[0].modes
        )
    ]
    t = target(c)
    assert t.uses_mounted_attitude()
    end = BEGIN + timedelta(hours=3)
    if planned:
        planner, options = _planner("priority")
        plan = planner(c, [t], BEGIN, end, include_passes=False, **options).schedule()
        run = DITL(c, plan=plan, begin=BEGIN, end=end)
    else:
        run = QueueDITL(c, begin=BEGIN, end=end)
        run.queue.add_request(t)
    assert run.calc()
    assert_sunlit(run)
