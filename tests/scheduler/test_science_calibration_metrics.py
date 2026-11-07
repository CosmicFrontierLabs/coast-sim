"""Executed calibration collection must not inflate science acceptance metrics."""

from datetime import timedelta

import pytest

from conops.benchmark import (
    BenchmarkScenario,
    TOOSpec,
    dispatch,
    planned,
    rolling,
    run_benchmark,
)
from conops.benchmark.harness import Run, _measure
from conops.ditl import DITL, QueueDITL
from conops.ditl.telemetry import Housekeeping
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.targets import ObservationSequence, Plan, Pointing, SequenceStep, TOORequest

from .planning_scenario import BEGIN, HOUR, MIN, T0, make_config


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


CALIBRATION = 155
SCIENCE = 421
MERIT = 37
END = BEGIN + timedelta(hours=3)


def configuration():
    c = make_config(3)
    c.constraint.sun_constraint = None
    c.constraint.earth_constraint = None
    return c


def targets(c):
    return [
        Pointing(
            config=c,
            obsid=100,
            ra=105,
            dec=10,
            merit=MERIT,
            visits=1,
            sequence=ObservationSequence(
                steps=[
                    SequenceStep(
                        name="reference calibration",
                        duration_seconds=CALIBRATION,
                        science=False,
                    ),
                    SequenceStep(name="target science", duration_seconds=SCIENCE),
                ]
            ),
        )
    ]


def scenario():
    return BenchmarkScenario(
        name="Calibration accounting",
        begin=BEGIN,
        end=END,
        make_config=configuration,
        make_targets=targets,
    )


@pytest.mark.parametrize(
    "kind,rolling_run",
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
def test_calibration_is_excluded_from_executed_science(kind, rolling_run):
    if kind == "dispatch":
        contender = dispatch()
    else:
        scheduler, options = _planner(kind)
        contender = (
            rolling(
                scheduler, replan_interval=timedelta(hours=1), scheduler_options=options
            )
            if rolling_run
            else planned(scheduler, **options)
        )
    [result] = run_benchmark(scenario(), [contender])
    assert result.error is None
    assert result.mismatches == 0
    assert result.science_hours == pytest.approx(SCIENCE / HOUR)
    assert result.calibration_hours == pytest.approx(CALIBRATION / HOUR)
    assert result.weighted_science == pytest.approx(SCIENCE * MERIT)
    assert result.observations == 1
    assert result.program_share == {"Observation": 1.0}


def test_saved_plan_replay_labels_the_flown_calibration(tmp_path):
    c = configuration()
    plan = PriorityScheduler(c, targets(c), BEGIN, END, include_passes=False).schedule()
    loaded = Plan.load(plan.save(tmp_path / "plan.json"))
    run = DITL(c, plan=loaded, begin=BEGIN, end=END)
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
    hk = run.telemetry.housekeeping
    calibration = [h for h in hk if h.calibration and h.collection_seconds]
    science = [h for h in hk if not h.calibration and h.collection_seconds]
    assert calibration and science
    assert sum(h.collection_seconds for h in calibration) == CALIBRATION
    assert sum(h.collection_seconds for h in science) == SCIENCE
    assert any(h.collection_seconds < MIN for h in calibration)
    assert all(
        Housekeeping.model_validate_json(h.model_dump_json()).calibration
        for h in calibration
    )


def test_sequence_too_response_starts_with_science_after_calibration():
    c = configuration()
    run = QueueDITL(c, begin=BEGIN, end=END)
    [t] = targets(c)
    run.submit_too(TOORequest(request=t, submit_time=T0))
    assert run.calc()
    metric_scenario = scenario()
    metric_scenario.toos = [
        TOOSpec(
            obsid=100,
            ra=105,
            dec=10,
            merit=MERIT,
            exptime=SCIENCE,
            name="Calibration then science",
            submit_time=T0,
            deadline=T0 + HOUR,
        )
    ]
    summary = _measure("dispatch", metric_scenario, Run(simulation=run), 0)
    first_science = next(
        h.timestamp.timestamp()
        for h in run.telemetry.housekeeping
        if h.collection_seconds and not h.calibration
    )
    first_calibration = next(
        h.timestamp.timestamp()
        for h in run.telemetry.housekeeping
        if h.collection_seconds and h.calibration
    )
    assert first_science > first_calibration
    assert summary.too_response_seconds[100] == first_science - T0
    assert summary.science_hours == pytest.approx(SCIENCE / HOUR)
    assert summary.calibration_hours == pytest.approx(CALIBRATION / HOUR)


def test_old_housekeeping_without_a_calibration_flag_keeps_its_default():
    record = Housekeeping(timestamp=BEGIN, collection_seconds=15)
    assert record.calibration is False
