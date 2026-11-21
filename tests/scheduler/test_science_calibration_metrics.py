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
from conops.ditl import QueueDITL
from conops.ditl.telemetry import Housekeeping
from conops.schedulers import PriorityScheduler
from conops.targets import ObservationSequence, Plan, Pointing, SequenceStep, TOORequest

from .planning_scenario import BEGIN, HOUR, MIN, T0, make_config, replay

CALIBRATION = 155
SCIENCE = 421
MERIT = 37
END = BEGIN + timedelta(hours=3)


def configuration():
    return make_config(3, open_sky=True)


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
    "contender",
    [
        dispatch(),
        planned(PriorityScheduler),
        rolling(PriorityScheduler, replan_interval=timedelta(hours=1)),
    ],
    ids=["dispatch", "planned", "rolling"],
)
def test_calibration_is_excluded_from_executed_science(contender):
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
    run = replay(c, Plan.load(plan.save(tmp_path / "plan.json")), BEGIN, END)
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
