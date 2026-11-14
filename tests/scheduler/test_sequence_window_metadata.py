"""Every planned sequence step retains the window of its indivisible visit."""

from datetime import timedelta

import pytest

from conops.config import ObservationTiming
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.targets import (
    ObservationSequence,
    Plan,
    Pointing,
    SequenceStep,
    VisitWindow,
)

from .planning_scenario import BEGIN, MIN, T0, make_config


@pytest.mark.parametrize("planner", ["priority", "local", "cp_sat"])
def test_sequence_steps_and_block_keep_their_visit_window(planner):
    if planner == "cp_sat":
        pytest.importorskip("ortools")
    config = make_config(3)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.payload.observation_timing = ObservationTiming()
    target = Pointing(
        config=config,
        obsid=700,
        ra=105,
        dec=10,
        ss_min=300,
        ss_max=300,
        sequence=ObservationSequence(
            steps=[
                SequenceStep(name="calibration", duration_seconds=120, science=False),
                SequenceStep(name="science", duration_seconds=300),
            ]
        ),
        visit_windows=[
            VisitWindow(start=T0 + 10 * MIN, deadline=T0 + 20 * MIN),
            VisitWindow(start=T0 + 60 * MIN, deadline=T0 + 70 * MIN),
        ],
    )
    args = (config, [target], BEGIN, BEGIN + timedelta(hours=2))
    if planner == "priority":
        scheduler = PriorityScheduler(*args, include_passes=False)
    elif planner == "local":
        scheduler = LocalSearchScheduler(
            *args, include_passes=False, max_iterations=20, time_limit=60, seed=2029
        )
    else:
        scheduler = CpSatScheduler(
            *args,
            include_passes=False,
            solver_deterministic_time=1,
            solver_time_limit=60,
            workers=1,
            seed=2029,
        )
    plan = scheduler.schedule()
    expected = [(0, "calibration"), (0, "science"), (1, "calibration"), (1, "science")]
    entries = [e for e in plan if e.obsid == target.obsid]
    assert [(e.visit, e.step) for e in entries] == expected
    assert [e.exposure for e in entries] == [120, 300, 120, 300]
    assert [
        b.entry.visit for b in scheduler.timeline if b.entry.obsid == target.obsid
    ] == [0, 1]
    reloaded = Plan.model_validate_json(plan.model_dump_json())
    assert [(e.visit, e.step) for e in reloaded if e.obsid == target.obsid] == expected
