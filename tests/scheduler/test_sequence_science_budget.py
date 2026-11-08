"""Calibration and inter-step motion consume time, not science budgets."""

from datetime import timedelta

import pytest

from conops.config import (
    ObservationCategories,
    ObservationCategory,
    ObservationGroup,
    ObservationGroups,
    ObservationTiming,
)
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.targets import ObservationSequence, Pointing, SequenceStep

from .planning_scenario import BEGIN, make_config


@pytest.mark.parametrize("planner", ["priority", "local_seed", "cp_seed"])
@pytest.mark.parametrize(
    "budget_source,budget,visits",
    [
        ("none", None, 2),
        ("category", 1200, 2),
        ("group", 1200, 2),
        ("both", 1200, 2),
        ("both", 600, 1),
    ],
)
def test_sequence_planning_counts_only_the_science_steps(
    planner, budget_source, budget, visits
):
    if planner == "cp_seed":
        pytest.importorskip("ortools")
    config = make_config(3)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.payload.observation_timing = ObservationTiming()
    config.observation_categories = ObservationCategories(
        categories=[
            ObservationCategory(
                name="Sequence",
                program="Science program",
                obsid_min=700,
                obsid_max=701,
                time_budget_seconds=budget
                if budget_source in ("category", "both")
                else None,
            )
        ]
    )
    config.observation_groups = ObservationGroups(
        groups=[
            ObservationGroup(
                name="Campaign",
                members=[700],
                time_budget_seconds=budget
                if budget_source in ("group", "both")
                else None,
            )
        ]
    )
    target = Pointing(
        config=config,
        obsid=700,
        ra=105,
        dec=10,
        visits=2,
        sequence=ObservationSequence(
            steps=[
                SequenceStep(name="science A", duration_seconds=300),
                SequenceStep(
                    name="calibration",
                    duration_seconds=1800,
                    science=False,
                    ra=107,
                    dec=11,
                ),
                SequenceStep(name="science B", duration_seconds=300),
            ]
        ),
    )
    args = (config, [target], BEGIN, BEGIN + timedelta(hours=3))
    if planner == "priority":
        scheduler = PriorityScheduler(*args, include_passes=False)
        entries = list(scheduler.schedule())
    else:
        cls = LocalSearchScheduler if planner == "local_seed" else CpSatScheduler
        scheduler = cls(*args, include_passes=False, time_limit=0, max_iterations=0)
        decoded = scheduler._start_from_priority_plan()
        entries = [e for b in decoded.final.timeline for e in b.steps or [b.entry]]
    assert [e.step for e in entries] == [
        "science A",
        "calibration",
        "science B",
    ] * visits
    science = sum(e.exposure for e in entries if not e.calibration)
    calibration = sum(e.exposure for e in entries if e.calibration)
    assert science == visits * 600
    assert calibration == visits * 1800
    assert scheduler._category_seconds == {"Sequence": science}
    assert scheduler._group_seconds == {"Campaign": science}
    assert scheduler._program_seconds == {"Science program": science}
