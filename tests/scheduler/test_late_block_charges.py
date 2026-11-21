"""A block that may begin late, before a sequence visit, on a plan that
charges in its idle time: the block settles after the charges inserted
before it, as the chain into it was checked, and its reservation stays
within the horizon."""

from datetime import timedelta

import pytest

from conops.schedulers import LocalSearchScheduler, PriorityScheduler
from conops.targets import ObservationSequence, Pointing, SequenceStep, VisitWindow
from conops.targets.activity import periodic_activities

from .planning_scenario import HOUR, PATCH, T0, activity_entries, replay
from .test_battery_constraint import (
    BEGIN,
    BORESIGHT,
    HOURS,
    _battery_config,
    _targets,
)

SEQUENCE = ObservationSequence(
    steps=[
        SequenceStep(name="ref", duration_seconds=600, science=False),
        SequenceStep(name="target", duration_seconds=1800),
        SequenceStep(name="ref", duration_seconds=600, science=False),
    ]
)


def _scenario(block_hour: float, slide_hours: float):  # type: ignore[no-untyped-def]
    config = _battery_config(BORESIGHT, stations=True)
    targets = _targets(config)
    for n, (ra, dec) in enumerate(PATCH[:3]):
        opens = T0 + (block_hour + 0.5 + n) * HOUR
        targets.append(
            Pointing(
                config=config,
                ra=ra,
                dec=dec,
                obsid=20000 + n,
                name=f"visit {n}",
                merit=300.0,
                sequence=SEQUENCE,
                visits=1,
                visit_windows=[VisitWindow(start=opens, deadline=opens + 3 * HOUR)],
            )
        )
    activities = periodic_activities(
        "Cal",
        T0 + block_hour * HOUR,
        6 * HOUR,
        30 * 60.0,
        until=T0 + HOURS * HOUR,
        slide_seconds=slide_hours * HOUR,
    )
    return config, targets, activities


@pytest.mark.parametrize(
    "scheduler, block_hour, slide_hours",
    [
        (PriorityScheduler, 1.0, 1.0),
        (PriorityScheduler, 4.0, 1.0),
        (LocalSearchScheduler, 1.0, 1.0),
        (LocalSearchScheduler, 4.0, 1.0),
        # A slide that would carry the block past the horizon is held within it.
        (PriorityScheduler, 4.0, 2.0),
    ],
)
def test_a_late_block_before_a_visit_plans_and_executes(  # type: ignore[no-untyped-def]
    scheduler, block_hour: float, slide_hours: float
) -> None:
    config, targets, activities = _scenario(block_hour, slide_hours)
    options = {"time_limit": 0.5} if scheduler is LocalSearchScheduler else {}
    planner = scheduler(
        config,
        targets,
        BEGIN,
        BEGIN + timedelta(hours=HOURS),
        activities=activities,
        **options,
    )
    plan = planner.schedule()
    blocks = activity_entries(plan)
    assert {e.obsid for e in blocks} == {
        a.obsid for a in planner.activities if a.begin < T0 + HOURS * HOUR
    }
    assert all(float(e.end) == e.activity_hold_end for e in blocks)
    assert all(float(e.end) <= T0 + HOURS * HOUR for e in blocks)
    replay(config, plan, BEGIN, BEGIN + timedelta(hours=HOURS))
