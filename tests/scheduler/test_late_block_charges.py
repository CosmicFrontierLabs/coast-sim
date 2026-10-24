"""A block that may begin late, before a sequence visit, on a plan that
charges in its idle time: the block settles after the charges inserted
before it, as the chain into it was checked, and its reservation stays
within the horizon."""

from datetime import timedelta

import pytest

from conops.schedulers import LocalSearchScheduler, PriorityScheduler
from conops.targets import ObservationSequence, Pointing, SequenceStep, VisitWindow
from conops.targets.activity import periodic_activities

from .planning_scenario import HOUR, PATCH, T0
from .test_battery_constraint import (
    BEGIN,
    BORESIGHT,
    HOURS,
    _battery_config,
    _execute,
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


@pytest.mark.parametrize("scheduler", [PriorityScheduler, LocalSearchScheduler])
@pytest.mark.parametrize("block_hour", [1.0, 2.0, 3.0, 4.0])
def test_a_late_block_before_a_visit_plans_and_executes(  # type: ignore[no-untyped-def]
    scheduler, block_hour: float
) -> None:
    config, targets, activities = _scenario(block_hour, slide_hours=1.0)
    options = {"time_limit": 0.5} if scheduler is LocalSearchScheduler else {}
    plan = scheduler(
        config,
        targets,
        BEGIN,
        BEGIN + timedelta(hours=HOURS),
        activities=activities,
        **options,
    ).schedule()
    blocks = [e for e in plan if e.obstype.name == "ACTIVITY"]
    assert blocks
    assert all(float(e.end) <= T0 + HOURS * HOUR for e in blocks)
    ditl = _execute(config, plan)
    assert ditl.validate_plan_matches_execution() == []


def test_a_late_block_near_the_horizon_end_is_reserved_within_it() -> None:
    """A block whose slide would carry its end past the horizon is held as
    far as the horizon goes; the plan builds and nothing is sampled past
    the ephemeris."""
    config, targets, activities = _scenario(4.0, slide_hours=2.0)
    plan = PriorityScheduler(
        config, targets, BEGIN, BEGIN + timedelta(hours=HOURS), activities=activities
    ).schedule()
    late = [e for e in plan if e.obstype.name == "ACTIVITY"]
    assert late and all(float(e.end) <= T0 + HOURS * HOUR for e in late)
