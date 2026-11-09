"""An inherited hold cannot consume science that was already collected."""

from datetime import timedelta

import pytest

from conops import QueueDITL
from conops.config import ObservationTiming
from conops.targets import ScheduledActivity

from .planning_scenario import BEGIN, HOUR, MIN, T0, make_config, make_target


@pytest.mark.parametrize("fixed", [False, True])
@pytest.mark.parametrize("collecting", [False, True])
def test_due_inherited_hold_preserves_collection_and_outstanding_teardown(
    fixed, collecting
):
    config = make_config(4)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.payload.observation_timing = ObservationTiming(
        cleanup_seconds=5 * MIN, handoff_seconds=2 * MIN
    )
    hold = ScheduledActivity(
        name="Calibration following an observation",
        begin=T0 + HOUR if fixed else T0 + 45 * MIN,
        duration_seconds=20 * MIN,
        fixed=fixed,
        slide_seconds=0 if fixed else 15 * MIN,
    )
    run = QueueDITL(
        config=config, begin=BEGIN, end=BEGIN + timedelta(hours=4), activities=[hold]
    )
    run.ustart, run.uend, run.step_size = T0, T0 + 4 * HOUR, MIN
    target = make_target(config, 100, 105, 10, minutes=90, ss_min=1)
    target.begin = T0
    target.collection_begin = T0 + 10 * MIN
    target.collection_end = T0 + (70 if collecting else 57) * MIN
    target.end = target.collection_end + 7 * MIN
    run.ppt = target
    run.plan.append(target.model_copy())
    original_end = target.collection_end
    now = T0 + HOUR

    run._manage_activities(now, run.acs.ra, run.acs.dec, run.acs.roll)

    assert run._active_activity is None
    assert run.ppt is target
    expected_stop = min(original_end, now)
    assert target.collection_end == expected_stop
    assert target.end == expected_stop + 7 * MIN
    assert run.plan[-1].collection_end == expected_stop
    assert run.plan[-1].end == target.end
    assert run.plan[-1].collection_begin == T0 + 10 * MIN
