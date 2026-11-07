"""A placement evaluates roll guidance once before applying its request."""

from datetime import timedelta
from unittest.mock import patch

import pytest

from conops.schedulers import PriorityScheduler
from conops.schedulers.priority_scheduler import StartState

from .planning_scenario import BEGIN, T0, make_config, make_target


@pytest.mark.parametrize("offset", [None, 0.0, 7.0])
def test_gap_attempt_does_not_repeat_roll_guidance(offset):
    config = make_config(1)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.constraint.invalidate_combined_constraint_cache()
    target = make_target(config, 701, 105, 10, minutes=5)
    target.roll_offset_deg = offset
    scheduler = PriorityScheduler(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(hours=1),
        include_passes=False,
        start_state=StartState(time=T0, attitude=(105, 10, 0)),
    )
    scheduler._reserve_fixed()
    request = scheduler._requests()[0]
    with patch.object(
        scheduler.ctx, "optimum_roll", wraps=scheduler.ctx.optimum_roll
    ) as guidance:
        trial = scheduler._fit_in_gap(request, scheduler._pred_at(0), None)
    assert trial is not None
    entry = trial[0].entry
    assert int(entry.obsid) == 701
    assert entry.exptime == 300
    assert guidance.call_count == 1
