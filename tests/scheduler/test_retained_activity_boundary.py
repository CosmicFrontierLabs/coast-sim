"""Momentum-prefix reconstruction preserves sliding holds at equal start times."""

from datetime import timedelta

import pytest

from conops.schedulers import PriorityScheduler
from conops.targets import ScheduledActivity

from .planning_scenario import (
    BEGIN,
    HOUR,
    MIN,
    T0,
    activity_entries,
    collected_seconds,
    make_config,
    make_target,
    replay,
)
from .planning_scenario import planner as _planner


@pytest.mark.parametrize("kind", ["priority", "local_search", "cp_sat"])
def test_retained_slew_at_settled_activity_start_precedes_the_whole_hold(kind):
    c = make_config(4, open_sky=True)
    targets = [
        make_target(c, 100, 105, 10, merit=100, minutes=20, snapshot=20, ss_min=20),
        make_target(c, 101, 110, 10, merit=50, minutes=20, snapshot=20, ss_min=20),
    ]
    activity = ScheduledActivity(
        obsid=900,
        name="Calibration",
        begin=T0 + MIN,
        duration_seconds=20 * MIN,
        slide_seconds=2 * HOUR,
    )
    cls, options = _planner(kind)
    s = cls(
        c,
        targets,
        BEGIN,
        BEGIN + timedelta(hours=4),
        include_passes=False,
        activities=[activity],
        **options,
    )
    s.schedule()
    science = [b for b in s.timeline if b.entry.obsid in (100, 101)]
    assert len(science) == 2
    assert science[1].slew_start == s.ctx.ceil_step(science[0].end)
    s._keep_before(T0 + 3 * HOUR)
    s._reserve_fixed()
    hold = next(b for b in s.timeline if b.is_activity)
    retained = next(b for b in s.timeline if b.entry.obsid == 101)
    assert s.timeline.index(retained) < s.timeline.index(hold)
    assert retained.end <= hold.ready
    plan = s._finish(s.timeline)
    run = replay(c, plan, BEGIN, BEGIN + timedelta(hours=4))
    flown = activity_entries(run.plan)
    assert len(flown) == 1 and flown[0].activity_finishes
    assert float(flown[0].end) - float(flown[0].begin) == 20 * MIN
    assert collected_seconds(run) == 40 * MIN


def test_rejected_retained_science_reopens_its_time_for_replacement():
    c = make_config(4, open_sky=True)
    t = make_target(c, 100, 105, 10, merit=100, minutes=20, snapshot=20, ss_min=20)
    s = PriorityScheduler(
        c, [t], BEGIN, BEGIN + timedelta(hours=4), include_passes=False
    )
    s.schedule()
    block = next(b for b in s.timeline if b.entry.obsid == 100)
    s._keep_before(T0 + 3 * HOUR)
    s._resume_from = T0 + 3 * HOUR
    s.activities = [
        ScheduledActivity(
            obsid=900, name="Fixed block", begin=T0, duration_seconds=HOUR, fixed=True
        )
    ]
    s._reserve_fixed()
    assert s._kept == []
    assert s._resume_from == block.slew_start
    plan = s._plan_once()
    science = [e for e in plan if e.obsid == 100]
    assert len(science) == 1
    assert float(science[0].begin) >= T0 + HOUR
    assert float(science[0].end) < T0 + 3 * HOUR
    replay(c, plan, BEGIN, BEGIN + timedelta(hours=4))
