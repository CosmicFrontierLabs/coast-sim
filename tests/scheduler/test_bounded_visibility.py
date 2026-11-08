"""A partial visibility query must not truncate a later full-horizon plan."""

from datetime import timedelta

from conops.schedulers import PriorityScheduler

from .planning_scenario import BEGIN, HOUR, T0, make_config, make_target


def test_priority_planner_extends_a_targets_partial_visibility():
    c = make_config(6)
    c.constraint.sun_constraint = None
    c.constraint.earth_constraint = None
    c.constraint.ignore_roll = True
    t = make_target(c, 100, 105, 10, minutes=30, snapshot=30)
    t.earliest_start = T0 + 3 * HOUR
    t.visibility(begin=T0, end=T0 + HOUR)
    assert t.windows == [[T0, T0 + HOUR]]
    planner = PriorityScheduler(
        c, [t], BEGIN, BEGIN + timedelta(hours=6), include_passes=False
    )
    plan = planner.schedule()
    assert any(e.obsid == 100 and e.collection_begin >= t.earliest_start for e in plan)
    assert planner.targets[0].windows == [[T0, T0 + 6 * HOUR]]
