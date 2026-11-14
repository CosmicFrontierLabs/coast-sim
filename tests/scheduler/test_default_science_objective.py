"""Valid late observations retain their science value unless explicitly discounted."""

from datetime import timedelta

import pytest

from conops import DITL, create_ditl
from conops.schedulers import CpSatScheduler, LocalSearchScheduler
from conops.targets import ScheduledActivity

from .planning_scenario import BEGIN, HOUR, T0, make_config, make_target


def _scenario():
    config = make_config(4)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    end = BEGIN + timedelta(hours=3, minutes=30)
    targets = [
        make_target(
            config,
            1,
            105,
            10,
            merit=1000,
            minutes=60,
            snapshot=60,
            ss_min=60,
            deadline=end.timestamp(),
        ),
        make_target(config, 2, 105, 10, merit=800, minutes=60, snapshot=60, ss_min=60),
    ]
    activity = ScheduledActivity(
        name="Reserved engineering hold",
        begin=T0,
        duration_seconds=2 * HOUR,
        fixed=True,
    )
    return config, targets, activity, end


def _science(run):
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
    return {
        obsid: sum(
            h.collection_seconds or 0
            for h in run.telemetry.housekeeping
            if h.obsid == obsid and not h.calibration
        )
        for obsid in (1, 2)
    }


@pytest.mark.parametrize("kind", ["local_search", "cp_sat"])
@pytest.mark.parametrize("discount", [None, 0.5], ids=["default", "explicit-penalty"])
def test_late_science_keeps_its_merit_by_default(kind, discount):
    config, targets, activity, end = _scenario()
    baseline = create_ditl(config, targets, begin=BEGIN, end=end, activities=[activity])
    assert _science(baseline) == {1: HOUR, 2: 0}

    # Rebuild the inputs independently of the completed dispatch run.
    config, targets, activity, end = _scenario()
    options = {} if discount is None else {"earliness_weight": discount}
    if kind == "cp_sat":
        pytest.importorskip("ortools")
        builder = CpSatScheduler(
            config,
            targets,
            BEGIN,
            end,
            activities=[activity],
            include_passes=False,
            solver_time_limit=10,
            solver_deterministic_time=0.2,
            workers=1,
            seed=2029,
            **options,
        )
    else:
        builder = LocalSearchScheduler(
            config,
            targets,
            BEGIN,
            end,
            activities=[activity],
            include_passes=False,
            time_limit=10,
            max_iterations=50,
            seed=2029,
            **options,
        )
    plan = builder.schedule()
    run = DITL(config, plan=plan, begin=BEGIN, end=end)
    science = _science(run)
    if discount is None:
        assert science == {1: HOUR, 2: 0}
        assert builder.score == 1000 * HOUR
    else:
        # An explicitly requested time preference may trade fixed-merit science.
        assert science == {1: 0, 2: HOUR}
        assert builder.score == 800 * HOUR
