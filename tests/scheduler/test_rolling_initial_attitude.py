"""An explicitly initialized spacecraft must be the origin of its first plan."""

from datetime import timedelta

import pytest

from conops.config import ObservationTiming, SpacecraftFrame
from conops.ditl import RollingHorizonDITL
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.targets import Pointing

from .planning_scenario import BEGIN, make_config


@pytest.mark.parametrize("declared_frame", [False, True])
@pytest.mark.parametrize(
    "planner", [PriorityScheduler, LocalSearchScheduler, CpSatScheduler]
)
def test_first_plan_uses_the_executed_initial_pose(
    declared_frame: bool, planner: type[PriorityScheduler]
) -> None:
    config = make_config(1)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.payload.observation_timing = ObservationTiming()
    config.battery.planning_level = None
    config.spacecraft_bus.attitude_control.max_slew_rate = 0.01
    if declared_frame:
        config.spacecraft_bus.frame = SpacecraftFrame(
            boresight=(0, 0, 1), sun_side=(1, 0, 0)
        )
    target = Pointing(
        config=config,
        obsid=701,
        ra=105,
        dec=10,
        roll_deg=0,
        ss_min=300,
        ss_max=300,
        exptime=300,
    )
    run = RollingHorizonDITL(
        config,
        [target],
        begin=BEGIN,
        end=BEGIN + timedelta(minutes=10),
        scheduler=planner,
        scheduler_options={} if planner is PriorityScheduler else {"time_limit": 0},
        include_passes=False,
        horizon=timedelta(minutes=10),
        replan_interval=timedelta(hours=1),
        commit_lead_time=timedelta(0),
    )
    run.step_size = 60
    run.acs.ra, run.acs.dec, run.acs.roll = target.target_body_attitude(0)
    assert run.calc()
    assert sum(h.collection_seconds or 0 for h in run.telemetry.housekeeping) == 300
    entry = next(e for e in run.plan if e.obsid == target.obsid)
    assert entry.slewtime == 0
    assert float(entry.collection_begin) == BEGIN.timestamp()
    assert run.validate_plan_matches_execution() == []
