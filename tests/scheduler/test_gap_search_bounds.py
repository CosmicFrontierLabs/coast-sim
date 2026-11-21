"""Later feasible departures and horizon-safe return forecasts."""

from datetime import timedelta

import numpy as np
import pytest
from rust_ephem import SunConstraint

from conops.common import ObsType, SlewAlgorithm
from conops.common.vector import (
    _quat_mul,
    attitude_to_quat,
    quat_to_attitude,
    quaternion_attitude_delta,
)
from conops.config import ObservationTiming
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.schedulers.priority_scheduler import StartState
from conops.simulation.attitude import AttitudeTrajectory
from conops.targets import (
    Plan,
    PlanEntry,
    Pointing,
    ScheduledActivity,
)

from .planning_scenario import BEGIN, MIN, T0, collected_seconds, replay
from .planning_scenario import make_config as _config


@pytest.mark.parametrize("case", ["fast-body-axes", "fixed-settle"])
def test_the_fixed_pose_bound_never_exceeds_a_feasible_slew(case: str) -> None:
    config = _config(2, open_sky=case == "fast-body-axes")
    limits = config.spacecraft_bus.attitude_control
    if case == "fast-body-axes":
        limits.max_slew_rate_body = (0.01, 1.0, 1.0)
        limits.slew_acceleration_body = (0.01, 1.0, 1.0)
        limits.settle_time = 0.0
        first = (0.0, 0.0, 0.0)
        middle_q = _quat_mul(np.asarray((0.0, 0.0, 1.0, 0.0)), attitude_to_quat(*first))
        last_q = _quat_mul(np.asarray((0.0, 0.0, 0.0, 1.0)), middle_q)
        middle, last = quat_to_attitude(middle_q), quat_to_attitude(last_q)
        # Two executable fast-axis turns reach the same pose as a slow-axis turn.
        outbound = AttitudeTrajectory.turn(T0, first, middle, limits)
        inbound = AttitudeTrajectory.turn(outbound.end, middle, last, limits)
        feasible_duration = inbound.end - T0
        assert feasible_duration < 400.0
    else:
        limits.settle_time = 5.0
        first, last = (240.0, 20.0, 0.0), (110.0, -20.0, 0.0)
        distance, axis = quaternion_attitude_delta(*first, *last)
        feasible_duration = limits.slew_time(distance, axis)
    scheduler = PriorityScheduler(config, [], BEGIN, BEGIN + timedelta(hours=2))
    assert scheduler._shortest_slew_seconds(first, last) <= feasible_duration


class TestGapSearchBound:
    @pytest.mark.parametrize(
        "planner_class", [PriorityScheduler, LocalSearchScheduler, CpSatScheduler]
    )
    def test_arrival_past_ephemeris_does_not_forecast_a_return(
        self, planner_class: type[PriorityScheduler]
    ) -> None:
        config = _config(1)
        config.constraint.sun_constraint = SunConstraint(min_angle=1)
        config.constraint.earth_constraint = None
        config.payload.observation_timing = ObservationTiming()
        config.battery.planning_level = None
        acs = config.spacecraft_bus.attitude_control
        acs.slew_algorithm = SlewAlgorithm.CONSTRAINT_AVOIDING
        acs.max_slew_rate = 0.1
        initial = (105.0, 10.0, 0.0)
        target = Pointing(
            config=config,
            obsid=702,
            ra=285,
            dec=-10,
            roll_deg=0,
            ss_min=60,
            ss_max=60,
            exptime=60,
        )
        hold = ScheduledActivity(
            name="Final required hold",
            obsid=991,
            begin=T0 + 55 * MIN,
            duration_seconds=5 * MIN,
            fixed=True,
        )
        # A locked observation supplies the fixed return pose. The following
        # inherited engineering hold preserves the original full-hold check.
        locked = PlanEntry(
            config=config,
            name="Locked return pose",
            obsid=703,
            obstype=ObsType.AT,
            ra=initial[0],
            dec=initial[1],
            roll=initial[2],
            begin=T0 + 54 * MIN,
            end=T0 + 55 * MIN,
            collection_begin=T0 + 54 * MIN,
            collection_end=T0 + 55 * MIN,
            ss_min=MIN,
        )
        begin, end = BEGIN + timedelta(minutes=50), BEGIN + timedelta(hours=1)
        options = {} if planner_class is PriorityScheduler else {"time_limit": 0}
        plan = planner_class(
            config,
            [target],
            begin,
            end,
            include_passes=False,
            start_state=StartState(time=begin.timestamp(), attitude=initial),
            activities=[hold],
            locked=[locked],
            **options,
        ).schedule()
        assert not [e for e in plan if e.obsid == target.obsid]
        activity = next(e for e in plan if e.obsid == hold.obsid)
        assert activity.activity_hold_begin == hold.begin
        assert activity.activity_hold_end == hold.begin + hold.duration_seconds
        replayed = Plan.model_validate_json(plan.model_dump_json())
        run = replay(config, replayed, begin, end, pose=initial)
        assert collected_seconds(run) == MIN
