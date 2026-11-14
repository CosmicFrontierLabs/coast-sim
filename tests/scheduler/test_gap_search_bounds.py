"""Later feasible departures and horizon-safe return forecasts."""

from datetime import timedelta
from unittest.mock import patch

import numpy as np
import pytest
from rust_ephem import SunConstraint

from conops import DITL
from conops.common import ObsType, SlewAlgorithm
from conops.common.ephemeris import sun_vector_change_deg
from conops.common.vector import _quat_mul, attitude_to_quat, quat_to_attitude
from conops.config import MissionConfig, ObservationTiming
from conops.config.acs import ThermalSettlingStep
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.schedulers.priority_scheduler import StartState
from conops.simulation.attitude import AttitudeTrajectory
from conops.targets import (
    ObservationSequence,
    Plan,
    PlanEntry,
    Pointing,
    ScheduledActivity,
    SequenceStep,
)

from .planning_scenario import BEGIN, MIN, T0
from .planning_scenario import make_config as _config


def test_fixed_pose_bound_allows_a_faster_route_on_other_body_axes() -> None:
    config = _config(2)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    limits = config.spacecraft_bus.attitude_control
    limits.max_slew_rate_body = (0.01, 1.0, 1.0)
    limits.slew_acceleration_body = (0.01, 1.0, 1.0)
    limits.settle_time = 0.0
    limits.thermal_settling = []
    first = (0.0, 0.0, 0.0)
    middle_q = _quat_mul(np.asarray((0.0, 0.0, 1.0, 0.0)), attitude_to_quat(*first))
    last_q = _quat_mul(np.asarray((0.0, 0.0, 0.0, 1.0)), middle_q)
    middle, last = quat_to_attitude(middle_q), quat_to_attitude(last_q)
    # Two executable fast-axis turns reach the same pose as a slow-axis turn.
    outbound = AttitudeTrajectory.turn(T0, first, middle, limits)
    inbound = AttitudeTrajectory.turn(outbound.end, middle, last, limits)
    feasible_duration = inbound.end - T0
    assert feasible_duration < 400.0
    scheduler = PriorityScheduler(config, [], BEGIN, BEGIN + timedelta(hours=2))
    assert scheduler._shortest_slew_seconds(first, last) <= feasible_duration


def test_fixed_pose_bound_includes_categories_skipping_the_settle_table() -> None:
    config = _config(2)
    limits = config.spacecraft_bus.attitude_control
    limits.settle_time = 5.0
    limits.thermal_settling = [
        ThermalSettlingStep(max_sun_vector_change_deg=1.0, settle_seconds=1800.0),
        ThermalSettlingStep(max_sun_vector_change_deg=180.0, settle_seconds=10.0),
    ]
    first, last = (240.0, 20.0, 0.0), (110.0, -20.0, 0.0)
    from conops.common.vector import quaternion_attitude_delta

    distance, axis = quaternion_attitude_delta(*first, *last)
    # No Sun-vector change selects the category's constant settling fallback.
    feasible_duration = limits.slew_time(distance, axis, None)
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
        run = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=Plan.model_validate_json(plan.model_dump_json()),
            begin=begin,
            end=end,
        )
        run.step_size = MIN
        run.acs.ra, run.acs.dec, run.acs.roll = initial
        assert run.calc()
        assert run.validate_plan_matches_execution() == []
        assert sum(h.collection_seconds or 0 for h in run.telemetry.housekeeping) == MIN

    @pytest.mark.parametrize("boundary", ["activity", "deadline"])
    @pytest.mark.parametrize("sequence", [False, True])
    def test_a_later_departure_can_cross_to_a_shorter_settle(
        self, boundary: str, sequence: bool
    ) -> None:
        config = _config(2)
        config.constraint.sun_constraint = None
        config.constraint.earth_constraint = None
        config.payload.observation_timing = ObservationTiming()
        config.battery.planning_level = None
        start, finish = (240.0, 20.0, 0.0), (110.0, -20.0, 0.0)
        ephem = config.constraint.ephem
        assert ephem is not None
        changes = [
            sun_vector_change_deg(ephem, time, start, finish) for time in (T0, T0 + MIN)
        ]
        assert changes[0] > changes[1]
        config.spacecraft_bus.attitude_control.thermal_settling = [
            ThermalSettlingStep(
                max_sun_vector_change_deg=sum(changes) / 2, settle_seconds=60
            ),
            ThermalSettlingStep(max_sun_vector_change_deg=180, settle_seconds=1800),
        ]
        target = Pointing(
            config=config,
            obsid=701,
            ra=finish[0],
            dec=finish[1],
            roll_deg=finish[2],
            ss_min=600,
            ss_max=600,
            exptime=600,
            deadline=T0 + 1000 if boundary == "deadline" else None,
            sequence=ObservationSequence(
                steps=[
                    SequenceStep(name="first", duration_seconds=300),
                    SequenceStep(name="second", duration_seconds=300),
                ]
            )
            if sequence
            else None,
        )
        activities = (
            [
                ScheduledActivity(
                    name="Required hold",
                    obsid=990,
                    begin=T0 + 20 * MIN,
                    duration_seconds=40 * MIN,
                    fixed=True,
                )
            ]
            if boundary == "activity"
            else []
        )
        end = BEGIN + timedelta(minutes=60)
        plan = PriorityScheduler(
            config,
            [target],
            BEGIN,
            end,
            include_passes=False,
            start_state=StartState(time=T0, attitude=start),
            activities=activities,
        ).schedule()
        science = [e for e in plan if e.obsid == target.obsid]
        assert len(science) == (2 if sequence else 1)
        assert float(science[0].begin) == T0 + MIN
        assert science[0].settle_seconds == 60
        replay = Plan.model_validate_json(plan.model_dump_json())
        run = DITL(config=config, ephem=ephem, plan=replay, begin=BEGIN, end=end)
        run.step_size = MIN
        run.acs.ra, run.acs.dec, run.acs.roll = start
        assert run.calc()
        assert run.validate_plan_matches_execution() == []
        assert sum(h.collection_seconds or 0 for h in run.telemetry.housekeeping) == 600
        if activities:
            hold = next(e for e in run.plan if e.obsid == 990)
            assert hold.activity_hold_begin == T0 + 20 * MIN
            assert hold.activity_hold_end == T0 + 60 * MIN

    def test_the_plan_is_the_one_an_exhaustive_scan_finds(self) -> None:
        """The gap search stops stepping through a window once not even the
        shortest possible slew leaves room for a snapshot; the plan is the
        plan found by trying every start, on a program under a thermal
        settling table, where the long settles make that scan long."""
        from conops.benchmark.scenarios import program

        base = program("examples/example.tle", hours=12)

        def with_table() -> MissionConfig:
            config = base.make_config()
            acs = config.spacecraft_bus.attitude_control
            acs.settle_time = 300.0
            acs.thermal_settling = [
                ThermalSettlingStep(
                    max_sun_vector_change_deg=3.0, settle_seconds=600.0
                ),
                ThermalSettlingStep(
                    max_sun_vector_change_deg=30.0, settle_seconds=1500.0
                ),
                ThermalSettlingStep(
                    max_sun_vector_change_deg=180.0, settle_seconds=2100.0
                ),
            ]
            return config

        def plan(exhaustive: bool) -> list[tuple[int, float, float, float]]:
            config = with_table()
            scheduler = PriorityScheduler(
                config,
                base.make_targets(config),
                base.begin,
                base.end,
                activities=base.activities,
            )
            if exhaustive:
                with patch.object(
                    PriorityScheduler, "_shortest_slew_seconds", return_value=0.0
                ):
                    entries = scheduler.schedule().entries
            else:
                entries = scheduler.schedule().entries
            return [
                (
                    int(e.obsid),
                    float(e.begin),
                    float(e.collection_begin or 0),
                    float(e.end),
                )
                for e in entries
            ]

        bounded = plan(exhaustive=False)
        assert len(bounded) > 5
        assert bounded == plan(exhaustive=True)
