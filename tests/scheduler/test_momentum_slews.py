"""The planners' forecast with the momentum of slews counted: a slew in
which the wheels would reach the capacity is flown slower, and an
observation whose slew fits at no rate is not planned."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from rust_ephem import EarthLimbConstraint, SunConstraint, TLEEphemeris

from conops import DITL
from conops.common import ObsType
from conops.common.vector import (
    attitude_to_quat,
    quaternion_attitude_delta,
    quaternion_to_rotation_matrix,
)
from conops.config import (
    AttitudeControlSystem,
    Battery,
    Constraint,
    GroundStationRegistry,
    MissionConfig,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
    StoredMomentumConfig,
    ThrusterDumpConfig,
)
from conops.schedulers import PriorityScheduler
from conops.targets import Pointing

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
HOURS = 2
END = BEGIN + timedelta(hours=HOURS)
INERTIA = 1000.0
RATE = 0.5
TRANSIENT = INERTIA * np.deg2rad(RATE)
CAPACITY = 6.5


def _config(**momentum: object) -> MissionConfig:
    ephem = TLEEphemeris(tle="examples/example.tle", begin=BEGIN, end=END, step_size=10)
    config = MissionConfig(
        constraint=Constraint(
            sun_constraint=SunConstraint(min_angle=45),
            earth_constraint=EarthLimbConstraint(min_angle=10),
        ),
        ground_stations=GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        battery=Battery(watthour=100_000.0),
        spacecraft_bus=SpacecraftBus(
            inertia_tensor_body_kg_m2=np.diag([INERTIA, INERTIA, INERTIA]),
            attitude_control=AttitudeControlSystem(
                max_slew_rate=RATE,
                slew_acceleration=0.25,
                settle_time=30.0,
                stored_momentum=StoredMomentumConfig(
                    slew_momentum_enabled=True,
                    capacity_n_m_s=CAPACITY,
                    **momentum,  # type: ignore[arg-type]
                ),
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
    )
    config.constraint.ephem = ephem
    return config


def _targets(config: MissionConfig) -> list[Pointing]:
    """Two targets far apart: each slew reaches the full rate."""
    targets = []
    for i, (ra, dec) in enumerate([(80.0, -20.0), (200.0, 30.0)]):
        target = Pointing(
            config=config,
            obsid=100 + i,
            name=f"T{i}",
            ra=ra,
            dec=dec,
            merit=90.0 - i,
            ss_min=600,
            ss_max=1800,
        )
        target.exptime = 1800
        targets.append(target)
    return targets


def _preloaded(config: MissionConfig, norm: float) -> tuple[float, float, float]:
    """A stored momentum of ``norm`` perpendicular to the first planned slew's
    axis, in inertial coordinates: what the wheels hold along that slew is
    then the root of the squares of it and the body's own, whatever the
    direction of the slew."""
    dry = PriorityScheduler(config, _targets(config), BEGIN, END, step_size=60)
    dry.schedule()
    slew = next(b.slew for b in dry.timeline if b.slew is not None)
    _, axis = quaternion_attitude_delta(
        slew.startra,
        slew.startdec,
        slew.startroll,
        slew.endra,
        slew.enddec,
        slew.endroll,
    )
    axis = np.asarray(axis, dtype=float)
    other = (
        np.array([1.0, 0.0, 0.0]) if abs(axis[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    )
    perpendicular_body = np.cross(axis, other)
    perpendicular_body /= np.linalg.norm(perpendicular_body)
    rotation = quaternion_to_rotation_matrix(
        attitude_to_quat(slew.startra, slew.startdec, slew.startroll)
    )
    eci = norm * (rotation.T @ perpendicular_body)
    return float(eci[0]), float(eci[1]), float(eci[2])


def _science(plan: list) -> list:
    return sorted(
        (e for e in plan if e.obstype not in (ObsType.ACTIVITY, ObsType.GSP)),
        key=lambda e: float(e.begin),
    )


class TestPriorityScheduler:
    def test_slews_in_which_the_wheels_would_reach_the_capacity_are_flown_slower(
        self,
    ) -> None:
        """Each slew at full rate carries more than the capacity: the planner
        caps each at half, where it fits, plans again, and the plan's
        entries carry the fraction. The forecast of what the wheels hold
        stays within the capacity, and the planned simulation agrees."""
        config = _config()
        scheduler = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=60
        )
        plan = scheduler.schedule()
        science = _science(list(plan))
        assert len(science) == 2
        assert all(e.slew_rate_fraction == 0.5 for e in science)
        assert scheduler._slew_caps == {100: 0.5, 101: 0.5}
        capped = [
            e.description
            for e in scheduler.log.events
            if "flown at 0.5 of the slew rate" in e.description
        ]
        assert len(capped) == 2
        assert scheduler._momentum is not None
        times, demands = scheduler._momentum.demand(scheduler.timeline)
        assert demands.max() < CAPACITY
        assert demands.max() > 0.4 * TRANSIENT
        assert not any(
            "reaches the 6.5 N m s capacity" in e.description
            and "flown at" not in e.description
            for e in scheduler.log.events
        )
        # A full-rate estimate of the same slew is shorter.
        full = Pointing(config=config, obsid=9, ra=80.0, dec=-20.0, roll=0.0)
        assert science[0].slewtime > full.calc_slewtime(0.0, 0.0, 0.0)
        ditl = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=plan,
            begin=BEGIN,
            end=END,
        )
        ditl.step_size = 10
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
        executed = np.array(
            [
                v
                for v in ditl.telemetry.housekeeping.actuator_momentum_norm_n_m_s
                if v is not None
            ]
        )
        assert executed.max() < CAPACITY
        assert executed.max() == pytest.approx(demands.max(), rel=0.05)

    def test_an_observation_whose_slew_fits_at_no_rate_is_not_planned(self) -> None:
        config = _config(slew_rate_fractions=(1.0,))
        scheduler = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=60
        )
        plan = scheduler.schedule()
        assert _science(list(plan)) == []
        assert scheduler._momentum_rejected == {100, 101}
        rejected = [
            e.description
            for e in scheduler.log.events
            if "not planned: its slew asks" in e.description
        ]
        assert len(rejected) == 2
        assert all("at 1 of the slew rate" in e for e in rejected)

    def test_without_thrusters_the_overload_is_logged_and_the_plan_stands(
        self,
    ) -> None:
        """At the slowest configured rate the first slew still carries more
        than the capacity from the wheels as forecast but fits from empty,
        and there are no thrusters: the plan stands, with the overload
        logged, and the capacity crossing with it."""
        config = _config(slew_rate_fractions=(1.0, 0.5))
        scheduler = PriorityScheduler(
            config,
            _targets(config),
            BEGIN,
            END,
            step_size=60,
            start_momentum_eci_n_m_s=_preloaded(config, 0.9 * CAPACITY),
        )
        plan = scheduler.schedule()
        assert len(_science(list(plan))) == 2
        assert scheduler._slew_caps[100] == 0.5
        logged = [
            e.description
            for e in scheduler.log.events
            if "no thrusters to empty them first" in e.description
        ]
        assert len(logged) == 1
        assert "in the slew to 100" in logged[0]
        assert any(
            "no thrusters to dump it" in e.description for e in scheduler.log.events
        )

    def test_a_dump_goes_before_the_slew_where_it_fits_from_empty(self) -> None:
        """With thrusters, the same overload has the wheels emptied before
        the observation: a dump is reserved before it, and the slew at the
        slowest rate then fits."""
        config = _config(
            slew_rate_fractions=(1.0, 0.5),
            thruster_dump=ThrusterDumpConfig(duration_seconds=120.0),
            planning_level=0.999,
        )
        scheduler = PriorityScheduler(
            config,
            _targets(config),
            BEGIN,
            END,
            step_size=60,
            start_momentum_eci_n_m_s=_preloaded(config, 0.9 * CAPACITY),
        )
        plan = scheduler.schedule()
        science = _science(list(plan))
        dumps = sorted((e for e in plan if e.desaturation), key=lambda e: e.begin)
        assert len(science) == 2 and len(dumps) == 1
        assert float(dumps[0].end) <= float(science[0].begin)
        assert scheduler._slew_caps == {100: 0.5, 101: 0.5}
        reserved = [
            e.description
            for e in scheduler.log.events
            if "whose slew would reach the capacity" in e.description
        ]
        assert len(reserved) == 1
        assert "in the slew to 100" in reserved[0]
        assert scheduler._momentum is not None
        _, demands = scheduler._momentum.demand(scheduler.timeline)
        assert demands.max() < CAPACITY
        assert demands.max() > 0.4 * TRANSIENT
