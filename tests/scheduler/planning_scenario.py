"""Shared real-orbit scenario for scheduler and replanning tests."""

from datetime import datetime, timedelta, timezone

import pytest
from rust_ephem import EarthLimbConstraint, SunConstraint, TLEEphemeris

from conops import DITL
from conops.common import ObsType
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
    TargetConfig,
)
from conops.config.observation_categories import (
    ObservationCategories,
    ObservationCategory,
)
from conops.schedulers import CpSatScheduler, LocalSearchScheduler, PriorityScheduler
from conops.targets import Plan, Pointing

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
T0 = BEGIN.timestamp()
MIN = 60
HOUR = 3600


def make_config(
    hours: int,
    *,
    stations: bool = False,
    weights: TargetConfig | None = None,
    categories: list[ObservationCategory] | None = None,
    open_sky: bool = False,
) -> MissionConfig:
    ephem = TLEEphemeris(
        tle="examples/example.tle",
        begin=BEGIN,
        end=BEGIN + timedelta(hours=hours),
        step_size=60,
    )
    config = MissionConfig(
        constraint=Constraint(
            sun_constraint=SunConstraint(min_angle=45),
            earth_constraint=EarthLimbConstraint(min_angle=10),
        ),
        ground_stations=GroundStationRegistry.default()
        if stations
        else GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        battery=Battery(watthour=100_000.0),
        spacecraft_bus=SpacecraftBus(
            attitude_control=AttitudeControlSystem(
                max_slew_rate=1.0, slew_acceleration=0.5, settle_time=30.0
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
        targets=weights or TargetConfig(),
        observation_categories=ObservationCategories(categories=categories or []),
    )
    config.random_seed = 1234
    config.constraint.ephem = ephem
    if open_sky:  # no sun or earth limits, so every target stays visible
        config.constraint.sun_constraint = config.constraint.earth_constraint = None
    return config


def planner(kind: str) -> tuple[type, dict]:
    """The planner class for ``kind`` and fast, deterministic search options."""
    if kind == "priority":
        return PriorityScheduler, {}
    if kind == "local_search":
        return LocalSearchScheduler, {"time_limit": 0.0}
    pytest.importorskip("ortools")
    return CpSatScheduler, {
        "time_limit": 0.0,
        "solver_time_limit": 2.0,
        "solver_deterministic_time": 0.05,
        "workers": 1,
        "seed": 1,
    }


def make_target(
    config: MissionConfig,
    obsid: int,
    ra: float,
    dec: float,
    *,
    merit: float = 50.0,
    minutes: float = 20,
    snapshot: float | None = None,
    ss_min: float | None = None,
    deadline: float | None = None,
    earliest_start: float | None = None,
) -> Pointing:
    snapshot = minutes if snapshot is None else snapshot
    target = Pointing(
        config=config,
        obsid=obsid,
        name=f"T{obsid}",
        ra=ra,
        dec=dec,
        merit=merit,
        fom=merit,
        ss_min=(snapshot if ss_min is None else ss_min) * MIN,
        ss_max=snapshot * MIN,
        deadline=deadline,
        earliest_start=earliest_start,
    )
    target.exptime = minutes * MIN
    return target


# A patch of sky this orbit keeps visible throughout, and some targets that are
# constrained for part of every orbit.
PATCH = [(105.0, 10.0), (110.0, 10.0), (115.0, 10.0), (105.0, 0.0), (110.0, 20.0)]
CONSTRAINED = [(30.0, 40.0), (300.0, -50.0), (170.0, 60.0), (250.0, 10.0)]


def patch_targets(
    config: MissionConfig,
    n: int = len(PATCH),
    *,
    obsid: int = 100,
    merit: float = 90,
    **kw,
) -> list[Pointing]:
    """``n`` targets on the visible patch, merit descending by one."""
    return [
        make_target(config, obsid + i, ra, dec, merit=merit - i, **kw)
        for i, (ra, dec) in enumerate(PATCH[:n])
    ]


def science_entries(plan: Plan | list) -> list:
    return [e for e in plan if e.obstype not in (ObsType.GSP, ObsType.ACTIVITY)]


def activity_entries(plan: Plan | list) -> list:
    return [e for e in plan if e.obstype == ObsType.ACTIVITY]


def collected_seconds(run, obsid: int | None = None, *, science: bool = False) -> float:
    """Seconds collected in a run, optionally for one obsid or excluding calibration."""
    return sum(
        h.collection_seconds or 0
        for h in run.telemetry.housekeeping
        if (obsid is None or h.obsid == obsid) and not (science and h.calibration)
    )


def replay(config, plan, begin=BEGIN, end=None, *, step=60, pose=None) -> DITL:
    """Execute ``plan`` and check it flew exactly as planned."""
    run = DITL(config, plan=plan, begin=begin, end=end or config.constraint.ephem.end)
    run.step_size = step
    if pose is not None:
        run.acs.ra, run.acs.dec, run.acs.roll = pose
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
    return run
