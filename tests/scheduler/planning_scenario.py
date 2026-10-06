"""Shared real-orbit scenario for planner and replanning tests."""

from datetime import datetime, timedelta, timezone

from rust_ephem import EarthLimbConstraint, SunConstraint, TLEEphemeris

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
from conops.targets import Pointing

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
    return config


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
