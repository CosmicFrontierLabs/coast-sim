"""Shared momentum-dumping scenario for the momentum planner tests."""

from datetime import datetime, timedelta, timezone

import numpy as np
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
    StoredMomentumConfig,
    ThrusterDumpConfig,
)
from conops.targets import Plan, Pointing

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
HOURS = 6
END = BEGIN + timedelta(hours=HOURS)
T0 = BEGIN.timestamp()
CAPACITY = 1.0
LEVEL = 0.6
DUMP = ThrusterDumpConfig(duration_seconds=120.0)


def momentum_config(**momentum: object) -> MissionConfig:
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
            inertia_tensor_body_kg_m2=np.diag([1000.0, 1200.0, 800.0]),
            attitude_control=AttitudeControlSystem(
                max_slew_rate=0.5,
                slew_acceleration=0.25,
                settle_time=30.0,
                stored_momentum=StoredMomentumConfig(
                    gravity_gradient_enabled=True,
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


def dumping_config() -> MissionConfig:
    return momentum_config(
        capacity_n_m_s=CAPACITY, planning_level=LEVEL, thruster_dump=DUMP
    )


def momentum_targets(config: MissionConfig) -> list[Pointing]:
    """Three targets close together, held long enough to fill the wheels."""
    targets = []
    for i, (ra, dec) in enumerate([(80.0, -20.0), (85.0, -22.0), (75.0, -18.0)]):
        target = Pointing(
            config=config,
            obsid=100 + i,
            name=f"T{i}",
            ra=ra,
            dec=dec,
            merit=90.0 - i,
            ss_min=1200,
            ss_max=3600,
        )
        target.exptime = 4 * 3600
        targets.append(target)
    return targets


def dump_entries(plan: Plan | list) -> list:
    return sorted((e for e in plan if e.desaturation), key=lambda e: float(e.begin))
