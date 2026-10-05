"""Standard benchmark scenarios, each stressing a different part of scheduling.

Every scenario uses one low-Earth orbit from a TLE, Sun and Earth-limb
avoidance, a battery that never limits operations, and seeded random targets:

* :func:`baseline`: a day of 200 targets and one ToO.
* :func:`too_heavy`: eight ToOs across the day, with mixed deadlines and two
  tiers.
* :func:`oversubscribed`: 600 targets, many with deadlines, for far more time
  than the day holds.
* :func:`cadence_and_programs`: monitoring targets that want regular revisits
  and programs with allocated shares of time, with those merit terms on.
* :func:`multi_day`: three days, for replanning cadence and planning time.
* :func:`long_range`: a week with deadlines spread across it, targets the Sun
  covers or uncovers as the week goes on, and ToOs arriving through it, for
  long-range allocation.

:data:`SCENARIOS` names them for scripts.
"""

from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from rust_ephem import EarthLimbConstraint, SunConstraint, TLEEphemeris

from ..config import (
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
from ..config.observation_categories import (
    ObservationCategories,
    ObservationCategory,
)
from ..targets import Pointing
from . import BenchmarkScenario, TOOSpec

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
"""Start of every standard scenario."""

HOUR = 3600.0


def mission_config(
    tle: str | Path,
    begin: datetime,
    end: datetime,
    *,
    seed: int,
    stations: bool = True,
    targets: TargetConfig | None = None,
    categories: list[ObservationCategory] | None = None,
) -> MissionConfig:
    """Return the standard spacecraft over ``[begin, end]``."""
    config = MissionConfig(
        constraint=Constraint(
            sun_constraint=SunConstraint(min_angle=45),
            earth_constraint=EarthLimbConstraint(min_angle=10),
        ),
        ground_stations=GroundStationRegistry.default()
        if stations
        else GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        # No solar panels, so the battery cannot recharge: without these
        # settings it drains to the recharge threshold within days, and
        # emergency charging, which cannot help, takes over the schedule.
        battery=Battery(
            watthour=100_000.0, recharge_threshold=0.0, max_depth_of_discharge=1.0
        ),
        spacecraft_bus=SpacecraftBus(
            attitude_control=AttitudeControlSystem(
                max_slew_rate=1.0, slew_acceleration=0.5, settle_time=30.0
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
        targets=targets or TargetConfig(),
        observation_categories=ObservationCategories(categories=categories or []),
    )
    config.random_seed = seed
    config.constraint.ephem = TLEEphemeris(
        tle=str(tle), begin=begin, end=end, step_size=60
    )
    return config


def random_targets(
    config: MissionConfig,
    count: int,
    seed: int,
    *,
    first_obsid: int = 10000,
    merit: tuple[int, int] = (10, 100),
    exposures: tuple[int, ...] = (600, 1200, 2400),
    deadline_fraction: float = 0.0,
    deadline_hours: tuple[float, float] = (2.0, 12.0),
) -> list[Pointing]:
    """Return targets spread uniformly over the sky.

    Args:
        config: Configuration the targets belong to.
        count: Number of targets.
        seed: Random seed.
        first_obsid: Obsid of the first target; the rest follow.
        merit: Range of base merit, drawn uniformly.
        exposures: Exposure times to draw from, in seconds.
        deadline_fraction: Fraction of targets given a deadline.
        deadline_hours: Range of those deadlines after the start, in hours.
    """
    ephem = config.constraint.ephem
    assert ephem is not None
    start = ephem.timestamp[0]
    start = start if start.tzinfo is not None else start.replace(tzinfo=timezone.utc)
    rng = np.random.default_rng(seed)
    targets = []
    for k in range(count):
        value = float(rng.integers(*merit))
        deadline = None
        if rng.random() < deadline_fraction:
            deadline = start.timestamp() + float(rng.uniform(*deadline_hours)) * HOUR
        target = Pointing(
            config=config,
            ra=float(rng.uniform(0, 360)),
            dec=float(np.degrees(np.arcsin(rng.uniform(-1, 1)))),
            obsid=first_obsid + k,
            name=f"t{first_obsid + k}",
            merit=value,
            fom=value,
            ss_min=300,
            ss_max=1200,
            deadline=deadline,
        )
        target.exptime = int(rng.choice(exposures))
        targets.append(target)
    return targets


def _too(
    obsid: int, submit: float, deadline: float | None, rng: np.random.Generator
) -> TOOSpec:
    return TOOSpec(
        obsid=obsid,
        ra=float(rng.uniform(0, 360)),
        dec=float(np.degrees(np.arcsin(rng.uniform(-1, 1)))),
        merit=500.0,
        exptime=900,
        name=f"ToO {obsid}",
        submit_time=submit,
        deadline=deadline,
    )


def baseline(
    tle: str | Path, *, hours: int = 24, targets: int = 200, seed: int = 1234
) -> BenchmarkScenario:
    """A day of targets and one ToO halfway through with a one-hour deadline."""
    end = BEGIN + timedelta(hours=hours)
    middle = BEGIN.timestamp() + hours * HOUR / 2
    return BenchmarkScenario(
        name=f"baseline: {hours}h, {targets} targets",
        begin=BEGIN,
        end=end,
        make_config=lambda: mission_config(tle, BEGIN, end, seed=seed),
        make_targets=lambda config: random_targets(config, targets, seed),
        toos=[
            TOOSpec(
                obsid=1_000_001,
                ra=105.0,
                dec=10.0,
                merit=500.0,
                exptime=900,
                name="ToO",
                submit_time=middle,
                deadline=middle + HOUR,
            )
        ],
    )


def too_heavy(
    tle: str | Path,
    *,
    hours: int = 24,
    targets: int = 200,
    toos: int = 8,
    seed: int = 1234,
) -> BenchmarkScenario:
    """ToOs spread across the run, with mixed deadlines and two tiers.

    Even obsids are urgent (tier 1, deadlines of 30 minutes to 2 hours); odd
    ones are routine (tier 0, deadlines of 4 to 12 hours).
    """
    end = BEGIN + timedelta(hours=hours)
    rng = np.random.default_rng(seed + 1)
    specs = []
    for k in range(toos):
        obsid = 1_000_000 + k
        submit = BEGIN.timestamp() + (k + 0.5) * hours * HOUR / toos
        urgent = k % 2 == 0
        window = rng.uniform(0.5, 2.0) if urgent else rng.uniform(4.0, 12.0)
        specs.append(_too(obsid, submit, submit + window * HOUR, rng))
    urgent_categories = [
        ObservationCategory(
            name=f"Urgent ToO {obsid}",
            obsid_min=obsid,
            obsid_max=obsid + 1,
            tier=1,
            program="ToO",
        )
        for obsid in range(1_000_000, 1_000_000 + toos, 2)
    ]
    return BenchmarkScenario(
        name=f"ToO-heavy: {hours}h, {targets} targets, {toos} ToOs",
        begin=BEGIN,
        end=end,
        make_config=lambda: mission_config(
            tle,
            BEGIN,
            end,
            seed=seed,
            targets=TargetConfig(urgency_weight=200.0),
            categories=urgent_categories,
        ),
        make_targets=lambda config: random_targets(config, targets, seed),
        toos=specs,
    )


def oversubscribed(
    tle: str | Path,
    *,
    hours: int = 24,
    targets: int = 600,
    seed: int = 1234,
) -> BenchmarkScenario:
    """Far more requested time than the run holds, a third with deadlines."""
    end = BEGIN + timedelta(hours=hours)
    return BenchmarkScenario(
        name=f"over-subscribed: {hours}h, {targets} targets",
        begin=BEGIN,
        end=end,
        make_config=lambda: mission_config(
            tle, BEGIN, end, seed=seed, targets=TargetConfig(urgency_weight=50.0)
        ),
        make_targets=lambda config: random_targets(
            config,
            targets,
            seed,
            exposures=(1200, 2400, 3600),
            deadline_fraction=1 / 3,
            deadline_hours=(2.0, hours * 0.75),
        ),
    )


def cadence_and_programs(
    tle: str | Path,
    *,
    hours: int = 24,
    seed: int = 1234,
) -> BenchmarkScenario:
    """Monitoring targets with a revisit interval, and two survey programs.

    Twelve monitoring targets want a visit every four hours; two survey
    programs of 150 targets each have equal time shares, but program B's
    targets have lower merit, so balance depends on the completion deficit.
    """
    end = BEGIN + timedelta(hours=hours)
    categories = [
        ObservationCategory(
            name="Monitoring",
            obsid_min=20000,
            obsid_max=20012,
            cadence_seconds=4 * HOUR,
            time_share=0.2,
        ),
        ObservationCategory(
            name="Survey A", obsid_min=10000, obsid_max=10150, time_share=0.4
        ),
        ObservationCategory(
            name="Survey B", obsid_min=30000, obsid_max=30150, time_share=0.4
        ),
    ]

    def make_targets(config: MissionConfig) -> list[Pointing]:
        monitoring = random_targets(
            config,
            12,
            seed + 2,
            first_obsid=20000,
            merit=(30, 40),
            exposures=(6 * 600,),
        )
        for target in monitoring:
            target.ss_min = target.ss_max = 600
        survey_a = random_targets(config, 150, seed, merit=(60, 100))
        survey_b = random_targets(
            config, 150, seed + 3, first_obsid=30000, merit=(20, 60)
        )
        return [*monitoring, *survey_a, *survey_b]

    return BenchmarkScenario(
        name=f"cadence and programs: {hours}h",
        begin=BEGIN,
        end=end,
        make_config=lambda: mission_config(
            tle,
            BEGIN,
            end,
            seed=seed,
            targets=TargetConfig(cadence_weight=60.0, completion_deficit_weight=60.0),
            categories=categories,
        ),
        make_targets=make_targets,
    )


def multi_day(
    tle: str | Path,
    *,
    days: int = 3,
    targets: int = 400,
    toos: int = 3,
    seed: int = 1234,
) -> BenchmarkScenario:
    """Several days, with a ToO a day, for replanning over a long run."""
    hours = days * 24
    end = BEGIN + timedelta(hours=hours)
    rng = np.random.default_rng(seed + 4)
    submits = [BEGIN.timestamp() + (k + 0.5) * hours * HOUR / toos for k in range(toos)]
    specs = [
        _too(1_000_000 + k, submit, submit + 2 * HOUR, rng)
        for k, submit in enumerate(submits)
    ]
    return BenchmarkScenario(
        name=f"multi-day: {days} days, {targets} targets",
        begin=BEGIN,
        end=end,
        make_config=lambda: mission_config(tle, BEGIN, end, seed=seed),
        make_targets=lambda config: random_targets(config, targets, seed),
        toos=specs,
    )


def _along_sun_path(
    config: MissionConfig, count: int, elongations: tuple[float, float], seed: int
) -> list[tuple[float, float]]:
    """Positions along the Sun's path over the run, at ``elongations`` from it.

    Positive elongations lie ahead of the Sun, which closes in on them; negative
    ones behind it, which it moves away from. Each position is offset a few
    degrees across the path.
    """
    ephem = config.constraint.ephem
    assert ephem is not None

    def unit(ra: float, dec: float) -> np.ndarray:
        r, d = np.radians(ra), np.radians(dec)
        return np.array([np.cos(d) * np.cos(r), np.cos(d) * np.sin(r), np.sin(d)])

    first = unit(float(ephem.sun_ra_deg[0]), float(ephem.sun_dec_deg[0]))
    last = unit(float(ephem.sun_ra_deg[-1]), float(ephem.sun_dec_deg[-1]))
    normal = np.cross(first, last)
    normal /= np.linalg.norm(normal)
    ahead = np.cross(normal, first)
    rng = np.random.default_rng(seed)
    positions = []
    for _ in range(count):
        angle = np.radians(rng.uniform(*elongations))
        across = np.radians(rng.uniform(-4.0, 4.0))
        vector = np.cos(angle) * first + np.sin(angle) * ahead
        vector = np.cos(across) * vector + np.sin(across) * normal
        vector /= np.linalg.norm(vector)
        ra = float(np.degrees(np.arctan2(vector[1], vector[0])) % 360.0)
        dec = float(np.degrees(np.arcsin(np.clip(vector[2], -1.0, 1.0))))
        positions.append((ra, dec))
    return positions


def long_range(
    tle: str | Path,
    *,
    days: int = 7,
    targets: int = 300,
    edge: int = 30,
    toos: int = 10,
    seed: int = 1234,
) -> BenchmarkScenario:
    """A week in which when each request can be observed changes.

    Besides random targets, half of them with deadlines spread across the
    week, two programs sit by the Sun's 45-degree avoidance zone: "Early"
    targets just ahead of the Sun, which it covers as the week goes on, and
    "Late" targets just behind it, which it uncovers. A scheduler that only
    looks a few hours ahead spends the early days on targets that could have
    waited, and the Early targets miss their chance.

    ToOs arrive evenly through the week: alternately urgent, due 2 to 6 hours
    after they arrive, and routine, due 1 to 3 days after.
    """
    hours = days * 24
    end = BEGIN + timedelta(hours=hours)
    categories = [
        ObservationCategory(
            name="Early", obsid_min=40000, obsid_max=40000 + edge, program="Early"
        ),
        ObservationCategory(
            name="Late", obsid_min=50000, obsid_max=50000 + edge, program="Late"
        ),
    ]

    def make_targets(config: MissionConfig) -> list[Pointing]:
        pool = random_targets(
            config,
            targets,
            seed,
            deadline_fraction=0.5,
            deadline_hours=(12.0, hours - 6.0),
        )
        groups = [
            (40000, (46.0, 52.0), seed + 5),
            (50000, (-44.0, -38.0), seed + 6),
        ]
        for first_obsid, elongations, group_seed in groups:
            rng = np.random.default_rng(group_seed)
            for k, (ra, dec) in enumerate(
                _along_sun_path(config, edge, elongations, group_seed)
            ):
                value = float(rng.integers(40, 70))
                target = Pointing(
                    config=config,
                    ra=ra,
                    dec=dec,
                    obsid=first_obsid + k,
                    name=f"t{first_obsid + k}",
                    merit=value,
                    fom=value,
                    ss_min=300,
                    ss_max=1200,
                )
                target.exptime = 2400
                pool.append(target)
        return pool

    rng = np.random.default_rng(seed + 7)
    specs = []
    for k in range(toos):
        submit = BEGIN.timestamp() + (k + 0.5) * hours * HOUR / toos
        window = rng.uniform(2.0, 6.0) if k % 2 == 0 else rng.uniform(24.0, 72.0)
        specs.append(_too(1_000_000 + k, submit, submit + window * HOUR, rng))
    return BenchmarkScenario(
        name=f"long-range: {days} days, {targets} targets, {toos} ToOs",
        begin=BEGIN,
        end=end,
        make_config=lambda: mission_config(
            tle, BEGIN, end, seed=seed, categories=categories
        ),
        make_targets=make_targets,
        toos=specs,
    )


SCENARIOS: dict[str, Callable[..., BenchmarkScenario]] = {
    "baseline": baseline,
    "too-heavy": too_heavy,
    "oversubscribed": oversubscribed,
    "cadence": cadence_and_programs,
    "multi-day": multi_day,
    "long-range": long_range,
}
"""Standard scenarios by name; each takes the TLE path first."""
