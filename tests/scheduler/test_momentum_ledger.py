"""The planners within the wheels' momentum: the forecast along a plan and
the thruster dumps reserved where it reaches the level."""

from datetime import datetime, timedelta, timezone

import numpy as np
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
    StoredMomentumConfig,
    ThrusterDumpConfig,
)
from conops.ditl import RollingHorizonDITL
from conops.schedulers import LocalSearchScheduler, PriorityScheduler
from conops.targets import Plan, Pointing

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
HOURS = 6
END = BEGIN + timedelta(hours=HOURS)
T0 = BEGIN.timestamp()
CAPACITY = 1.0
LEVEL = 0.6
DUMP = ThrusterDumpConfig(duration_seconds=120.0)


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


def _dumping() -> MissionConfig:
    return _config(capacity_n_m_s=CAPACITY, planning_level=LEVEL, thruster_dump=DUMP)


def _targets(config: MissionConfig) -> list[Pointing]:
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


def _dumps(plan: Plan | list) -> list:
    return sorted((e for e in plan if e.desaturation), key=lambda e: float(e.begin))


def _science(plan: Plan | list) -> list:
    return sorted(
        (e for e in plan if e.obstype not in (ObsType.ACTIVITY, ObsType.GSP)),
        key=lambda e: float(e.begin),
    )


class TestPriorityScheduler:
    def test_dumps_are_reserved_where_the_forecast_reaches_the_level(self) -> None:
        """Each pass reserves a dump where the forecast along the plan reaches
        the level; the final plan's forecast never passes the level by more
        than the dump's own build-up, science runs up to each dump, and the
        executed plan matches the forecast."""
        config = _dumping()
        scheduler = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=60
        )
        plan = scheduler.schedule()
        dumps = _dumps(plan)
        assert len(dumps) >= 4
        assert [e.obsid for e in dumps] == [d.obsid for d in scheduler.dumps]
        assert all(float(e.end) - float(e.begin) == pytest.approx(120.0) for e in dumps)
        assert all(e.slewtime == 0 for e in dumps)
        reserved = [
            e for e in scheduler.log.events if "thruster dump reserved" in e.description
        ]
        assert len(reserved) == len(dumps)
        times, norms = scheduler._momentum.profile(scheduler.timeline)
        assert norms.max() < CAPACITY
        assert norms.max() >= LEVEL
        # Each dump begins at the level and the wheels are empty after it.
        for dump in dumps:
            before = norms[times < float(dump.begin)][-1]
            after = norms[times > float(dump.end)][0]
            assert before >= LEVEL - 0.05
            assert after < 0.05
        # Science before a dump ends at the dump and takes its attitude.
        science = _science(plan)
        assert len(science) >= 4
        adjoining = [
            (s, d)
            for d in dumps
            for s in science
            if abs(float(s.end) - float(d.begin)) < 1e-6
        ]
        assert adjoining
        for entry, dump in adjoining:
            assert (float(dump.ra), float(dump.dec)) == pytest.approx(
                (float(entry.ra), float(entry.dec))
            )
        replay = Plan.model_validate_json(plan.model_dump_json())
        ditl = DITL(
            config=_dumping(),
            ephem=config.constraint.ephem,
            plan=replay,
            begin=BEGIN,
            end=END,
        )
        ditl.step_size = 10
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
        executed = np.array(
            ditl.telemetry.housekeeping.stored_momentum_norm_n_m_s, dtype=float
        )
        assert executed.max() == pytest.approx(norms.max(), abs=0.05)
        unloads = [
            e for e in ditl.log.events if "Thruster dump unloads" in e.description
        ]
        assert len(unloads) == len(dumps)

    def test_without_thrusters_the_forecast_past_the_capacity_is_logged(self) -> None:
        config = _config(capacity_n_m_s=CAPACITY)
        scheduler = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=60
        )
        plan = scheduler.schedule()
        assert _dumps(plan) == [] and scheduler.dumps == []
        assert any(
            "no thrusters to dump it" in e.description for e in scheduler.log.events
        )
        _, norms = scheduler._momentum.profile(scheduler.timeline)
        assert norms.max() > CAPACITY

    def test_without_tracking_nothing_changes(self) -> None:
        config = _config()
        config.spacecraft_bus.attitude_control.stored_momentum = StoredMomentumConfig()
        scheduler = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=60
        )
        plan = scheduler.schedule()
        assert scheduler._momentum is None and _dumps(plan) == []


class TestLocalSearch:
    def test_the_search_keeps_within_the_level(self) -> None:
        config = _dumping()
        scheduler = LocalSearchScheduler(
            config, _targets(config), BEGIN, END, step_size=60, time_limit=1.0
        )
        plan = scheduler.schedule()
        assert len(_dumps(plan)) >= 4
        _, norms = scheduler._momentum.profile(scheduler.timeline)
        assert norms.max() < CAPACITY


class TestRolling:
    def test_each_plan_starts_from_the_momentum_it_has(self) -> None:
        """Replans forecast from the stored momentum at the replan, so the
        executed run stays within the capacity with the dumps the plans
        reserved, and the planned simulation unloads at each."""
        config = _dumping()
        ditl = RollingHorizonDITL(
            config,
            _targets(config),
            begin=BEGIN,
            end=END,
            horizon=timedelta(hours=3),
            replan_interval=timedelta(hours=2),
        )
        ditl.step_size = 10
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
        dumps = _dumps(ditl.plan)
        assert len(dumps) >= 4
        norm = np.array(
            ditl.telemetry.housekeeping.stored_momentum_norm_n_m_s, dtype=float
        )
        assert norm.max() < CAPACITY
        assert sum(
            1 for e in ditl.log.events if "Thruster dump unloads" in e.description
        ) == len(dumps)
        assert any(
            r.utime > T0 and ditl._momentum_start() is not None for r in ditl.replans
        )
