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
from conops.schedulers.momentum import MomentumForecast
from conops.targets import Plan, Pointing, ScheduledActivity
from conops.targets.sequence import ObservationSequence, SequenceStep

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
        """Each pass reserves a dump for the forecast's next crossing of the
        level, between observations: after the observation the crossing falls
        in, or before it where the capacity would be reached inside it. The
        final plan's forecast stays within the capacity, no dump cuts an
        observation, and the executed plan matches the forecast."""
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
        assert not any(
            "forecast reaches the" in e.description for e in scheduler.log.events
        )
        # The wheels are empty after each dump.
        for dump in dumps:
            after = norms[times > float(dump.end)][0]
            assert after < 0.05
        placements = [e.description for e in reserved]
        assert any("after observation" in d for d in placements)
        assert any("before observation" in d for d in placements)
        # Dumps go between observations: none cuts one, and each follows one
        # within a step or precedes one within a slew, holding the attitude
        # of what precedes it.
        science = _science(plan)
        assert len(science) >= 4
        for dump in dumps:
            assert not any(
                float(s.begin) < float(dump.end) and float(s.end) > float(dump.begin)
                for s in science
            )
            assert any(
                0.0 <= float(dump.begin) - float(s.end) <= 60.0
                or 0.0 <= float(s.begin) - float(dump.end) <= 600.0
                for s in science
            )
        for dump in dumps:
            preceding = [s for s in science if float(s.end) <= float(dump.begin)]
            if preceding:
                entry = max(preceding, key=lambda s: float(s.end))
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

    def test_a_visit_is_forecast_over_all_its_steps(self) -> None:
        """The planner's loading of a visit from empty wheels integrates
        every step, so a visit of three short steps that together exceed
        the capacity is not planned, on the loading of all three."""
        config = _config(capacity_n_m_s=0.3, planning_level=0.6, thruster_dump=DUMP)
        step = 15 * 60.0

        def visit(obsid: int, steps: int) -> Pointing:
            return Pointing(
                config=config,
                obsid=obsid,
                name=f"Visit {steps}",
                ra=80.0,
                dec=-20.0,
                merit=100.0,
                sequence=ObservationSequence(
                    steps=[
                        SequenceStep(name=f"s{k}", duration_seconds=step)
                        for k in range(steps)
                    ]
                ),
                visits=1,
            )

        scheduler = PriorityScheduler(config, [visit(2, 3)], BEGIN, END, step_size=60)
        plan = scheduler.schedule()
        assert not any(e.obsid == 2 for e in plan)
        not_planned = [
            e.description
            for e in scheduler.log.events
            if "Visit of 2 not planned" in e.description
        ]
        assert len(not_planned) == 1
        loading = float(not_planned[0].split("loading of ")[1].split(" N m s")[0])
        # Three steps' worth: one step of this hold stores about 0.2 N m s.
        assert loading > 0.5

    def test_the_forecast_reads_integrated_spans_from_its_cache(self) -> None:
        """A plan's passes share the spans integrated so far: the profile
        from the cache is the profile integrated afresh, and a pass
        integrates only from the dump it reserves onwards."""
        config = _dumping()
        scheduler = PriorityScheduler(
            config, _targets(config), BEGIN, END, step_size=60
        )
        scheduler.schedule()
        assert scheduler._momentum is not None
        cache = scheduler._momentum.cache
        assert cache and cache is scheduler._momentum_cache
        times, norms = scheduler._momentum.profile(scheduler.timeline)
        fresh = MomentumForecast(config, scheduler.ctx)
        assert fresh.cache == {}
        fresh_times, fresh_norms = fresh.profile(scheduler.timeline)
        assert np.array_equal(times, fresh_times)
        assert np.array_equal(norms, fresh_norms)
        # The spans before the last dump were integrated once, in the pass
        # before it: a span is keyed on its times, attitude and the state it
        # starts from, so the passes hit the same entries.
        spans = scheduler._momentum._spans(scheduler.timeline)
        assert len(cache) < 2 * len(spans) * len(scheduler.dumps)

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
        assert any(r.utime > T0 for r in ditl.replans)
        assert ditl._momentum_start(ditl.uend, ditl.uend, []) is not None


class TestResolution:
    @pytest.mark.parametrize("step", [60, 120])
    def test_plans_and_tracks_at_the_ephemeris_resolution(self, step: int) -> None:
        """At a coarse ephemeris the run is kept within the capacity as at a
        fine one: slews are integrated from the attitude the ACS flew within
        each step, and each replan forecasts from the attitude held and the
        momentum stored when its horizon starts, the committed entries'
        contribution included."""
        config = _dumping()
        config.constraint.ephem = TLEEphemeris(
            tle="examples/example.tle", begin=BEGIN, end=END, step_size=step
        )
        ditl = RollingHorizonDITL(
            config,
            _targets(config),
            begin=BEGIN,
            end=END,
            horizon=timedelta(hours=3),
            replan_interval=timedelta(hours=2),
        )
        ditl.step_size = step
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
        norm = np.array(
            ditl.telemetry.housekeeping.stored_momentum_norm_n_m_s, dtype=float
        )
        assert len(norm) == HOURS * 3600 // step
        assert norm.max() < CAPACITY
        assert norm.max() < LEVEL + 0.15
        assert len(_dumps(ditl.plan)) >= 5


class TestDumpsAroundBlocks:
    def test_a_crossing_inside_an_activity_puts_the_dump_before_or_after_it(
        self,
    ) -> None:
        """Where the forecast reaches the level inside a scheduled activity,
        the dump goes before the block where the capacity would be reached
        inside it, else after; dumps are never stacked after the block."""
        config = _dumping()
        bare = PriorityScheduler(config, _targets(config), BEGIN, END, step_size=60)
        bare.schedule()
        first = _dumps(bare.plan)[0]
        crossing = float(first.begin)
        block = ScheduledActivity(
            name="Calibration",
            begin=crossing - 20 * 60.0,
            duration_seconds=60 * 60.0,
        )
        config = _dumping()
        scheduler = PriorityScheduler(
            config,
            _targets(config),
            BEGIN,
            END,
            step_size=60,
            activities=[block],
        )
        plan = scheduler.schedule()
        dumps = _dumps(plan)
        assert dumps
        for dump in dumps:
            assert float(dump.end) <= block.begin or float(dump.begin) >= block.end
            assert float(dump.end) > float(dump.begin)
        gaps = [float(b.begin) - float(a.end) for a, b in zip(dumps, dumps[1:])]
        assert all(gap > 20 * 60.0 for gap in gaps), "dumps stacked back to back"
        assert any(
            "activity Calibration" in e.description for e in scheduler.log.events
        )
        assert scheduler._momentum is not None
        _, norms = scheduler._momentum.profile(scheduler.timeline)
        assert norms.max() < CAPACITY

    def test_a_dump_that_may_begin_late_keeps_its_times_with_charges(self) -> None:
        """A dump displaced by a sliding block settles after what precedes
        it, and a plan that charges its battery in idle time keeps the
        dump's entry where it settled: it begins before it ends, so the
        battery ledger can take it and the plan executes."""
        from .test_battery_constraint import BORESIGHT, _battery_config

        config = _battery_config(BORESIGHT, stations=False)
        config.spacecraft_bus.inertia_tensor_body_kg_m2 = np.diag(
            [1000.0, 1200.0, 800.0]
        )
        config.spacecraft_bus.attitude_control.stored_momentum = StoredMomentumConfig(
            gravity_gradient_enabled=True,
            capacity_n_m_s=CAPACITY,
            planning_level=LEVEL,
            thruster_dump=DUMP,
        )
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
            target.exptime = 6 * 3600
            targets.append(target)
        sliding = ScheduledActivity(
            name="Cal",
            begin=T0 + 3 * 3600,
            duration_seconds=1800.0,
            slide_seconds=3600.0,
        )
        end = BEGIN + timedelta(hours=14)
        scheduler = PriorityScheduler(
            config, targets, BEGIN, end, step_size=60, activities=[sliding]
        )
        plan = scheduler.schedule()
        for entry in plan:
            assert float(entry.end) > float(entry.begin), entry
        assert _dumps(plan)
        ditl = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=plan,
            begin=BEGIN,
            end=end,
        )
        ditl.step_size = 60
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []
