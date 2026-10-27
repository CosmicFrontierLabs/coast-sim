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
from conops.schedulers.context import SchedulingContext
from conops.schedulers.momentum import MomentumForecast
from conops.simulation.attitude import Attitude
from conops.simulation.slew import Slew
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
        # The forecast reads the plan returned, searched and charged.
        assert [
            e for b in scheduler.timeline for e in (b.steps or [b.entry])
        ] == plan.entries
        _, norms = scheduler._momentum.profile(scheduler.timeline)
        assert norms.max() < CAPACITY
        # Each pass after a dump kept the searched plan before it.
        assert scheduler._kept
        assert all(block.entry in plan.entries for block in scheduler._kept)


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

    def test_a_slew_timed_into_a_late_block_leaves_its_settled_times(self) -> None:
        """A block that may begin late begins and ends where it settled,
        whatever start the slew into it was timed at: the slew was timed
        against the block where it was reserved, the settle moved it."""
        from conops.schedulers.priority_scheduler import PriorityScheduler, _Block
        from conops.simulation.slew import Slew

        config = _dumping()
        dump = ScheduledActivity(
            name="Desaturation",
            begin=T0 + 7200.0,
            duration_seconds=120.0,
            desaturation=True,
            obsid=998000,
        )
        entry = dump.plan_entry(config, (80.0, -20.0, 0.0))
        block = _Block(
            entry=entry,
            attitude_in=(80.0, -20.0, 0.0),
            attitude_out=(80.0, -20.0, 0.0),
            ready=T0 + 7200.0,
            end=T0 + 7320.0,
            follows=True,
            activity=dump,
            latest_ready=T0 + 7200.0 + 3600.0,
        )
        slew = Slew(config=config)
        slew.slewstart = T0 + 10800.0
        slew.slewend = T0 + 10800.0
        slew.slewtime = 0.0
        slew.slewdist = 0.0
        slew.endra, slew.enddec, slew.endroll = 80.0, -20.0, 0.0
        PriorityScheduler._apply_slew(block, slew)
        assert float(block.entry.begin) == T0 + 7200.0
        assert float(block.entry.end) == T0 + 7320.0
        assert float(block.entry.end) > float(block.entry.begin)


def _science_key(plan: Plan) -> list[tuple[int, float, float, float, float]]:
    return [
        (
            int(e.obsid),
            float(e.begin),
            float(e.collection_begin or 0.0),
            float(e.collection_end or 0.0),
            float(e.end),
        )
        for e in _science(plan)
    ]


def _given_dumps(plan: Plan) -> list[ScheduledActivity]:
    """The plan's dumps as activities a configuration could schedule."""
    return [
        ScheduledActivity(
            name="Desaturation",
            begin=float(d.begin),
            duration_seconds=float(d.end) - float(d.begin),
            desaturation=True,
        )
        for d in _dumps(plan)
    ]


class TestPlanningOnFromADump:
    """A pass after a dump keeps the science before the dump and plans on."""

    HORIZON = BEGIN + timedelta(hours=2)

    def test_the_pass_after_a_dump_keeps_the_plan_before_it(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """With one dump, the plan is the plan a single pass builds with that
        dump scheduled as an activity; the science before the dump is kept
        from the pass before, and the pass after the dump searches no slew
        start before the first slew it let go."""
        config = _dumping()
        targets = _targets(config)
        science = {int(t.obsid) for t in targets}
        scheduler = PriorityScheduler(
            config, targets, BEGIN, self.HORIZON, step_size=60
        )
        searched: list[tuple[float | None, float]] = []
        original = SchedulingContext.slew

        def slew(
            ctx: SchedulingContext,
            start: Attitude,
            end: Attitude,
            slewstart: float,
            *,
            obstype: ObsType = ObsType.PPT,
            obsid: int = 0,
        ) -> Slew:
            if obsid in science:
                searched.append((scheduler._resume_from, float(slewstart)))
            return original(ctx, start, end, slewstart, obstype=obstype, obsid=obsid)

        monkeypatch.setattr(SchedulingContext, "slew", slew)
        plan = scheduler.schedule()
        dumps = _dumps(plan)
        assert len(dumps) == 1
        assert len(scheduler._kept) == 1
        kept = scheduler._kept[0]
        assert float(kept.end) <= float(dumps[0].begin)
        assert kept.entry in plan.entries
        assert scheduler._resume_from is not None
        assert scheduler._resume_from >= float(dumps[0].end)
        # The second pass placed only from where the first pass let go.
        floored = [(floor, start) for floor, start in searched if floor is not None]
        assert floored
        assert all(start >= floor for floor, start in floored)
        assert min(start for _, start in floored) == scheduler._resume_from

        whole = PriorityScheduler(
            config,
            _targets(config),
            BEGIN,
            self.HORIZON,
            step_size=60,
            activities=_given_dumps(plan),
        )
        whole_plan = whole.schedule()
        assert whole.dumps == []
        assert _science_key(whole_plan) == _science_key(plan)
        assert [float(d.begin) for d in _dumps(whole_plan)] == [
            float(d.begin) for d in dumps
        ]

    def test_a_kept_visit_counts_for_its_steps(self) -> None:
        """A sequence visit before the dump is kept whole and counts as its
        science steps: the pass after the dump does not plan it again."""
        config = _dumping()
        visit = Pointing(
            config=config,
            obsid=2,
            name="Visit",
            ra=80.0,
            dec=-20.0,
            merit=100.0,
            sequence=ObservationSequence(
                steps=[
                    SequenceStep(name=f"s{k}", duration_seconds=600.0) for k in range(3)
                ]
            ),
            visits=1,
        )
        targets = [visit, *_targets(config)]
        scheduler = PriorityScheduler(
            config, targets, BEGIN, self.HORIZON, step_size=60
        )
        plan = scheduler.schedule()
        dumps = _dumps(plan)
        assert len(dumps) >= 1
        steps = [e for e in plan if int(e.obsid) == 2]
        assert [e.step for e in steps] == ["s0", "s1", "s2"]
        assert all(float(e.end) <= float(dumps[0].begin) for e in steps)
        assert [b.is_visit for b in scheduler._kept].count(True) == 1
        assert visit.obsid not in {int(t.obsid) for t in scheduler.unplaced}

        whole = PriorityScheduler(
            config,
            [visit, *_targets(config)],
            BEGIN,
            self.HORIZON,
            step_size=60,
            activities=_given_dumps(plan),
        )
        assert _science_key(whole.schedule()) == _science_key(plan)
