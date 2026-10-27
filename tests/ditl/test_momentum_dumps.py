"""Thruster momentum dumps called for by the stored momentum reaching the
planning level of the wheels' capacity."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from pydantic import ValidationError
from rust_ephem import EarthLimbConstraint, SunConstraint, TLEEphemeris

from conops import DITL, QueueDITL, create_ditl
from conops.common import ACSMode, ObsType
from conops.config import (
    AttitudeConstraintScope,
    AttitudeControlSystem,
    Battery,
    Constraint,
    GroundStationRegistry,
    Instrument,
    InstrumentMode,
    MissionConfig,
    ObservationTiming,
    Payload,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
    StoredMomentumConfig,
    ThrusterDumpConfig,
)
from conops.ditl.ditl_mixin import DITLMixin
from conops.simulation.momentum import StoredMomentumTracker
from conops.targets import DESATURATION_OBSID, Plan, Pointing
from conops.targets.sequence import ObservationSequence, SequenceStep

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
HOURS = 4
END = BEGIN + timedelta(hours=HOURS)
STEP = 10
CAPACITY = 1.0


def _config(step: int = STEP, **momentum: object) -> MissionConfig:
    ephem = TLEEphemeris(
        tle="examples/example.tle", begin=BEGIN, end=END, step_size=step
    )
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


def _held(config: MissionConfig) -> Pointing:
    """One target held through the run: an inertial hold builds momentum."""
    target = Pointing(
        config=config,
        obsid=1,
        name="Held",
        ra=80.0,
        dec=-20.0,
        merit=100.0,
        ss_min=600,
        ss_max=HOURS * 3600,
    )
    target.exptime = HOURS * 3600
    return target


def _dispatch(
    config: MissionConfig, step: int = STEP, targets: list[Pointing] | None = None
) -> QueueDITL:
    ditl = create_ditl(config, targets or [_held(config)], BEGIN, END)
    assert isinstance(ditl, QueueDITL)
    ditl.step_size = step
    assert ditl.calc()
    return ditl


def _visit(config: MissionConfig, hours: float, obsid: int = 2) -> Pointing:
    """A one-step visit held for ``hours``: indivisible."""
    return Pointing(
        config=config,
        obsid=obsid,
        name=f"Visit {hours}h",
        ra=80.0,
        dec=-20.0,
        merit=100.0,
        sequence=ObservationSequence(
            steps=[SequenceStep(name="hold", duration_seconds=hours * 3600)]
        ),
        visits=1,
    )


def _momentum_events(ditl: QueueDITL) -> list[str]:
    return [e.description for e in ditl.log.events if e.event_type == "MOMENTUM"]


def _capacity_faults(ditl: QueueDITL) -> list:
    return [
        e for e in ditl.fault_management.events if e.event_type == "momentum_capacity"
    ]


def _load_to_capacity_at(monkeypatch: pytest.MonkeyPatch, load_at: float) -> None:
    """Load the wheels to the capacity at ``load_at``, as no forecast can
    foresee: the emergency path."""
    update = DITLMixin._update_stored_momentum

    def loaded(self: DITLMixin, utime: float, position, attitude, index=None):  # type: ignore[no-untyped-def]
        sample = update(self, utime, position, attitude, index=index)
        tracker = self._stored_momentum_tracker
        if tracker is not None and utime == load_at:
            current = tracker.momentum_eci_n_m_s
            assert current is not None
            norm = float(np.linalg.norm(current))
            tracker.start_from_eci(current * (1.001 * CAPACITY / norm))
        return sample

    monkeypatch.setattr(DITLMixin, "_update_stored_momentum", loaded)


class TestConfiguration:
    def test_desat_holds_under_hardware_safety_unless_configured(self) -> None:
        """A thruster dump holds as idle does: the DESAT mode is checked
        against hardware safety by default, also for a configuration that
        sets scopes for other modes only, and against its own where set."""
        config = _config()
        assert config.attitude_constraint_scopes_for_mode(ACSMode.DESAT) == [
            AttitudeConstraintScope.HARDWARE_SAFETY
        ]
        older = _config()
        older.attitude_constraint_scopes = {
            ACSMode.IDLE.name: [
                AttitudeConstraintScope.HARDWARE_SAFETY,
                AttitudeConstraintScope.POWER_GENERATION,
            ]
        }
        assert older.attitude_constraint_scopes_for_mode(ACSMode.DESAT) == [
            AttitudeConstraintScope.HARDWARE_SAFETY
        ]
        own = _config()
        own.attitude_constraint_scopes = {ACSMode.DESAT.name: []}
        assert own.attitude_constraint_scopes_for_mode(ACSMode.DESAT) == []

    def test_dumps_need_a_capacity(self) -> None:
        with pytest.raises(ValidationError, match="capacity"):
            StoredMomentumConfig(thruster_dump=ThrusterDumpConfig())
        config = StoredMomentumConfig(
            capacity_n_m_s=2.0, planning_level=0.75, thruster_dump=ThrusterDumpConfig()
        )
        assert config.dump_level_n_m_s == pytest.approx(1.5)
        assert StoredMomentumConfig(capacity_n_m_s=2.0).dump_level_n_m_s is None

    def test_the_tracker_unloads_to_the_residual(self) -> None:
        tracker = StoredMomentumTracker(
            np.eye(3), initial_momentum_body_n_m_s=(3.0, 0.0, 4.0), sources=[]
        )
        assert tracker.stored_momentum_norm_n_m_s == pytest.approx(5.0)
        tracker.update(
            utime=0.0,
            position_eci_km=np.array([7000.0, 0.0, 0.0]),
            attitude_quaternion_eci_to_body=np.array([1.0, 0.0, 0.0, 0.0]),
        )
        tracker.unload(0.1)
        assert tracker.stored_momentum_norm_n_m_s == pytest.approx(0.5)
        tracker.unload()
        assert tracker.stored_momentum_norm_n_m_s == 0.0
        with pytest.raises(ValueError, match="residual_fraction"):
            tracker.unload(1.0)


class TestDispatch:
    def test_a_dump_is_called_for_at_the_planning_level_and_unloads(self) -> None:
        """Holding one attitude, the gravity gradient fills a small capacity
        within the run. A dump called for at the planning level goes between
        observations: the observation under way runs on, or is bounded at
        selection to end where the level is reached, and the dump follows
        it; nothing is interrupted and the capacity is never reached."""
        config = _config(
            capacity_n_m_s=CAPACITY,
            planning_level=0.6,
            thruster_dump=ThrusterDumpConfig(duration_seconds=120.0),
        )
        ditl = _dispatch(config)
        records = ditl.telemetry.housekeeping
        norm = np.array(records.stored_momentum_norm_n_m_s, dtype=float)
        assert norm.max() < CAPACITY
        assert _capacity_faults(ditl) == []
        dumps = [e for e in ditl.plan if e.obstype == ObsType.ACTIVITY]
        assert dumps and all(e.desaturation and e.name == "Desaturation" for e in dumps)
        assert [e.obsid for e in dumps] == [
            DESATURATION_OBSID + k for k in range(len(dumps))
        ]
        assert all(
            float(e.end) - float(e.begin) == pytest.approx(120.0, abs=STEP)
            for e in dumps
        )
        times = np.array([hk.timestamp.timestamp() for hk in records])
        for dump in dumps:
            before = norm[times < float(dump.begin)][-1]
            after = norm[times >= float(dump.end)][0]
            assert before >= 0.6 * CAPACITY - 0.05
            assert after < 0.1 * before
        events = _momentum_events(ditl)
        assert sum(e.startswith("Thruster dump unloads") for e in events) == len(dumps)
        assert any("planning level" in e and "thruster dump" in e for e in events)
        assert not any("emergency" in e or "interrupt" in e for e in events)
        # Science before a dump ends at the dump, or earlier: none is cut by one.
        science = [e for e in ditl.plan if e.obstype != ObsType.ACTIVITY]
        assert len(dumps) >= 2 and len(science) >= 3
        for dump in dumps:
            assert not any(
                float(e.begin) < float(dump.end) and float(e.end) > float(dump.begin)
                for e in science
            )
        assert not any(
            e.event_type == "OBSERVATION" and "Desaturation begins" in e.description
            for e in ditl.log.events
        )

    @pytest.mark.parametrize("step", [STEP, 120])
    def test_an_observation_is_bounded_where_the_level_would_be_reached(
        self, step: int
    ) -> None:
        """Where the forecast over the observation reaches the capacity, a
        plain observation ends where the level is reached and the dump
        follows; the exported plan replays without a mismatch."""
        config = _config(
            step,
            capacity_n_m_s=CAPACITY,
            planning_level=0.6,
            thruster_dump=ThrusterDumpConfig(duration_seconds=120.0),
        )
        config.payload.observation_timing = ObservationTiming(cleanup_seconds=60.0)
        ditl = _dispatch(config, step)
        events = _momentum_events(ditl)
        bounded = [e for e in events if e.startswith("Observation 1 ends at")]
        assert bounded and all("the thruster dump follows it" in e for e in bounded)
        assert _capacity_faults(ditl) == []
        norm = np.array(
            ditl.telemetry.housekeeping.stored_momentum_norm_n_m_s, dtype=float
        )
        assert norm.max() < CAPACITY
        plan = list(ditl.plan)
        dumps = [e for e in plan if e.desaturation]
        assert len(dumps) >= 2
        for dump in dumps:
            assert (float(dump.begin) - ditl.ustart) % step == 0
        replay = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=Plan.model_validate_json(ditl.plan.model_dump_json()),
            begin=BEGIN,
            end=END,
        )
        replay.step_size = step
        assert replay.calc()
        assert replay.validate_plan_matches_execution() == []

    @pytest.mark.parametrize("step", [STEP, 120])
    def test_the_capacity_reached_during_an_observation_is_an_emergency(
        self, step: int, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The wheels reach the capacity while an observation collects:
        fault management records it against the observation, the emergency
        dump interrupts the observation once its cleanup budget is paid,
        and the unload clears the fault. No safe mode."""
        cleanup = 60.0
        config = _config(
            step,
            capacity_n_m_s=CAPACITY,
            planning_level=0.6,
            thruster_dump=ThrusterDumpConfig(duration_seconds=120.0),
        )
        config.payload.observation_timing = ObservationTiming(cleanup_seconds=cleanup)
        load_at = BEGIN.timestamp() + 1800.0
        _load_to_capacity_at(monkeypatch, load_at)
        ditl = _dispatch(config, step)
        faults = _capacity_faults(ditl)
        assert len(faults) == 2
        reached, cleared = faults
        assert reached.utime >= load_at
        assert "during observation 1" in reached.cause
        assert reached.metadata == {
            "stored_n_m_s": pytest.approx(reached.metadata["stored_n_m_s"]),
            "capacity_n_m_s": CAPACITY,
            "obsid": 1,
            "during_observation": 1,
        }
        assert "unloaded" in cleared.cause
        assert ditl.fault_management.states["stored_momentum"].current == "nominal"
        assert not ditl.fault_management.safe_mode_requested
        assert not any(
            e.event_type == "safe_mode_trigger" for e in ditl.fault_management.events
        )
        events = _momentum_events(ditl)
        emergency = [e for e in events if "emergency thruster dump" in e]
        assert len(emergency) == 1 and "interrupts observation 1" in emergency[0]
        plan = list(ditl.plan)
        dumps = [e for e in plan if e.desaturation]
        first = dumps[0]
        before = plan[plan.index(first) - 1]
        assert before.obsid == 1 and float(before.end) == float(first.begin)
        assert before.collection_end == float(first.begin) - cleanup
        assert (float(first.begin) - ditl.ustart) % step == 0
        assert any(
            e.description.startswith("Reserved thruster dump cleanup")
            for e in ditl.log.events
        )
        assert float(first.end) - float(first.begin) == pytest.approx(120.0)
        replay = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=Plan.model_validate_json(ditl.plan.model_dump_json()),
            begin=BEGIN,
            end=END,
        )
        replay.step_size = step
        assert replay.calc()
        assert replay.validate_plan_matches_execution() == []

    def test_a_visit_that_would_fill_the_wheels_is_rejected(self) -> None:
        """A visit is indivisible: one whose loading from empty wheels
        exceeds the capacity is rejected and marked done, with the reason
        logged, and nothing of it is observed."""
        config = _config(
            capacity_n_m_s=0.3,
            planning_level=0.6,
            thruster_dump=ThrusterDumpConfig(duration_seconds=120.0),
        )
        visit = _visit(config, hours=0.9)
        ditl = _dispatch(config, targets=[visit])
        rejected = [e for e in _momentum_events(ditl) if "Visit of 2 rejected" in e]
        assert len(rejected) == 1 and "exceeds the 0.3 N m s capacity" in rejected[0]
        queued = next(t for t in ditl.queue.targets if t.obsid == visit.obsid)
        assert queued.done
        assert not any(e.obsid == 2 for e in ditl.plan)

    def test_the_selection_forecast_covers_the_whole_visit(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A visit of several steps is forecast at selection over all its
        steps, the slews between them and its cleanup, not over its first
        step alone: its loading from empty is the whole visit's, and one
        that exceeds the capacity is rejected on that."""
        spans: list[float] = []
        forecast = DITLMixin._forecast_momentum

        def probed(self: DITLMixin, begin: float, end: float, attitude_at, **kw):  # type: ignore[no-untyped-def]
            spans.append(end - begin)
            return forecast(self, begin, end, attitude_at, **kw)

        monkeypatch.setattr(DITLMixin, "_forecast_momentum", probed)
        config = _config(
            capacity_n_m_s=0.3,
            planning_level=0.6,
            thruster_dump=ThrusterDumpConfig(duration_seconds=120.0),
        )
        step = 15 * 60.0
        visit = Pointing(
            config=config,
            obsid=2,
            name="Three steps",
            ra=80.0,
            dec=-20.0,
            merit=100.0,
            sequence=ObservationSequence(
                steps=[
                    SequenceStep(name="a", duration_seconds=step),
                    SequenceStep(name="b", duration_seconds=step, ra=82.0, dec=-20.0),
                    SequenceStep(name="c", duration_seconds=step),
                ]
            ),
            visits=1,
        )
        ditl = _dispatch(config, targets=[visit])
        rejected = [e for e in _momentum_events(ditl) if "Visit of 2 rejected" in e]
        assert len(rejected) == 1
        loading = float(rejected[0].split("loading of ")[1].split(" N m s")[0])
        # Three steps' worth, well over one step's.
        assert loading > 0.5
        assert max(spans) >= 3 * step

    def test_wheels_past_the_level_are_emptied_before_a_visit(self) -> None:
        """Wheels already past the level when the run starts: the dump comes
        first, and the visit then runs uninterrupted with its own dump after."""
        config = _config(
            capacity_n_m_s=0.7,
            planning_level=0.6,
            thruster_dump=ThrusterDumpConfig(duration_seconds=120.0),
            initial_momentum_body_n_m_s=(0.0, 0.6, 0.0),
        )
        visit = _visit(config, hours=0.9)
        ditl = _dispatch(config, targets=[visit])
        events = _momentum_events(ditl)
        assert (
            events[0].startswith("Stored momentum 0.600")
            and "thruster dump" in events[0]
        )
        assert any("thruster dump after observation 2" in e for e in events)
        plan = list(ditl.plan)
        assert plan[0].desaturation
        assert plan[1].obsid == 2
        assert float(plan[1].end) - float(plan[1].begin) >= 0.9 * 3600 - STEP
        assert _capacity_faults(ditl) == []
        norm = np.array(
            ditl.telemetry.housekeeping.stored_momentum_norm_n_m_s, dtype=float
        )
        assert norm.max() < 0.7

    def test_a_planned_simulation_records_the_capacity_against_the_entry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The planned simulation has no dump to call for: the wheels at
        the capacity during a plan entry are a fault management event
        against that entry, cleared by the plan's next dump."""
        config = _config(
            capacity_n_m_s=CAPACITY,
            planning_level=0.6,
            thruster_dump=ThrusterDumpConfig(duration_seconds=120.0),
        )
        dispatched = _dispatch(config)
        first = next(e for e in dispatched.plan if e.obstype != ObsType.ACTIVITY)
        load_at = float(first.begin) + 1800.0
        _load_to_capacity_at(monkeypatch, load_at)
        replay = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=Plan.model_validate_json(dispatched.plan.model_dump_json()),
            begin=BEGIN,
            end=END,
        )
        replay.step_size = STEP
        assert replay.calc()
        faults = [
            e
            for e in replay.fault_management.events
            if e.event_type == "momentum_capacity"
        ]
        assert len(faults) == 2
        assert faults[0].utime == load_at
        assert faults[0].metadata is not None and faults[0].metadata["obsid"] == 1
        assert "during observation 1" in faults[0].cause
        next_dump = next(
            e for e in dispatched.plan if e.desaturation and float(e.begin) > load_at
        )
        assert faults[1].utime == float(next_dump.end)
        assert not replay.fault_management.safe_mode_requested

    def test_a_dump_holds_in_desat_mode(self) -> None:
        """Telemetry reports DESAT through each dump and never outside one,
        in dispatch and in the planned simulation replaying its plan, and
        an instrument that defines a DESAT mode is in it, at its power,
        while the dump holds."""
        config = _config(
            capacity_n_m_s=CAPACITY,
            planning_level=0.6,
            thruster_dump=ThrusterDumpConfig(duration_seconds=120.0),
        )
        config.payload = Payload(
            instruments=[
                Instrument(
                    name="camera",
                    modes=[
                        InstrumentMode(name="IDLE", power_w=5.0),
                        InstrumentMode(name="SCIENCE", power_w=50.0),
                        InstrumentMode(name="DESAT", power_w=1.0),
                    ],
                )
            ]
        )
        ditl = _dispatch(config)
        dumps = [e for e in ditl.plan if e.desaturation]
        assert len(dumps) >= 2
        replay = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=Plan.model_validate_json(ditl.plan.model_dump_json()),
            begin=BEGIN,
            end=END,
        )
        replay.step_size = STEP
        assert replay.calc()
        assert replay.validate_plan_matches_execution() == []
        for run in (ditl, replay):
            records = run.telemetry.housekeeping
            times = np.array([hk.timestamp.timestamp() for hk in records])
            modes = [ACSMode(int(m)) for m in records.acs_mode if m is not None]
            assert len(modes) == len(times)
            inside = np.zeros(len(times), dtype=bool)
            for dump in dumps:
                inside |= (times >= float(dump.begin)) & (times < float(dump.end))
            assert inside.any()
            assert all(modes[i] is ACSMode.DESAT for i in np.flatnonzero(inside))
            assert not any(modes[i] is ACSMode.DESAT for i in np.flatnonzero(~inside))
            instrument_modes = records.instrument_modes
            assert all(
                instrument_modes[i] == {"camera": "DESAT"}
                for i in np.flatnonzero(inside)
            )
            assert all(
                float(run.power_payload[i]) == pytest.approx(1.0)
                for i in np.flatnonzero(inside)
            )
            assert any(
                float(run.power_payload[i]) > 1.0 for i in np.flatnonzero(~inside)
            )

    def test_without_thrusters_the_wheels_fill(self) -> None:
        config = _config(capacity_n_m_s=CAPACITY)
        ditl = _dispatch(config)
        assert not [e for e in ditl.plan if e.obstype == ObsType.ACTIVITY]
        norm = np.array(
            ditl.telemetry.housekeeping.stored_momentum_norm_n_m_s, dtype=float
        )
        assert norm.max() > CAPACITY

    def test_the_executed_plan_reproduces_the_dumps(self) -> None:
        """Replaying the dispatched plan, the planned simulation unloads the
        wheels at each desaturation entry's end."""
        config = _config(
            capacity_n_m_s=CAPACITY,
            thruster_dump=ThrusterDumpConfig(duration_seconds=120.0),
        )
        dispatched = _dispatch(config)
        dumps = [e for e in dispatched.plan if e.desaturation]
        assert dumps
        replay = Plan.model_validate_json(dispatched.plan.model_dump_json())
        assert [e.desaturation for e in replay if e.obstype == ObsType.ACTIVITY] == [
            True
        ] * len(dumps)
        again = DITL(
            config=_config(
                capacity_n_m_s=CAPACITY,
                thruster_dump=ThrusterDumpConfig(duration_seconds=120.0),
            ),
            ephem=config.constraint.ephem,
            plan=replay,
            begin=BEGIN,
            end=END,
        )
        again.step_size = STEP
        assert again.calc()
        assert again.validate_plan_matches_execution() == []
        norm = np.array(
            again.telemetry.housekeeping.stored_momentum_norm_n_m_s, dtype=float
        )
        times = np.array(
            [hk.timestamp.timestamp() for hk in again.telemetry.housekeeping]
        )
        for dump in dumps:
            before = norm[times < float(dump.begin)][-1]
            after = norm[times >= float(dump.end)][0]
            assert after < 0.1 * before
        assert sum(
            1
            for e in again.log.events
            if e.description.startswith("Thruster dump unloads")
        ) == len(dumps)
