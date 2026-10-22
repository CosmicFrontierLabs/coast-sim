"""Thruster momentum dumps called for by the stored momentum reaching the
planning level of the wheels' capacity."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from pydantic import ValidationError
from rust_ephem import EarthLimbConstraint, SunConstraint, TLEEphemeris

from conops import DITL, QueueDITL, create_ditl
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
from conops.simulation.momentum import StoredMomentumTracker
from conops.targets import DESATURATION_OBSID, Plan, Pointing

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
HOURS = 4
END = BEGIN + timedelta(hours=HOURS)
STEP = 10
CAPACITY = 1.0


def _config(**momentum: object) -> MissionConfig:
    ephem = TLEEphemeris(
        tle="examples/example.tle", begin=BEGIN, end=END, step_size=STEP
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


def _dispatch(config: MissionConfig) -> QueueDITL:
    ditl = create_ditl(config, [_held(config)], BEGIN, END)
    assert isinstance(ditl, QueueDITL)
    ditl.step_size = STEP
    assert ditl.calc()
    return ditl


class TestConfiguration:
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
        within the run: dispatch ends the observation, holds for the dump,
        unloads the wheels, and observes again. The dumps are in the plan.
        The level's margin covers what builds up during the dump itself."""
        config = _config(
            capacity_n_m_s=CAPACITY,
            planning_level=0.6,
            thruster_dump=ThrusterDumpConfig(duration_seconds=120.0),
        )
        ditl = _dispatch(config)
        records = ditl.telemetry.housekeeping
        norm = np.array(records.stored_momentum_norm_n_m_s, dtype=float)
        assert norm.max() < CAPACITY
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
            assert before >= 0.6 * CAPACITY - 1e-6
            assert after < 0.1 * before
        events = [e.description for e in ditl.log.events if e.event_type == "MOMENTUM"]
        assert len(events) == 2 * len(dumps)
        assert events[0].startswith("Stored momentum") and "thruster dump" in events[0]
        assert events[1].startswith("Thruster dump unloads")
        # The observation under way ended for the dumps and resumed after
        # them; none runs through a dump.
        science = [e for e in ditl.plan if e.obstype != ObsType.ACTIVITY]
        assert len(dumps) >= 2 and len(science) >= 3
        assert any(float(e.begin) >= float(dumps[0].end) for e in science)
        for dump in dumps:
            assert not any(
                float(e.begin) < float(dump.end) and float(e.end) > float(dump.begin)
                for e in science
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
