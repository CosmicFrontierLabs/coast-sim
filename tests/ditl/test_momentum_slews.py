"""Slews admitted against the wheels' capacity in dispatch: the momentum the
body carries while it turns, I·ω, counts where ``slew_momentum_enabled``,
and a slew is flown slower where it does not fit."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
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
from conops.ditl.ditl_mixin import DITLMixin, MomentumProfile
from conops.targets import Plan, Pointing

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
HOURS = 2
END = BEGIN + timedelta(hours=HOURS)
STEP = 10
INERTIA = 1000.0
RATE = 0.5
TRANSIENT = INERTIA * np.deg2rad(RATE)
"""What the body carries at the full slew rate about any axis: the inertia
is isotropic, so the direction of the slew does not matter."""
CAPACITY = 6.5
"""Less than the transient at full rate, more than at half."""


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


def _target(config: MissionConfig) -> Pointing:
    """One target well away from the initial attitude: the slew to it
    reaches the full rate."""
    target = Pointing(
        config=config,
        obsid=1,
        name="Far",
        ra=80.0,
        dec=-20.0,
        merit=100.0,
        ss_min=600,
        ss_max=HOURS * 3600,
    )
    target.exptime = HOURS * 3600
    return target


def _dispatch(config: MissionConfig) -> QueueDITL:
    ditl = create_ditl(config, [_target(config)], BEGIN, END)
    assert isinstance(ditl, QueueDITL)
    ditl.step_size = STEP
    assert ditl.calc()
    return ditl


def _momentum_events(ditl: QueueDITL) -> list[str]:
    return [e.description for e in ditl.log.events if e.event_type == "MOMENTUM"]


def _science(ditl: QueueDITL | DITL) -> list:
    return [e for e in ditl.plan if e.obstype not in (ObsType.ACTIVITY, ObsType.GSP)]


class TestDispatch:
    def test_a_slew_that_does_not_fit_is_flown_slower(self) -> None:
        """At the full rate the body carries more than the capacity; at
        half it does not. The slew is flown at half, the entry records it,
        the wheels never reach the capacity, and a planned simulation flies
        the plan the same way."""
        config = _config()
        ditl = _dispatch(config)
        science = _science(ditl)
        assert science and science[0].obsid == 1
        assert science[0].slew_rate_fraction == 0.5
        assert any("flown at 0.5 of the slew rate" in e for e in _momentum_events(ditl))
        records = ditl.telemetry.housekeeping
        demand = np.array(
            [v for v in records.actuator_momentum_norm_n_m_s if v is not None]
        )
        assert len(demand) == len(records)
        assert demand.max() < CAPACITY
        assert demand.max() > 0.4 * TRANSIENT
        assert not any(
            e.event_type == "momentum_capacity" for e in ditl.fault_management.events
        )
        # The executed slew is the slower one: longer than at full rate.
        full = Pointing(config=config, obsid=9, ra=80.0, dec=-20.0, roll=0.0)
        assert science[0].slewtime > full.calc_slewtime(0.0, 0.0, 0.0)
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
        replayed = np.array(
            [
                v
                for v in replay.telemetry.housekeeping.actuator_momentum_norm_n_m_s
                if v is not None
            ]
        )
        assert replayed.max() < CAPACITY
        assert replayed.max() == pytest.approx(demand.max(), rel=0.05)

    def test_a_slew_that_fits_at_no_rate_is_held_out(self) -> None:
        """With the full rate the only one configured, the slew asks more
        of the wheels than the capacity even from empty: the target is held
        out and marked done, and nothing is observed."""
        config = _config(slew_rate_fractions=(1.0,))
        ditl = _dispatch(config)
        assert _science(ditl) == []
        held = [e for e in _momentum_events(ditl) if "held out" in e]
        assert len(held) == 1
        assert "exceeds the 6.5 N m s capacity from empty wheels" in held[0]
        assert all(t.done for t in ditl.queue.targets)

    def test_without_slew_momentum_the_slew_is_flown_at_full_rate(self) -> None:
        config = _config()
        config.spacecraft_bus.attitude_control.stored_momentum.slew_momentum_enabled = (
            False
        )
        config.spacecraft_bus.attitude_control.stored_momentum.gravity_gradient_enabled = True
        ditl = _dispatch(config)
        science = _science(ditl)
        assert science and science[0].slew_rate_fraction == 1.0
        assert not any("slew rate" in e for e in _momentum_events(ditl))
        records = ditl.telemetry.housekeeping
        # The rate is reported regardless; it is not checked.
        demand = np.array(
            [v for v in records.actuator_momentum_norm_n_m_s if v is not None]
        )
        assert demand.max() > CAPACITY

    def test_the_wheels_are_emptied_first_where_the_slowest_rate_fits_from_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Where the stored momentum leaves the slew no room at any rate
        but the slowest fits from empty wheels, a thruster dump goes first
        and the observation follows it. The forecast is stood in for: the
        wheels hold more than the capacity over the slew from the momentum
        they have until the first dump is called for, and little from empty."""
        config = _config(
            thruster_dump=ThrusterDumpConfig(duration_seconds=120.0),
            planning_level=0.99,
        )
        forecast = DITLMixin._forecast_momentum
        asked: list[float] = []

        def stood_in(self, begin, end, attitude_at, *, start_eci_n_m_s, slew=None):  # type: ignore[no-untyped-def]
            profile = forecast(
                self,
                begin,
                end,
                attitude_at,
                start_eci_n_m_s=start_eci_n_m_s,
                slew=slew,
            )
            if slew is None or self._desaturations:
                return profile
            asked.append(float(slew.rate_fraction))
            loaded = 0.5 if start_eci_n_m_s is None else 2.0
            return MomentumProfile(
                profile.times,
                profile.norms,
                np.full_like(profile.demands, loaded * CAPACITY),
                profile.end_eci_n_m_s,
            )

        monkeypatch.setattr(DITLMixin, "_forecast_momentum", stood_in)
        ditl = _dispatch(config)
        plan = list(ditl.plan)
        dumps = [e for e in plan if e.desaturation]
        science = _science(ditl)
        assert len(dumps) == 1 and science
        assert float(dumps[0].end) <= float(science[0].begin)
        assert asked[:5] == [1.0, 0.75, 0.5, 0.25, 0.125]
        events = _momentum_events(ditl)
        assert any(
            "no room within the 6.5 N m s capacity at 0.125 of the slew rate: "
            "thruster dump first" in e
            for e in events
        )
        assert any(e.startswith("Thruster dump unloads") for e in events)
