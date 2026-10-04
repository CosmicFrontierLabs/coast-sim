"""SchedulingContext answers planning questions with the models DITL executes."""

from datetime import datetime, timedelta, timezone

import pytest
from rust_ephem import SunConstraint, TLEEphemeris

from conops.common import ACSMode
from conops.config import (
    AttitudeControlSystem,
    Constraint,
    GroundStationRegistry,
    MissionConfig,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
)
from conops.schedulers import SchedulingContext
from conops.simulation.acs import ACS
from conops.targets import Pointing

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
END = BEGIN + timedelta(hours=2)
T0 = BEGIN.timestamp()


@pytest.fixture
def config() -> MissionConfig:
    config = MissionConfig(
        constraint=Constraint(sun_constraint=SunConstraint(min_angle=45)),
        ground_stations=GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        spacecraft_bus=SpacecraftBus(
            attitude_control=AttitudeControlSystem(
                max_slew_rate=1.0, slew_acceleration=0.5, settle_time=30.0
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
    )
    config.constraint.ephem = TLEEphemeris(
        tle="examples/example.tle", begin=BEGIN, end=END, step_size=60
    )
    return config


@pytest.fixture
def ctx(config: MissionConfig) -> SchedulingContext:
    return SchedulingContext(config, BEGIN, END)


class TestStepGrid:
    def test_step_defaults_to_ephemeris_step(self, ctx: SchedulingContext) -> None:
        assert ctx.step_size == 60

    @pytest.mark.parametrize(
        ("utime", "ceil", "floor"),
        [(T0, T0, T0), (T0 + 1, T0 + 60, T0), (T0 + 60, T0 + 60, T0 + 60)],
    )
    def test_rounds_to_steps(
        self, ctx: SchedulingContext, utime: float, ceil: float, floor: float
    ) -> None:
        assert ctx.ceil_step(utime) == ceil
        assert ctx.floor_step(utime) == floor

    def test_steps_are_half_open(self, ctx: SchedulingContext) -> None:
        assert list(ctx.steps(T0 + 1, T0 + 180)) == [T0 + 60, T0 + 120]

    def test_rejects_nonpositive_step(self, config: MissionConfig) -> None:
        with pytest.raises(ValueError, match="step_size"):
            SchedulingContext(config, BEGIN, END, step_size=0)

    def test_requires_an_ephemeris(self, config: MissionConfig) -> None:
        config.constraint.ephem = None

        with pytest.raises(ValueError, match="ephem"):
            SchedulingContext(config, BEGIN, END)


class TestSlews:
    def test_retimed_slew_matches_a_fresh_one(self, ctx: SchedulingContext) -> None:
        start, end = (10.0, 20.0, 30.0), (80.0, -10.0, 120.0)
        first = ctx.slew(start, end, T0)

        retimed = ctx.slew(start, end, T0 + 600)
        fresh = SchedulingContext(ctx.config, BEGIN, END).slew(start, end, T0 + 600)

        assert retimed is not first
        assert retimed.slewstart == T0 + 600
        assert retimed.slewend == fresh.slewend
        assert retimed.attitude(T0 + 650) == pytest.approx(fresh.attitude(T0 + 650))
        assert first.slewstart == T0

    def test_initial_attitude_matches_a_fresh_acs(
        self, ctx: SchedulingContext, config: MissionConfig
    ) -> None:
        acs = ACS(config=config)
        expected = [acs.pointing(T0 + 60 * i)[:3] for i in range(4)]

        assert ctx.initial_attitude(T0 + 180) == pytest.approx(expected[3])
        assert ctx.initial_attitude(T0 + 60) == pytest.approx(expected[1])


class TestConstraints:
    def test_attitude_at_the_sun_is_not_allowed(self, ctx: SchedulingContext) -> None:
        sun = (
            float(ctx.ephem.sun_ra_deg[0]),
            float(ctx.ephem.sun_dec_deg[0]),
            0.0,
        )

        assert ctx.attitude_violation(sun, T0, ACSMode.SCIENCE) is not None
        assert ctx.first_hold_violation(sun, T0, T0 + 180, ACSMode.SCIENCE) == T0

    def test_visibility_windows_exclude_the_sun(self, ctx: SchedulingContext) -> None:
        sun = Pointing(
            ra=float(ctx.ephem.sun_ra_deg[0]), dec=float(ctx.ephem.sun_dec_deg[0])
        )
        anti_sun = Pointing(ra=(sun.ra + 180.0) % 360.0, dec=-sun.dec)

        assert ctx.visibility_windows(sun) == []
        assert ctx.visibility_windows(anti_sun) != []


class TestPasses:
    def test_only_passes_inside_the_horizon(self, config: MissionConfig) -> None:
        config.ground_stations = GroundStationRegistry.default()
        ctx = SchedulingContext(config, BEGIN, END)

        passes = ctx.predict_passes()

        assert all(T0 <= p.begin and p.end <= END.timestamp() for p in passes)
