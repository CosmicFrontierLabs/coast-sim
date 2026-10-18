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
from conops.schedulers import PriorityScheduler, SchedulingContext
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


class TestHardwareEphemeris:
    """A plan can be built before any DITL exists, so the context gives the
    star trackers and radiators the ephemeris, as a DITL does."""

    @pytest.fixture
    def default_config(self) -> MissionConfig:
        # The default spacecraft has a star tracker and a radiator.
        config = MissionConfig()
        config.constraint.ephem = TLEEphemeris(
            tle="examples/example.tle", begin=BEGIN, end=END, step_size=60
        )
        return config

    def test_star_trackers_and_radiators_get_the_ephemeris(
        self, default_config: MissionConfig
    ) -> None:
        ephem = default_config.constraint.ephem
        bus = default_config.spacecraft_bus
        assert bus.star_trackers.num_trackers() > 0
        assert bus.radiators.num_radiators() > 0

        SchedulingContext(default_config, BEGIN, END)

        hardware = [
            constraint
            for tracker in bus.star_trackers.star_trackers
            for constraint in (tracker.hard_constraint, tracker.soft_constraint)
        ] + [radiator.hard_constraint for radiator in bus.radiators.radiators]
        attached = [constraint for constraint in hardware if constraint is not None]
        assert attached
        assert all(constraint.ephem is ephem for constraint in attached)

    def test_a_scheduler_plans_before_any_ditl_exists(
        self, default_config: MissionConfig
    ) -> None:
        target = Pointing(
            config=default_config, ra=105.0, dec=10.0, obsid=1, ss_min=60, ss_max=600
        )
        target.exptime = 600

        plan = PriorityScheduler(default_config, [target], BEGIN, END).schedule()

        assert any(int(e.obsid) == 1 for e in plan)


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

    def test_initial_attitude_replay_does_not_print_its_events(
        self,
        ctx: SchedulingContext,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        pointing = ACS.pointing

        def pointing_with_event(
            acs: ACS, utime: float
        ) -> tuple[float, float, float, int]:
            acs._log_or_print(utime, "ACS", "replay event")
            return pointing(acs, utime)

        monkeypatch.setattr(ACS, "pointing", pointing_with_event)

        ctx.initial_attitude(T0 + 120)

        assert "replay event" not in capsys.readouterr().out


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

    def test_predicts_passes_after_midnight(self, config: MissionConfig) -> None:
        begin = BEGIN + timedelta(hours=18)
        end = begin + timedelta(hours=18)
        config.ground_stations = GroundStationRegistry.default()
        # PassTimes.get counts days from midnight, where this ephemeris starts.
        config.constraint.ephem = TLEEphemeris(
            tle="examples/example.tle", begin=BEGIN, end=end, step_size=60
        )
        ctx = SchedulingContext(config, begin, end)

        passes = ctx.predict_passes()

        midnight = (BEGIN + timedelta(days=1)).timestamp()
        assert any(p.begin >= midnight for p in passes)
