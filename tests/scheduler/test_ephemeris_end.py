"""Nothing is planned, or checked, past the ephemeris's last sample."""

from datetime import timedelta

from conops import QueueDITL, create_ditl
from conops.common import ACSMode
from conops.ditl import RollingHorizonDITL
from conops.schedulers import PriorityScheduler, SchedulingContext
from conops.simulation.slew import Slew

from .planning_scenario import BEGIN, MIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 6
STEP = 60
EPHEM_END = T0 + HOURS * 3600


def _targets(config):  # type: ignore[no-untyped-def]
    return [
        _target(config, 1 + k, ra, dec, merit=60.0 - k, minutes=400, snapshot=20)
        for k, (ra, dec) in enumerate(PATCH[:3])
    ]


class TestContext:
    def test_the_context_ends_with_the_ephemeris(self) -> None:
        """A horizon reaching past the ephemeris ends at its last sample,
        and the step grid never samples beyond it."""
        config = _config(HOURS)
        ctx = SchedulingContext(
            config, BEGIN, BEGIN + timedelta(hours=HOURS + 2), step_size=STEP
        )
        assert ctx.ephem_end == EPHEM_END
        assert ctx.uend == EPHEM_END
        assert ctx.end.timestamp() == EPHEM_END
        steps = ctx.steps(EPHEM_END - 5 * MIN, EPHEM_END + 10 * MIN)
        assert len(steps) == 6 and steps[-1] == EPHEM_END
        assert len(ctx.steps(EPHEM_END + STEP, EPHEM_END + 10 * MIN)) == 0
        within = SchedulingContext(
            config, BEGIN, BEGIN + timedelta(hours=2), step_size=STEP
        )
        assert within.uend == T0 + 2 * 3600

    def test_a_slew_reaching_past_the_ephemeris_is_checked_up_to_it(self) -> None:
        config = _config(HOURS)
        ctx = SchedulingContext(config, BEGIN, BEGIN + timedelta(hours=HOURS))
        slew = Slew(config=config, slewstart=EPHEM_END - 2 * MIN)
        slew.startra, slew.startdec, slew.startroll = PATCH[0][0], PATCH[0][1], 0.0
        slew.endra, slew.enddec, slew.endroll = PATCH[3][0], PATCH[3][1], 0.0
        slew.slewtime = 10 * MIN
        slew.slewend = slew.slewstart + slew.slewtime
        assert ctx.first_slew_violation(slew, ACSMode.SLEWING) is None
        assert (
            ctx.first_hold_violation(
                (PATCH[0][0], PATCH[0][1], 0.0),
                EPHEM_END - 2 * MIN,
                EPHEM_END + 10 * MIN,
                ACSMode.IDLE,
            )
            is None
        )


class TestPlanners:
    def test_the_priority_plan_stops_at_the_ephemeris(self) -> None:
        """Asked to plan two hours past the ephemeris, the planner plans to
        its end and no further, without sampling past it."""
        config = _config(HOURS)
        scheduler = PriorityScheduler(
            config,
            _targets(config),
            BEGIN,
            BEGIN + timedelta(hours=HOURS + 2),
            step_size=STEP,
        )
        plan = scheduler.schedule()
        assert scheduler.ctx.uend == EPHEM_END
        assert len(plan) >= 3
        assert all(float(e.end) <= EPHEM_END for e in plan)
        assert max(float(e.end) for e in plan) > EPHEM_END - 3600

    def test_a_rolling_run_replans_near_the_ephemeris_end(self) -> None:
        """A target of opportunity late in a rolling run forces a rapid
        replan whose horizon reaches past the ephemeris: the run completes
        and plans nothing beyond it."""
        config = _config(HOURS)
        ditl = RollingHorizonDITL(
            config,
            _targets(config),
            begin=BEGIN,
            end=BEGIN + timedelta(hours=HOURS),
            horizon=timedelta(hours=5),
            replan_interval=timedelta(hours=2),
        )
        ditl.step_size = STEP
        ditl.submit_too(
            obsid=1_000_001,
            name="Late",
            ra=PATCH[4][0],
            dec=PATCH[4][1],
            merit=500.0,
            exptime=600,
            submit_time=EPHEM_END - 40 * MIN,
            deadline=EPHEM_END - 10 * MIN,
        )
        assert ditl.calc()
        assert any(r.utime >= EPHEM_END - 40 * MIN for r in ditl.replans)
        assert all(float(e.end) <= EPHEM_END for e in ditl.plan)
        assert ditl.validate_plan_matches_execution() == []

    def test_dispatch_is_unchanged(self) -> None:
        config = _config(HOURS)
        ditl = create_ditl(
            config, _targets(config), BEGIN, BEGIN + timedelta(hours=HOURS)
        )
        assert isinstance(ditl, QueueDITL)
        ditl.step_size = STEP
        assert ditl.calc()
        assert all(float(e.end) <= EPHEM_END for e in ditl.plan)
