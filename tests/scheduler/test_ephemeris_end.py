"""Nothing is planned, or checked, past the ephemeris's last sample."""

from datetime import timedelta

from conops.ditl import RollingHorizonDITL
from conops.schedulers import PriorityScheduler

from .planning_scenario import BEGIN, MIN, PATCH, T0, patch_targets
from .planning_scenario import make_config as _config

HOURS = 6
STEP = 60
EPHEM_END = T0 + HOURS * 3600


def _targets(config):  # type: ignore[no-untyped-def]
    return patch_targets(config, 3, obsid=1, merit=60, minutes=400, snapshot=20)


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
