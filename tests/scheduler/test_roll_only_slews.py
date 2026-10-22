"""Roll-only slews: their own duration, and sequence steps at roll offsets."""

from datetime import timedelta

import pytest

from conops import DITL, create_ditl
from conops.config import MissionConfig, ObservationTiming
from conops.schedulers import PriorityScheduler
from conops.simulation.slew import Slew
from conops.targets import ObservationSequence, Plan, Pointing, SequenceStep

from .planning_scenario import BEGIN, MIN, PATCH, T0
from .planning_scenario import make_config as _config

HOURS = 4
STEP = 60
END = BEGIN + timedelta(hours=HOURS)
ROLL_STEPS = ObservationSequence(
    steps=[
        SequenceStep(name="roll A", duration_seconds=10 * MIN),
        SequenceStep(name="roll B", duration_seconds=10 * MIN, roll_offset_deg=-15.0),
        SequenceStep(name="roll C", duration_seconds=10 * MIN, roll_offset_deg=15.0),
    ]
)
TIMING = ObservationTiming(setup_seconds=120, cleanup_seconds=60)


def _delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _roll_config(hours: int = HOURS, roll_only: float | None = 300.0) -> MissionConfig:
    config = _config(hours)
    config.payload.observation_timing = TIMING
    acs = config.spacecraft_bus.attitude_control
    acs.roll_only_slew_seconds = roll_only
    acs.max_roll_from_optimum_deg = 15.0
    return config


def _roll_target(config: MissionConfig) -> Pointing:
    return Pointing(
        config=config,
        obsid=600,
        name="Roll campaign",
        ra=PATCH[0][0],
        dec=PATCH[0][1],
        merit=95.0,
        sequence=ROLL_STEPS,
    )


def _slew(config: MissionConfig, start: tuple, end: tuple) -> Slew:
    slew = Slew(config=config)
    slew.ephem = config.constraint.ephem
    slew.slewrequest = slew.slewstart = T0
    slew.startra, slew.startdec, slew.startroll = start
    slew.endra, slew.enddec, slew.endroll = end
    slew.calc_slewtime()
    return slew


class TestSlewModel:
    def test_a_roll_only_slew_is_recognised(self) -> None:
        config = _roll_config()
        assert _slew(config, (10.0, 20.0, 30.0), (10.0, 20.0, 45.0)).is_roll_only
        assert not _slew(config, (10.0, 20.0, 30.0), (12.0, 20.0, 45.0)).is_roll_only
        assert not _slew(config, (10.0, 20.0, 30.0), (10.0, 20.0, 30.0)).is_roll_only

    def test_a_roll_only_slew_takes_its_own_time_plus_settle(self) -> None:
        config = _roll_config(roll_only=300.0)
        settle = config.spacecraft_bus.attitude_control.settle_time
        roll = _slew(config, (10.0, 20.0, 30.0), (10.0, 20.0, 45.0))
        assert roll.slewtime == 300 + settle
        assert roll.slewdist == pytest.approx(15.0)
        # A slew that moves the boresight keeps the kinematic time.
        moved = _slew(config, (10.0, 20.0, 30.0), (12.0, 20.0, 45.0))
        kinematic = config.spacecraft_bus.attitude_control.slew_time(moved.slewdist)
        assert moved.slewtime == pytest.approx(kinematic, abs=1.0)
        # Without the setting a roll is timed by the rates like any rotation.
        plain = _roll_config(roll_only=None)
        roll = _slew(plain, (10.0, 20.0, 30.0), (10.0, 20.0, 45.0))
        assert roll.slewtime == pytest.approx(
            plain.spacecraft_bus.attitude_control.slew_time(15.0), abs=1.0
        )

    def test_the_duration_bound_covers_a_roll_only_slew(self) -> None:
        config = _roll_config(roll_only=3000.0)
        acs = config.spacecraft_bus.attitude_control
        assert Slew.duration_upper_bound(acs) >= 3000 + acs.settle_time
        assert Slew.duration_upper_bound(acs) >= Slew._kinematic_duration_upper_bound(
            acs
        )
        assert Slew.duration_upper_bound(
            _roll_config(roll_only=None).spacecraft_bus.attitude_control
        ) == Slew._kinematic_duration_upper_bound(acs)


class TestStepRolls:
    def test_step_pointings_carry_their_roll_requests(self) -> None:
        config = _roll_config()
        target = _roll_target(config)
        a, b, c = (target.step_pointing(i) for i in range(3))
        assert (a.roll_deg, a.roll_offset_deg) == (None, None)
        assert (b.roll_deg, b.roll_offset_deg) == (None, -15.0)
        assert (c.roll_deg, c.roll_offset_deg) == (None, 15.0)
        target.roll_offset_deg = 5.0
        assert target.step_pointing(1).roll_offset_deg == -10.0
        target.roll_offset_deg = None
        target.roll_deg = 100.0
        assert target.step_pointing(2).roll_deg == 115.0
        assert target.step_pointing(0).roll_deg == 100.0

    def test_the_plan_rolls_between_steps_and_executes(self) -> None:
        config = _roll_config()
        scheduler = PriorityScheduler(
            config, [_roll_target(config)], BEGIN, END, step_size=STEP
        )
        plan = scheduler.schedule()
        steps = [e for e in plan if e.obsid == 600]
        assert [e.step for e in steps] == ["roll A", "roll B", "roll C"]
        rolls = [float(e.roll) for e in steps]
        assert _delta(rolls[1], rolls[0] - 15.0) < 1e-6
        assert _delta(rolls[2], rolls[0] + 15.0) < 1e-6
        # The steps stay on the pointing and slew in roll only, each taking
        # the roll-only time plus settling.
        settle = config.spacecraft_bus.attitude_control.settle_time
        assert all((e.ra, e.dec) == PATCH[0] for e in steps)
        assert steps[1].slewtime == 300 + settle
        assert steps[2].slewtime == 300 + settle
        assert steps[1].slewdist == pytest.approx(15.0)
        assert steps[2].slewdist == pytest.approx(30.0)
        replay = Plan.model_validate_json(plan.model_dump_json())
        ditl = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=replay,
            begin=BEGIN,
            end=END,
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []

    def test_a_visit_arrives_at_its_first_step(self) -> None:
        """A visit whose first step points elsewhere than the target is
        reached at that step's attitude."""
        config = _roll_config()
        away = ObservationSequence(
            steps=[
                SequenceStep(
                    name="reference",
                    duration_seconds=5 * MIN,
                    ra=PATCH[2][0],
                    dec=PATCH[2][1],
                ),
                SequenceStep(name="target", duration_seconds=10 * MIN),
            ]
        )
        target = _roll_target(config)
        target = Pointing(
            config=config,
            obsid=601,
            name="Away",
            ra=PATCH[0][0],
            dec=PATCH[0][1],
            merit=90.0,
            sequence=away,
        )
        scheduler = PriorityScheduler(config, [target], BEGIN, END, step_size=STEP)
        plan = scheduler.schedule()
        steps = [e for e in plan if e.obsid == 601]
        assert (steps[0].ra, steps[0].dec) == PATCH[2]
        visit = next(b for b in scheduler.timeline if b.is_visit)
        assert visit.attitude_in[:2] == pytest.approx(PATCH[2])
        replay = Plan.model_validate_json(plan.model_dump_json())
        ditl = DITL(
            config=config,
            ephem=config.constraint.ephem,
            plan=replay,
            begin=BEGIN,
            end=END,
        )
        ditl.step_size = STEP
        assert ditl.calc()
        assert ditl.validate_plan_matches_execution() == []

    def test_dispatch_rolls_between_steps(self) -> None:
        config = _roll_config()
        ditl = create_ditl(config, [_roll_target(config)], BEGIN, END)
        ditl.step_size = STEP
        assert ditl.calc()
        steps = [e for e in ditl.plan if e.obsid == 600][:3]
        assert [e.step for e in steps] == ["roll A", "roll B", "roll C"]
        rolls = [float(e.roll) for e in steps]
        assert _delta(rolls[1], rolls[0] - 15.0) < 0.5
        assert _delta(rolls[2], rolls[0] + 15.0) < 0.5
        settle = config.spacecraft_bus.attitude_control.settle_time
        assert steps[1].slewtime == 300 + settle and steps[2].slewtime == 300 + settle
        assert not any("cut short" in e.description for e in ditl.log.events)
