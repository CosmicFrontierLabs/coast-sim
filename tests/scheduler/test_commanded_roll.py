"""Commanded rolls: a target observed at an offset from the power-optimal
roll, or at an absolute roll, within the ACS roll limit."""

from datetime import timedelta

import pytest
from pydantic import ValidationError

from conops import DITL, QueueDITL, create_ditl
from conops.config import MissionConfig
from conops.schedulers import PriorityScheduler
from conops.schedulers.context import SchedulingContext
from conops.targets import Plan, Pointing

from .planning_scenario import BEGIN, PATCH, T0
from .planning_scenario import make_config as _config
from .planning_scenario import make_target as _target

HOURS = 3
STEP = 60
END = BEGIN + timedelta(hours=HOURS)


def _delta(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def _target_with(config: MissionConfig, **roll: float) -> Pointing:
    target = _target(
        config, 100, PATCH[0][0], PATCH[0][1], merit=90, minutes=40, snapshot=20
    )
    for key, value in roll.items():
        setattr(target, key, value)
    return target


class TestRequest:
    def test_one_request_at_a_time(self) -> None:
        config = _config(1)
        with pytest.raises(ValueError, match="not both"):
            Pointing(
                config=config,
                obsid=1,
                ra=10.0,
                dec=10.0,
                roll_deg=5.0,
                roll_offset_deg=5.0,
            )

    def test_the_requested_roll(self) -> None:
        config = _config(1)
        plain = _target_with(config)
        assert not plain.commands_roll and plain.requested_roll(100.0, 15.0) == 100.0
        offset = _target_with(config, roll_offset_deg=-15.0)
        assert offset.requested_roll(10.0, 15.0) == pytest.approx(355.0)
        with pytest.raises(ValueError, match="exceeds"):
            _target_with(config, roll_offset_deg=20.0).requested_roll(10.0, 15.0)
        assert (
            _target_with(config, roll_offset_deg=20.0).requested_roll(10.0, None)
            == 30.0
        )
        absolute = _target_with(config, roll_deg=350.0)
        assert absolute.requested_roll(0.0, 15.0) == 350.0
        assert absolute.requested_roll(30.0, 15.0) is None
        assert absolute.requested_roll(30.0, None) == 350.0

    def test_round_trips_and_leaves_defaults_out(self) -> None:
        config = _config(1)
        target = _target_with(config, roll_offset_deg=15.0)
        loaded = Pointing.model_validate_json(target.model_dump_json())
        assert loaded.roll_offset_deg == 15.0 and loaded.roll_deg is None
        dumped = _target_with(config).model_dump()
        assert "roll_deg" not in dumped and "roll_offset_deg" not in dumped


class TestContext:
    def test_the_context_applies_the_request(self) -> None:
        config = _config(HOURS)
        config.spacecraft_bus.attitude_control.max_roll_from_optimum_deg = 15.0
        ctx = SchedulingContext(config, BEGIN, END, step_size=STEP)
        plain = _target_with(config)
        optimum = ctx.optimum_roll(plain, T0)
        assert ctx.instrument_roll(plain, T0) == optimum
        offset = _target_with(config, roll_offset_deg=10.0)
        assert _delta(ctx.instrument_roll(offset, T0), optimum + 10.0) < 1e-9
        assert ctx.roll_allowed(offset, T0)
        far = _target_with(config, roll_deg=optimum + 90.0)
        assert not ctx.roll_allowed(far, T0)
        near = _target_with(config, roll_deg=optimum + 5.0)
        assert ctx.roll_allowed(near, T0)
        assert _delta(ctx.instrument_roll(near, T0), optimum + 5.0) < 1e-9


class TestPlanning:
    def test_the_plan_observes_at_the_offset_and_executes(self) -> None:
        config = _config(HOURS)
        config.spacecraft_bus.attitude_control.max_roll_from_optimum_deg = 15.0
        targets = [_target_with(config, roll_offset_deg=12.0)]
        scheduler = PriorityScheduler(config, targets, BEGIN, END, step_size=STEP)
        plan = scheduler.schedule()
        entries = [e for e in plan if e.obsid == 100]
        assert entries
        for entry in entries:
            optimum = scheduler.ctx.optimum_roll(targets[0], float(entry.begin))
            assert _delta(float(entry.roll), optimum + 12.0) < 1e-6
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

    def test_an_absolute_roll_beyond_the_limit_is_not_placed(self) -> None:
        config = _config(HOURS)
        config.spacecraft_bus.attitude_control.max_roll_from_optimum_deg = 15.0
        ctx = SchedulingContext(config, BEGIN, END, step_size=STEP)
        probe = _target_with(config)
        optimum = ctx.optimum_roll(probe, T0)
        far = _target_with(config, roll_deg=optimum + 120.0)
        plan = PriorityScheduler(config, [far], BEGIN, END, step_size=STEP).schedule()
        # The optimum barely moves in three hours: never within reach.
        assert [e for e in plan if e.obsid == 100] == []
        near = _target_with(config, roll_deg=optimum + 10.0)
        plan = PriorityScheduler(config, [near], BEGIN, END, step_size=STEP).schedule()
        entries = [e for e in plan if e.obsid == 100]
        assert entries and all(
            _delta(float(e.roll), optimum + 10.0) < 1e-6 for e in entries
        )


class TestDispatch:
    def test_the_queue_adds_a_target_with_its_roll_request(self) -> None:
        config = _config(1)
        ditl = create_ditl(config, [], BEGIN, BEGIN + timedelta(hours=1))
        assert isinstance(ditl, QueueDITL)
        offset = ditl.queue.add(
            ra=PATCH[0][0], dec=PATCH[0][1], obsid=1, roll_offset_deg=10.0
        )
        absolute = ditl.queue.add(
            ra=PATCH[1][0], dec=PATCH[1][1], obsid=2, roll_deg=200.0
        )
        plain = ditl.queue.add(ra=PATCH[2][0], dec=PATCH[2][1], obsid=3)
        assert (offset.roll_deg, offset.roll_offset_deg) == (None, 10.0)
        assert (absolute.roll_deg, absolute.roll_offset_deg) == (200.0, None)
        assert not plain.commands_roll
        with pytest.raises(ValidationError, match="not both"):
            ditl.queue.add(ra=0.0, dec=0.0, obsid=4, roll_deg=1.0, roll_offset_deg=1.0)

    def test_dispatch_observes_at_the_requested_roll(self) -> None:
        config = _config(HOURS)
        config.spacecraft_bus.attitude_control.max_roll_from_optimum_deg = 15.0
        offset = _target_with(config, roll_offset_deg=-8.0)
        absolute = _target(
            config, 101, PATCH[1][0], PATCH[1][1], merit=80, minutes=40, snapshot=20
        )
        ditl = create_ditl(config, [offset, absolute], BEGIN, END)
        ditl.step_size = STEP
        probe_ctx = SchedulingContext(config, BEGIN, END, step_size=STEP)
        absolute_roll = probe_ctx.optimum_roll(absolute, T0) + 6.0
        queued = next(t for t in ditl.queue.targets if t.obsid == 101)
        queued.roll_deg = absolute_roll
        assert ditl.calc()
        for entry in ditl.plan:
            if entry.obsid == 100:
                optimum = probe_ctx.optimum_roll(
                    offset, float(entry.begin) + entry.slewtime
                )
                assert _delta(float(entry.roll), optimum - 8.0) < 0.5
            if entry.obsid == 101:
                assert _delta(float(entry.roll), absolute_roll) < 1e-6
        observed = {e.obsid for e in ditl.plan}
        assert {100, 101} <= observed

    def test_dispatch_skips_a_roll_beyond_the_limit(self) -> None:
        config = _config(HOURS)
        config.spacecraft_bus.attitude_control.max_roll_from_optimum_deg = 15.0
        probe = _target_with(config)
        probe_ctx = SchedulingContext(config, BEGIN, END, step_size=STEP)
        far = _target_with(config, roll_deg=probe_ctx.optimum_roll(probe, T0) + 150.0)
        ditl = create_ditl(config, [far], BEGIN, END)
        ditl.step_size = STEP
        assert ditl.calc()
        assert [e for e in ditl.plan if e.obsid == 100] == []
        assert any("beyond the ACS limit" in e.description for e in ditl.log.events)
