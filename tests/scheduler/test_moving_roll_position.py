"""Roll guidance uses a moving request's position at its evaluation time."""

from datetime import timedelta

import pytest

from conops.schedulers.context import SchedulingContext
from conops.targets import MovingPointing, Pointing, track_from_table

from .planning_scenario import BEGIN, T0, make_config


@pytest.mark.parametrize("offset_seconds", [900, 1800, 3600])
@pytest.mark.parametrize("roll_offset", [None, 10.0])
def test_moving_roll_matches_a_fixed_target_at_the_same_instantaneous_position(
    offset_seconds: int, roll_offset: float | None
) -> None:
    config = make_config(1)
    ctx = SchedulingContext(config, BEGIN, BEGIN + timedelta(hours=1))
    moving = MovingPointing(
        config=config,
        obsid=701,
        ra=105,
        dec=10,
        roll_offset_deg=roll_offset,
        track=track_from_table([T0, T0 + 3600], [105, 115], [10, 10]),
    )
    at = T0 + offset_seconds
    ra, dec = moving.position_at(at)
    instantaneous = Pointing(
        config=config, obsid=702, ra=ra, dec=dec, roll_offset_deg=roll_offset
    )
    assert ctx.optimum_roll(moving, at) == ctx.optimum_roll(instantaneous, at)
    assert ctx.instrument_roll(moving, at) == ctx.instrument_roll(instantaneous, at)
    assert (moving.ra, moving.dec) == (105, 10)


def test_absolute_moving_roll_is_checked_against_the_current_position() -> None:
    config = make_config(1)
    config.spacecraft_bus.attitude_control.max_roll_from_optimum_deg = 0.1
    ctx = SchedulingContext(config, BEGIN, BEGIN + timedelta(hours=1))
    moving = MovingPointing(
        config=config,
        obsid=701,
        ra=105,
        dec=10,
        track=track_from_table([T0, T0 + 3600], [105, 115], [10, 10]),
    )
    at = T0 + 1800
    ra, dec = moving.position_at(at)
    instantaneous = Pointing(config=config, obsid=702, ra=ra, dec=dec)
    current_roll = ctx.optimum_roll(instantaneous, at)
    moving.roll_deg = current_roll
    assert ctx.roll_allowed(moving, at)
    moving.roll_deg = ctx.optimum_roll(
        Pointing(config=config, obsid=703, ra=105, dec=10), at
    )
    assert not ctx.roll_allowed(moving, at)
