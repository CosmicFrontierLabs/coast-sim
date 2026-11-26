"""CP-SAT's representative moving-target pose and power use one epoch."""

from datetime import timedelta

import numpy as np
import pytest

pytest.importorskip("ortools")

from conops.common import ACSMode  # noqa: E402
from conops.config import SolarPanel, SolarPanelSet  # noqa: E402
from conops.schedulers import CpSatScheduler  # noqa: E402
from conops.targets import MovingPointing, track_from_table  # noqa: E402

from .planning_scenario import BEGIN, T0, make_config  # noqa: E402


def _scenario() -> tuple[CpSatScheduler, MovingPointing, float]:
    config = make_config(1)
    config.solar_panel = SolarPanelSet(
        panels=[SolarPanel(name="panel", normal=(0, 1, 0), max_power=600)]
    )
    config.battery.planning_level = 0.5
    moving = MovingPointing(
        config=config,
        obsid=701,
        ra=105,
        dec=10,
        exptime=600,
        ss_min=600,
        ss_max=600,
        track=track_from_table([T0, T0 + 3600], [105, 115], [10, 10]),
    )
    scheduler = CpSatScheduler(
        config,
        [moving],
        BEGIN,
        BEGIN + timedelta(hours=1),
        include_passes=False,
        max_iterations=0,
        time_limit=0,
    )
    scheduler._start_from_priority_plan()
    when = next(
        float(t)
        for t in scheduler._battery.grid
        if float(t) >= T0 + 600 and not scheduler._battery.in_eclipse(float(t))
    )
    return scheduler, moving, when


def test_candidate_power_uses_the_tracked_pointing() -> None:
    scheduler, moving, when = _scenario()
    ledger = scheduler._battery
    pose = moving.target_body_attitude(
        scheduler.ctx.instrument_roll(moving, when), at=when
    )
    modes = moving.effective_instrument_modes()
    expected = (
        int(
            np.ceil(ledger._watts(ACSMode.SCIENCE, False, modes))
            - np.floor(ledger.generation_watts(pose, when))
        ),
        int(np.ceil(ledger._watts(ACSMode.SCIENCE, True, modes))),
    )
    assert scheduler._candidate_power(moving, when) == expected
    assert (moving.ra, moving.dec) == (105, 10)


def test_candidate_slew_pose_uses_the_same_epoch_as_its_roll() -> None:
    scheduler, moving, when = _scenario()
    state = scheduler._initial_state()
    tasks = scheduler._candidates(
        state, dict(state.remaining), when, T0 + 3600, T0 + 3600, {}
    )
    assert tasks
    expected = moving.target_body_attitude(
        scheduler.ctx.instrument_roll(moving, when), at=when
    )
    assert moving.position_at(when) != (moving.ra, moving.dec)
    for task in tasks:
        assert task.attitude_in == pytest.approx(expected, abs=1e-9, rel=0)
        assert task.attitude_out == pytest.approx(expected, abs=1e-9, rel=0)
    assert (moving.ra, moving.dec) == (105, 10)
