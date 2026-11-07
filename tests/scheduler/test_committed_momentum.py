"""Rolling resource forecasts follow committed spacecraft trajectories."""

from datetime import timedelta

import numpy as np
import pytest

from conops.ditl import RollingHorizonDITL
from conops.schedulers import PriorityScheduler
from conops.schedulers.context import SchedulingContext
from conops.schedulers.momentum import MomentumForecast
from conops.schedulers.priority_scheduler import StartState
from conops.targets import Pointing

from .test_momentum_ledger import BEGIN, T0, _config


@pytest.mark.parametrize("idle_seconds", [0, 180])
@pytest.mark.parametrize("in_progress", [False, True])
def test_committed_momentum_matches_the_planned_slew_and_final_hold(
    idle_seconds, in_progress
):
    config = _config()
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    origin = (10.0, 20.0, 30.0)
    initial = (0.1, -0.2, 0.3)
    target = Pointing(
        config=config,
        obsid=1,
        name="Committed exposure",
        ra=150.0,
        dec=-20.0,
        exptime=300,
        ss_min=300,
        ss_max=300,
    )
    scheduler = PriorityScheduler(
        config,
        [target],
        BEGIN,
        BEGIN + timedelta(hours=1),
        start_state=StartState(time=T0, attitude=origin),
        start_momentum_eci_n_m_s=initial,
        include_passes=False,
    )
    [entry] = scheduler.schedule().entries
    assert entry.slewtime > 120
    stop = float(entry.end) + idle_seconds
    context = SchedulingContext(config, BEGIN, BEGIN + timedelta(seconds=stop - T0))
    physical = MomentumForecast(
        config,
        context,
        start_attitude=origin,
        start_momentum_eci_n_m_s=initial,
    )
    physical.profile(scheduler.timeline)
    expected = physical.tracker.momentum_eci_n_m_s.copy()
    rolling = RollingHorizonDITL(config, [], begin=BEGIN, end=context.end)
    rolling.step_size = context.step_size
    rolling.acs.ra, rolling.acs.dec, rolling.acs.roll = origin
    at = T0
    if in_progress:
        # A replan can begin part-way through the already installed ingress.
        # Its past torque is in the wheel state and must not be integrated twice.
        slew = scheduler.timeline[0].slew
        at = T0 + 120
        prefix = rolling._forecast_momentum(
            T0, at, slew.attitude, start_eci_n_m_s=initial, slew=slew
        )
        rolling.acs.last_slew = slew
        rolling.acs.ra, rolling.acs.dec, rolling.acs.roll = slew.attitude(at)
        initial = prefix.end_eci_n_m_s
    rolling._stored_momentum_tracker.start_from_eci(initial)
    before = rolling._stored_momentum_tracker.momentum_eci_n_m_s.copy()
    actual = rolling._momentum_start(at, stop, [entry])
    np.testing.assert_allclose(actual, expected, rtol=1e-9, atol=1e-10)
    # Forecasting must leave the live wheel state at the replan origin.
    np.testing.assert_array_equal(
        rolling._stored_momentum_tracker.momentum_eci_n_m_s, before
    )
