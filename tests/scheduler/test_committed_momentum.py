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


@pytest.mark.parametrize(
    "handover,idle_seconds,cut_seconds",
    [
        (0.0, 0, None),
        (300.0, 0, None),
        (0.0, 120, None),
        (300.0, 120, None),
        (1200.01, 120, None),
        (30.0, 120, 45),
        (300.0, 120, 319),
    ],
)
def test_committed_moving_momentum_tracks_and_brakes(
    handover, idle_seconds, cut_seconds
):
    from conops.common import ObsType
    from conops.simulation.attitude import AttitudeTrajectory
    from conops.simulation.slew import Slew
    from conops.targets import PlanEntry, track_from_table

    config = _config()
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    limits = config.spacecraft_bus.attitude_control
    origin = (80.0, -20.0, 10.0)
    initial = (0.1, -0.2, 0.3)
    end = T0 + 1200
    stop = end + idle_seconds
    entry = PlanEntry(
        config=config,
        obsid=702,
        obstype=ObsType.PPT,
        ra=80,
        dec=-20,
        roll=10,
        begin=T0,
        end=end,
        slewtime=0,
        track=track_from_table([T0, T0 + 3600], [80, 100], [-20, -20]),
    )
    run = RollingHorizonDITL(
        config, [], begin=BEGIN, end=BEGIN + timedelta(seconds=stop - T0)
    )
    run._entry_passes = {}
    run.step_size = 10
    run.acs.ra, run.acs.dec, run.acs.roll = origin
    slew = Slew.idle_hold(config, *origin, T0)
    slew.obstype, slew.at, slew.obsid = ObsType.PPT, entry, entry.obsid
    ingress = AttitudeTrajectory.from_slew(slew, limits)
    # This is the continuous trajectory ACS installs, including its physical
    # terminal brake. Comparing two resource forecasters would miss a shared
    # assumption that a moving observation holds one attitude.
    tracking = run.acs._tracking_after(slew, ingress, limits)
    assert tracking is not None and tracking.end > end
    if cut_seconds is not None:
        end = T0 + cut_seconds
        entry = entry.model_copy(update={"end": end})
        stop = end + idle_seconds
        brake = AttitudeTrajectory.braking(tracking.state(end), limits)
        assert brake is not None
    else:
        brake = None

    def physical_at(at):
        if at > end and brake is not None:
            return brake.state(at).attitude
        return tracking.state(at).attitude

    at = T0 + handover
    if handover:
        executor = run.acs._advance_attitude(T0)
        executor.install(tracking)
        run.acs._tracked_slew = slew
        initial = run._forecast_momentum(
            T0,
            at,
            physical_at,
            start_eci_n_m_s=initial,
        ).end_eci_n_m_s
    # Integrate through the physical stop, then the following stationary hold.
    # The stop is a torque-integration boundary, even when it falls off-grid.
    rest = float(brake.end if brake is not None else tracking.end)
    expected = run._forecast_momentum(
        at,
        min(stop, rest),
        physical_at,
        start_eci_n_m_s=initial,
    ).end_eci_n_m_s
    if stop > rest:
        expected = run._forecast_momentum(
            rest, stop, physical_at, start_eci_n_m_s=expected
        ).end_eci_n_m_s
    run._stored_momentum_tracker.start_from_eci(initial)
    before = run._stored_momentum_tracker.momentum_eci_n_m_s.copy()
    actual = run._momentum_start(at, stop, [entry])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(
        run._stored_momentum_tracker.momentum_eci_n_m_s, before
    )
