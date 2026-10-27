"""Momentum forecasts follow the physical trajectories ACS installs: committed
slews and holds, ground contacts, and moving-target tracking."""

from datetime import timedelta

import numpy as np
import pytest

from conops.common import ACSCommandType, ObsType
from conops.ditl import RollingHorizonDITL
from conops.schedulers import PriorityScheduler
from conops.schedulers.context import SchedulingContext
from conops.schedulers.momentum import MomentumForecast
from conops.schedulers.priority_scheduler import StartState, _Block
from conops.simulation.acs_command import ACSCommand
from conops.simulation.attitude import AttitudeExecutor, AttitudeTrajectory
from conops.simulation.passes import Pass
from conops.simulation.slew import Slew
from conops.targets import PlanEntry, Pointing, track_from_table

from .momentum_scenario import BEGIN, T0
from .momentum_scenario import momentum_config as _config

INITIAL = (0.1, -0.2, 0.3)


def _contact(late=0, wait=0):
    config = _config()
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    gspass = Pass(
        config=config,
        station="Test station",
        obsid=900,
        begin=T0,
        length=600,
        utime=[T0 + k * 60 for k in range(11)],
        ra=[80 + k * 0.8 for k in range(11)],
        dec=[-20] * 11,
        roll=[10] * 11,
    )
    acquire = T0 + late
    begin = acquire - wait
    profile = list(zip(gspass.ra, gspass.dec, gspass.roll, strict=True))
    origin = gspass.attitude_for_profile_at(profile, acquire)
    assert origin is not None
    entry = PlanEntry(
        config=config,
        obstype=ObsType.GSP,
        obsid=gspass.obsid,
        begin=begin,
        end=gspass.end,
        contact_begin=acquire,
        contact_end=gspass.end,
        slewtime=0,
        ra=origin[0],
        dec=origin[1],
        roll=origin[2],
    )
    run = RollingHorizonDITL(
        config, [], begin=BEGIN, end=BEGIN + timedelta(seconds=720)
    )
    run.step_size = 10
    run._entry_passes = {id(entry): gspass}
    run.acs.ra, run.acs.dec, run.acs.roll = origin
    run.acs._executor = AttitudeExecutor(begin, origin)
    run.acs.passrequests.passes = [gspass]
    run.acs._start_pass(
        ACSCommand(command_type=ACSCommandType.START_PASS, execution_time=acquire),
        acquire,
    )
    motion = run.acs._executor._trajectory
    assert motion is not None and motion.end > entry.end
    context = SchedulingContext(
        config, BEGIN + timedelta(seconds=begin - T0), BEGIN + timedelta(seconds=720)
    )
    block = _Block(
        entry=entry,
        attitude_in=origin,
        attitude_out=motion.state(motion.end).attitude,
        ready=acquire,
        end=motion.end,
        gspass=gspass,
    )
    return config, context, block, motion, run


def _physical(run, motion, origin, begin, end, initial):
    # The ACS motion's start and physical stop are integration boundaries,
    # including a wait before acquisition and an off-grid terminal brake.
    cursor = begin
    for stop in sorted({motion.start, motion.end, end}):
        stop = min(stop, end)
        if stop <= cursor:
            continue
        initial = run._forecast_momentum(
            cursor,
            stop,
            lambda t: origin if t < motion.start else motion.state(t).attitude,
            start_eci_n_m_s=initial,
        ).end_eci_n_m_s
        cursor = stop
    return initial


@pytest.mark.parametrize("model", ["committed", "planner"])
@pytest.mark.parametrize("late,wait", [(0, 0), (30, 0), (60, 60)])
def test_contact_forecast_tracks_acquisition_and_terminal_braking(model, late, wait):
    config, context, block, motion, run = _contact(late, wait)
    expected = _physical(
        run, motion, block.attitude_in, context.ustart, context.uend, INITIAL
    )
    run._stored_momentum_tracker.start_from_eci(INITIAL)
    before = run._stored_momentum_tracker.momentum_eci_n_m_s.copy()
    if model == "committed":
        # This variant predicts a contact that has not yet been installed.
        run.acs.current_pass = None
        actual = run._momentum_start(context.ustart, context.uend, [block.entry])
    else:
        forecast = MomentumForecast(config, context, start_momentum_eci_n_m_s=INITIAL)
        forecast.profile([block])
        actual = forecast.tracker.momentum_eci_n_m_s
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(
        run._stored_momentum_tracker.momentum_eci_n_m_s, before
    )


@pytest.mark.parametrize("late", [0, 30])
@pytest.mark.parametrize("at", [300, 600.02])
def test_contact_forecast_during_tracking_or_braking(late, at):
    _, context, block, motion, run = _contact(late)
    begin = T0 + at
    initial = _physical(run, motion, block.attitude_in, context.ustart, begin, INITIAL)
    expected = _physical(run, motion, block.attitude_in, begin, context.uend, initial)
    run._stored_momentum_tracker.start_from_eci(initial)
    before = run._stored_momentum_tracker.momentum_eci_n_m_s.copy()
    if begin > block.entry.end:
        # END_PASS has run, but its physical brake has not finished yet.
        run.acs.current_pass = None
    actual = run._momentum_start(begin, context.uend, [block.entry])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    np.testing.assert_array_equal(
        run._stored_momentum_tracker.momentum_eci_n_m_s, before
    )


@pytest.mark.parametrize("model", ["committed", "planner"])
def test_contact_forecast_can_end_during_the_acquisition_wait(model):
    config, _, block, motion, run = _contact(late=60, wait=60)
    stop = T0 + 30
    expected = _physical(run, motion, block.attitude_in, T0, stop, INITIAL)
    if model == "committed":
        run._stored_momentum_tracker.start_from_eci(INITIAL)
        actual = run._momentum_start(T0, stop, [block.entry])
    else:
        context = SchedulingContext(config, BEGIN, BEGIN + timedelta(seconds=30))
        forecast = MomentumForecast(config, context, start_momentum_eci_n_m_s=INITIAL)
        times, _ = forecast.profile([block])
        assert times[-1] == stop
        actual = forecast.tracker.momentum_eci_n_m_s
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)


def test_installed_contact_profile_survives_a_new_selected_profile():
    _, context, block, motion, run = _contact()
    profile = block.gspass.ra.copy()
    profile[5] += 1
    # The plan's new pass object names the same contact, but selecting its
    # future profile cannot rewrite the motion already installed in ACS.
    replacement = block.gspass.model_copy(update={"ra": profile})
    run._entry_passes[id(block.entry)] = replacement
    run._stored_momentum_tracker.start_from_eci(INITIAL)
    before = run.acs._executor.state
    expected = _physical(
        run, motion, block.attitude_in, context.ustart, context.uend, INITIAL
    )
    actual = run._momentum_start(context.ustart, context.uend, [block.entry])
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-10)
    assert run.acs._executor.state == before


def _tracking(idle_seconds=120, middle=84):
    config = _config()
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    origin = (80.0, -20.0, 10.0)
    entry = PlanEntry(
        config=config,
        obsid=702,
        obstype=ObsType.PPT,
        ra=80,
        dec=-20,
        roll=10,
        begin=T0,
        end=T0 + 1200,
        slewtime=0,
        track=track_from_table(
            [T0, T0 + 600, T0 + 1200, T0 + 3600],
            [80, middle, 88, 100],
            [-20] * 4,
        ),
    )
    limits = config.spacecraft_bus.attitude_control
    end = BEGIN + timedelta(seconds=1200 + idle_seconds)
    run = RollingHorizonDITL(config, [], begin=BEGIN, end=end)
    run.step_size = 10
    slew = Slew.idle_hold(config, *origin, T0)
    slew.obstype, slew.at, slew.obsid = ObsType.PPT, entry, entry.obsid
    ingress = AttitudeTrajectory.from_slew(slew, limits)
    motion = run.acs._tracking_after(slew, ingress, limits)
    assert motion is not None
    context = SchedulingContext(config, BEGIN, end)
    block = _Block(
        entry=entry,
        attitude_in=origin,
        attitude_out=motion.state(motion.end).attitude,
        ready=T0,
        end=motion.end,
    )
    return config, context, block, motion, run


@pytest.mark.parametrize("idle_seconds", [0, 120])
def test_planner_integrates_tracking_and_terminal_stop(idle_seconds):
    config, context, block, motion, run = _tracking(idle_seconds)
    initial = INITIAL
    stop = context.uend
    rest = min(stop, motion.end)
    expected = run._forecast_momentum(
        T0, rest, lambda t: motion.state(t).attitude, start_eci_n_m_s=initial
    ).end_eci_n_m_s
    if stop > rest:
        expected = run._forecast_momentum(
            rest, stop, lambda t: motion.state(t).attitude, start_eci_n_m_s=expected
        ).end_eci_n_m_s
    forecast = MomentumForecast(config, context, start_momentum_eci_n_m_s=initial)
    times, _ = forecast.profile([block])
    assert times[-1] == stop
    np.testing.assert_allclose(
        forecast.tracker.momentum_eci_n_m_s, expected, rtol=0, atol=1e-10
    )


def test_loading_covers_the_whole_block_beyond_a_short_context():
    config, context, block, _, _ = _tracking()
    full = MomentumForecast(config, context).loading(block)
    short = SchedulingContext(config, BEGIN, BEGIN + timedelta(seconds=300))
    assert full > 0.5
    assert MomentumForecast(config, short).loading(block) == pytest.approx(
        full, rel=0, abs=1e-10
    )


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
        (300.0, 120, None),
        (1200.01, 120, None),
        (300.0, 120, 319),
    ],
)
def test_committed_moving_momentum_tracks_and_brakes(
    handover, idle_seconds, cut_seconds
):
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


@pytest.mark.parametrize("kind,peak", [("contact", 0.1), ("tracking", 0.05)])
def test_wheel_demand_includes_the_body_rate(kind, peak):
    config, context, block, motion, _ = _contact() if kind == "contact" else _tracking()
    momentum = config.spacecraft_bus.attitude_control.stored_momentum
    momentum.gravity_gradient_enabled = False
    momentum.capacity_n_m_s = 1
    momentum.slew_momentum_enabled = True
    forecast = MomentumForecast(config, context, start_momentum_eci_n_m_s=(0, 0, 0))
    times, demand = forecast.demand([block])
    inertia = np.asarray(config.spacecraft_bus.inertia_tensor_body_kg_m2)
    # No external torque and zero initial H: the wheels carry the body's I*omega.
    expected = [
        np.linalg.norm(inertia @ np.deg2rad(motion.state(t).angular_velocity_body))
        for t in times
    ]
    assert max(expected) > peak
    np.testing.assert_allclose(demand, expected, rtol=0, atol=1e-10)
    assert demand[-1] == pytest.approx(0)
    assert forecast.loading(block) >= max(expected) - 1e-10


@pytest.mark.parametrize("kind", ["contact", "tracking"])
def test_cache_distinguishes_paths_with_the_same_endpoints(kind, monkeypatch):
    if kind == "contact":
        config, context, first, _, _ = _contact()
        ra = first.gspass.ra.copy()
        ra[5] += 1
        second = first.model_copy(
            update={"gspass": first.gspass.model_copy(update={"ra": ra})}
        )
    else:
        config, context, first, _, _ = _tracking(middle=83)
        second = _tracking(middle=86)[2]
        second.entry.config = config
    forecast = MomentumForecast(config, context, start_momentum_eci_n_m_s=INITIAL)
    forecast.profile([first])
    first_h = forecast.tracker.momentum_eci_n_m_s.copy()
    with monkeypatch.context() as patch:

        def uncached_update(**kwargs):
            pytest.fail("An equivalent rebuilt block should reuse its integration")

        patch.setattr(forecast.tracker, "update", uncached_update)
        forecast.profile([first.model_copy(update={"entry": first.entry.model_copy()})])
    forecast.profile([second])
    second_h = forecast.tracker.momentum_eci_n_m_s.copy()
    fresh = MomentumForecast(config, context, start_momentum_eci_n_m_s=INITIAL)
    fresh.profile([second])
    np.testing.assert_array_equal(second_h, fresh.tracker.momentum_eci_n_m_s)
    assert np.linalg.norm(first_h - second_h) > 1e-4
