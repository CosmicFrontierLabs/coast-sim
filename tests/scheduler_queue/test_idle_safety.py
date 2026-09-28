"""Predictive holds reserve finite escape motion without relaxing keepouts."""

from datetime import datetime, timedelta, timezone
from math import inf
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from conops import ACSMode, AttitudeControlSystem, Constraint, MissionConfig
from conops.common import ACSCommandType, ObsType, SlewAlgorithm
from conops.simulation.acs import ACS
from conops.simulation.idle_safety import IdleSafetyPlanner
from conops.simulation.passes import Pass
from conops.simulation.slew import Slew


def make_planner(step=2, violation_offset=4000):
    config = MissionConfig(constraint=Constraint())
    config.spacecraft_bus.attitude_control = AttitudeControlSystem(
        max_slew_rate_body=(0.2, 2, 0.2),
        slew_acceleration_body=(0.02, 0.2, 0.02),
        settle_time=0,
    )
    epoch = datetime(2027, 1, 1, tzinfo=timezone.utc)
    times = [epoch + timedelta(seconds=i) for i in range(0, 6000 + step, step)]
    config.constraint.ephem = SimpleNamespace(timestamp=times, step_size=step)
    tree = SimpleNamespace(
        evaluate=Mock(
            return_value=SimpleNamespace(
                constraint_array=np.array(
                    [
                        time >= epoch + timedelta(seconds=violation_offset)
                        for time in times
                    ]
                )
            )
        )
    )
    config.constraint.__dict__["hardware_safety_constraint_config"] = tree
    return IdleSafetyPlanner(config, times[-1].timestamp())


@pytest.fixture
def planner():
    return make_planner()


def test_forecast_reserves_slowest_axis_and_two_ticks(planner):
    start = planner.times[0]
    assert planner.reserve == 914
    assert planner.first_violation((0.0, 0.0, 0.0), start) == start + 4000
    assert planner.departure_deadline((0.0, 0.0, 0.0), start) == start + 3086
    assert planner.first_violation((0.0, 0.0, 0.0), start + 4000) == start + 4000
    # Repeated scheduling callbacks reuse a single full-ephemeris forecast.
    planner.config.constraint.hardware_safety_constraint_config.evaluate.assert_called_once()


def test_safe_endpoint_alone_is_not_a_safe_hold(planner):
    start = planner.times[0]
    assert planner.hold_is_safe((0.0, 0.0, 0.0), start + 2000)
    assert not planner.hold_is_safe((0.0, 0.0, 0.0), start + 3000)


def test_repeated_large_queue_scan_reuses_hold_forecasts(planner):
    start = planner.times[0]
    for _ in range(2):
        for index in range(500):
            assert (
                planner.first_violation((index * 0.5, 0.0, 0.0), start) == start + 4000
            )
    assert (
        planner.config.constraint.hardware_safety_constraint_config.evaluate.call_count
        == 500
    )


@pytest.mark.parametrize("step", [2, 60])
@pytest.mark.parametrize("end_offset", [0, -0.5, -1.0])
def test_final_interval_crossing_requires_escape_and_rejects_hold(step, end_offset):
    planner = make_planner(step, violation_offset=6000)
    start = planner.times[0]
    planner.end = start + 6000 + end_offset
    attitude = (0.0, 0.0, 0.0)
    # The last safe sample is before the end, but the first failing one is at
    # or after it. Neither an exact-grid nor an off-grid end erases that risk.
    assert planner.departure_deadline(attitude, start) == start + 6000 - planner.reserve
    assert not planner.hold_is_safe(attitude, start + 5800)


@pytest.mark.parametrize("step", [2, 60])
def test_crossing_interval_entirely_after_end_needs_no_escape(step):
    planner = make_planner(step, violation_offset=6000)
    start = planner.times[0]
    planner.end = start + 6000 - step
    attitude = (0.0, 0.0, 0.0)
    assert planner.departure_deadline(attitude, start) == inf
    assert planner.hold_is_safe(attitude, start + 5800)


@pytest.mark.parametrize("step", [2, 60])
def test_recovery_dwell_must_end_before_crossing_interval(step):
    planner = make_planner(step, violation_offset=3000)
    attitude = (0.0, 0.0, 0.0)
    last_safe = planner.times[0] + 3000 - step
    dwell = (
        planner.reserve + planner.config.spacecraft_bus.attitude_control.idle_min_hold_s
    )
    assert planner.hold_is_safe(attitude, last_safe - dwell)
    assert not planner.hold_is_safe(attitude, last_safe - dwell + 0.5)


def test_empty_scopes_do_not_create_a_safety_deadline(planner):
    planner.scopes = []
    assert planner.departure_deadline((25.0, 0.0, 0.0), planner.times[0]) == inf


@pytest.fixture
def recovery_queue(queue_ditl, monkeypatch):
    ditl = queue_ditl
    start = ditl.begin.timestamp()
    ditl.uend = start + 6000
    ditl.step_size = 2
    ditl.ephem.step_size = 2
    ditl.acs.ra, ditl.acs.dec, ditl.acs.roll = 0.0, 0.0, 0.0
    ditl.acs.get_mode = Mock(return_value=ACSMode.IDLE)
    ditl.acs._idle_safe_attitude_candidates = Mock(return_value=[(20.0, 0.0)])
    ditl.acs._idle_safe_roll_candidates = Mock(return_value=[0.0])
    ditl._idle_safety = SimpleNamespace(
        departure_deadline=lambda *args: start,
        first_violation=lambda *args: start + 900,
        hold_is_safe=Mock(return_value=True),
    )
    ditl._slew_attitude_constraint_violation = Mock(return_value=None)
    monkeypatch.setattr(
        "conops.ditl.queue_ditl.optimum_roll", lambda *args, **kwargs: 0.0
    )
    # Real kinematics and quaternion trajectory, not a teleporting test double.
    ditl.config.spacecraft_bus.attitude_control = AttitudeControlSystem(
        max_slew_rate_body=(0.2, 2, 0.2),
        slew_acceleration_body=(0.02, 0.2, 0.02),
        settle_time=0,
    )
    return ditl, start


def test_recovery_commands_finite_validated_slew(recovery_queue):
    ditl, start = recovery_queue
    ditl._schedule_idle_recovery(start)
    command = ditl.acs.enqueue_command.call_args.args[0]
    assert command.command_type == ACSCommandType.SLEW_TO_TARGET
    assert command.execution_time == start
    slew = command.slew
    assert isinstance(slew, Slew)
    assert slew.obstype == ObsType.IDLE
    assert slew.slewtime == pytest.approx(110)
    assert slew.attitude(start) == pytest.approx((0.0, 0.0, 0.0))
    assert slew.attitude(slew.slewend) == pytest.approx((20.0, 0.0, 0.0))
    assert (ditl.acs.ra, ditl.acs.dec, ditl.acs.roll) == (0.0, 0.0, 0.0)
    ditl._slew_attitude_constraint_violation.assert_called_once_with(
        slew, ACSMode.SLEWING
    )


@pytest.mark.parametrize("failure", ["path", "hold", "already_unsafe"])
def test_recovery_fails_closed(recovery_queue, failure):
    ditl, start = recovery_queue
    if failure == "path":
        ditl._slew_attitude_constraint_violation.return_value = (
            start + 2,
            "keepout",
            "hardware_safety",
        )
    elif failure == "hold":
        ditl._idle_safety.hold_is_safe.return_value = False
    else:
        ditl._idle_safety.first_violation = lambda *args: start
    with pytest.raises(RuntimeError):
        ditl._schedule_idle_recovery(start)
    ditl.acs.enqueue_command.assert_not_called()


def test_safe_hold_does_not_schedule_premature_recovery(recovery_queue):
    ditl, start = recovery_queue
    ditl._idle_safety.departure_deadline = lambda *args: start + 200
    ditl._schedule_idle_recovery(start)
    ditl.acs.enqueue_command.assert_not_called()


@pytest.mark.parametrize(
    "pass_offset, rolls, routed, accepted",
    [
        (114, [0.0], False, False),
        (115, [0.0], False, True),
        (200, [0.0, 20.0], False, False),
        (400, [0.0, 20.0], False, True),
        (200, [0.0], True, False),
    ],
)
def test_recovery_reserves_all_pass_ingress_deadlines(
    recovery_queue, pass_offset, rolls, routed, accepted
):
    ditl, start = recovery_queue
    if routed:
        ditl.config.spacecraft_bus.attitude_control.slew_algorithm = (
            SlewAlgorithm.CONSTRAINT_AVOIDING
        )
    contact = Pass(
        config=ditl.config,
        ephem=ditl.ephem,
        station="TEST",
        begin=start + pass_offset,
        length=100,
        gsstartra=20.0,
        gsstartdec=0.0,
        gsstartroll=0.0,
        utime=[start + pass_offset],
        tracking_attitude_profiles=[[(20.0, 0.0, roll)] for roll in rolls],
    )
    ditl.acs.passrequests.next_pass.return_value = contact
    # Recovery takes 110 s; the aligned pass needs no turn but still needs
    # the two-tick ingress buffer. The alternative 20-degree roll needs another
    # 110 s: its deadline is +86, not the aligned profile's +196.
    if not accepted:
        with pytest.raises(RuntimeError, match="No path-validated"):
            ditl._schedule_idle_recovery(start)
        ditl.acs.enqueue_command.assert_not_called()
    else:
        ditl._schedule_idle_recovery(start)
        ditl.acs.enqueue_command.assert_called_once()


@pytest.mark.parametrize(
    "step, delay, accepted",
    [
        (2, 1, True),
        (2, 3, False),
        (60, 0, True),
        (60, 90, True),
        (60, 120, True),
        (60, 120.01, False),
        (60, 150, False),
        (60, 180, False),
    ],
)
@pytest.mark.parametrize("active_slew", [False, True])
def test_delayed_science_slew_checks_wait_through_execution_tick(
    recovery_queue, step, delay, accepted, active_slew
):
    ditl, start = recovery_queue
    ditl.step_size = ditl.ephem.step_size = step
    ditl.ephem.timestamp = [
        datetime.fromtimestamp(start + i * step, timezone.utc)
        for i in range(6000 // step + 1)
    ]
    ditl.ephem.index = lambda time: int((time.timestamp() - start) // step)
    unsafe_at = start + (4 if step == 2 else 180)
    first_violation = Mock(return_value=unsafe_at)
    ditl._idle_safety = SimpleNamespace(
        step=step,
        first_violation=first_violation,
        departure_deadline=lambda *args: inf,
    )
    # The preceding slew finishes after half a tick. Only its endpoint is held
    # while waiting; evaluating its current attitude would check the wrong hold.
    hold_attitude = (10.0, 0.0, 0.0) if active_slew else (0.0, 0.0, 0.0)
    hold_start = start + step / 2 if active_slew else start
    if active_slew:
        ditl.acs.last_slew = SimpleNamespace(
            is_slewing=lambda time: True,
            slewstart=start,
            slewtime=step / 2,
            endra=10.0,
            enddec=0.0,
            endroll=0.0,
        )
    ditl.acs._is_actively_slewing = Mock(return_value=active_slew)
    target = Mock(
        exptime=None,
        exposure=3600.0,
        collection_begin=None,
        collection_end=None,
        ra=45.0,
        dec=30.0,
        obsid=1001,
        ss_max=3600.0,
        ss_min=300.0,
        windows=[[start, start + 6000]],
    )
    target.next_vis.return_value = start + delay
    ditl.queue.get.return_value = target
    ditl.queue.targets = [target]
    ditl._fetch_new_ppt(start, 0.0, 0.0)
    if not accepted:
        ditl.acs.enqueue_command.assert_not_called()
        assert ditl.ppt is None
    else:
        command = ditl.acs.enqueue_command.call_args.args[0]
        expected_start = start + np.ceil(max(delay, hold_start - start) / step) * step
        assert command.execution_time == expected_start
        assert command.execution_time <= unsafe_at - step
        executor = ACS(config=ditl.config)
        executor.ra, executor.dec, executor.roll = hold_attitude
        predicted = command.slew.attitude(expected_start + step)
        executor._start_slew(command.slew, expected_start)
        assert command.slew.attitude(expected_start + step) == pytest.approx(predicted)
    first_violation.assert_called_once_with(hold_attitude, hold_start)


def test_active_recovery_is_not_replaced_by_target_selection(recovery_queue):
    ditl, start = recovery_queue
    ditl.acs.current_slew = SimpleNamespace(
        obstype=ObsType.IDLE, is_slewing=lambda time: True
    )
    ditl._should_initiate_charging = Mock(return_value=False)
    ditl._check_too_interrupt = Mock(return_value=False)
    ditl._manage_ppt_lifecycle = Mock()
    ditl._fetch_new_ppt = Mock()
    ditl._handle_science_mode(start, 0.0, 0.0, ACSMode.SLEWING)
    ditl._fetch_new_ppt.assert_not_called()
    ditl._check_too_interrupt.assert_not_called()
