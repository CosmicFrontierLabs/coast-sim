"""Physical execution invariants, independently of mode and sampling cadence."""

from dataclasses import FrozenInstanceError
from unittest.mock import patch

import numpy as np
import pytest

from conops import ACSCommand, ACSCommandType, AttitudeControlSystem, Pass, Slew
from conops.common import ACSMode, ObsType
from conops.common.vector import quaternion_attitude_delta
from conops.simulation.attitude import (
    AttitudeExecutionError,
    AttitudeExecutor,
    AttitudeTrajectory,
)


def assert_attitude(actual, expected, tolerance=1e-7):
    assert quaternion_attitude_delta(*actual, *expected)[0] < tolerance


@pytest.fixture
def limits():
    return AttitudeControlSystem(
        max_slew_rate=2,
        slew_acceleration=0.5,
        settle_time=3,
        max_slew_rate_body=(2, 1, 0.5),
        slew_acceleration_body=(0.5, 0.25, 0.125),
    )


@pytest.mark.parametrize(
    "first,last",
    [
        ((0, 0, 0), (0, 0, 90)),
        ((30, 40, 350), (30, 40, 10)),
        ((0, 0, 0), (2, 0, 0)),
        ((0, 0, 0), (0, 0, 0.001)),
        ((0, 89, 40), (180, 89, 40)),
        ((12, -30, 40), (220, 55, 310)),
        ((0, 0, 0), (180, 0, 0)),
    ],
)
def test_motion_obeys_directional_rate_and_acceleration(limits, first, last):
    trajectory = AttitudeTrajectory.turn(0, first, last, limits)
    times = np.linspace(0, trajectory.end, 2001)
    states = [trajectory.state(float(time)) for time in times]
    rates = np.asarray([state.angular_velocity_body for state in states])
    assert (
        np.max(np.linalg.norm(rates / limits.max_slew_rate_body, axis=1)) <= 1 + 1e-10
    )
    accelerations = np.diff(rates, axis=0) / np.diff(times)[:, None]
    assert (
        np.max(np.linalg.norm(accelerations / limits.slew_acceleration_body, axis=1))
        <= 1 + 1e-9
    )
    assert_attitude(states[0].attitude, first)
    assert_attitude(states[-1].attitude, last)
    assert (
        states[0].angular_velocity_body == states[-1].angular_velocity_body == (0, 0, 0)
    )
    # Independently differentiate the executed orientations, not only reported rates.
    for i in range(10, len(times) - 1, 53):
        angle, axis = quaternion_attitude_delta(
            *states[i].attitude, *states[i + 1].attitude
        )
        rate = angle / (times[i + 1] - times[i])
        assert rate <= limits.effective_max_slew_rate(axis) * (1 + 1e-7)
        np.testing.assert_allclose(
            rate * np.asarray(axis), (rates[i] + rates[i + 1]) / 2, atol=1e-7
        )


def test_tracking_joins_with_continuous_nonzero_rate(limits):
    track = AttitudeTrajectory.tracking(
        [
            (100, (0, 0, 0)),
            (130, (4, 2, 8)),
            (160, (8, 4, 0)),
        ],
        limits,
    )
    for time, attitude in [(100, (0, 0, 0)), (130, (4, 2, 8)), (160, (8, 4, 0))]:
        assert_attitude(track.state(time).attitude, attitude)
        assert np.linalg.norm(track.state(time).angular_velocity_body) > 0
    assert 0 < track.state(115).attitude[0] < 4
    np.testing.assert_allclose(
        track.state(130 - 1e-5).angular_velocity_body,
        track.state(130).angular_velocity_body,
        atol=1e-5,
    )
    np.testing.assert_allclose(
        track.state(130 + 1e-5).angular_velocity_body,
        track.state(130).angular_velocity_body,
        atol=1e-5,
    )
    assert track.state(track.end).angular_velocity_body == (0, 0, 0)


@pytest.mark.parametrize(
    "samples",
    [
        [(0, (0, 0, 0)), (0, (1, 0, 0))],
        [(0, (0, 0, 0)), (1, (90, 0, 0))],
        [(0, (0, 0, 0)), (100, (float("nan"), 0, 0))],
    ],
)
def test_invalid_tracking_rejected(limits, samples):
    with pytest.raises(AttitudeExecutionError):
        AttitudeTrajectory.tracking(samples, limits)


def test_executor_rejects_discontinuity_and_time_reversal(limits):
    executor = AttitudeExecutor(0, (0, 0, 0))
    with pytest.raises(AttitudeExecutionError, match="join"):
        executor.install(AttitudeTrajectory.turn(0, (30, 0, 0), (40, 0, 0), limits))
    executor.install(AttitudeTrajectory.turn(0, (0, 0, 0), (0, 0, 90), limits))
    executor.advance(10)
    state = executor.state
    with pytest.raises(AttitudeExecutionError, match="instantaneously"):
        executor.hold(10)
    with pytest.raises(AttitudeExecutionError, match="join"):
        executor.install(AttitudeTrajectory.turn(10, state.attitude, (0, 0, 0), limits))
    with pytest.raises(AttitudeExecutionError, match="monotonic"):
        executor.advance(9)
    assert executor.state == state
    with pytest.raises(FrozenInstanceError):
        executor.state.utime = 11


def test_stop_request_brakes_before_next_tracking_knot(limits):
    executor = AttitudeExecutor(0, (0, 0, 0), (0, 0, 4 / 30))
    executor.install(
        AttitudeTrajectory.tracking(
            [
                (0, (0, 0, 0)),
                (30, (4, 0, 0)),
                (60, (8, 0, 0)),
            ],
            limits,
        )
    )
    executor.advance(10)
    assert executor.request_stop(10, limits) == pytest.approx(10 + (4 / 30) / 0.125)
    executor.advance(65)
    stop_angle = (4 / 30) * 10 + 0.5 * (4 / 30) ** 2 / 0.125
    assert_attitude(executor.state.attitude, (stop_angle, 0, 0))
    assert executor.state.angular_velocity_body == (0, 0, 0)


def test_snapshot_and_sampling_cadence_do_not_change_physics(limits):
    trajectory = AttitudeTrajectory.turn(0, (30, 20, 10), (240, -15, 70), limits)
    fine = AttitudeExecutor(0, (30, 20, 10))
    coarse = AttitudeExecutor(0, (30, 20, 10))
    fine.install(trajectory)
    coarse.install(trajectory)
    limits.max_slew_rate_body = (90, 90, 90)
    for time in range(1, 61):
        fine.advance(time)
    coarse.advance(60)
    assert fine.state == coarse.state


@pytest.fixture
def physical_acs(acs, limits):
    acs.config.spacecraft_bus.attitude_control = limits
    acs.ra, acs.dec, acs.roll = 0, 0, 0
    return acs


def new_slew(acs, target, obstype=ObsType.PPT):
    slew = Slew(config=acs.config)
    slew.endra, slew.enddec, slew.endroll = target
    slew.obstype = obstype
    return slew


def test_coarse_slew_handoff_advances_old_motion_first(physical_acs):
    acs = physical_acs
    first = new_slew(acs, (10, 0, 0))
    acs._start_slew(first, 1000)
    acs.pointing(1010)
    second = new_slew(acs, (20, 0, 0))
    acs.enqueue_command(
        ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=first.slewend,
            slew=second,
        )
    )
    acs.pointing(1060)
    assert_attitude((acs.ra, acs.dec, acs.roll), (10, 0, 0))
    assert_attitude((second.startra, second.startdec, second.startroll), (10, 0, 0))
    assert second.slewstart == 1060
    assert acs.angular_velocity_body == (0, 0, 0)


def test_midmotion_command_waits_without_resetting_rate(physical_acs):
    acs = physical_acs
    first = new_slew(acs, (10, 0, 0))
    acs._start_slew(first, 1000)
    second = new_slew(acs, (20, 0, 0))
    acs.enqueue_command(
        ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET, execution_time=1005, slew=second
        )
    )
    acs.pointing(1005)
    assert acs.current_slew is first
    assert np.linalg.norm(acs.angular_velocity_body) > 0
    assert len(acs.command_queue) == 1
    assert acs.command_queue[0].execution_time == pytest.approx(1009)
    acs.pointing(1060)
    assert acs.current_slew is second
    assert_attitude((second.startra, second.startdec, second.startroll), (2.5, 0, 0))


@pytest.mark.parametrize("safe", [False, True])
def test_dwell_guidance_cannot_teleport(physical_acs, safe):
    acs = physical_acs
    acs.in_safe_mode = safe
    with (
        patch.object(acs, "_is_in_charging_mode", return_value=not safe),
        patch(
            "conops.simulation.acs.optimum_roll",
            return_value=90,
        ),
    ):
        acs.pointing(1000)
        assert_attitude((acs.ra, acs.dec, acs.roll), (0, 0, 0))
        assert acs.angular_velocity_body == (0, 0, 0)
        acs.pointing(1000)
        assert_attitude((acs.ra, acs.dec, acs.roll), (0, 0, 0))
        acs.pointing(1001)
        assert np.linalg.norm(acs.angular_velocity_body) > 0
        assert quaternion_attitude_delta(0, 0, 0, acs.ra, acs.dec, acs.roll)[0] < 0.26


def test_execution_attitude_is_read_only_after_initialization(physical_acs):
    acs = physical_acs
    acs.ra = 42
    acs.pointing(1000)
    for attribute in ("ra", "dec", "roll"):
        with pytest.raises(AttitudeExecutionError, match="read-only"):
            setattr(acs, attribute, 0)


def test_future_charge_end_does_not_freeze_solar_guidance(physical_acs):
    acs = physical_acs
    acs.enqueue_command(
        ACSCommand(command_type=ACSCommandType.END_BATTERY_CHARGE, execution_time=1100)
    )
    with (
        patch.object(acs, "_is_in_charging_mode", return_value=True),
        patch("conops.simulation.acs.optimum_roll", return_value=90),
    ):
        acs.pointing(1000)
        acs.pointing(1001)
    assert np.linalg.norm(acs.angular_velocity_body) > 0
    assert acs.command_queue[0].execution_time == 1100


def test_terminal_tracking_brake_is_not_idle(physical_acs):
    acs = physical_acs
    executor = acs._advance_attitude(1000)
    trajectory = AttitudeTrajectory.tracking(
        [(1000, (0, 0, 0)), (1060, (6, 0, 0))],
        acs.config.spacecraft_bus.attitude_control,
        initial_rate=(0, 0, 0),
    )
    executor.install(trajectory)
    acs.pointing(1060.1)
    assert acs.get_mode(1060.1) == ACSMode.SLEWING
    acs.pointing(trajectory.end)
    assert acs.get_mode(trajectory.end) == ACSMode.IDLE


@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["max_slew_rate", "slew_acceleration"])
def test_invalid_physical_limits_rejected(field, value):
    limits = AttitudeControlSystem(**{field: value})
    with pytest.raises(AttitudeExecutionError, match="finite and positive"):
        AttitudeTrajectory.turn(0, (0, 0, 0), (1, 0, 0), limits)


def test_small_turn_at_unix_epoch_is_not_compressed(limits):
    start = 1_700_000_000.0
    track = AttitudeTrajectory.turn(start, (0, 0, 0), (0, 0, 0.001), limits)
    leg = track.legs[0]
    assert track.end - track.start >= leg.motion.duration
    assert leg.scale <= 1


def test_slew_snapshot_survives_command_mutation(physical_acs):
    acs = physical_acs
    slew = new_slew(acs, (10, 0, 0))
    acs._start_slew(slew, 1000)
    expected = acs.predicted_attitude(1010)
    slew.endra, slew.enddec, slew.endroll = 270, 80, 30
    slew.calc_slewtime()
    assert_attitude(acs.predicted_attitude(1010), expected)


def test_safe_entry_during_motion_cannot_reset_velocity(physical_acs):
    acs = physical_acs
    slew = new_slew(acs, (10, 0, 0))
    acs._start_slew(slew, 1000)
    expected = acs._executor.predict(1005)
    acs.request_safe_mode(1005)
    acs.pointing(1005)
    assert acs.in_safe_mode
    assert_attitude((acs.ra, acs.dec, acs.roll), expected.attitude)
    assert acs.angular_velocity_body == expected.angular_velocity_body
    acs.pointing(1060)
    assert acs.current_slew.obstype == ObsType.SAFE
    assert_attitude((acs.ra, acs.dec, acs.roll), (2.5, 0, 0))


def test_pass_acquisition_tracking_and_exit_use_executor(physical_acs):
    acs = physical_acs
    gspass = Pass(
        config=acs.config,
        station="Test station",
        begin=1000,
        length=60,
        utime=[1000, 1030, 1060],
        ra=[0, 4, 8],
        dec=[0, 0, 0],
        roll=[0, 0, 0],
    )
    acs.passrequests.current_pass.return_value = gspass
    acs._start_pass(
        ACSCommand(command_type=ACSCommandType.START_PASS, execution_time=1000), 1000
    )
    acs.pointing(1015)
    assert 0 < acs.ra < 4
    stopping = AttitudeTrajectory.braking(
        acs._executor.state, acs.config.spacecraft_bus.attitude_control
    )
    expected_stop = stopping.state(stopping.end).attitude
    acs.enqueue_command(
        ACSCommand(command_type=ACSCommandType.END_PASS, execution_time=1015)
    )
    acs.pointing(1015)
    acs.pointing(1070)
    assert_attitude((acs.ra, acs.dec, acs.roll), expected_stop)
    assert acs.current_pass is None
    assert acs.angular_velocity_body == (0, 0, 0)


@pytest.mark.parametrize("profile", [[20, 21], [0, 90]])
def test_unacquired_or_infeasible_pass_cannot_change_execution(physical_acs, profile):
    acs = physical_acs
    acs.pointing(1000)
    gspass = Pass(
        config=acs.config,
        station="Test station",
        begin=1000,
        length=1,
        utime=[1000, 1001],
        ra=profile,
        dec=[0, 0],
        roll=[0, 0],
    )
    acs.passrequests.current_pass.return_value = gspass
    with pytest.raises(AttitudeExecutionError):
        acs._start_pass(
            ACSCommand(command_type=ACSCommandType.START_PASS, execution_time=1000),
            1000,
        )
    assert acs.current_pass is None
    assert_attitude(acs.predicted_attitude(1060), (0, 0, 0))
