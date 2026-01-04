import numpy as np
import pytest

from conops.simulation.reaction_wheel import ReactionWheel
from conops.simulation.slew import Slew
from conops.simulation.wheel_dynamics import WheelDynamics


def _add_test_wheels(acs, max_torque=1.0, max_momentum=10.0):
    acs.reaction_wheels = []
    for i, orient in enumerate(
        [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)], start=1
    ):
        # Construct wheels via ReactionWheel class to avoid fragile mocks
        from conops.simulation.reaction_wheel import ReactionWheel

        acs.reaction_wheels.append(
            ReactionWheel(
                max_torque=max_torque,
                max_momentum=max_momentum,
                orientation=orient,
                name=f"rw{i}",
            )
        )
    # Sync with WheelDynamics
    acs.wheel_dynamics.wheels = acs.reaction_wheels


def test_update_wheel_momentum_first_call_sets_time(acs):
    _add_test_wheels(acs)
    utime = 100.0
    acs._last_pointing_time = None
    acs._update_wheel_momentum(utime)
    assert acs._last_pointing_time == pytest.approx(utime)
    assert all(abs(w.current_momentum) == 0.0 for w in acs.reaction_wheels)


def test_hold_torque_matches_disturbance_when_unclamped(acs):
    _add_test_wheels(acs, max_torque=0.1, max_momentum=10.0)
    disturbance = np.array([1e-5, -2e-5, 3e-5], dtype=float)
    acs._apply_hold_wheel_torque(disturbance, dt=1.0, utime=0.0)
    assert acs._last_hold_torque_target_mag == pytest.approx(
        np.linalg.norm(-disturbance), rel=1e-6
    )
    target_mag = np.linalg.norm(-disturbance)
    # Regularization can slightly reduce applied magnitude; ensure it's close and not larger.
    assert acs._last_hold_torque_actual_mag <= target_mag + 1e-12
    assert acs._last_hold_torque_actual_mag == pytest.approx(target_mag, rel=2e-2)


def test_slew_accel_profile_triangular_and_trapezoidal(acs):
    # Triangular: small angle
    slew = Slew(config=acs.config)
    slew.slewstart = 0.0
    slew.slewdist = 0.1
    a = acs.acs_config.slew_acceleration
    t_peak = (slew.slewdist / a) ** 0.5
    assert acs._slew_accel_profile(slew, t_peak * 0.5) == pytest.approx(a)
    assert acs._slew_accel_profile(slew, t_peak * 1.5) == pytest.approx(-a)
    assert acs._slew_accel_profile(slew, 10.0) == 0.0

    # Trapezoidal: large angle
    slew = Slew(config=acs.config)
    slew.slewstart = 0.0
    slew.slewdist = 10.0
    t_accel = acs.acs_config.max_slew_rate / a
    t_cruise = (slew.slewdist - 2 * 0.5 * a * t_accel**2) / acs.acs_config.max_slew_rate
    assert acs._slew_accel_profile(slew, t_accel * 0.5) == pytest.approx(a)
    assert acs._slew_accel_profile(slew, t_accel + t_cruise * 0.5) == 0.0
    assert acs._slew_accel_profile(
        slew, t_accel + t_cruise + t_accel * 0.5
    ) == pytest.approx(-a)


def test_disturbance_vector_telemetry_present(acs):
    torque = acs._compute_disturbance_torque(0.0)
    assert isinstance(torque, np.ndarray)
    comps = acs._last_disturbance_components
    assert "vector" in comps
    assert len(comps["vector"]) == 3


def test_slew_torque_updates_wheel_momentum_consistently(acs, monkeypatch):
    _add_test_wheels(acs, max_torque=1.0, max_momentum=10.0)
    slew = Slew(config=acs.config)
    slew.slewstart = 0.0
    slew.slewend = 10.0
    slew.slewdist = 1.0
    slew.rotation_axis = (1.0, 0.0, 0.0)
    acs.current_slew = slew
    acs.last_slew = slew

    monkeypatch.setattr(
        acs, "_compute_disturbance_torque", lambda _ut: np.zeros(3, dtype=float)
    )
    monkeypatch.setattr(acs, "_slew_accel_profile", lambda _slew, _t: 1.0)

    acs._last_pointing_time = None
    acs._update_wheel_momentum(0.0)
    before = {w.name: w.current_momentum for w in acs.reaction_wheels}
    dt = 1.0
    acs._update_wheel_momentum(dt)
    after = {w.name: w.current_momentum for w in acs.reaction_wheels}
    snapshot = acs.wheel_snapshot()

    moi_cfg = acs.config.spacecraft_bus.attitude_control.spacecraft_moi
    if isinstance(moi_cfg, (list, tuple)):
        if len(moi_cfg) == 3 and any(isinstance(x, (list, tuple)) for x in moi_cfg):
            i_mat = np.array(moi_cfg, dtype=float)
        elif len(moi_cfg) == 3:
            i_mat = np.diag([float(x) for x in moi_cfg])
        else:
            val = float(sum(moi_cfg) / len(moi_cfg))
            i_mat = np.diag([val, val, val])
    else:
        val = float(moi_cfg)
        i_mat = np.diag([val, val, val])

    axis = np.array([1.0, 0.0, 0.0], dtype=float)
    i_axis = float(axis.dot(i_mat.dot(axis)))
    expected_torque = (1.0 * np.pi / 180.0) * i_axis

    assert snapshot.t_actual_mag == pytest.approx(expected_torque, rel=1e-6)

    # Verify wheel momentum goes to the correct wheel based on frame transformation.
    # Default pointing is (RA=180°, Dec=0°). With pole-reference rotation matrix:
    #   z_b = (-1, 0, 0), y_b = (0, 0, 1), x_b = (0, -1, 0)
    # Inertial X-axis (1,0,0) transforms to body frame as:
    #   R @ (1,0,0) = (0, 0, -1) = body -Z
    # So Z-wheel should absorb momentum (positive, since body torque is in -Z direction).
    assert acs.reaction_wheels[2].current_momentum == pytest.approx(
        expected_torque, rel=1e-6
    )
    assert abs(acs.reaction_wheels[0].current_momentum) < 1e-8
    assert abs(acs.reaction_wheels[1].current_momentum) < 1e-8

    assert snapshot.wheels
    for reading in snapshot.wheels:
        delta = after[reading.name] - before[reading.name]
        assert delta == pytest.approx(reading.torque_applied * dt, rel=1e-6, abs=1e-9)


def test_slew_headroom_clamp_respects_margin(acs, monkeypatch):
    _add_test_wheels(acs, max_torque=10.0, max_momentum=0.5)
    acs._wheel_mom_margin = 0.1
    acs.wheel_dynamics._momentum_margin = 0.1  # Sync margin
    for w in acs.reaction_wheels:
        w.current_momentum = w.max_momentum * 0.095

    slew = Slew(config=acs.config)
    slew.slewstart = 0.0
    slew.slewend = 10.0
    slew.slewdist = 1.0
    slew.rotation_axis = (1.0, 0.0, 0.0)
    acs.current_slew = slew
    acs.last_slew = slew

    monkeypatch.setattr(
        acs, "_compute_disturbance_torque", lambda _ut: np.zeros(3, dtype=float)
    )
    monkeypatch.setattr(acs, "_slew_accel_profile", lambda _slew, _t: 100.0)

    acs._last_pointing_time = None
    acs._update_wheel_momentum(0.0)
    acs._update_wheel_momentum(1.0)

    for w in acs.reaction_wheels:
        assert w.current_momentum <= w.max_momentum * acs._wheel_mom_margin + 1e-9


def test_hold_headroom_clamp_respects_margin(acs):
    _add_test_wheels(acs, max_torque=10.0, max_momentum=0.5)
    acs._wheel_mom_margin = 0.1
    acs.wheel_dynamics._momentum_margin = 0.1  # Sync margin
    for w in acs.reaction_wheels:
        w.current_momentum = w.max_momentum * 0.095

    disturbance = np.array([0.1, 0.0, 0.0], dtype=float)
    acs._apply_hold_wheel_torque(disturbance, dt=1.0, utime=0.0)

    for w in acs.reaction_wheels:
        assert w.current_momentum <= w.max_momentum * acs._wheel_mom_margin + 1e-9


class TestComputeSlewParams:
    """Tests for compute_slew_params physics correctness."""

    def test_torque_limited_accel_independent_of_distance(self):
        """In torque-limited regime, acceleration should not depend on slew distance.

        Physics:
        - Torque-limited: accel = τ_max / I (constant)
        - Momentum-limited: accel = H_avail / (I * t) (varies with time/distance)

        When wheels have ample momentum headroom, the acceleration should be
        determined solely by max torque and inertia, not by slew distance.
        """
        # Create wheel dynamics with plenty of momentum headroom (torque-limited)
        wheels = [
            ReactionWheel(
                max_torque=1.0,
                max_momentum=1000.0,  # Huge headroom - torque limited
                orientation=(1.0, 0.0, 0.0),
                current_momentum=0.0,
                name="rw_x",
            )
        ]
        inertia = np.diag([10.0, 10.0, 10.0])
        wheel_dynamics = WheelDynamics(wheels, inertia)

        axis = np.array([1.0, 0.0, 0.0])

        # Compute slew params for different distances
        accel_1deg, _, _ = wheel_dynamics.compute_slew_params(axis, 1.0)
        accel_10deg, _, _ = wheel_dynamics.compute_slew_params(axis, 10.0)
        accel_90deg, _, _ = wheel_dynamics.compute_slew_params(axis, 90.0)

        # All accelerations should be the same (torque-limited)
        # Expected: accel = τ / I = 1.0 / 10.0 = 0.1 rad/s² = 5.73 deg/s²
        expected_accel = (1.0 / 10.0) * (180.0 / np.pi)

        assert np.isclose(accel_1deg, expected_accel, rtol=0.01), (
            f"1° slew: accel={accel_1deg:.4f}, expected={expected_accel:.4f}"
        )
        assert np.isclose(accel_10deg, expected_accel, rtol=0.01), (
            f"10° slew: accel={accel_10deg:.4f}, expected={expected_accel:.4f}"
        )
        assert np.isclose(accel_90deg, expected_accel, rtol=0.01), (
            f"90° slew: accel={accel_90deg:.4f}, expected={expected_accel:.4f}"
        )

        # Verify they're all equal to each other
        assert np.isclose(accel_1deg, accel_10deg, rtol=0.001), (
            f"Accel varies with distance in torque-limited regime: "
            f"1°={accel_1deg:.4f}, 10°={accel_10deg:.4f}"
        )
        assert np.isclose(accel_10deg, accel_90deg, rtol=0.001), (
            f"Accel varies with distance in torque-limited regime: "
            f"10°={accel_10deg:.4f}, 90°={accel_90deg:.4f}"
        )

    def test_momentum_limited_accel_varies_correctly(self):
        """In momentum-limited regime, acceleration should vary with motion time.

        When momentum headroom is limited, longer slews allow lower acceleration
        (spreading the impulse over more time), while shorter slews may be
        constrained by the momentum budget.
        """
        # Create wheel dynamics with limited momentum (momentum-limited)
        wheels = [
            ReactionWheel(
                max_torque=100.0,  # High torque - not the limit
                max_momentum=1.0,  # Low headroom - momentum limited
                orientation=(1.0, 0.0, 0.0),
                current_momentum=0.0,
                name="rw_x",
            )
        ]
        inertia = np.diag([10.0, 10.0, 10.0])
        wheel_dynamics = WheelDynamics(wheels, inertia)

        axis = np.array([1.0, 0.0, 0.0])

        # For a short slew, we need high accel to use momentum in short time
        accel_short, _, time_short = wheel_dynamics.compute_slew_params(axis, 1.0)
        # For a long slew, we can use lower accel over longer time
        accel_long, _, time_long = wheel_dynamics.compute_slew_params(axis, 10.0)

        # Both should be valid
        assert accel_short > 0, "Short slew should have valid accel"
        assert accel_long > 0, "Long slew should have valid accel"

        # Motion time should be longer for longer distance
        assert time_long > time_short, (
            f"Longer slew should take more time: "
            f"1°={time_short:.2f}s, 10°={time_long:.2f}s"
        )

    def test_accel_uses_motion_time_not_distance(self):
        """Verify get_axis_accel_limit uses motion_time correctly, not distance.

        The momentum-limited torque calculation should use actual motion time
        (in seconds), not slew distance (in degrees). This tests that the
        parameter is being interpreted correctly.

        Bug to catch: passing distance_deg where motion_time is expected would
        give wrong acceleration limits in momentum-constrained regimes.
        """
        # Setup: wheel with moderate torque and momentum limits
        # Choose values where the distinction matters
        wheels = [
            ReactionWheel(
                max_torque=0.5,  # Moderate torque
                max_momentum=5.0,  # Moderate momentum
                orientation=(1.0, 0.0, 0.0),
                current_momentum=0.0,
                name="rw_x",
            )
        ]
        inertia = np.diag([10.0, 10.0, 10.0])
        wheel_dynamics = WheelDynamics(wheels, inertia)
        axis = np.array([1.0, 0.0, 0.0])

        # Direct call to get_axis_accel_limit with explicit motion times
        # If working correctly:
        # - Short motion time → momentum limit may constrain (H/t is large)
        # - Long motion time → torque limit applies (H/t is small)

        # Torque-limited accel = τ/I = 0.5/10 = 0.05 rad/s² = 2.86 deg/s²
        torque_limited_accel = (0.5 / 10.0) * (180.0 / np.pi)

        # With 1 second motion time:
        # Mom-limited torque = H_avail / t = (5.0 * 0.9) / 1.0 = 4.5 N·m
        # This exceeds max_torque (0.5), so should be torque-limited
        accel_1s = wheel_dynamics.get_axis_accel_limit(axis, motion_time=1.0)
        assert np.isclose(accel_1s, torque_limited_accel, rtol=0.01), (
            f"With 1s motion time, should be torque-limited: "
            f"got {accel_1s:.4f}, expected {torque_limited_accel:.4f}"
        )

        # With 100 second motion time:
        # Mom-limited torque = (5.0 * 0.9) / 100 = 0.045 N·m
        # This is LESS than max_torque (0.5), so should be momentum-limited
        # Expected accel = 0.045 / 10 * (180/π) = 0.258 deg/s²
        expected_mom_limited_accel = (0.045 / 10.0) * (180.0 / np.pi)
        accel_100s = wheel_dynamics.get_axis_accel_limit(axis, motion_time=100.0)
        assert np.isclose(accel_100s, expected_mom_limited_accel, rtol=0.01), (
            f"With 100s motion time, should be momentum-limited: "
            f"got {accel_100s:.4f}, expected {expected_mom_limited_accel:.4f}"
        )

        # Verify the momentum-limited case has lower accel than torque-limited
        assert accel_100s < accel_1s, (
            f"Long motion time should give lower accel (momentum-limited): "
            f"1s={accel_1s:.4f}, 100s={accel_100s:.4f}"
        )

    def test_compute_slew_params_trapezoidal_uses_torque_limited_accel(self):
        """Verify trapezoidal slews use torque-limited acceleration, not momentum.

        For trapezoidal profiles (when triangular peak exceeds rate limit):
        - Acceleration is torque-limited (max torque / inertia)
        - Rate is capped by wheel momentum headroom
        - Peak momentum at coast phase = I × rate_limit

        This differs from triangular profiles where peak rate is lower.
        """
        # Setup: low momentum means rate limit will cap, forcing trapezoidal profile
        wheels = [
            ReactionWheel(
                max_torque=1.0,
                max_momentum=0.5,  # Low momentum - will limit rate
                orientation=(1.0, 0.0, 0.0),
                current_momentum=0.0,
                name="rw_x",
            )
        ]
        inertia = np.diag([10.0, 10.0, 10.0])
        wheel_dynamics = WheelDynamics(wheels, inertia)
        axis = np.array([1.0, 0.0, 0.0])
        distance = 10.0

        # Compute expected values from first principles
        # WheelDynamics applies two margins:
        #   momentum_margin (0.9) in get_headroom_along_axis
        #   budget_margin (0.85) in get_axis_rate_limit
        # headroom = max_momentum * momentum_margin * budget_margin = 0.5 * 0.9 * 0.85
        headroom = 0.5 * 0.9 * 0.85
        moi = 10.0  # moment of inertia

        # rate_limit = headroom / moi * (180/π) = 2.192 deg/s
        rate_limit = headroom / moi * (180.0 / np.pi)

        # torque_limited_accel = τ/moi * (180/π) = 5.73 deg/s²
        torque_limited_accel = (1.0 / moi) * (180.0 / np.pi)

        # Verify we're in trapezoidal regime (triangular peak would exceed rate limit)
        triangular_peak = np.sqrt(torque_limited_accel * distance)
        assert triangular_peak > rate_limit, (
            f"Test requires trapezoidal regime: peak {triangular_peak:.2f} > "
            f"limit {rate_limit:.2f}"
        )

        # Get what compute_slew_params returns
        accel, rate, motion_time = wheel_dynamics.compute_slew_params(axis, distance)

        # For trapezoidal profile: acceleration should be torque-limited
        assert np.isclose(accel, torque_limited_accel, rtol=1e-6), (
            f"Acceleration should be torque-limited at {torque_limited_accel:.4f} deg/s², "
            f"got {accel:.4f} deg/s²."
        )

        # Rate should be capped by momentum headroom
        assert np.isclose(rate, rate_limit, rtol=1e-6), (
            f"Rate should be momentum-limited at {rate_limit:.4f} deg/s, "
            f"got {rate:.4f} deg/s."
        )

        # Verify coast-phase momentum equals usable headroom (with both margins applied)
        omega_coast = rate * (np.pi / 180.0)  # rad/s
        coast_momentum = moi * omega_coast
        # Note: headroom already includes both margins
        assert np.isclose(coast_momentum, headroom, rtol=1e-5), (
            f"Coast momentum {coast_momentum:.4f} should equal headroom {headroom:.4f}"
        )


def test_pre_slew_headroom_uses_body_frame_axis(acs):
    """Verify pre-slew feasibility checks transform rotation axis to body frame.

    The rotation axis from slew geometry is in inertial frame, but MOI calculations
    need body-frame axis. At non-standard pointings, these frames differ significantly.

    At (RA=0, Dec=0):
    - Body-Z points at (1, 0, 0) inertial
    - Body-Y aligned with celestial pole (0, 0, 1) inertial
    - Body-X = (0, 1, 0) inertial

    A slew around inertial Z-axis transforms to body-Y axis for MOI calculation.
    """
    _add_test_wheels(acs, max_torque=0.1, max_momentum=10.0)

    # Set ACS to non-default pointing (RA=0, Dec=0)
    acs.ra = 0.0
    acs.dec = 0.0

    # Create slew from this pointing
    slew = Slew(config=acs.config)
    slew.startra = 0.0
    slew.startdec = 0.0
    slew.endra = 5.0  # Small slew in RA
    slew.enddec = 0.0
    slew.slewdist = 5.0
    slew.slewtime = 30.0
    # Inertial Z-axis rotation (around celestial pole)
    slew.rotation_axis = (0.0, 0.0, 1.0)
    slew._accel_override = 0.5

    # Mock disturbance model
    from unittest.mock import Mock

    acs.disturbance_model = Mock()
    acs.disturbance_model.compute = Mock(return_value=(np.zeros(3), {}))

    # The check should pass and use the correct body-frame axis
    feasible, msg = acs._check_slew_momentum_budget(slew, 1000.0)
    assert feasible, f"Slew should be feasible: {msg}"
    assert "Budget OK" in msg

    # Verify the returned axis is in body frame
    _, returned_axis = acs._compute_slew_peak_momentum(slew)

    # At (RA=0, Dec=0), inertial Z (0,0,1) transforms to body Y (0,1,0)
    # The returned axis should be close to body-Y
    expected_body_axis = np.array([0.0, 1.0, 0.0])
    dot_product = abs(np.dot(returned_axis, expected_body_axis))
    assert dot_product > 0.99, (
        f"Axis should transform to body-Y at (RA=0, Dec=0): "
        f"got {returned_axis}, expected ~{expected_body_axis}, dot={dot_product}"
    )
