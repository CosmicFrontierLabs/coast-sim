import numpy as np
import pytest

from conops.simulation.slew import Slew


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
    # Wheel momentum is opposite to body torque (Newton's 3rd law)
    # +x body torque → -x wheel momentum
    assert acs.reaction_wheels[0].current_momentum == pytest.approx(
        -expected_torque, rel=1e-6
    )
    assert abs(acs.reaction_wheels[1].current_momentum) < 1e-8
    assert abs(acs.reaction_wheels[2].current_momentum) < 1e-8

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
