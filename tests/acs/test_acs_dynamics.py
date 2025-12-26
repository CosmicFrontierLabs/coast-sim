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
