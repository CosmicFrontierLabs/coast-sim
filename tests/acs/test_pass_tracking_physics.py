import math
from dataclasses import dataclass

import numpy as np
import pytest

from conops.simulation.reaction_wheel import ReactionWheel


@dataclass
class DummyPass:
    ra_deg: float
    dec_deg: float

    def ra_dec(self, utime: float) -> tuple[float, float]:
        return self.ra_deg, self.dec_deg


def _make_orthogonal_wheels(max_torque: float = 10.0, max_momentum: float = 10.0):
    return [
        ReactionWheel(
            max_torque=max_torque,
            max_momentum=max_momentum,
            orientation=(1.0, 0.0, 0.0),
            current_momentum=0.0,
            name="rw_x",
        ),
        ReactionWheel(
            max_torque=max_torque,
            max_momentum=max_momentum,
            orientation=(0.0, 1.0, 0.0),
            current_momentum=0.0,
            name="rw_y",
        ),
        ReactionWheel(
            max_torque=max_torque,
            max_momentum=max_momentum,
            orientation=(0.0, 0.0, 1.0),
            current_momentum=0.0,
            name="rw_z",
        ),
    ]


def test_pass_tracking_torque_matches_inertia(acs):
    acs.ra = 0.0
    acs.dec = 0.0
    acs.reaction_wheels = _make_orthogonal_wheels()
    acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
    acs._compute_disturbance_torque = lambda _ut: np.zeros(3)
    acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)
    acs.current_pass = DummyPass(1.0, 0.0)

    acs._apply_pass_wheel_update(dt=1.0, utime=0.0)

    expected_torque = (math.pi / 180.0) * 10.0
    assert math.isclose(acs._last_pass_rate_deg_s, 1.0, rel_tol=1e-6)
    assert math.isclose(acs._last_pass_torque_target_mag, expected_torque, rel_tol=1e-6)


def test_pass_tracking_updates_wheel_momentum_axis(acs):
    acs.ra = 0.0
    acs.dec = 0.0
    acs.reaction_wheels = _make_orthogonal_wheels()
    acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
    acs._compute_disturbance_torque = lambda _ut: np.zeros(3)
    acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)
    acs.current_pass = DummyPass(1.0, 0.0)

    acs._apply_pass_wheel_update(dt=1.0, utime=0.0)

    expected_torque = (math.pi / 180.0) * 10.0
    # Wheel momentum is opposite to body torque (Newton's 3rd law)
    assert math.isclose(
        acs.reaction_wheels[2].current_momentum, -expected_torque, rel_tol=2e-2
    )
    assert math.isclose(acs.reaction_wheels[0].current_momentum, 0.0, abs_tol=1e-9)
    assert math.isclose(acs.reaction_wheels[1].current_momentum, 0.0, abs_tol=1e-9)


def test_pass_tracking_no_change_when_no_motion(acs):
    acs.ra = 0.0
    acs.dec = 0.0
    acs.reaction_wheels = _make_orthogonal_wheels()
    acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
    acs._compute_disturbance_torque = lambda _ut: np.zeros(3)
    acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)
    acs.current_pass = DummyPass(0.0, 0.0)

    acs._apply_pass_wheel_update(dt=1.0, utime=0.0)

    assert acs._last_pass_torque_target_mag == 0.0
    assert acs._last_pass_torque_actual_mag == 0.0
    for wheel in acs.reaction_wheels:
        assert wheel.current_momentum == 0.0


def test_pass_tracking_momentum_delta_matches_torque(acs):
    acs.ra = 0.0
    acs.dec = 0.0
    acs.reaction_wheels = _make_orthogonal_wheels()
    acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
    acs._compute_disturbance_torque = lambda _ut: np.zeros(3)
    acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)
    acs.current_pass = DummyPass(1.0, 0.0)

    before = {w.name: w.current_momentum for w in acs.reaction_wheels}
    dt = 0.5
    acs._apply_pass_wheel_update(dt=dt, utime=0.0)

    after = {w.name: w.current_momentum for w in acs.reaction_wheels}
    snapshot = acs.wheel_snapshot()
    assert snapshot.wheels
    for reading in snapshot.wheels:
        delta = after[reading.name] - before[reading.name]
        assert delta == pytest.approx(reading.torque_applied * dt, rel=1e-6, abs=1e-9)


def test_pass_tracking_torque_uses_rate_delta(acs):
    acs.ra = 0.0
    acs.dec = 0.0
    acs.reaction_wheels = _make_orthogonal_wheels()
    acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
    acs._compute_disturbance_torque = lambda _ut: np.zeros(3)
    acs.config.spacecraft_bus.attitude_control.spacecraft_moi = (10.0, 10.0, 10.0)
    acs.current_pass = DummyPass(1.0, 0.0)
    acs._last_pass_rate_deg_s = 0.2

    acs._apply_pass_wheel_update(dt=1.0, utime=0.0)

    accel_req = 0.8  # (1.0 - 0.2) deg/s^2
    expected_torque = (math.pi / 180.0) * 10.0 * accel_req
    assert acs._last_pass_rate_deg_s == pytest.approx(1.0, rel=1e-6)
    assert acs._last_pass_torque_target_mag == pytest.approx(expected_torque, rel=1e-6)
