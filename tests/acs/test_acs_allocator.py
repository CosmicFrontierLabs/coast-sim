from unittest.mock import Mock

import numpy as np

from conops.simulation.reaction_wheel import ReactionWheel


def test_allocate_wheel_torques_clamps_by_margin(acs):
    acs.reaction_wheels = [
        ReactionWheel(
            max_torque=1.0,
            max_momentum=1.0,
            orientation=(1.0, 0.0, 0.0),
            current_momentum=0.9,
        )
    ]
    acs._wheel_mom_margin = 0.95

    taus, taus_allowed, _, clamped = acs._allocate_wheel_torques(
        np.array([0.2, 0.0, 0.0]), dt=1.0
    )
    assert clamped is True
    assert np.isclose(taus_allowed[0], 0.05)

    taus, taus_allowed, _, clamped = acs._allocate_wheel_torques(
        np.array([-0.2, 0.0, 0.0]), dt=1.0
    )
    assert clamped is False
    assert np.isclose(taus_allowed[0], -0.2)


def test_allocate_wheel_torques_weighting_prefers_low_momentum(acs):
    acs.reaction_wheels = [
        ReactionWheel(
            max_torque=1.0,
            max_momentum=1.0,
            orientation=(1.0, 0.0, 0.0),
            current_momentum=0.9,
            name="rw_high",
        ),
        ReactionWheel(
            max_torque=1.0,
            max_momentum=1.0,
            orientation=(1.0, 0.0, 0.0),
            current_momentum=0.0,
            name="rw_low",
        ),
    ]

    taus, taus_allowed, _, _ = acs._allocate_wheel_torques(
        np.array([1.0, 0.0, 0.0]), dt=1.0, use_weights=True
    )

    assert abs(taus_allowed[1]) > abs(taus_allowed[0])


def test_pass_wheel_update_respects_momentum_margin(acs):
    wheel = ReactionWheel(
        max_torque=10.0,
        max_momentum=1.0,
        orientation=(1.0, 0.0, 0.0),
        current_momentum=0.9,
        name="rw_x",
    )
    acs.reaction_wheels = [wheel]
    acs._wheel_mom_margin = 0.95
    acs._compute_disturbance_torque = Mock(return_value=np.zeros(3))

    acs.ra = 90.0
    acs.dec = 0.0
    acs.current_pass = Mock()
    acs.current_pass.ra_dec = Mock(return_value=(0.0, 90.0))

    acs._apply_pass_wheel_update(dt=1.0, utime=0.0)

    assert wheel.current_momentum <= 0.95 + 1e-6
