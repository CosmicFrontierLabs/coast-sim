import numpy as np

from conops.simulation.reaction_wheel import ReactionWheel
from conops.simulation.torque_allocator import allocate_wheel_torques


def test_allocate_wheel_torques_clamps_by_margin():
    wheels = [
        ReactionWheel(
            max_torque=1.0,
            max_momentum=1.0,
            orientation=(1.0, 0.0, 0.0),
            current_momentum=0.9,
        )
    ]

    taus, taus_allowed, _, clamped = allocate_wheel_torques(
        wheels,
        np.array([0.2, 0.0, 0.0]),
        dt=1.0,
        mom_margin=0.95,
    )
    assert clamped is True
    assert np.isclose(taus_allowed[0], 0.05)

    taus, taus_allowed, _, clamped = allocate_wheel_torques(
        wheels,
        np.array([-0.2, 0.0, 0.0]),
        dt=1.0,
        mom_margin=0.95,
    )
    assert clamped is False
    assert np.isclose(taus_allowed[0], -0.2)


def test_allocate_wheel_torques_weighting_prefers_low_momentum():
    wheels = [
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

    _, taus_allowed, _, _ = allocate_wheel_torques(
        wheels,
        np.array([1.0, 0.0, 0.0]),
        dt=1.0,
        use_weights=True,
    )

    assert abs(taus_allowed[1]) > abs(taus_allowed[0])
