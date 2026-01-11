import numpy as np

from conops.simulation.reaction_wheel import ReactionWheel
from conops.simulation.torque_allocator import allocate_wheel_torques


def test_allocate_wheel_torques_clamps_by_margin():
    """Test that wheel torques are clamped when approaching momentum limits.

    Wheel at +0.9 momentum (max=1.0, margin=0.95 → headroom=0.05).
    - Positive body torque → negative wheel torque (spins down) → away from max → no clamp
    - Negative body torque → positive wheel torque (spins up) → toward max → clamped
    """
    wheels = [
        ReactionWheel(
            max_torque=1.0,
            max_momentum=1.0,
            orientation=(1.0, 0.0, 0.0),
            current_momentum=0.9,
        )
    ]

    # Positive body torque → wheel spins DOWN (away from max) → no clamping
    taus, taus_allowed, _, clamped = allocate_wheel_torques(
        wheels,
        np.array([0.2, 0.0, 0.0]),
        dt=1.0,
        mom_margin=0.95,
    )
    assert clamped is False
    assert np.isclose(taus_allowed[0], -0.2)  # negative wheel torque

    # Negative body torque → wheel spins UP (toward max) → clamped
    taus, taus_allowed, _, clamped = allocate_wheel_torques(
        wheels,
        np.array([-0.2, 0.0, 0.0]),
        dt=1.0,
        mom_margin=0.95,
    )
    assert clamped is True
    assert np.isclose(taus_allowed[0], 0.05)  # limited by headroom


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
