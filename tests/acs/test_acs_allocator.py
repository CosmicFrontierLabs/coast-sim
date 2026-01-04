from unittest.mock import Mock

import numpy as np

from conops.simulation.reaction_wheel import ReactionWheel


def test_allocate_wheel_torques_clamps_by_margin(acs):
    """Test that wheel torques are clamped when approaching momentum limits.

    Wheel at +0.9 momentum (max=1.0, margin=0.95 → headroom=0.05).
    - Positive body torque → negative wheel torque (spins down) → away from max → no clamp
    - Negative body torque → positive wheel torque (spins up) → toward max → clamped
    """
    acs.reaction_wheels = [
        ReactionWheel(
            max_torque=1.0,
            max_momentum=1.0,
            orientation=(1.0, 0.0, 0.0),
            current_momentum=0.9,
        )
    ]
    acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
    acs.wheel_dynamics._momentum_margin = 0.95

    # Positive body torque → wheel spins DOWN (away from max) → no clamping
    taus, taus_allowed, _, clamped = acs._allocate_wheel_torques(
        np.array([0.2, 0.0, 0.0]), dt=1.0
    )
    assert clamped is False
    assert np.isclose(taus_allowed[0], -0.2)  # negative wheel torque

    # Negative body torque → wheel spins UP (toward max) → clamped
    taus, taus_allowed, _, clamped = acs._allocate_wheel_torques(
        np.array([-0.2, 0.0, 0.0]), dt=1.0
    )
    assert clamped is True
    assert np.isclose(taus_allowed[0], 0.05)  # limited by headroom


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
    acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync

    taus, taus_allowed, _, _ = acs._allocate_wheel_torques(
        np.array([1.0, 0.0, 0.0]), dt=1.0, use_weights=True
    )

    assert abs(taus_allowed[1]) > abs(taus_allowed[0])


def test_apply_control_torque_sign_convention(acs):
    """Verify that allocate + apply produces body momentum in the desired direction.

    Physics:
    - We request a positive body torque along +x
    - Allocator finds wheel torques to produce that body torque
    - apply_wheel_torques applies reaction to body
    - body_momentum should increase along +x (same direction as requested torque)
    """
    # Single wheel along x-axis, starting at zero momentum
    wheel = ReactionWheel(
        max_torque=1.0,
        max_momentum=10.0,
        orientation=(1.0, 0.0, 0.0),
        current_momentum=0.0,
        name="rw_x",
    )
    acs.reaction_wheels = [wheel]
    acs.wheel_dynamics.wheels = acs.reaction_wheels
    acs.wheel_dynamics.body_momentum = np.zeros(3)

    # Request positive body torque along x-axis
    desired_body_torque = np.array([1.0, 0.0, 0.0])
    dt = 1.0

    # Allocate wheel torques
    _, taus_allowed, _, _ = acs._allocate_wheel_torques(desired_body_torque, dt=dt)

    # Apply wheel torques (this updates body_momentum)
    actual_torque = acs.wheel_dynamics.apply_wheel_torques(taus_allowed, dt)

    # Body momentum should have increased along +x (same sign as desired torque)
    # Expected: body_momentum[0] > 0 after requesting +x torque
    assert acs.wheel_dynamics.body_momentum[0] > 0, (
        f"Sign mismatch: requested +x body torque but body_momentum[0] = "
        f"{acs.wheel_dynamics.body_momentum[0]:.4f} (expected > 0)"
    )

    # The actual torque returned should also be in the +x direction
    assert actual_torque[0] > 0, (
        f"Sign mismatch: actual_torque[0] = {actual_torque[0]:.4f} (expected > 0)"
    )


def test_disturbance_rejection_cancels_external_torque(acs):
    """Verify that wheel torque can cancel an external disturbance.

    Physics:
    - External disturbance applies +τ to the body
    - Wheels apply -τ to counter it
    - Net body momentum change should be zero (attitude hold)

    This is the core of attitude hold mode: external torques change total
    system momentum, but wheels redistribute it to keep body stationary.
    """
    # Single wheel along x-axis
    wheel = ReactionWheel(
        max_torque=10.0,
        max_momentum=100.0,
        orientation=(1.0, 0.0, 0.0),
        current_momentum=0.0,
        name="rw_x",
    )
    acs.reaction_wheels = [wheel]
    acs.wheel_dynamics.wheels = acs.reaction_wheels
    acs.wheel_dynamics.body_momentum = np.zeros(3)

    dt = 1.0
    disturbance = np.array([0.5, 0.0, 0.0])  # External +x torque

    # Step 1: Apply external disturbance (changes total system momentum)
    acs.wheel_dynamics.apply_external_torque(disturbance, dt, source="test_disturbance")
    body_after_disturbance = acs.wheel_dynamics.body_momentum.copy()

    # Body should have gained momentum from disturbance
    assert body_after_disturbance[0] > 0, "Disturbance should add +x momentum to body"

    # Step 2: Counter with wheels (request -disturbance to cancel effect on body)
    counter_torque = -disturbance
    _, taus_allowed, _, _ = acs._allocate_wheel_torques(counter_torque, dt=dt)
    acs.wheel_dynamics.apply_wheel_torques(taus_allowed, dt)

    # Body momentum should return to near-zero (disturbance canceled by wheel reaction)
    assert np.allclose(acs.wheel_dynamics.body_momentum, 0.0, atol=1e-10), (
        f"Disturbance rejection failed: body_momentum = "
        f"{acs.wheel_dynamics.body_momentum} (expected ~0)"
    )

    # Wheel should have absorbed the momentum
    assert wheel.current_momentum > 0, (
        f"Wheel should have absorbed +x momentum, got {wheel.current_momentum}"
    )


def test_slew_acceleration_direction(acs):
    """Verify that requesting positive acceleration produces positive velocity change.

    Physics:
    - To accelerate spacecraft rotation about +x axis
    - We need +x torque on the body
    - Wheels spin in -x direction (absorb -x momentum)
    - Body gains +x momentum (rotates faster about +x)
    """
    # Single wheel along x-axis
    wheel = ReactionWheel(
        max_torque=10.0,
        max_momentum=100.0,
        orientation=(1.0, 0.0, 0.0),
        current_momentum=0.0,
        name="rw_x",
    )
    acs.reaction_wheels = [wheel]
    acs.wheel_dynamics.wheels = acs.reaction_wheels
    acs.wheel_dynamics.body_momentum = np.zeros(3)

    dt = 1.0
    # Slew wants to accelerate about +x axis
    slew_axis = np.array([1.0, 0.0, 0.0])
    requested_accel_torque = 0.5  # positive = accelerate in +x direction
    desired_body_torque = slew_axis * requested_accel_torque

    _, taus_allowed, _, _ = acs._allocate_wheel_torques(desired_body_torque, dt=dt)
    acs.wheel_dynamics.apply_wheel_torques(taus_allowed, dt)

    # Body should have +x momentum (accelerating in +x direction)
    assert acs.wheel_dynamics.body_momentum[0] > 0, (
        f"Slew acceleration wrong direction: body_momentum[0] = "
        f"{acs.wheel_dynamics.body_momentum[0]:.4f} (expected > 0)"
    )

    # Wheel should have -x momentum (conservation)
    assert wheel.current_momentum < 0, (
        f"Wheel momentum wrong sign: {wheel.current_momentum:.4f} (expected < 0)"
    )


def test_pass_wheel_update_respects_momentum_margin(acs):
    wheel = ReactionWheel(
        max_torque=10.0,
        max_momentum=1.0,
        orientation=(1.0, 0.0, 0.0),
        current_momentum=0.9,
        name="rw_x",
    )
    acs.reaction_wheels = [wheel]
    acs.wheel_dynamics.wheels = acs.reaction_wheels  # Keep in sync
    acs.wheel_dynamics._momentum_margin = 0.95
    acs._compute_disturbance_torque = Mock(return_value=np.zeros(3))

    acs.ra = 90.0
    acs.dec = 0.0
    acs.current_pass = Mock()
    acs.current_pass.ra_dec = Mock(return_value=(0.0, 90.0))

    acs._apply_pass_wheel_update(dt=1.0, utime=0.0)

    assert wheel.current_momentum <= 0.95 + 1e-6
