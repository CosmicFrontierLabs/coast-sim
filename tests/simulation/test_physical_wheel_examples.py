from conops.simulation.reaction_wheel import ReactionWheel


def test_angular_momentum_conservation_simple():
    """Applying internal wheel torque should produce equal and opposite spacecraft impulse."""
    I = 5.0  # spacecraft moment of inertia (kg*m^2)
    rw = ReactionWheel(max_torque=0.2, max_momentum=10.0)
    rw.current_momentum = 0.0

    T = 0.05  # N*m applied to wheel
    dt = 10.0  # seconds

    # Apply torque to wheel (this increases wheel stored momentum by T*dt)
    rw.apply_torque(T, dt)

    L_wheel = rw.current_momentum
    # Spacecraft receives equal-and-opposite impulse: delta_L = -T * dt
    L_spacecraft = -T * dt

    # Total angular momentum should be conserved (close to zero)
    assert abs((L_wheel + L_spacecraft)) < 1e-9


def test_max_momentum_storage_clamps():
    rw = ReactionWheel(max_torque=1.0, max_momentum=0.5)
    # apply a large torque over time that would exceed capacity
    rw.apply_torque(1.0, 1.0)
    assert abs(rw.current_momentum) <= rw.max_momentum + 1e-12
    # applying opposite torque should allow movement back within bounds
    rw.apply_torque(-2.0, 1.0)
    assert abs(rw.current_momentum) <= rw.max_momentum + 1e-12


def test_sequential_slew_impulse_exhaustion():
    rw = ReactionWheel(max_torque=0.1, max_momentum=1.0)
    motion_time = 20.0
    req = 0.1

    first_adj = rw.reserve_impulse(req, motion_time)
    rw.apply_torque(first_adj, motion_time)

    second_adj = rw.reserve_impulse(req, motion_time)

    # After applying the first adjusted torque, available momentum should be reduced,
    # so the second adjusted torque should be <= first adjusted torque
    assert abs(second_adj) <= abs(first_adj) + 1e-12
    # If first_adj already used some capacity, second_adj will typically be smaller
    assert second_adj <= first_adj + 1e-12
