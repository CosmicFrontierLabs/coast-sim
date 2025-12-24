"""Demo showing sequential slew impulse usage and conservation checks.

Run: python3 examples/wheel_sequence_demo.py
"""
from conops.simulation.reaction_wheel import ReactionWheel


def main():
    rw = ReactionWheel(max_torque=0.1, max_momentum=1.0)
    motion_time = 15.0
    req_torque = 0.1

    print("Initial wheel momentum:", rw.current_momentum)
    for i in range(1, 5):
        adj = rw.reserve_impulse(req_torque, motion_time)
        print(f"Slew {i}: requested={req_torque:.3f} N*m, adjusted={adj:.6f} N*m")
        rw.apply_torque(adj, motion_time)
        print(f"  Wheel momentum after slew {i}: {rw.current_momentum:.6f} N*m*s (max {rw.max_momentum})")

    # Conservation check example: apply small torque and compute implied spacecraft L
    T = 0.05
    dt = 10.0
    rw2 = ReactionWheel(max_torque=0.1, max_momentum=10.0)
    rw2.current_momentum = 0.0
    I = 5.0
    rw2.apply_torque(T, dt)
    Lw = rw2.current_momentum
    Ls = -T * dt
    print(f"Conservation demo: wheel L={Lw:.6f}, spacecraft L={Ls:.6f}, total={Lw+Ls:.6e}")


if __name__ == "__main__":
    main()
