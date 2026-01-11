import numpy as np

from conops.simulation.reaction_wheel import ReactionWheel
from conops.simulation.wheel_dynamics import WheelDynamics


def test_mtq_desat_flips_dipole_to_reduce_negative_momentum() -> None:
    wheel = ReactionWheel(
        max_torque=0.1,
        max_momentum=1.0,
        orientation=(1.0, 0.0, 0.0),
        current_momentum=-0.5,
    )
    magnetorquers = [
        {"orientation": (0.0, 1.0, 0.0), "dipole_strength": 1.0, "power_draw": 1.0}
    ]
    wd = WheelDynamics([wheel], np.eye(3), magnetorquers=magnetorquers)

    wd.apply_magnetorquer_desat(np.array([0.0, 0.0, 1.0]), dt=1.0)

    assert abs(wheel.current_momentum) < 0.5
