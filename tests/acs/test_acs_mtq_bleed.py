from unittest.mock import Mock

import numpy as np

from conops import ACSMode
from conops.simulation.reaction_wheel import ReactionWheel


def test_mtq_bleed_in_science_toggle(acs):
    acs.reaction_wheels = [
        ReactionWheel(
            max_torque=0.1,
            max_momentum=1.0,
            orientation=(1.0, 0.0, 0.0),
        )
    ]
    acs.magnetorquers = [
        {"orientation": (1.0, 0.0, 0.0), "dipole": 32.0, "power_draw": 5.0}
    ]
    # Sync with WheelDynamics
    acs.wheel_dynamics.wheels = acs.reaction_wheels
    acs.wheel_dynamics.magnetorquers = acs.magnetorquers
    acs.acsmode = ACSMode.SCIENCE
    acs._desat_active = False
    acs._desat_use_mtq = True
    acs._last_pointing_time = 100.0

    # Mock disturbance model for MTQ calculations
    acs.disturbance_model = Mock()
    acs.disturbance_model.local_bfield_vector = Mock(
        return_value=(np.array([0.0, 0.0, 3e-5]), 3e-5)
    )
    acs.disturbance_model.compute = Mock(return_value=(np.zeros(3), {}))

    # Set via acs_config since _update_wheel_momentum reads from there
    acs.acs_config.mtq_bleed_in_science = False
    acs._update_wheel_momentum(101.0)
    assert acs.mtq_power_w == 0.0

    acs.acs_config.mtq_bleed_in_science = True
    acs._update_wheel_momentum(102.0)
    assert acs.mtq_power_w > 0.0


def test_mtq_cycle_on_off(acs):
    acs.reaction_wheels = [
        ReactionWheel(
            max_torque=0.1,
            max_momentum=1.0,
            orientation=(1.0, 0.0, 0.0),
            current_momentum=0.5,
        )
    ]
    acs.magnetorquers = [
        {"orientation": (1.0, 0.0, 0.0), "dipole": 32.0, "power_draw": 5.0}
    ]
    acs.wheel_dynamics.wheels = acs.reaction_wheels
    acs.wheel_dynamics.magnetorquers = acs.magnetorquers
    acs.acsmode = ACSMode.DESAT
    acs._desat_active = True
    acs._desat_use_mtq = True
    acs._last_pointing_time = 100.0

    acs.acs_config.mtq_cycle_on_s = 9.0
    acs.acs_config.mtq_cycle_off_s = 1.0

    acs.disturbance_model = Mock()
    acs.disturbance_model.local_bfield_vector = Mock(
        return_value=(np.array([0.0, 0.0, 3e-5]), 3e-5)
    )
    acs.disturbance_model.compute = Mock(return_value=(np.zeros(3), {}))

    acs._update_wheel_momentum(109.0)
    assert acs.mtq_power_w > 0.0

    acs._update_wheel_momentum(110.0)
    assert acs.mtq_power_w == 0.0
    assert acs._last_mtq_torque_mag == 0.0
