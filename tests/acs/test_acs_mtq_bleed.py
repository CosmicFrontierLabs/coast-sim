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
    acs.acsmode = ACSMode.SCIENCE
    acs._desat_active = False
    acs._desat_use_mtq = True
    acs._last_pointing_time = 100.0

    acs._mtq_bleed_in_science = False
    acs._update_wheel_momentum(101.0)
    assert acs.mtq_power_w == 0.0

    acs._mtq_bleed_in_science = True
    acs._update_wheel_momentum(102.0)
    assert acs.mtq_power_w > 0.0
