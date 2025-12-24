"""Small demo showing effect of wheel torque on predicted slew time.

Run with: python3 examples/wheel_slew_demo.py
"""
from conops.config import MissionConfig
from conops.simulation.slew import Slew


def run_demo():
    cfg = MissionConfig()
    # set a reasonable ephemeris stub so Slew can initialize
    class DummyAngle:
        def __init__(self, deg):
            self.deg = deg

    class DummyBody:
        def __init__(self, ra, dec):
            self.ra = DummyAngle(ra)
            self.dec = DummyAngle(dec)

    class DummyEphem:
        def __init__(self):
            self.earth = [DummyBody(0.0, 0.0)]
            self.sun = [DummyBody(0.0, 0.0)]

        def index(self, dt):
            return 0

    cfg.constraint.ephem = DummyEphem()

    # baseline accel
    base_accel = cfg.spacecraft_bus.attitude_control.slew_acceleration

    # Create a slew between two nearby points
    slew = Slew(config=cfg)
    slew.startra = 0.0
    slew.startdec = 0.0
    slew.endra = 10.0
    slew.enddec = 0.0
    slew.slewstart = 0

    # Compute without wheel limits
    slew._accel_override = None
    slew._vmax_override = None
    t_nom = slew.calc_slewtime()

    # Now simulate a wheel-imposed lower accel
    slew._accel_override = base_accel / 4.0
    t_wheel = slew.calc_slewtime()

    print(f"Nominal slew time: {t_nom}s")
    print(f"Wheel-limited slew time (accel={slew._accel_override}): {t_wheel}s")


if __name__ == "__main__":
    run_demo()
