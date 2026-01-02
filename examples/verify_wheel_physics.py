from math import pi

from conops.config import MissionConfig
from conops.simulation.acs import ACS
from conops.simulation.slew import Slew

# Setup config
cfg = MissionConfig()


# minimal ephem stub
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


# stub eclipse check on class
def _in_eclipse_stub(self, ra, dec, time):
    return False


setattr(cfg.constraint.__class__, "in_eclipse", _in_eclipse_stub)

acs_cfg = cfg.spacecraft_bus.attitude_control
# set known MOI and wheel params
acs_cfg.spacecraft_moi = 5.0  # kg*m^2
acs_cfg.wheel_enabled = True
acs_cfg.wheel_max_torque = 0.2  # N*m
acs_cfg.wheel_max_momentum = 1.0  # N*m*s
# request a high ACS accel so wheel torque cap is used
acs_cfg.slew_acceleration = 10.0  # deg/s^2

acs = ACS(config=cfg, log=None)

# Create a single slew of 30 deg
slew = Slew(config=cfg)
slew.startra = acs.ra
slew.startdec = acs.dec
slew.endra = slew.startra + 30.0
slew.enddec = slew.startdec
slew.slewstart = 0.0
acs._start_slew(slew, utime=0.0)

wheel = acs.reaction_wheels[0]
print(
    "Wheel params: max_torque=", wheel.max_torque, "max_momentum=", wheel.max_momentum
)

# Step through slew
dt = 0.5
t = 0.0
end_t = slew.slewend
prev_m = wheel.current_momentum
ok = True
print("\nTimestep diagnostics:")
print(
    "t\taccel_req(deg/s2)\treq_torque(Nm)\tallowed_calc(Nm)\tdelta_exp(Nm*s)\tdelta_act(Nm*s)\tcurr_mom(Nm*s)"
)
while t <= end_t + 1e-12:
    # compute pre-step expected values
    accel_req = getattr(acs.current_slew, "_accel_override", 0.0)
    moi = cfg.spacecraft_bus.attitude_control.spacecraft_moi
    req_torque = (accel_req * (pi / 180.0)) * float(moi) if accel_req and moi else 0.0
    available = max(0.0, wheel.max_momentum - abs(wheel.current_momentum))
    max_torque_by_momentum = available / dt if dt > 0 else 0.0
    allowed_calc = min(abs(req_torque), wheel.max_torque, max_torque_by_momentum)
    allowed_calc = allowed_calc if req_torque >= 0 else -allowed_calc
    expected_delta = allowed_calc * dt

    acs.pointing(t + dt)  # advance to t+dt (pointing uses last_time tracking)
    curr_m = wheel.current_momentum
    actual_delta = curr_m - prev_m
    prev_m = curr_m

    print(
        f"{t:.2f}\t{accel_req:.4f}\t{req_torque:.6f}\t{allowed_calc:.6f}\t{expected_delta:.6f}\t{actual_delta:.6f}\t{curr_m:.6f}"
    )
    # compare
    if abs(actual_delta - expected_delta) > 1e-6 + 1e-6 * abs(expected_delta):
        ok = False
    t += dt

print("\nResult: Physics check", "PASSED" if ok else "FAILED")
print("Final wheel momentum:", wheel.current_momentum)
