from conops.config import MissionConfig
from conops.simulation.acs import ACS
from conops.simulation.slew import Slew


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


def test_acs_adds_wheel_from_config_and_limits_slew_accel():
    cfg = MissionConfig()
    # enable legacy single-wheel support
    acs_cfg = cfg.spacecraft_bus.attitude_control
    acs_cfg.wheel_enabled = True
    acs_cfg.wheel_max_torque = 0.05
    acs_cfg.wheel_max_momentum = 0.5
    # ensure ephem present for Slew/ACS
    cfg.constraint.ephem = DummyEphem()

    acs = ACS(config=cfg, log=None)
    # Create a real Slew and start it - ensure accel override is set
    slew = Slew(config=cfg)
    slew.endra = 10.0
    slew.enddec = 0.0
    # start slew - should compute overrides
    acs._start_slew(slew, utime=1000.0)
    # When a wheel is present, accel override should be set (or zero)
    assert hasattr(slew, "_accel_override")
    # accel override should not exceed configured ACS accel
    orig = cfg.spacecraft_bus.attitude_control.slew_acceleration
    if slew._accel_override is not None:
        assert slew._accel_override <= orig


def test_multi_wheel_parsing_creates_wheels():
    cfg = MissionConfig()
    cfg.constraint.ephem = DummyEphem()
    acs_cfg = cfg.spacecraft_bus.attitude_control
    acs_cfg.wheels = [
        {
            "orientation": [1.0, 0.0, 0.0],
            "max_torque": 0.1,
            "max_momentum": 1.0,
            "name": "w0",
        },
        {
            "orientation": [0.0, 1.0, 0.0],
            "max_torque": 0.1,
            "max_momentum": 1.0,
            "name": "w1",
        },
    ]
    acs = ACS(config=cfg, log=None)
    assert len(acs.reaction_wheels) == 2


def test_wheel_config_validation_full_rank():
    """Test that a proper 3-axis wheel config has rank 3."""
    cfg = MissionConfig()
    cfg.constraint.ephem = DummyEphem()
    acs_cfg = cfg.spacecraft_bus.attitude_control
    acs_cfg.wheels = [
        {"orientation": [1.0, 0.0, 0.0], "max_torque": 0.1, "max_momentum": 1.0},
        {"orientation": [0.0, 1.0, 0.0], "max_torque": 0.1, "max_momentum": 1.0},
        {"orientation": [0.0, 0.0, 1.0], "max_torque": 0.1, "max_momentum": 1.0},
    ]
    acs = ACS(config=cfg, log=None)
    assert acs._wheel_config_rank == 3
    assert acs._wheel_config_n_wheels == 3


def test_wheel_config_validation_parallel_wheels_rank_deficient():
    """Test that parallel wheels result in rank < 3."""
    cfg = MissionConfig()
    cfg.constraint.ephem = DummyEphem()
    acs_cfg = cfg.spacecraft_bus.attitude_control
    # Two wheels along X axis, one along Y - no Z-axis control
    acs_cfg.wheels = [
        {"orientation": [1.0, 0.0, 0.0], "max_torque": 0.1, "max_momentum": 1.0},
        {"orientation": [1.0, 0.0, 0.0], "max_torque": 0.1, "max_momentum": 1.0},
        {"orientation": [0.0, 1.0, 0.0], "max_torque": 0.1, "max_momentum": 1.0},
    ]
    acs = ACS(config=cfg, log=None)
    assert acs._wheel_config_rank == 2  # Only 2 independent axes
    assert acs._wheel_config_n_wheels == 3


def test_wheel_config_validation_single_wheel():
    """Test that single wheel logs a warning (under-actuated)."""
    cfg = MissionConfig()
    cfg.constraint.ephem = DummyEphem()
    acs_cfg = cfg.spacecraft_bus.attitude_control
    acs_cfg.wheel_enabled = True
    acs_cfg.wheel_max_torque = 0.1
    acs_cfg.wheel_max_momentum = 1.0
    acs = ACS(config=cfg, log=None)
    assert acs._wheel_config_n_wheels == 1
    assert acs._wheel_config_rank == 1


def test_wheel_config_four_wheel_redundant():
    """Test that a 4-wheel pyramid config has rank 3 and good conditioning."""
    cfg = MissionConfig()
    cfg.constraint.ephem = DummyEphem()
    acs_cfg = cfg.spacecraft_bus.attitude_control
    # Classic pyramid configuration (4 wheels, redundant)
    import math

    theta = math.radians(54.74)  # angle from vertical for pyramid
    acs_cfg.wheels = [
        {
            "orientation": [math.sin(theta), 0.0, math.cos(theta)],
            "max_torque": 0.1,
            "max_momentum": 1.0,
        },
        {
            "orientation": [0.0, math.sin(theta), math.cos(theta)],
            "max_torque": 0.1,
            "max_momentum": 1.0,
        },
        {
            "orientation": [-math.sin(theta), 0.0, math.cos(theta)],
            "max_torque": 0.1,
            "max_momentum": 1.0,
        },
        {
            "orientation": [0.0, -math.sin(theta), math.cos(theta)],
            "max_torque": 0.1,
            "max_momentum": 1.0,
        },
    ]
    acs = ACS(config=cfg, log=None)
    assert acs._wheel_config_rank == 3
    assert acs._wheel_config_n_wheels == 4
    # Should be well-conditioned
    assert hasattr(acs, "_wheel_config_condition")
    assert acs._wheel_config_condition < 10  # Well-conditioned
