"""Charging search and tracking describe the same physical pose in either frame."""

from datetime import timedelta
from unittest.mock import Mock

import pytest
from rust_ephem import SunConstraint

from conops import QueueDITL
from conops.common import ACSMode, ObsType
from conops.common.vector import attitude_to_quat, quaternion_to_rotation_matrix
from conops.config import SolarPanel, SolarPanelSet, SpacecraftFrame
from conops.simulation.acs import ACS
from conops.simulation.emergency_charging import EmergencyCharging
from conops.targets import Pointing

from .planning_scenario import BEGIN, T0, make_config

FRAME = SpacecraftFrame(boresight=(0, 0, 1), sun_side=(1, 0, 0))


def configuration(frame):
    config = make_config(3)
    config.spacecraft_bus.frame = frame
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.solar_panel = SolarPanelSet(
        panels=[SolarPanel(normal=frame.body_vector((0, 1, 0)), max_power=3000)]
    )
    return config


def rotation(attitude):
    return quaternion_to_rotation_matrix(attitude_to_quat(*attitude))


@pytest.mark.parametrize("frame", [SpacecraftFrame(), FRAME])
@pytest.mark.parametrize(
    "pose", [(105, 10, 37), (-40, 20, 720), (10, 90, 37), (50, -90, 0)]
)
def test_reference_and_body_attitudes_round_trip(frame, pose):
    body = frame.body_attitude(*pose)
    restored = frame.canonical_attitude(*body)
    assert rotation(restored) == pytest.approx(rotation(pose), abs=1e-12)


def test_charging_search_is_independent_of_body_coordinates(monkeypatch):
    poses = []
    for frame in [SpacecraftFrame(), FRAME]:
        config = configuration(frame)
        charger = EmergencyCharging(config)
        monkeypatch.setattr(charger, "_is_in_sunlight", lambda *args: True)
        pointing = charger.create_charging_pointing(T0, config.constraint.ephem)
        assert pointing is not None
        poses.append((pointing.ra, pointing.dec, pointing.roll))
    assert rotation(poses[1]) == pytest.approx(
        FRAME.body_from_canonical @ rotation(poses[0]), abs=1e-12
    )


def test_charging_dwell_guidance_is_independent_of_body_coordinates():
    poses = []
    for frame in [SpacecraftFrame(), FRAME]:
        config = configuration(frame)
        acs = ACS(config)
        acs.ra, acs.dec, acs.roll = frame.mounting.body_attitude(105, 10, 37)
        acs.last_slew.obstype = ObsType.CHARGE
        acs._advance_attitude(T0)
        acs._update_dwell_guidance(T0 + 60)
        poses.append(acs._executor.predict(T0 + 3600).attitude)
    assert rotation(poses[1]) == pytest.approx(
        FRAME.body_from_canonical @ rotation(poses[0]), abs=1e-12
    )


@pytest.mark.parametrize("mode", [ACSMode.IDLE, ACSMode.CHARGING])
def test_charge_validation_does_not_mix_destination_and_executed_axes(mode):
    config = configuration(FRAME)
    config.constraint.safety_constraint = SunConstraint(min_angle=45)
    config.constraint.invalidate_combined_constraint_cache()
    sim = QueueDITL(config=config, begin=BEGIN, end=BEGIN + timedelta(hours=3))
    ra = (float(config.constraint.ephem.sun_ra_deg[0]) + 90) % 360
    destination = (ra, 0.0, 0.0)
    assert (
        sim._attitude_constraint_name_for_attitude(*destination, T0, ACSMode.CHARGING)
        is None
    )
    for roll in range(360):
        if (
            sim._attitude_constraint_name_for_attitude(
                destination[0], destination[1], roll, T0, ACSMode.CHARGING
            )
            is not None
        ):
            break
    else:
        pytest.fail("test requires a mixed-axis pose that violates the Sun cone")
    executed = ((ra - 90) % 360, 0.0, float(roll))
    assert (
        sim._attitude_constraint_name_for_attitude(*executed, T0, ACSMode.CHARGING)
        is None
    )
    sim.acs.ra, sim.acs.dec, sim.acs.roll = executed
    pointing = Pointing(
        config=config,
        ra=destination[0],
        dec=destination[1],
        roll=destination[2],
        obstype=ObsType.CHARGE,
    )
    sim.ppt = sim.charging_ppt = pointing
    sim._terminate_emergency_charging = Mock()
    sim._manage_ppt_lifecycle(T0, mode)
    sim._terminate_emergency_charging.assert_not_called()
