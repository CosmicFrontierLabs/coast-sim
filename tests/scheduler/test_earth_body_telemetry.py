"""Earth telemetry uses the same physical body frame as the executed quaternion."""

from datetime import timedelta

import numpy as np
import pytest

from conops import DITL, create_ditl
from conops.common import ACSMode
from conops.common.ephemeris import position_vectors
from conops.common.vector import quaternion_to_rotation_matrix
from conops.config import (
    InstrumentMounting,
    ObservationTiming,
    Payload,
    SpacecraftFrame,
    Telescope,
)
from conops.schedulers import PriorityScheduler
from conops.targets import Pointing

from .planning_scenario import BEGIN, make_config


@pytest.mark.parametrize("planned", [False, True])
@pytest.mark.parametrize("payload_frame", [False, True])
@pytest.mark.parametrize("offset_mount", [False, True])
def test_earth_direction_matches_the_executed_quaternion(
    planned: bool, payload_frame: bool, offset_mount: bool
) -> None:
    config = make_config(2)
    config.constraint.sun_constraint = None
    config.constraint.earth_constraint = None
    config.payload.observation_timing = ObservationTiming()
    config.battery.planning_level = None
    if payload_frame:
        config.spacecraft_bus.frame = SpacecraftFrame(
            boresight=(0, 0, 1), sun_side=(1, 0, 0)
        )
    if offset_mount:
        config.payload = Payload(
            instruments=[
                Telescope(
                    name="OffsetScope",
                    mounting=InstrumentMounting(
                        body_from_instrument_quaternion_wxyz=(
                            2**-0.5,
                            0,
                            0,
                            2**-0.5,
                        )
                    ),
                )
            ],
            observation_timing=ObservationTiming(),
        )
    target = Pointing(
        config=config,
        obsid=700,
        ra=150,
        dec=-25,
        roll_deg=37,
        instrument_name="OffsetScope" if offset_mount else None,
        ss_min=600,
        ss_max=600,
        exptime=600,
    )
    end = BEGIN + timedelta(minutes=30)
    ephem = config.constraint.ephem
    assert ephem is not None
    if planned:
        plan = PriorityScheduler(
            config, [target], BEGIN, end, include_passes=False
        ).schedule()
        run = DITL(config=config, ephem=ephem, plan=plan, begin=BEGIN, end=end)
    else:
        run = create_ditl(config, [target], BEGIN, end, ephem=ephem)
    run.step_size = 60
    assert run.calc()
    assert run.validate_plan_matches_execution() == []
    for entry in run.plan:
        if entry.obsid == target.obsid:
            assert entry.instrument_name == target.science_instrument_name()
    positions = position_vectors(ephem, "gcrs")
    changed = False
    differs_from_instrument = False
    assert sum(h.collection_seconds or 0 for h in run.telemetry.housekeeping) == 600
    for sample in run.telemetry.housekeeping:
        position = np.asarray(positions[ephem.index(sample.timestamp)], dtype=float)
        inertial = -position / np.linalg.norm(position)
        rotation = quaternion_to_rotation_matrix(
            (sample.quat_w, sample.quat_x, sample.quat_y, sample.quat_z)
        )
        expected = rotation @ inertial
        assert sample.earth_body_vector == pytest.approx(expected, abs=1e-12)
        changed |= bool(np.linalg.norm(expected - inertial) > 0.1)
        if (
            offset_mount
            and sample.collection_seconds
            and sample.acs_mode == ACSMode.SCIENCE
        ):
            # The explicit 90-degree mount puts the science boresight on body
            # +Y. Earth telemetry still describes the physical spacecraft,
            # rather than the instrument frame whose +X points at the target.
            # Partial collection in the interval after a telemetry timestamp
            # can begin after that timestamp's slew has finished.
            ra, dec = np.radians([target.ra, target.dec])
            direction = np.array(
                [np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)]
            )
            assert rotation @ direction == pytest.approx((0, 1, 0), abs=1e-12)
            instrument_earth = np.array([expected[1], -expected[0], expected[2]])
            differs_from_instrument |= bool(
                np.linalg.norm(expected - instrument_earth) > 0.1
            )
    assert changed
    if offset_mount:
        assert differs_from_instrument
