"""Changing body coordinates must preserve the physical observation and power."""

from datetime import timedelta

import numpy as np
import pytest
from rust_ephem import SunConstraint

from conops import QueueDITL
from conops.common.enums import SlewAlgorithm
from conops.common.vector import (
    attitude_to_quat,
    quaternion_to_rotation_matrix,
    radec2vec,
)
from conops.config import (
    Instrument,
    Payload,
    SolarPanel,
    SolarPanelSet,
    SpacecraftFrame,
    Telescope,
)
from conops.simulation.slew import Slew
from conops.targets import MovingPointing, Pointing, track_from_table

from .planning_scenario import BEGIN, HOUR, T0, make_config


@pytest.mark.parametrize("instrument_name", ["Camera", "Scope", None])
@pytest.mark.parametrize("moving", [False, True])
def test_frame_change_preserves_execution_and_physical_vectors(instrument_name, moving):
    runs = []
    frame = SpacecraftFrame(boresight=(0, 0, 1), sun_side=(1, 0, 0))
    for axes in [SpacecraftFrame(), frame]:
        config = make_config(3)
        config.constraint.sun_constraint = None
        config.constraint.earth_constraint = None
        config.spacecraft_bus.frame = axes
        config.solar_panel = SolarPanelSet(
            panels=[SolarPanel(normal=axes.sun_side, max_power=3000)]
        )
        config.payload = Payload(
            instruments=[
                Telescope(name="Scope", mounting=axes.mounting),
                Instrument(name="Camera"),
            ]
        )
        fields = dict(
            config=config,
            obsid=100,
            ra=105,
            dec=10,
            roll_offset_deg=7,
            instrument_name=instrument_name,
            ss_min=1200,
            ss_max=1200,
        )
        target = (
            MovingPointing(
                **fields,
                track=track_from_table([T0, T0 + 3 * HOUR], [105, 106], [10, 10.1]),
            )
            if moving
            else Pointing(**fields)
        )
        target.exptime = 1200
        run = QueueDITL(config=config, begin=BEGIN, end=BEGIN + timedelta(hours=3))
        run.queue.append(target)
        assert run.calc()
        assert not run.validate_plan_matches_execution()
        runs.append((run, target))
    (canonical, _), (changed, target) = runs
    assert [(e.begin, e.end, e.slewtime) for e in canonical.plan] == [
        (e.begin, e.end, e.slewtime) for e in changed.plan
    ]
    for original, transformed in zip(
        canonical.telemetry.housekeeping, changed.telemetry.housekeeping, strict=True
    ):
        q0 = attitude_to_quat(original.ra, original.dec, original.roll)
        q1 = attitude_to_quat(transformed.ra, transformed.dec, transformed.roll)
        matrix = quaternion_to_rotation_matrix(q1)
        assert matrix == pytest.approx(
            frame.body_from_canonical @ quaternion_to_rotation_matrix(q0), abs=1e-9
        )
        assert transformed.sun_body_vector == pytest.approx(
            frame.body_from_canonical @ np.asarray(original.sun_body_vector), abs=1e-9
        )
        assert transformed.power_usage == pytest.approx(original.power_usage)
        assert transformed.panel_illumination == pytest.approx(
            original.panel_illumination
        )
        if transformed.collection_seconds:
            ra, dec = target.position_at(transformed.timestamp.timestamp())
            direction = radec2vec(np.radians(ra), np.radians(dec))
            # Tracking interpolates minute-spaced guidance knots during motion.
            assert matrix @ direction == pytest.approx(frame.boresight, abs=2e-5)


def test_plain_instrument_mount_cache_tracks_frame_changes():
    config = make_config(1)
    config.payload = Payload(instruments=[Instrument(name="Camera")])
    target = Pointing(config=config, instrument_name="Camera", ra=105, dec=10)
    assert target.science_telescope() is None
    config.spacecraft_bus.frame = SpacecraftFrame(
        boresight=(0, 0, 1), sun_side=(1, 0, 0)
    )
    mount = target.science_telescope()
    assert mount is not None
    assert mount is target.science_telescope()
    assert mount.boresight == pytest.approx((0, 0, 1))
    config.spacecraft_bus.frame = SpacecraftFrame(
        boresight=(0, 1, 0), sun_side=(0, 0, 1)
    )
    assert target.science_telescope().boresight == pytest.approx((0, 1, 0))


@pytest.mark.parametrize("explicit_route", [False, True])
@pytest.mark.parametrize(
    "frame",
    [
        SpacecraftFrame(boresight=(0, 0, 1), sun_side=(1, 0, 0)),
        SpacecraftFrame(
            boresight=(np.sqrt(0.75), 0.5, 0), sun_side=(-0.5, np.sqrt(0.75), 0)
        ),
    ],
)
def test_sun_avoidance_and_hardware_safety_are_frame_invariant(explicit_route, frame):
    paths = []
    for axes in [SpacecraftFrame(), frame]:
        config = make_config(2)
        config.spacecraft_bus.frame = axes
        config.constraint.sun_constraint = SunConstraint(min_angle=45)
        config.constraint.earth_constraint = None
        config.constraint.safety_constraint = SunConstraint(min_angle=45)
        acs = config.spacecraft_bus.attitude_control
        acs.slew_algorithm = SlewAlgorithm.CONSTRAINT_AVOIDING
        if explicit_route:
            acs.slew_constraint = SunConstraint(min_angle=45)
        ra, dec = (
            float(config.constraint.ephem.sun_ra_deg[0]),
            float(config.constraint.ephem.sun_dec_deg[0]),
        )
        unsafe = axes.mounting.body_attitude(ra, dec, 27)
        safe = axes.mounting.body_attitude((ra + 50) % 360, dec, 0)
        assert config.constraint.in_explicit_safety(
            *unsafe[:2], T0, target_roll=unsafe[2]
        )
        assert not config.constraint.in_explicit_safety(
            *safe[:2], T0, target_roll=safe[2]
        )
        assert config.constraint.in_safety_constraint_batch(
            [unsafe[0], safe[0]],
            [unsafe[1], safe[1]],
            T0,
            target_rolls=[unsafe[2], safe[2]],
        ).tolist() == [True, False]
        start = axes.mounting.body_attitude((ra - 50) % 360, dec, 0)
        slew = Slew(
            config=config,
            slewstart=T0,
            startra=start[0],
            startdec=start[1],
            startroll=start[2],
            endra=safe[0],
            enddec=safe[1],
            endroll=safe[2],
        )
        slew.calc_slewtime()
        assert len(slew.quaternion_waypoints()) == 3  # a real avoidance detour
        paths.append(slew)
    assert paths[0].slewtime == paths[1].slewtime
    assert paths[0].slewdist == pytest.approx(paths[1].slewdist)
    assert paths[0].great_circle_deg == pytest.approx(paths[1].great_circle_deg)
    for original, changed in zip(
        paths[0].quaternion_waypoints(), paths[1].quaternion_waypoints(), strict=True
    ):
        assert quaternion_to_rotation_matrix(changed) == pytest.approx(
            frame.body_from_canonical @ quaternion_to_rotation_matrix(original),
            abs=1e-9,
        )


def test_explicit_hardware_offsets_are_not_rotated_twice():
    frame = SpacecraftFrame(boresight=(0, 0, 1), sun_side=(1, 0, 0))
    hardware = SunConstraint(min_angle=30).boresight_offset(pitch_deg=-90)
    assert frame.constraint_in_body(hardware) is hardware
    combined = SunConstraint(min_angle=45) | hardware
    transformed = frame.constraint_in_body(combined)
    assert transformed.constraints[1] is hardware
    assert transformed.constraints[0] is not combined.constraints[0]
