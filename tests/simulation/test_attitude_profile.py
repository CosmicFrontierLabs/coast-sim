import numpy as np
import pytest
from pydantic import ValidationError

from conops.common.vector import attitude_to_quat, quaternion_attitude_delta
from conops.config import AttitudeControlSystem
from conops.simulation.attitude_profile import SlewMotionProfile, same_rotation
from conops.simulation.slew import Slew, _SlewSegment


@pytest.mark.parametrize(
    "end",
    [
        (0, 0, 0),
        (0.001, 0, 0),
        (90, 10, 80),
        (180, 0, 0),
        (30, 90, 45),
        (359, -89.99, 359),
    ],
)
@pytest.mark.parametrize("directional", [False, True])
def test_profile_matches_runtime_and_snapshots_config(end, directional):
    config = AttitudeControlSystem(
        max_slew_rate=2, slew_acceleration=0.5, settle_time=13
    )
    if directional:
        config.max_slew_rate_body = (0.2, 0.2, 2.0)
        config.slew_acceleration_body = (0.02, 0.04, 0.5)
    slew = Slew(
        acs_config=config, slewstart=1000, endra=end[0], enddec=end[1], endroll=end[2]
    )
    slew.calc_slewtime()
    motion = slew.motion_profile()
    assert motion is not None
    restored = SlewMotionProfile.model_validate_json(motion.model_dump_json())
    times = np.linspace(999.0, slew.slewend + 50, 97)
    expected = [attitude_to_quat(*slew.attitude(time)) for time in times]
    for time, quaternion in zip(times, expected):
        assert same_rotation(restored.quaternion_at(time), quaternion)
    config.max_slew_rate = 10.0
    config.max_slew_rate_body = None
    slew.endra = 42.0
    for time, quaternion in zip(times, expected):
        assert same_rotation(restored.quaternion_at(time), quaternion)
    with pytest.raises(ValidationError, match="frozen"):
        restored.start_utime = 0


def test_waypoint_profile_reuses_each_segments_rest_to_rest_law():
    config = AttitudeControlSystem(max_slew_rate_body=(0.2, 0.3, 2))
    slew = Slew(acs_config=config, slewstart=1000, endra=80, enddec=30, endroll=50)
    slew.calc_slewtime()
    attitudes = [(0, 0, 0), (45, 45, 20), (80, 30, 50)]
    segments = []
    for start, end in zip(attitudes, attitudes[1:]):
        distance, axis = quaternion_attitude_delta(*start, *end)
        segments.append(
            _SlewSegment(
                distance, axis, attitude_to_quat(*start), attitude_to_quat(*end)
            )
        )
    slew._slew_segments = segments
    slew.slewdist = sum(s.distance_deg for s in segments)
    motion = slew.motion_profile()
    assert motion is not None
    for time in np.linspace(1000, motion.breakpoints[-1] + 20, 201):
        assert same_rotation(
            motion.quaternion_at(time), attitude_to_quat(*slew.attitude(time))
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("start_quaternion", (0, 0, 0, 0)),
        ("distance_deg", 180),
        ("max_rate_deg_s", 0),
        ("acceleration_deg_s2", float("nan")),
    ],
)
def test_invalid_motion_rejected(field, value):
    slew = Slew(acs_config=AttitudeControlSystem(), endra=90)
    slew.calc_slewtime()
    payload = slew.motion_profile().model_dump()
    payload["segments"][0][field] = value
    with pytest.raises(ValueError):
        SlewMotionProfile.model_validate(payload)


def test_discontinuous_segment_chain_rejected():
    slew = Slew(acs_config=AttitudeControlSystem(), endra=90)
    slew.calc_slewtime()
    payload = slew.motion_profile().model_dump()
    payload["segments"] *= 2
    with pytest.raises(ValueError, match="continuous"):
        SlewMotionProfile.model_validate(payload)


def test_legacy_nonkinematic_motion_remains_unresolved():
    slew = Slew(acs_config=AttitudeControlSystem(max_slew_rate=0), endra=90)
    slew.calc_slewtime()
    assert slew.motion_profile() is None


def test_snapshot_cache_invalidates_when_motion_or_limits_change():
    config = AttitudeControlSystem()
    slew = Slew(acs_config=config, endra=90)
    slew.calc_slewtime()
    first = slew.motion_profile()
    assert slew.motion_profile() is first
    config.max_slew_rate = 1.0
    second = slew.motion_profile()
    assert second is not first
    assert second.segments[0].max_rate_deg_s == 1.0
    assert first.segments[0].max_rate_deg_s == 0.25
    slew.slewstart = 1000
    third = slew.motion_profile()
    assert third is not second
    assert third.start_utime == 1000
    slew.endroll = 40
    slew.calc_slewtime()
    assert slew.motion_profile() is not third
