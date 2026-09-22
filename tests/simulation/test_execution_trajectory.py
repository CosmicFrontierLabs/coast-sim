import json
from datetime import datetime, timezone

import numpy as np
import pytest

from conops.common.vector import attitude_to_quat
from conops.config import AttitudeControlSystem
from conops.simulation.attitude_profile import ExecutedAttitudeInterval, same_rotation
from conops.simulation.execution_trajectory import ExecutionTrajectory
from conops.simulation.momentum import StoredMomentumTracker
from conops.simulation.slew import Slew
from conops.targets.plan import (
    AttitudeSampleSchema,
    AttitudeTimeseriesSchema,
    OrbitStateSampleSchema,
    OrbitStateTimeseriesSchema,
    Plan,
)

RADIUS = 6878.0
MEAN_MOTION = (398600.4418 / RADIUS**3) ** 0.5
START = 1_800_000_000.0


def _orbit(elapsed):
    angle = MEAN_MOTION * elapsed
    return (
        (RADIUS * np.cos(angle), RADIUS * np.sin(angle), 0.0),
        (
            -RADIUS * MEAN_MOTION * np.sin(angle),
            RADIUS * MEAN_MOTION * np.cos(angle),
            0.0,
        ),
    )


def _sidecars(*, resolved=True, step=60.0):
    slew = Slew(
        acs_config=AttitudeControlSystem(
            max_slew_rate=2, slew_acceleration=1, settle_time=0
        ),
        slewstart=START,
        endra=90.0 + np.degrees(MEAN_MOTION * 60),
    )
    slew.calc_slewtime()
    attitude = AttitudeTimeseriesSchema(version=1 if resolved else 0)
    orbit = OrbitStateTimeseriesSchema()
    for elapsed in np.arange(0.0, 60.0 + step / 2, step):
        time = START + elapsed
        timestamp = datetime.fromtimestamp(time, timezone.utc).isoformat()
        q = attitude_to_quat(*slew.attitude(time))
        attitude.samples.append(
            AttitudeSampleSchema(
                utime=time,
                timestamp=timestamp,
                quat_w=q[0],
                quat_x=q[1],
                quat_y=q[2],
                quat_z=q[3],
            )
        )
        position, velocity = _orbit(elapsed)
        orbit.samples.append(
            OrbitStateSampleSchema(
                utime=time,
                timestamp=timestamp,
                position_km=position,
                velocity_km_s=velocity,
            )
        )
    if resolved:
        attitude.resolved_intervals.append(
            ExecutedAttitudeInterval(
                start_utime=START,
                end_utime=START + 60,
                motion=slew.motion_profile(),
            )
        )
    return attitude, orbit, slew


def _trajectory(attitude, orbit):
    return ExecutionTrajectory(attitude, orbit, max_attitude_gap_s=2.5)


def test_legacy_dense_sidecars_roundtrip_read_only(tmp_path):
    attitude, orbit, _ = _sidecars(resolved=False, step=2.0)
    plan = Plan(attitude_timeseries=attitude, orbit_state_timeseries=orbit)
    path = plan.save(tmp_path / "plan.json")
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    trajectory = ExecutionTrajectory.load(path, max_attitude_gap_s=2.5)
    assert trajectory.start_utime == START
    assert trajectory.end_utime == START + 60
    # Random-access queries have no controller state or ordering requirement.
    first = trajectory.state_at(START + 17.5)
    list(trajectory.samples(0.37))
    assert trajectory.state_at(START + 17.5) == first
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before
    assert Plan.load(path).attitude_timeseries is None  # Existing load stays unchanged.


def test_sparse_motion_is_not_recovered_by_denser_interpolation():
    attitude, orbit, _ = _sidecars(resolved=False)
    with pytest.raises(ValueError, match="unresolved attitude input gap"):
        _trajectory(attitude, orbit)


def test_resolved_slew_and_hold_match_execution_between_coarse_samples(tmp_path):
    attitude, orbit, slew = _sidecars()
    path = Plan(attitude_timeseries=attitude, orbit_state_timeseries=orbit).save(
        tmp_path / "plan.json"
    )
    trajectory = ExecutionTrajectory.load(path, max_attitude_gap_s=2.5)
    for time in START + np.linspace(0, 60, 121):
        assert same_rotation(
            trajectory.state_at(time).quaternion, attitude_to_quat(*slew.attitude(time))
        )
    assert slew.slewtime == 49
    samples = list(trajectory.samples(2.5))
    assert max(np.diff([s.utime for s in samples])) <= 2.5 + 1e-6
    assert set(attitude.resolved_intervals[0].motion.breakpoints) <= {
        s.utime for s in samples
    }


def test_sparse_slew_impulse_converges_to_fine_execution_reference():
    attitude, orbit, slew = _sidecars()
    trajectory = _trajectory(attitude, orbit)
    inertia = np.diag((1000.0, 800.0, 600.0))
    reference = StoredMomentumTracker(inertia)
    for elapsed in np.linspace(0, 60, 2401):
        expected = reference.update(
            utime=START + elapsed,
            position_eci_km=_orbit(elapsed)[0],
            attitude_quaternion_eci_to_body=attitude_to_quat(
                *slew.attitude(START + elapsed)
            ),
        )
    errors = []
    for step in (2.5, 1.0, 0.25):
        tracker = StoredMomentumTracker(inertia, max_sample_interval_s=step)
        for state in trajectory.samples(step):
            actual = tracker.update(
                utime=state.utime,
                position_eci_km=state.position_km,
                attitude_quaternion_eci_to_body=state.quaternion,
            )
        errors.append(
            abs(actual.stored_momentum_norm_n_m_s - expected.stored_momentum_norm_n_m_s)
        )
    assert expected.stored_momentum_norm_n_m_s == pytest.approx(0.0108199, rel=1e-4)
    assert errors[-1] < errors[0]
    assert errors[-1] / expected.stored_momentum_norm_n_m_s < 1e-4


@pytest.mark.parametrize(
    "field,value",
    [
        ("plan_file", "wrong.json"),
        ("plan_version", 3),
        ("plan_start", 100),
        ("frame", "ITRS"),
        ("version", 99),
        ("order", "xyzw"),
        ("direction", "body_to_inertial"),
    ],
)
def test_reader_rejects_wrong_provenance_and_conventions(tmp_path, field, value):
    attitude, orbit, _ = _sidecars()
    path = Plan(attitude_timeseries=attitude, orbit_state_timeseries=orbit).save(
        tmp_path / "plan.json"
    )
    sidecar_path = tmp_path / "plan_attitude_timeseries.json"
    payload = json.loads(sidecar_path.read_text())
    payload[field] = value
    sidecar_path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        ExecutionTrajectory.load(path, max_attitude_gap_s=2.5)


@pytest.mark.parametrize(
    "link", [None, "../elsewhere.json", "/tmp/elsewhere.json", "plan.json"]
)
def test_reader_rejects_missing_or_escaping_sidecars(tmp_path, link):
    path = tmp_path / "plan.json"
    path.write_text(Plan(attitude_timeseries_file=link).model_dump_json())
    with pytest.raises(ValueError, match="sibling"):
        ExecutionTrajectory.load(path, max_attitude_gap_s=2.5)


def test_reader_rejects_symlink_outside_plan_directory(tmp_path):
    outside = tmp_path / "outside.json"
    outside.write_text("{}")
    directory = tmp_path / "export"
    directory.mkdir()
    (directory / "attitude.json").symlink_to(outside)
    path = directory / "plan.json"
    path.write_text(Plan(attitude_timeseries_file="attitude.json").model_dump_json())
    with pytest.raises(ValueError, match="sibling"):
        ExecutionTrajectory.load(path, max_attitude_gap_s=2.5)


@pytest.mark.parametrize(
    "change,match",
    [
        (lambda a, o: setattr(a.samples[1], "utime", START), "strictly increasing"),
        (
            lambda a, o: setattr(o.samples[1], "utime", float("nan")),
            "strictly increasing",
        ),
        (lambda a, o: setattr(a.samples[0], "quat_z", None), "complete wxyz"),
        (lambda a, o: setattr(a.samples[0], "quat_w", 2.0), "unit quaternion"),
        (
            lambda a, o: setattr(o.samples[0], "position_km", (0, 0, 0)),
            "nonzero position",
        ),
        (
            lambda a, o: setattr(o.samples[0], "velocity_km_s", (float("nan"), 0, 0)),
            "finite r/v",
        ),
        (
            lambda a, o: setattr(a.samples[0], "timestamp", "2026-01-01T00:00:00"),
            "timezone",
        ),
        (lambda a, o: setattr(a, "version", 0), "require sidecar version 1"),
    ],
)
def test_rejects_invalid_samples(change, match):
    attitude, orbit, _ = _sidecars()
    change(attitude, orbit)
    with pytest.raises(ValueError, match=match):
        _trajectory(attitude, orbit)


def test_rejects_orbit_gap_independently_of_exact_attitude():
    attitude, orbit, _ = _sidecars()
    with pytest.raises(ValueError, match="orbit input gap"):
        ExecutionTrajectory(attitude, orbit, max_attitude_gap_s=2.5, max_orbit_gap_s=10)


def test_partial_profile_does_not_cover_unresolved_tail():
    attitude, orbit, _ = _sidecars()
    attitude.resolved_intervals[0] = attitude.resolved_intervals[0].model_copy(
        update={"end_utime": START + 49}
    )
    with pytest.raises(ValueError, match="unresolved attitude input gap"):
        _trajectory(attitude, orbit)


def test_disagreement_and_overlapping_profiles_rejected():
    attitude, orbit, _ = _sidecars()
    attitude.resolved_intervals *= 2
    with pytest.raises(ValueError, match="nonoverlapping"):
        _trajectory(attitude, orbit)
    attitude.resolved_intervals.pop()
    attitude.samples[1].quat_w, attitude.samples[1].quat_x = 1.0, 0.0
    attitude.samples[1].quat_y, attitude.samples[1].quat_z = 0.0, 0.0
    with pytest.raises(ValueError, match="disagrees"):
        _trajectory(attitude, orbit)


def test_coverage_no_extrapolation_and_invalid_step():
    attitude, orbit, _ = _sidecars()
    trajectory = _trajectory(attitude, orbit)
    for time in (START - 1, START + 60.001, float("nan")):
        with pytest.raises(ValueError, match="coverage"):
            trajectory.state_at(time)
    for step in (0, -1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="step"):
            list(trajectory.samples(step))
    with pytest.raises(ValueError, match="precede"):
        list(trajectory.samples(1, start_utime=START + 1, end_utime=START))


def test_hermite_position_and_velocity_reproduce_cubic():
    attitude, orbit, _ = _sidecars()
    for sample in orbit.samples:
        t = sample.utime - START
        sample.position_km = (7000 + 0.01 * t**3, 2 * t**2, 3 * t)
        sample.velocity_km_s = (0.03 * t**2, 4 * t, 3)
    trajectory = _trajectory(attitude, orbit)
    for t in (0, 1, 23, 59.1, 60):
        state = trajectory.state_at(START + t)
        actual_t = state.utime - START
        assert state.position_km == pytest.approx(
            (7000 + 0.01 * actual_t**3, 2 * actual_t**2, 3 * actual_t)
        )
        assert state.velocity_km_s == pytest.approx(
            (0.03 * actual_t**2, 4 * actual_t, 3)
        )


def test_common_coverage_excludes_uncovered_attitude_head_and_tail():
    attitude, orbit, _ = _sidecars()
    for sample, elapsed in zip(orbit.samples, (10, 50)):
        sample.utime = START + elapsed
        sample.timestamp = datetime.fromtimestamp(
            sample.utime, timezone.utc
        ).isoformat()
        sample.position_km, sample.velocity_km_s = _orbit(elapsed)
    trajectory = _trajectory(attitude, orbit)
    assert (trajectory.start_utime, trajectory.end_utime) == (START + 10, START + 50)
    with pytest.raises(ValueError, match="coverage"):
        list(trajectory.samples(1, end_utime=START + 60))
    for sample in orbit.samples:
        sample.utime += 100
        sample.timestamp = datetime.fromtimestamp(
            sample.utime, timezone.utc
        ).isoformat()
    with pytest.raises(ValueError, match="no common"):
        _trajectory(attitude, orbit)


def test_adapter_detaches_from_mutable_input_schemas():
    attitude, orbit, _ = _sidecars()
    trajectory = _trajectory(attitude, orbit)
    expected = trajectory.state_at(START + 30)
    attitude.samples.clear()
    attitude.resolved_intervals.clear()
    orbit.samples[0].position_km = (1, 2, 3)
    assert trajectory.state_at(START + 30) == expected
