"""Engineering attitude exports preserve executed motion and frame conventions."""

import csv
import json
from datetime import datetime, timedelta, timezone

import erfa
import numpy as np
import pytest
import rust_ephem

from conops.common.vector import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from conops.config import AttitudeControlSystem, SpacecraftFrame
from conops.simulation.attitude import AttitudeExecutor, AttitudeTrajectory
from conops.targets import AttitudeSampleSchema, AttitudeTimeseriesSchema, Plan
from conops.targets.attitude_export import (
    euler123,
    j2000_from_gcrs,
    sample_executed_attitude,
)

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
T0 = BEGIN.timestamp()


def matrix123(angles):
    a, b, c = np.radians(angles)
    rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
    return (rx @ ry @ rz).T


def series(matrices):
    samples = []
    for k, m in enumerate(matrices):
        w, x, y, z = rotation_matrix_to_quaternion(m)
        samples.append(
            AttitudeSampleSchema(
                utime=T0 + k,
                timestamp=datetime.fromtimestamp(T0 + k, tz=timezone.utc).isoformat(),
                quat_w=w,
                quat_x=x,
                quat_y=y,
                quat_z=z,
                ut1_utc_seconds=0.1,
                sun_body_vector=(1, 0, 0),
            )
        )
    return AttitudeTimeseriesSchema(samples=samples)


def read_stk(path):
    text = path.read_text()
    assert text.startswith("stk.v.12.0\nBEGIN Attitude\n")
    assert text.rstrip().endswith("END Attitude")
    data = text.split("AttitudeTimeQuatScalarFirst\n")[1].split("END Attitude")[0]
    values = np.asarray(
        [[float(v) for v in line.split()] for line in data.splitlines() if line.strip()]
    )
    assert f"NumberOfAttitudePoints {len(values)}\n" in text
    return values


def stk_matrix(q):
    # Independent Shuster DCM equations from STK's orientation documentation.
    w, x, y, z = q
    return np.array(
        [
            [x * x - y * y - z * z + w * w, 2 * (x * y + z * w), 2 * (x * z - y * w)],
            [2 * (x * y - z * w), -x * x + y * y - z * z + w * w, 2 * (y * z + x * w)],
            [2 * (x * z + y * w), 2 * (y * z - x * w), -x * x - y * y + z * z + w * w],
        ]
    )


@pytest.mark.parametrize(
    "angles",
    [
        (0, 0, 0),
        (90, 0, 0),
        (0, 90, 0),
        (0, -90, 0),
        (0, 0, 90),
        (20, 30, 40),
        (-120, 89.9, 179),
        (20, 90, 40),
        (20, -90, 40),
    ],
)
def test_euler123_known_rotations_and_singularities(angles):
    m = matrix123(angles)
    a, b, c, singular = euler123(m)
    np.testing.assert_allclose(matrix123((a, b, c)), m, atol=1e-12)
    assert singular == (abs(angles[1]) == 90)
    if singular:
        assert c == 0


@pytest.mark.parametrize("reference_frame", ["ICRF", "J2000"])
def test_stk_scalar_order_and_shuster_action_preserve_every_body_axis(
    tmp_path, reference_frame
):
    original = [
        matrix123(angles)
        for angles in [(0, 0, 0), (90, 0, 0), (0, 90, 0), (0, 0, 90), (20, 30, 40)]
    ]
    data = series(original)
    path = data.save_stk(tmp_path / "attitude.a", reference_frame=reference_frame)
    rows = read_stk(path)
    assert f"CoordinateAxes {reference_frame}\n" in path.read_text()
    for sample, row, expected in zip(data.samples, rows, original, strict=True):
        if reference_frame == "J2000":
            expected = (
                expected @ j2000_from_gcrs(sample.utime, sample.ut1_utc_seconds).T
            )
        np.testing.assert_allclose(stk_matrix(row[1:]), expected, atol=1e-12)
    np.testing.assert_allclose(rows[:, 0], np.arange(len(original)))
    if reference_frame == "ICRF":
        # A physical +90-degree X rotation has positive STK vector component.
        np.testing.assert_allclose(rows[1, 1:], [2**-0.5, 2**-0.5, 0, 0], atol=1e-12)
    meta = json.loads(path.with_suffix(".a.metadata.json").read_text())
    assert meta["representation"] == "STK Shuster scalar-first wxyz"
    assert meta["direction"] == "reference_to_body"


def test_quaternion_signs_stay_continuous_through_180_degrees(tmp_path):
    data = series([matrix123((angle, 0, 0)) for angle in [170, 179, 180, 181, 190]])
    rows = read_stk(data.save_stk(tmp_path / "attitude.a", reference_frame="ICRF"))
    assert np.all(np.sum(rows[1:, 1:] * rows[:-1, 1:], axis=1) > 0)


@pytest.mark.parametrize("reference_frame", ["ICRF", "J2000"])
def test_euler_csv_round_trip_units_sun_angles_and_translation_metadata(
    tmp_path, reference_frame
):
    data = series([matrix123((20, 30, 40)), matrix123((20, 90, 40))])
    data.origin_offset_m = (0, 0, 1.110)
    path = data.save_euler_csv(tmp_path / "euler.csv", reference_frame=reference_frame)
    with path.open() as stream:
        rows = list(csv.DictReader(stream))
    for sample, row in zip(data.samples, rows, strict=True):
        m = quaternion_to_rotation_matrix(
            [sample.quat_w, sample.quat_x, sample.quat_y, sample.quat_z]
        )
        if reference_frame == "J2000":
            m = m @ j2000_from_gcrs(sample.utime, sample.ut1_utc_seconds).T
        angles = [float(row[f"euler_{k}_deg"]) for k in (1, 2, 3)]
        np.testing.assert_allclose(matrix123(angles), m, atol=1e-12)
        assert float(row["sun_body_x_angle_deg"]) == 0
        assert float(row["sun_body_z_angle_deg"]) == 90
        assert row["timestamp_utc"].endswith("+00:00")
    meta = json.loads(path.with_suffix(".csv.metadata.json").read_text())
    assert meta["origin_offset_m"] == [0, 0, 1.110]
    assert meta["origin_offset_applied_to_rotation"] is False


@pytest.mark.parametrize("seconds", [0, 10000000])
@pytest.mark.parametrize("dut1", [-0.3, 0.1])
def test_j2000_conversion_matches_both_earth_fixed_realizations(seconds, dut1):
    time = T0 + seconds
    utc1, utc2 = erfa.dtf2d(
        "UTC", *datetime.fromtimestamp(time, timezone.utc).timetuple()[:6]
    )
    tt1, tt2 = erfa.taitt(*erfa.utctai(utc1, utc2))
    ut11, ut12 = erfa.utcut1(utc1, utc2, dut1)
    gcrs_to_fixed = erfa.c2t06a(tt1, tt2, ut11, ut12, 0, 0)
    a = erfa.gmst82(ut11, ut12) + erfa.eqeq94(tt1, tt2)
    spin = np.array([[np.cos(a), np.sin(a), 0], [-np.sin(a), np.cos(a), 0], [0, 0, 1]])
    j2000_to_fixed = spin @ erfa.nutm80(tt1, tt2) @ erfa.pmat76(tt1, tt2)
    result = j2000_from_gcrs(time, dut1)
    np.testing.assert_allclose(j2000_to_fixed @ result, gcrs_to_fixed, atol=1e-12)
    assert np.linalg.norm(result - np.eye(3)) > 1e-8  # Native axes are not relabeled.


@pytest.fixture(scope="module")
def ephem():
    return rust_ephem.TLEEphemeris(
        tle="examples/example.tle",
        begin=BEGIN,
        end=BEGIN + timedelta(hours=1),
        step_size=60,
    )


def limits():
    return AttitudeControlSystem(max_slew_rate=2, slew_acceleration=0.5, settle_time=3)


def test_fine_sampling_recalls_interrupted_slew_braking_and_stationary_hold(ephem):
    executor = AttitudeExecutor(T0, (10, 20, 30))
    executor.install(
        AttitudeTrajectory.turn(T0, (10, 20, 30), (170, -10, 90), limits())
    )
    executor.advance(T0 + 7.25)
    rest = executor.request_stop(T0 + 7.25, limits())
    executor.advance(rest + 10)
    data = sample_executed_attitude(
        executor, ephem, SpacecraftFrame(), interval_seconds=1
    )
    assert data.sample_source == "executed_trajectory"
    assert data.samples[0].utime == T0
    assert data.samples[-1].utime == rest + 10
    assert T0 + 7.25 in [s.utime for s in data.samples]
    assert max(np.diff([s.utime for s in data.samples])) <= 1
    for sample in data.samples:
        actual = [sample.quat_w, sample.quat_x, sample.quat_y, sample.quat_z]
        expected = executor.flown(sample.utime).quaternion
        assert abs(np.dot(actual, expected)) == pytest.approx(1, abs=1e-12)
    # Final held state must not rewrite the earlier motion as coarse interpolation.
    assert data.samples[5].ra != pytest.approx(data.samples[-1].ra)
    np.testing.assert_allclose(
        [s.quat_w for s in data.samples[-5:]], data.samples[-1].quat_w
    )


def test_tracking_export_exact_endpoints_and_saved_plan_lifecycle(ephem, tmp_path):
    executor = AttitudeExecutor(T0, (10, 0, 0), (0, 0, 0.1))
    trajectory = AttitudeTrajectory.tracking(
        [(T0, (10, 0, 0)), (T0 + 60, (16, 0, 0)), (T0 + 120, (22, 0, 0))], limits()
    )
    executor.install(trajectory)
    executor.advance(T0 + 100)
    data = sample_executed_attitude(
        executor,
        ephem,
        SpacecraftFrame(),
        begin=T0 + 20.25,
        end=T0 + 81.75,
        interval_seconds=1,
    )
    assert data.samples[0].utime == T0 + 20.25 and data.samples[-1].utime == T0 + 81.75
    assert T0 + 60 in [s.utime for s in data.samples]
    assert data.samples[0].ra == pytest.approx(12.025, abs=1e-9)
    plan = Plan(attitude_timeseries=data)
    loaded = Plan.load(plan.save(tmp_path / "plan.json"), load_timeseries=True)
    assert loaded.attitude_timeseries.samples == data.samples
    loaded.attitude_timeseries.save_stk(tmp_path / "saved.a")
    assert len(read_stk(tmp_path / "saved.a")) == len(data.samples)


def test_payload_frame_is_already_in_the_executed_quaternion(ephem):
    frame = SpacecraftFrame(boresight=(0, 0, 1), sun_side=(1, 0, 0))
    from conops.config import MissionConfig, Payload, SpacecraftBus, Telescope

    c = MissionConfig(
        spacecraft_bus=SpacecraftBus(frame=frame),
        payload=Payload(instruments=[Telescope(name="Scope")]),
    )
    attitude = c.payload.instruments[0].target_body_attitude(60, 20, 15)
    executor = AttitudeExecutor(T0, attitude)
    executor.advance(T0 + 10)
    data = sample_executed_attitude(executor, ephem, frame)
    q = executor.flown(T0).quaternion
    np.testing.assert_allclose(
        [
            data.samples[0].quat_w,
            data.samples[0].quat_x,
            data.samples[0].quat_y,
            data.samples[0].quat_z,
        ],
        q,
        atol=1e-12,
    )
    r = quaternion_to_rotation_matrix(q)
    line = np.array(
        [
            np.cos(np.radians(20)) * np.cos(np.radians(60)),
            np.cos(np.radians(20)) * np.sin(np.radians(60)),
            np.sin(np.radians(20)),
        ]
    )
    np.testing.assert_allclose(r @ line, [0, 0, 1], atol=1e-12)
    assert data.spacecraft_frame == frame


@pytest.mark.parametrize(
    "changes",
    [
        {"interval_seconds": 0},
        {"interval_seconds": float("nan")},
        {"begin": T0 - 1},
        {"end": T0 + 11},
        {"begin": T0 + 9, "end": T0 + 8},
    ],
)
def test_sampling_rejects_unflown_or_invalid_intervals(ephem, changes):
    executor = AttitudeExecutor(T0, (0, 0, 0))
    executor.advance(T0 + 10)
    with pytest.raises(ValueError):
        sample_executed_attitude(executor, ephem, SpacecraftFrame(), **changes)


def test_incomplete_series_is_rejected_without_overwriting_an_existing_file(tmp_path):
    dest = tmp_path / "existing.a"
    dest.write_text("keep")
    data = AttitudeTimeseriesSchema(
        samples=[AttitudeSampleSchema(utime=T0, timestamp=BEGIN.isoformat())]
    )
    with pytest.raises(ValueError, match="quaternion"):
        data.save_stk(dest)
    assert dest.read_text() == "keep"
    data = series([np.eye(3)])
    data.samples[0].ut1_utc_seconds = None
    with pytest.raises(ValueError, match="UT1"):
        data.save_stk(dest)
    assert dest.read_text() == "keep"


def test_invalid_sun_vector_does_not_overwrite_existing_csv(tmp_path):
    dest = tmp_path / "existing.csv"
    dest.write_text("keep")
    data = series([np.eye(3)])
    data.samples[0].sun_body_vector = (0, 0, 0)
    with pytest.raises(ValueError, match="Sun body vector"):
        data.save_euler_csv(dest)
    assert dest.read_text() == "keep"


def test_sidecar_loading_is_opt_in_and_loads_both_streams(tmp_path):
    from conops.targets import OrbitStateSampleSchema, OrbitStateTimeseriesSchema

    orbit = OrbitStateTimeseriesSchema(
        samples=[
            OrbitStateSampleSchema(
                utime=T0,
                timestamp=BEGIN.isoformat(),
                position_km=(7000, 0, 0),
                velocity_km_s=(0, 7, 0),
            )
        ]
    )
    data = series([np.eye(3)])
    path = Plan(attitude_timeseries=data, orbit_state_timeseries=orbit).save(
        tmp_path / "plan.json"
    )
    plain = Plan.load(path)
    assert plain.attitude_timeseries is None and plain.orbit_state_timeseries is None
    loaded = Plan.load(path, load_timeseries=True)
    assert loaded.attitude_timeseries.samples == data.samples
    assert loaded.orbit_state_timeseries.samples == orbit.samples
    (tmp_path / plain.attitude_timeseries_file).unlink()
    assert Plan.load(path).attitude_timeseries is None
    with pytest.raises(FileNotFoundError):
        Plan.load(path, load_timeseries=True)


def test_sidecar_link_must_name_a_sibling_file(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text(Plan(attitude_timeseries_file="../other.json").model_dump_json())
    with pytest.raises(ValueError, match="sibling"):
        Plan.load(path, load_timeseries=True)
