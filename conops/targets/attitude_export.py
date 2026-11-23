"""Executed attitude sampling and explicitly defined engineering-file exports."""

from __future__ import annotations

import csv
import io
import json
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import erfa  # type: ignore[import-untyped]
import numpy as np
import numpy.typing as npt
import rust_ephem

from ..common import unixtime_from
from ..common.ephemeris import position_vectors
from ..common.vector import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from .plan import AttitudeSampleSchema, AttitudeTimeseriesSchema

if TYPE_CHECKING:
    from ..config.frame import SpacecraftFrame
    from ..simulation.attitude import AttitudeExecutor

ReferenceFrame = Literal["ICRF", "J2000"]
_SINGULAR_COSINE = 1e-10
_MONTHS = (
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "May",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Oct",
    "Nov",
    "Dec",
)


def sample_executed_attitude(
    executor: AttitudeExecutor,
    ephem: rust_ephem.Ephemeris,
    frame: SpacecraftFrame,
    *,
    begin: float | int | str | datetime | None = None,
    end: float | int | str | datetime | None = None,
    interval_seconds: float = 1.0,
    include_transitions: bool = True,
    origin_offset_m: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> AttitudeTimeseriesSchema:
    """Recall actual flown motion while the run's executor remains available.

    Retain the returned series (or save it with a plan) before discarding the
    run. A loaded coarse sidecar cannot recover finer executed motion.
    """
    first, last = executor.flown_interval
    begin = first if begin is None else unixtime_from(begin)
    end = last if end is None else unixtime_from(end)
    if not all(np.isfinite(x) for x in (begin, end, interval_seconds)):
        raise ValueError("Export times and interval must be finite")
    if interval_seconds <= 0 or begin > end or begin < first or end > last:
        raise ValueError(
            "Export must lie within the flown interval with a positive step"
        )
    if not all(np.isfinite(value) for value in origin_offset_m):
        raise ValueError("Origin offset must contain finite coordinates")
    count = int(np.floor((end - begin) / interval_seconds))
    times = {begin + index * interval_seconds for index in range(count + 1)}
    times.add(end)
    if include_transitions:
        times.update(t for t in executor.flown_boundaries() if begin <= t <= end)
    grid = np.asarray([t.timestamp() for t in ephem.timestamp], dtype=float)
    if begin < grid[0] or end > grid[-1]:
        raise ValueError("Export lies outside the ephemeris coverage")
    ordered = sorted(times)
    # The Sun's direction at every sample, interpolated once from the
    # ephemeris points that bracket the export. Doing this for each sample
    # over the whole ephemeris made a sample cost more the longer the
    # ephemeris was, whatever the length exported.
    first_point = max(0, int(np.searchsorted(grid, begin, side="right")) - 1)
    last_point = min(len(grid), int(np.searchsorted(grid, end, side="left")) + 1)
    window = grid[first_point:last_point]
    separation = (
        np.asarray(position_vectors(ephem, "sun"), dtype=float)[first_point:last_point]
        - np.asarray(position_vectors(ephem, "gcrs"), dtype=float)[
            first_point:last_point
        ]
    )
    directions = np.column_stack(
        [np.interp(ordered, window, separation[:, axis]) for axis in range(3)]
    )
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    samples = []
    previous: npt.NDArray[np.float64] | None = None
    for utime, direction in zip(ordered, directions, strict=True):
        state = executor.flown(utime)
        q = np.asarray(state.quaternion, dtype=float)
        q /= np.linalg.norm(q)
        if previous is not None and np.dot(previous, q) < 0:
            q = -q
        previous = q
        body_sun = quaternion_to_rotation_matrix(q) @ direction
        timestamp = datetime.fromtimestamp(utime, tz=timezone.utc)
        ra, dec, roll = state.attitude
        dut1 = rust_ephem.get_ut1_utc_offset(timestamp)
        samples.append(
            AttitudeSampleSchema(
                utime=utime,
                timestamp=timestamp.isoformat(),
                ra=ra,
                dec=dec,
                roll=roll,
                quat_w=float(q[0]),
                quat_x=float(q[1]),
                quat_y=float(q[2]),
                quat_z=float(q[3]),
                sun_body_vector=(
                    float(body_sun[0]),
                    float(body_sun[1]),
                    float(body_sun[2]),
                ),
                ut1_utc_seconds=dut1,
            )
        )
    return AttitudeTimeseriesSchema(
        samples=samples,
        spacecraft_frame=frame.model_copy(deep=True),
        sample_source="executed_trajectory",
        interval_seconds=interval_seconds,
        origin_offset_m=origin_offset_m,
        plan_start=begin,
        plan_end=end,
    )


def j2000_from_gcrs(utime: float, dut1: float) -> npt.NDArray[np.float64]:
    """Relate ICRF/GCRS axes to the FK5/IAU76 J2000 realization via Earth fixed.

    Match IAU2006/2000A GCRS-to-fixed and IAU76/80 J2000-to-fixed rotations,
    including the IERS equation-of-equinoxes correction. Common polar motion
    cancels. UTC/TT/UT1 are kept distinct; no catalog aberration is applied to
    attitude axes. The explicit DUT1 is saved with newly sampled series.
    """
    return np.asarray(_j2000_from_gcrs_many([utime], [dut1])[0], dtype=float)


def _j2000_from_gcrs_many(
    utimes: Sequence[float], dut1s: Sequence[float]
) -> npt.NDArray[np.float64]:
    """:func:`j2000_from_gcrs` for many times at once, as an ``(N, 3, 3)``
    array: one call of each ERFA routine for the whole series, not one for
    each sample."""
    times = np.asarray(utimes, dtype=float)
    dut1 = np.asarray(dut1s, dtype=float)
    if not np.all(np.isfinite(times)) or not np.all(np.isfinite(dut1)):
        raise ValueError("Frame conversion requires finite UTC and UT1-UTC")
    stamps = [datetime.fromtimestamp(float(t), tz=timezone.utc) for t in times]
    utc1, utc2 = erfa.dtf2d(
        "UTC",
        [dt.year for dt in stamps],
        [dt.month for dt in stamps],
        [dt.day for dt in stamps],
        [dt.hour for dt in stamps],
        [dt.minute for dt in stamps],
        [dt.second + dt.microsecond / 1e6 for dt in stamps],
    )
    tt1, tt2 = erfa.taitt(*erfa.utctai(utc1, utc2))
    ut11, ut12 = erfa.utcut1(utc1, utc2, dut1)
    identity = np.eye(3)
    # gst06a builds the same precession-nutation matrix itself, which is most
    # of the cost of a sample; gst06 takes the one already made.
    precession_nutation = erfa.pnm06a(tt1, tt2)
    modern = erfa.c2teqx(
        precession_nutation,
        erfa.gst06(ut11, ut12, tt1, tt2, precession_nutation),
        identity,
    )
    classic = erfa.c2teqx(
        erfa.nutm80(tt1, tt2) @ erfa.pmat76(tt1, tt2),
        erfa.gmst82(ut11, ut12) + erfa.eqeq94(tt1, tt2),
        identity,
    )
    return np.asarray(np.swapaxes(classic, -1, -2) @ modern, dtype=float)


def euler123(matrix: npt.ArrayLike) -> tuple[float, float, float, bool]:
    """Intrinsic physical X, then Y', then Z'' rotations, in degrees.

    ``matrix`` maps reference coordinates into body coordinates. At gimbal
    lock set the final rotation to zero and retain the equivalent orientation.
    """
    m = np.asarray(matrix, dtype=float).T
    cosine = float(np.hypot(m[0, 0], m[0, 1]))
    beta = float(np.arctan2(m[0, 2], cosine))
    singular = cosine < _SINGULAR_COSINE
    if singular:
        alpha, gamma = float(np.arctan2(m[2, 1], m[1, 1])), 0.0
    else:
        alpha = float(np.arctan2(-m[1, 2], m[2, 2]))
        gamma = float(np.arctan2(-m[0, 1], m[0, 0]))
    return (
        float(np.degrees(alpha)),
        float(np.degrees(beta)),
        float(np.degrees(gamma)),
        singular,
    )


def _rotations(
    series: AttitudeTimeseriesSchema, reference_frame: ReferenceFrame
) -> list[tuple[AttitudeSampleSchema, npt.NDArray[np.float64]]]:
    if reference_frame not in ("ICRF", "J2000"):
        raise ValueError("Reference frame must be ICRF or J2000")
    if not series.samples:
        raise ValueError("Cannot export an empty attitude series")
    if not all(np.isfinite(value) for value in series.origin_offset_m):
        raise ValueError("Origin offset must contain finite coordinates")
    matrices = []
    previous = -float("inf")
    for sample in series.samples:
        if not np.isfinite(sample.utime) or sample.utime <= previous:
            raise ValueError(
                "Attitude sample times must be finite and strictly increasing"
            )
        previous = sample.utime
        q = (sample.quat_w, sample.quat_x, sample.quat_y, sample.quat_z)
        if any(value is None for value in q):
            raise ValueError("Attitude export requires every quaternion component")
        matrices.append(quaternion_to_rotation_matrix(np.asarray(q, dtype=float)))
    if reference_frame == "J2000":
        offsets = [sample.ut1_utc_seconds for sample in series.samples]
        if any(offset is None for offset in offsets):
            raise ValueError(
                "J2000 export requires saved UT1-UTC; sample the executed trajectory first"
            )
        # One rotation for each sample, from one call of each ERFA routine.
        to_j2000 = _j2000_from_gcrs_many(
            [sample.utime for sample in series.samples],
            [float(offset) for offset in offsets if offset is not None],
        )
        matrices = [
            matrix @ rotation.T
            for matrix, rotation in zip(matrices, to_j2000, strict=True)
        ]
    return list(zip(series.samples, matrices, strict=True))


def _metadata(
    series: AttitudeTimeseriesSchema,
    path: Path,
    reference_frame: ReferenceFrame,
    representation: str,
) -> None:
    metadata = {
        "reference_frame": reference_frame,
        "direction": "reference_to_body",
        "representation": representation,
        "spacecraft_frame": series.spacecraft_frame.model_dump(),
        "sample_source": series.sample_source,
        "interval_seconds": series.interval_seconds,
        "origin_offset_m": series.origin_offset_m,
        "origin_offset_applied_to_rotation": False,
        "begin": series.samples[0].utime,
        "end": series.samples[-1].utime,
        "frame_conversion": "identity axes from GCRS"
        if reference_frame == "ICRF"
        else "IAU2006/2000A and FK5/IAU76/80 Earth-fixed realization, saved DUT1",
    }
    path.with_suffix(path.suffix + ".metadata.json").write_text(
        json.dumps(metadata, indent=2, allow_nan=False) + "\n"
    )


def save_stk(
    series: AttitudeTimeseriesSchema,
    path: str | Path,
    *,
    reference_frame: ReferenceFrame = "J2000",
) -> Path:
    """Write STK's scalar-first Shuster quaternions (not Hamilton wxyz)."""
    rotations = _rotations(series, reference_frame)
    epoch = series.samples[0].utime
    dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    epoch_text = (
        f"{dt.day} {_MONTHS[dt.month - 1]} {dt.year} {dt:%H:%M:%S}.{dt.microsecond:06d}"
    )
    lines = [
        "stk.v.12.0",
        "BEGIN Attitude",
        f"NumberOfAttitudePoints {len(rotations)}",
        f"ScenarioEpoch {epoch_text}",
        "CentralBody Earth",
        f"CoordinateAxes {reference_frame}",
        "InterpolationOrder 1",
        "AttitudeTimeQuatScalarFirst",
    ]
    previous = None
    for sample, matrix in rotations:
        q = rotation_matrix_to_quaternion(matrix)
        # STK uses the Shuster DCM convention: conjugate the Hamilton vector.
        q[1:] *= -1
        if previous is not None and np.dot(previous, q) < 0:
            q *= -1
        previous = q
        lines.append(
            " ".join(f"{float(value):.17g}" for value in (sample.utime - epoch, *q))
        )
    lines.extend(["END Attitude", ""])
    dest = Path(path)
    dest.write_text("\n".join(lines))
    _metadata(series, dest, reference_frame, "STK Shuster scalar-first wxyz")
    return dest


def save_euler_csv(
    series: AttitudeTimeseriesSchema,
    path: str | Path,
    *,
    reference_frame: ReferenceFrame = "J2000",
) -> Path:
    """Write UTC and intrinsic Euler 1-2-3 angles for thermal-analysis import."""
    rotations = _rotations(series, reference_frame)
    dest = Path(path)
    with io.StringIO(newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "timestamp_utc",
                "elapsed_seconds",
                "euler_1_deg",
                "euler_2_deg",
                "euler_3_deg",
                "gimbal_lock",
                "sun_body_x_angle_deg",
                "sun_body_z_angle_deg",
            ]
        )
        for sample, matrix in rotations:
            angles = euler123(matrix)
            sun_angles: list[float | str] = ["", ""]
            if sample.sun_body_vector is not None:
                sun = np.asarray(sample.sun_body_vector, dtype=float)
                if not np.all(np.isfinite(sun)) or np.linalg.norm(sun) == 0:
                    raise ValueError("Sun body vector must be finite and nonzero")
                sun /= np.linalg.norm(sun)
                sun_angles = [
                    float(np.degrees(np.arccos(np.clip(sun[k], -1, 1)))) for k in (0, 2)
                ]
            writer.writerow(
                [
                    datetime.fromtimestamp(sample.utime, tz=timezone.utc).isoformat(),
                    sample.utime - series.samples[0].utime,
                    *angles,
                    *sun_angles,
                ]
            )
        payload = stream.getvalue()
    dest.write_text(payload)
    _metadata(
        series,
        dest,
        reference_frame,
        "intrinsic Euler 123 degrees; R_reference_to_body=(Rx(a)Ry(b)Rz(c)).T",
    )
    return dest
