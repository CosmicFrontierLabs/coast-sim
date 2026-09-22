"""Read-only prescribed trajectory adapter for external analysis.

This consumes the existing plan sidecars; it does not load a mission, propagate
an orbit, or call the scheduler/controller. Interpolation is an approximation
except within exported resolved attitude intervals.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt

from ..common.vector import quat_slerp
from ..targets.plan import (
    AttitudeSampleSchema,
    AttitudeTimeseriesSchema,
    OrbitStateSampleSchema,
    OrbitStateTimeseriesSchema,
    Plan,
)
from .attitude_profile import (
    ExecutedAttitudeInterval,
    Quaternion,
    same_rotation,
    unit_quaternion,
)


@dataclass(frozen=True)
class TrajectorySample:
    """GCRS Earth-centered km, km/s; scalar-first GCRS-to-body quaternion."""

    utime: float
    position_km: tuple[float, float, float]
    velocity_km_s: tuple[float, float, float]
    quaternion: Quaternion


def _sample_times(
    samples: Sequence[AttitudeSampleSchema | OrbitStateSampleSchema],
    name: str,
) -> npt.NDArray[np.float64]:
    times = np.asarray([sample.utime for sample in samples], dtype=float)
    if len(times) < 2 or not np.isfinite(times).all() or not (np.diff(times) > 0).all():
        raise ValueError(
            f"{name} needs at least two finite, strictly increasing timestamps"
        )
    for sample in samples:
        timestamp = datetime.fromisoformat(sample.timestamp)
        if timestamp.tzinfo is None or abs(timestamp.timestamp() - sample.utime) > 1e-6:
            raise ValueError(f"{name} timestamp disagrees with utime or lacks timezone")
    return times


def _bracket(times: npt.NDArray[np.float64], utime: float) -> int:
    return min(
        len(times) - 2, max(0, int(np.searchsorted(times, utime, side="right")) - 1)
    )


class ExecutionTrajectory:
    """Validated adapter for exported samples and optional resolved motion.

    ``max_attitude_gap_s`` bounds *unresolved input gaps*, not output sampling.
    Choose at most min(10 s, 5 deg / fastest bus rate) for momentum analysis and
    demonstrate convergence. Orbit interpolation is cubic Hermite using r/v;
    its separate input-gap limit defaults to 60 s and also needs convergence
    checks. Neither gap limit is a guarantee against all possible aliasing.
    """

    @classmethod
    def load(
        cls,
        plan_path: str | Path,
        *,
        max_attitude_gap_s: float,
        max_orbit_gap_s: float = 60.0,
    ) -> "ExecutionTrajectory":
        path = Path(plan_path).resolve()
        plan = Plan.load(path)
        sidecars = []
        for link, schema in (
            (plan.attitude_timeseries_file, AttitudeTimeseriesSchema),
            (plan.orbit_state_timeseries_file, OrbitStateTimeseriesSchema),
        ):
            if not link or Path(link).name != link:
                raise ValueError("plan must link sibling attitude and orbit sidecars")
            sidecar_path = (path.parent / link).resolve()
            if sidecar_path.parent != path.parent or sidecar_path == path:
                raise ValueError("sidecar must resolve to a sibling of the plan")
            sidecar = schema.model_validate_json(
                sidecar_path.read_text(encoding="utf-8")
            )
            if (
                sidecar.plan_file != path.name
                or sidecar.plan_version != plan.version
                or sidecar.plan_start != plan._start_ts
                or sidecar.plan_end != plan._end_ts
            ):
                raise ValueError(f"sidecar provenance does not match plan: {link}")
            sidecars.append(sidecar)
        attitude, orbit = sidecars
        assert isinstance(attitude, AttitudeTimeseriesSchema)
        assert isinstance(orbit, OrbitStateTimeseriesSchema)
        return cls(
            attitude,
            orbit,
            max_attitude_gap_s=max_attitude_gap_s,
            max_orbit_gap_s=max_orbit_gap_s,
        )

    def __init__(
        self,
        attitude: AttitudeTimeseriesSchema,
        orbit: OrbitStateTimeseriesSchema,
        *,
        max_attitude_gap_s: float,
        max_orbit_gap_s: float = 60.0,
    ) -> None:
        # Validate a detached snapshot even if the caller mutated a schema in memory.
        attitude = AttitudeTimeseriesSchema.model_validate(attitude.model_dump())
        orbit = OrbitStateTimeseriesSchema.model_validate(orbit.model_dump())
        for name, limit in (
            ("attitude", max_attitude_gap_s),
            ("orbit", max_orbit_gap_s),
        ):
            if not np.isfinite(limit) or limit <= 0.0:
                raise ValueError(f"{name} input-gap limit must be finite and positive")
        if attitude.version not in (0, 1) or orbit.version != 0:
            raise ValueError("unsupported execution sidecar version")
        if attitude.version == 0 and attitude.resolved_intervals:
            raise ValueError("resolved attitude intervals require sidecar version 1")
        attitude_times = _sample_times(attitude.samples, "attitude")
        self._orbit_times = _sample_times(orbit.samples, "orbit")
        if np.any(np.diff(self._orbit_times) > max_orbit_gap_s + 1e-6):
            raise ValueError(
                "orbit input gap exceeds max_orbit_gap_s; export finer ephemeris"
            )
        self._positions = np.asarray(
            [s.position_km for s in orbit.samples], dtype=float
        )
        self._velocities = np.asarray(
            [s.velocity_km_s for s in orbit.samples], dtype=float
        )
        if (
            not np.isfinite(self._positions).all()
            or not np.isfinite(self._velocities).all()
            or np.any(np.linalg.norm(self._positions, axis=1) == 0.0)
        ):
            raise ValueError("orbit needs finite r/v and nonzero position")
        knots: dict[float, npt.NDArray[np.float64]] = {}
        for sample in attitude.samples:
            values = (sample.quat_w, sample.quat_x, sample.quat_y, sample.quat_z)
            w, x, y, z = values
            if w is None or x is None or y is None or z is None:
                raise ValueError("attitude samples require complete wxyz quaternions")
            knots[sample.utime] = np.asarray(unit_quaternion((w, x, y, z)), dtype=float)

        intervals = tuple(attitude.resolved_intervals)
        for index, interval in enumerate(intervals):
            within_samples = (
                attitude_times[0]
                <= interval.start_utime
                < interval.end_utime
                <= attitude_times[-1]
            )
            overlaps_previous = (
                index > 0 and intervals[index - 1].end_utime > interval.start_utime
            )
            if not within_samples or overlaps_previous:
                raise ValueError(
                    "resolved intervals must be ordered, nonoverlapping and within samples"
                )
            for time in attitude_times[
                np.searchsorted(attitude_times, interval.start_utime) : np.searchsorted(
                    attitude_times, interval.end_utime, side="right"
                )
            ]:
                if not same_rotation(knots[time], interval.motion.quaternion_at(time)):
                    raise ValueError(
                        "resolved motion disagrees with an executed attitude sample"
                    )
            for time in (interval.start_utime, interval.end_utime):
                quaternion = interval.motion.quaternion_at(time)
                if time in knots and not same_rotation(knots[time], quaternion):
                    raise ValueError(
                        "resolved motion is discontinuous at an interval boundary"
                    )
                knots[time] = quaternion
        self._attitude_times = np.asarray(sorted(knots), dtype=float)
        self._quaternions = np.asarray([knots[t] for t in self._attitude_times])
        self._motion: list[ExecutedAttitudeInterval | None] = []
        interval_index = 0
        for start, end in zip(self._attitude_times, self._attitude_times[1:]):
            while (
                interval_index < len(intervals)
                and intervals[interval_index].end_utime <= start
            ):
                interval_index += 1
            covering = (
                intervals[interval_index] if interval_index < len(intervals) else None
            )
            if (
                covering is not None
                and covering.start_utime <= start
                and end <= covering.end_utime
            ):
                self._motion.append(covering)
            else:
                if end - start > max_attitude_gap_s + 1e-6:
                    raise ValueError(
                        f"unresolved attitude input gap {start:g}..{end:g} "
                        f"({end - start:g} s) exceeds {max_attitude_gap_s:g} s; "
                        "export finer attitudes or resolved motion, not denser interpolation"
                    )
                self._motion.append(None)
        self.start_utime = float(max(self._attitude_times[0], self._orbit_times[0]))
        self.end_utime = float(min(self._attitude_times[-1], self._orbit_times[-1]))
        if self.end_utime <= self.start_utime:
            raise ValueError("attitude and orbit have no common time interval")
        breakpoints = set(self._attitude_times) | set(self._orbit_times)
        for interval in intervals:
            breakpoints.update(
                t
                for t in interval.motion.breakpoints
                if interval.start_utime <= t <= interval.end_utime
            )
        self._breakpoints = tuple(
            sorted(
                float(t) for t in breakpoints if self.start_utime <= t <= self.end_utime
            )
        )

    def state_at(self, utime: float) -> TrajectorySample:
        """Evaluate without extrapolation or any scheduler/controller side effects."""
        if not np.isfinite(utime) or not self.start_utime <= utime <= self.end_utime:
            raise ValueError(
                f"time must lie in common coverage [{self.start_utime}, {self.end_utime}]"
            )
        index = _bracket(self._attitude_times, utime)
        interval = self._motion[index]
        if interval is not None:
            quaternion = interval.motion.quaternion_at(utime)
        else:
            start, end = self._attitude_times[index : index + 2]
            quaternion = quat_slerp(
                self._quaternions[index],
                self._quaternions[index + 1],
                float((utime - start) / (end - start)),
            )
        index = _bracket(self._orbit_times, utime)
        start, end = self._orbit_times[index : index + 2]
        dt = end - start
        u = (utime - start) / dt
        r0, r1 = self._positions[index : index + 2]
        v0, v1 = self._velocities[index : index + 2]
        position = (
            (2 * u**3 - 3 * u**2 + 1) * r0
            + (u**3 - 2 * u**2 + u) * dt * v0
            + (-2 * u**3 + 3 * u**2) * r1
            + (u**3 - u**2) * dt * v1
        )
        velocity = (
            (6 * u**2 - 6 * u) * r0 / dt
            + (3 * u**2 - 4 * u + 1) * v0
            + (-6 * u**2 + 6 * u) * r1 / dt
            + (3 * u**2 - 2 * u) * v1
        )
        return TrajectorySample(
            utime=float(utime),
            position_km=tuple(position),
            velocity_km_s=tuple(velocity),
            quaternion=tuple(quaternion),
        )

    def samples(
        self,
        step_s: float,
        *,
        start_utime: float | None = None,
        end_utime: float | None = None,
    ) -> Iterator[TrajectorySample]:
        """Quadrature samples with bounded gaps, including source/phase boundaries.

        Defaults to actual common sample coverage, NOT the plan's declared end.
        Thin the *analysis results* separately when choosing plotting cadence.
        """
        if not np.isfinite(step_s) or step_s <= 0.0:
            raise ValueError("analysis step must be finite and positive")
        start = self.start_utime if start_utime is None else start_utime
        end = self.end_utime if end_utime is None else end_utime
        self.state_at(start)
        self.state_at(end)
        if end < start:
            raise ValueError("end must not precede start")
        if start + step_s == start:
            raise ValueError("analysis step is smaller than timestamp resolution")
        boundaries = sorted(
            {start, end, *(t for t in self._breakpoints if start < t < end)}
        )
        yield self.state_at(start)
        for left, right in zip(boundaries, boundaries[1:]):
            count = max(1, int(np.ceil((right - left) / step_s)))
            for time in np.linspace(left, right, count + 1)[1:]:
                yield self.state_at(float(time))
