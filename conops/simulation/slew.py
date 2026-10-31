from bisect import bisect_right
from collections.abc import Sequence
from typing import TYPE_CHECKING, NamedTuple

import numpy as np
import numpy.typing as npt
import rust_ephem
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr, model_validator

from ..common import roll_over_angle, separation, unixtime2date
from ..common.enums import ObsType, SlewAlgorithm
from ..common.ephemeris import sun_vector_change_deg
from ..common.numerics import ATTITUDE_TOLERANCE_DEG
from ..common.vector import (
    _batch_quat_to_attitudes,
    angular_separation,
    attitude_to_quat,
    quat_slerp,
    quat_slerp_at,
    quat_to_attitude,
    quaternion_attitude_delta,
    quaternion_slew_path,
)
from ..config import AttitudeControlSystem, Constraint, MissionConfig
from ..config.acs import ThermalSettlingStep, scheduled_slew_time
from ..config.constants import DTOR
from ..config.observation_categories import ObservationCategories

if TYPE_CHECKING:
    from ..targets import PlanEntry
    from .attitude import AttitudeTrajectory


# The constraint-avoiding planner inserts at most one waypoint.
_MAX_CONSTRAINT_AVOIDING_SEGMENTS = 2


class _SlewSegment(NamedTuple):
    distance_deg: float
    axis_body: tuple[float, float, float]
    start_quat: npt.NDArray[np.float64]
    end_quat: npt.NDArray[np.float64]


class Slew(BaseModel):
    """Class defines a Spacecraft Slew. Calculates slew time and slew path.

    Supports two path algorithms selected via the ACS configuration:
    - QUATERNION (default): full SO(3) SLERP coupling pointing and roll changes.
    - CONSTRAINT_AVOIDING: generalized constraint-avoiding SLERP.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: MissionConfig | None = Field(default=None, exclude=True)
    ephem: rust_ephem.Ephemeris | None = Field(default=None, exclude=True)
    constraint: Constraint | None = Field(default=None, exclude=True)
    acs_config: AttitudeControlSystem | None = Field(default=None, exclude=True)
    slewrequest: float = 0.0  # When was the slew requested
    slewstart: float = 0.0
    slewend: float = 0.0
    startra: float = 0.0
    startdec: float = 0.0
    startroll: float = 0.0
    endra: float = 0.0
    enddec: float = 0.0
    endroll: float = 0.0
    slewtime: float = 0.0
    slewdist: float = 0.0
    sun_vector_change_deg: float = 0.0
    """How far the Sun's direction moves in the body frame between the
    start and end attitudes, in degrees; computed with a thermal settling
    table, zero without one."""
    settle_seconds: float | None = None
    """The settle the slew's time includes, from the thermal settling table
    or the ACS's ``settle_time``; None until the time is calculated."""
    slewpath: tuple[list[float], list[float]] = Field(default_factory=lambda: ([], []))
    obstype: ObsType = ObsType.PPT
    obsid: int = 0
    instrument_name: str | None = None
    instrument_roll: float | None = None
    at: "PlanEntry | None" = None  # In quotes to avoid circular import
    rate_fraction: float = Field(default=1.0, gt=0.0, le=1.0, allow_inf_nan=False)
    _momentum_trajectory: "AttitudeTrajectory | None" = PrivateAttr(default=None)
    _momentum_hold_end: float | None = PrivateAttr(default=None)
    _momentum_collection_begin: float | None = PrivateAttr(default=None)
    _momentum_collection_end: float | None = PrivateAttr(default=None)
    _motion_limits_key: tuple[object, ...] | None = PrivateAttr(default=None)
    _motion_limits_cache: AttitudeControlSystem | None = PrivateAttr(default=None)
    # Private defaults are copied per instance, so these lists are never
    # shared. A default_factory would cost far more: pydantic inspects the
    # factory's signature for every instance, and schedulers build many slews.
    # Quaternion SLERP: intermediate roll values along the path
    _quat_roll_path: list[float] = PrivateAttr(default=[])
    # Shortest maneuver axis resolved in the initial spacecraft body frame.
    _rotation_axis_body: tuple[float, float, float] | None = PrivateAttr(default=None)
    # Maneuvers are rest-to-rest SLERP segments, each with its own angular
    # distance, initial body-frame rotation axis, and exact quaternion endpoints.
    _slew_segments: list[_SlewSegment] = PrivateAttr(default=[])

    @model_validator(mode="after")
    def _derive_from_config(self) -> "Slew":
        """Populate ephem/constraint/acs_config from config, when not already set.

        Uses ``getattr`` (not ``self.config``) because pydantic re-runs this
        "after" validator on any object that merely satisfies ``isinstance``
        against ``Slew`` — e.g. a test double such as ``Mock(spec=Slew)``
        embedded in another model's ``Slew``-typed field — without going
        through real field population first, so ``config`` may not exist.
        """
        config = getattr(self, "config", None)
        if config is None:
            return self
        if self.constraint is None:
            self.constraint = config.constraint
        assert self.constraint is not None, "Constraint must be set for Slew class"
        if self.ephem is None:
            self.ephem = self.constraint.ephem
        assert self.ephem is not None, "Ephemeris must be set for Slew class"
        if self.acs_config is None:
            self.acs_config = config.spacecraft_bus.attitude_control
        assert self.acs_config is not None, "ACS config must be set for Slew class"
        return self

    def __eq__(self, other: object) -> bool:
        """Compare by identity, matching plain-object semantics (Slew instances are used as dict keys)."""
        return self is other

    @property
    def motion_limits(self) -> AttitudeControlSystem:
        """A speed cap never changes acceleration, settling, or shared config."""
        assert self.acs_config is not None
        # Slew is mutable, so construction-time field validation is insufficient.
        if not np.isfinite(self.rate_fraction) or not 0 < self.rate_fraction <= 1:
            raise ValueError("Slew rate fraction must be finite and within (0, 1]")
        if self.rate_fraction == 1:
            return self.acs_config
        # Include the source fields by value: ACS permits assignment, so a
        # cached cap must follow changes to rates, acceleration and settling.
        key = (
            self.rate_fraction,
            id(self.acs_config),
            self.acs_config.max_slew_rate,
            self.acs_config.max_slew_rate_body,
            self.acs_config.slew_acceleration,
            self.acs_config.slew_acceleration_body,
            self.acs_config.settle_time,
        )
        if key == self._motion_limits_key and self._motion_limits_cache is not None:
            return self._motion_limits_cache
        limits = self.acs_config.model_copy(
            update={
                "max_slew_rate": self.acs_config.max_slew_rate * self.rate_fraction,
                "max_slew_rate_body": tuple(
                    x * self.rate_fraction for x in self.acs_config.max_slew_rate_body
                )
                if self.acs_config.max_slew_rate_body is not None
                else None,
            }
        )
        self._motion_limits_key, self._motion_limits_cache = key, limits
        return limits

    def __hash__(self) -> int:
        """Hash by identity, matching plain-object semantics (Slew instances are used as dict keys)."""
        return id(self)

    @classmethod
    def idle_hold(
        cls,
        config: MissionConfig,
        ra: float,
        dec: float,
        roll: float,
        utime: float,
    ) -> "Slew":
        """Create a zero-duration IDLE hold at the given attitude."""
        hold = cls(
            config=config,
            slewrequest=utime,
            slewstart=utime,
            slewend=utime,
            slewtime=0.0,
            slewdist=0.0,
            startra=ra,
            startdec=dec,
            startroll=roll,
            endra=ra,
            enddec=dec,
            endroll=roll,
            obstype=ObsType.IDLE,
            obsid=0,
            at=None,
        )
        return hold

    def __str__(self) -> str:
        return f"Slew from {self.startra:.3f},{self.startdec:.3f},{self.startroll:.3f}° to {self.endra:.3f},{self.enddec:.3f},{self.endroll:.3f}° @ {unixtime2date(self.slewstart)}"

    def is_slewing(self, utime: float) -> bool:
        """For a given utime, are we slewing?"""
        return self.slewstart <= utime < self.slewend

    def quaternion_waypoints(self) -> list[tuple[float, float, float, float]]:
        """Snapshot the analytic route without a singular coordinate round trip."""
        start = (self.startra, self.startdec, self.startroll)
        end = (self.endra, self.enddec, self.endroll)
        if self._slew_segments:
            points = [
                self._slew_segments[0].start_quat,
                *(segment.end_quat for segment in self._slew_segments),
            ]
        else:
            distance, _ = quaternion_attitude_delta(*start, *end)
            if distance > ATTITUDE_TOLERANCE_DEG:
                raise ValueError(
                    "Physical execution requires an analytic slew trajectory"
                )
            points = [attitude_to_quat(*start), attitude_to_quat(*end)]
        return [(float(q[0]), float(q[1]), float(q[2]), float(q[3])) for q in points]

    def ra_dec(self, utime: float) -> tuple[float, float]:
        return self.slew_ra_dec(utime)

    def roll(self, utime: float) -> float:
        """Return roll angle at the given time during the slew."""
        return self.slew_roll(utime)

    @property
    def great_circle_deg(self) -> float:
        """The great-circle angle between the pointings the slew joins, in
        degrees: what a slew requirement is written in, where ``slewdist`` is
        the rotation flown, which the roll lengthens."""
        return angular_separation(
            float(self.startra),
            float(self.startdec),
            float(self.endra),
            float(self.enddec),
        )

    def sun_vector_change(self) -> float:
        """The angle between the Sun's directions in the body frame at the
        start and end attitudes, in degrees, with the Sun where it is at the
        slew's start."""
        assert self.ephem is not None, "Ephemeris must be set for the Sun vector"
        return sun_vector_change_deg(
            self.ephem,
            float(self.slewstart),
            (self.startra, self.startdec, self.startroll),
            (self.endra, self.enddec, self.endroll),
        )

    def _thermal_settling_table(self) -> list[ThermalSettlingStep]:
        """The thermal settling steps that apply to this slew: none without
        a table, for a test double's configuration, or for a slew into an
        observation whose category skips the thermal settle."""
        acs = self.acs_config
        table = getattr(acs, "thermal_settling", None)
        if not isinstance(table, list) or not table:
            return []
        categories = getattr(self.config, "observation_categories", None)
        if isinstance(
            categories, ObservationCategories
        ) and categories.skips_thermal_settling(int(self.obsid)):
            return []
        return table

    def attitude(self, utime: float) -> tuple[float, float, float]:
        """Return the complete executed attitude from one trajectory evaluation."""
        if self._slew_segments:
            t = utime - self.slewstart
            if t <= 0 or self.slewdist <= 0:
                return self.startra, self.startdec, self.startroll
            return self._quaternion_attitude(self._slew_fraction(t))

        # Compatibility for manually constructed sampled paths that predate
        # quaternion segment metadata.
        ra, dec = self.slew_ra_dec(utime)
        return ra, dec, self.slew_roll(utime)

    def attitudes(self, times: Sequence[float]) -> list[tuple[float, float, float]]:
        """The attitude at each of ``times``, as :meth:`attitude` gives it,
        interpolated together along a direct quaternion slew: a battery
        ledger samples a slew at every step of its path, and one evaluation
        per step costs more than the interpolation."""
        if len(self._slew_segments) != 1 or self.slewdist <= 0:
            return [self.attitude(float(utime)) for utime in times]
        segment = self._slew_segments[0]
        start = (self.startra, self.startdec, self.startroll)
        elapsed = [float(utime) - self.slewstart for utime in times]
        moving = [k for k, t in enumerate(elapsed) if t > 0]
        result: list[tuple[float, float, float]] = [start] * len(elapsed)
        if not moving:
            return result
        fractions = np.array(
            [self._slew_fraction(elapsed[k]) for k in moving], dtype=np.float64
        )
        if segment.distance_deg <= 0.0:
            fractions = np.zeros_like(fractions)
        else:
            fractions = np.clip(
                fractions * float(self.slewdist) / segment.distance_deg, 0.0, 1.0
            )
        ra, dec, roll = _batch_quat_to_attitudes(
            quat_slerp_at(segment.start_quat, segment.end_quat, fractions)
        )
        for k, index in enumerate(moving):
            result[index] = (
                float(ra[k]) % 360.0,
                float(dec[k]),
                float(roll[k]) % 360.0,
            )
        return result

    def _slew_fraction(self, t: float) -> float:
        """Return progress fraction [0,1] along the bang-bang profile at elapsed time t."""
        assert self.acs_config is not None, (
            "ACS config must be set to calculate slew fraction"
        )
        total_dist = float(self.slewdist)
        limits = self.motion_limits
        if self._slew_segments:
            remaining_time = max(0.0, float(t))
            s = 0.0
            for segment in self._slew_segments:
                segment_time = limits.motion_time(
                    segment.distance_deg, segment.axis_body
                )
                if remaining_time >= segment_time:
                    s += segment.distance_deg
                    remaining_time -= segment_time
                    continue
                s += limits.s_of_t(
                    segment.distance_deg, remaining_time, segment.axis_body
                )
                break
        else:
            s = limits.s_of_t(total_dist, t, self._rotation_axis_body)
        return 0.0 if total_dist == 0 else max(0.0, min(1.0, s / total_dist))

    @staticmethod
    def _shortest_roll_diff(start: float, end: float) -> float:
        """Return end - start adjusted to take the shortest path around the circle."""
        return float(roll_over_angle([start, end])[1] - start)

    def _quaternion_attitude(self, fraction: float) -> tuple[float, float, float]:
        """Evaluate the direct or waypoint-routed quaternion trajectory."""
        remaining_distance = fraction * float(self.slewdist)
        attitude: tuple[float, float, float] | None = None
        for index, segment in enumerate(self._slew_segments):
            is_last = index == len(self._slew_segments) - 1
            if remaining_distance <= segment.distance_deg or is_last:
                segment_fraction = (
                    0.0
                    if segment.distance_deg <= 0.0
                    else min(1.0, max(0.0, remaining_distance / segment.distance_deg))
                )
                ra, dec, roll = quat_to_attitude(
                    quat_slerp(segment.start_quat, segment.end_quat, segment_fraction)
                )
                attitude = float(ra) % 360.0, float(dec), float(roll) % 360.0
                break
            remaining_distance -= segment.distance_deg

        if attitude is None:
            raise RuntimeError("slew has no quaternion segment to evaluate")
        return attitude

    def slew_ra_dec(self, utime: float) -> tuple[float, float]:
        """Return RA/Dec at time using bang-bang slew profile when configured.

        If an AttitudeControlSystem config is present, advance along the
        pre-computed path according to bang-bang kinematics (accel → cruise → decel).
        The path may be a great-circle arc, quaternion SLERP, or sun-avoiding arc
        depending on the configured slew algorithm.
        """
        if self._slew_segments:
            ra, dec, _roll = self.attitude(utime)
            return ra, dec

        t = utime - self.slewstart
        if t <= 0:
            return self.startra, self.startdec

        if len(self.slewpath[0]) == 0 or self.slewdist <= 0:
            return self.startra, self.startdec

        f = self._slew_fraction(t)

        ra_path, dec_path = self.slewpath
        n = len(ra_path)
        if n <= 1:
            return float(ra_path[0]) % 360, float(dec_path[0])

        idx = f * (n - 1)
        x = np.arange(n, dtype=float)
        ras = roll_over_angle(ra_path)
        ra = np.interp(idx, x, ras) % 360
        dec = np.interp(idx, x, dec_path)
        return ra, dec

    def slew_roll(self, utime: float) -> float:
        """Return roll angle at time during slew, drawn from the SLERP path."""
        if self._slew_segments:
            return self.attitude(utime)[2]

        t = utime - self.slewstart
        if t <= 0:
            return self.startroll
        if self.slewtime <= 0:
            return self.startroll

        if self._quat_roll_path:
            f = self._slew_fraction(t)
            n = len(self._quat_roll_path)
            idx = f * (n - 1)
            rolls = roll_over_angle(self._quat_roll_path)
            return float(np.interp(idx, np.arange(n, dtype=float), rolls)) % 360

        # Fallback: shortest-path linear interpolation
        f = max(0.0, min(1.0, t / self.slewtime))
        roll_diff = self._shortest_roll_diff(self.startroll, self.endroll)
        return (self.startroll + f * roll_diff) % 360

    @staticmethod
    def duration_upper_bound(acs_config: AttitudeControlSystem) -> int:
        """Bound any path from the selected algorithm, independent of epoch.

        Each shortest quaternion segment is at most 180 degrees. Using the
        slowest rate and acceleration bounds every direction of the ellipsoidal
        body-axis limits, even when their minima occur on different axes.
        """
        if acs_config.slew_algorithm == SlewAlgorithm.QUATERNION:
            segments = 1
        elif acs_config.slew_algorithm == SlewAlgorithm.CONSTRAINT_AVOIDING:
            segments = _MAX_CONSTRAINT_AVOIDING_SEGMENTS
        else:
            raise ValueError("No slew duration bound for the configured algorithm")

        slowest = acs_config.model_copy(
            update={
                "max_slew_rate": min(acs_config.max_slew_rate_body)
                if acs_config.max_slew_rate_body is not None
                else acs_config.max_slew_rate,
                "slew_acceleration": min(acs_config.slew_acceleration_body)
                if acs_config.slew_acceleration_body is not None
                else acs_config.slew_acceleration,
                "max_slew_rate_body": None,
                "slew_acceleration_body": None,
            }
        )
        if not all(
            np.isfinite(value) and value > 0.0
            for value in (slowest.max_slew_rate, slowest.slew_acceleration)
        ) or not (
            np.isfinite(slowest.max_settle_seconds)
            and slowest.max_settle_seconds >= 0.0
        ):
            raise ValueError("Cannot bound slew duration with invalid kinematic limits")
        # Directional floating-point arithmetic can cross an integer boundary
        # before execution rounds up; reserve that extra second as well.
        return (
            scheduled_slew_time(
                segments * slowest.motion_time(180.0) + slowest.max_settle_seconds
            )
            + 1
        )

    def calc_slewtime(self) -> float:
        """Calculate time to slew between 2 coordinates, given in degrees.

        Uses the AttitudeControlSystem configuration for accurate slew time
        calculation with bang-bang control profile.
        """
        # Calculate slew distance along great circle path
        self.predict_slew()
        if (
            self.acs_config is not None
            and self.acs_config.slew_algorithm == SlewAlgorithm.CONSTRAINT_AVOIDING
            and len(self._slew_segments) > _MAX_CONSTRAINT_AVOIDING_SEGMENTS
        ):
            raise ValueError("Slew path exceeds the segment bound used by admission")
        distance = self.slewdist

        # Handle invalid distances
        if np.isnan(distance) or distance < 0:
            raise ValueError(
                f"Invalid slew distance: {distance} (start={self.startra},{self.startdec} end={self.endra},{self.enddec})"
            )
        return self._time_along_path()

    def retime(self, slewstart: float) -> float:
        """Time the slew again from ``slewstart`` along the path it has.

        A quaternion slew's path does not depend on when it starts; its
        settle does, where a thermal settling table follows the Sun. A
        scheduler trying many starts for the same pair of attitudes
        predicts the path once and retimes it for each start, which gives
        the slew ``calc_slewtime`` would from that start.
        """
        self.slewrequest = slewstart
        self.slewstart = slewstart
        return self._time_along_path()

    def _time_along_path(self) -> float:
        """The slew's settle and duration from its predicted path and its start."""
        distance = self.slewdist
        # Calculate slew time using AttitudeControlSystem
        assert self.acs_config is not None, (
            "ACS config must be set to calculate slew time"
        )
        limits = self.motion_limits
        # The settle depends on how far the slew moves the Sun in the body
        # frame when a thermal settling table is configured. A test double's
        # configuration has no table.
        acs = limits
        table = self._thermal_settling_table()
        self.sun_vector_change_deg = self.sun_vector_change() if table else 0.0
        if distance <= 0.0:
            self.settle_seconds = 0.0
        elif table:
            self.settle_seconds = acs.settle_seconds(self.sun_vector_change_deg)
        elif isinstance(acs.settle_time, (int, float)):
            self.settle_seconds = float(acs.settle_time)
        else:
            # A test double's configuration: the executor uses its limits.
            self.settle_seconds = None
        if len(self._slew_segments) > 1:
            if distance <= 0.0:
                self.slewtime = 0
            else:
                motion_time = sum(
                    acs.motion_time(segment.distance_deg, segment.axis_body)
                    for segment in self._slew_segments
                )
                self.slewtime = scheduled_slew_time(motion_time + self.settle_seconds)
        elif table:
            self.slewtime = scheduled_slew_time(
                acs.slew_time(
                    distance, self._rotation_axis_body, self.sun_vector_change_deg
                )
            )
        else:
            self.slewtime = scheduled_slew_time(
                acs.slew_time(distance, self._rotation_axis_body)
            )

        self.slewend = self.slewstart + self.slewtime
        return self.slewtime

    def predict_slew(self) -> None:
        """Compute slew distance and path according to the configured algorithm.

        QUATERNION (default):
            Full SO(3) SLERP.  The RA/Dec path is derived from the boresight
            component of the SLERP; roll is stored in _quat_roll_path so
            slew_roll() returns the properly coupled value.

        CONSTRAINT_AVOIDING:
            Generalized constraint-avoiding SLERP using the combined rust-ephem
            constraint configuration.  Routes around any combination of Sun,
            Earth, Moon, and other exclusion zones.  Falls back to QUATERNION
            when no constraint violation is detected on the direct arc.

        In all cases self.slewdist is the total angular distance (degrees).
        self.slewpath retains sampled RA/Dec values for route inspection and
        visualization; executed attitudes are evaluated from quaternion segment
        endpoints rather than interpolated in Euler coordinates.
        """
        assert self.acs_config is not None, "ACS config must be set to predict slew"
        self._slew_segments = []
        steps = 100
        if self.acs_config.slew_algorithm == SlewAlgorithm.CONSTRAINT_AVOIDING:
            self._predict_slew_constraint_avoiding(steps)
        else:
            self._predict_slew_quaternion(steps)

    def _predict_slew_quaternion(self, steps: int) -> None:
        """Compute slew path via full quaternion SLERP."""

        ras, decs, rolls = quaternion_slew_path(
            self.startra,
            self.startdec,
            self.startroll,
            self.endra,
            self.enddec,
            self.endroll,
            steps=steps,
        )
        self.slewpath = (ras, decs)
        self._quat_roll_path = rolls

        self.slewdist, rotation_axis_body = quaternion_attitude_delta(
            self.startra,
            self.startdec,
            self.startroll,
            self.endra,
            self.enddec,
            self.endroll,
        )
        self._rotation_axis_body = rotation_axis_body
        self._slew_segments = [
            _SlewSegment(
                distance_deg=self.slewdist,
                axis_body=rotation_axis_body,
                start_quat=attitude_to_quat(
                    self.startra, self.startdec, self.startroll
                ),
                end_quat=attitude_to_quat(self.endra, self.enddec, self.endroll),
            )
        ]

    def _predict_slew_constraint_avoiding(self, steps: int) -> None:
        """Compute constraint-avoiding slew path using quaternion SLERP segments.

        Uses the ACS slew_constraint if configured, otherwise falls back to the
        spacecraft's general pointing constraint. rust-ephem's route_around
        finds a validated waypoint around a constraint crossing (Sun, Earth,
        Moon, etc.), then two SLERP segments connect it to the endpoints.
        Check their actual pointing and roll samples in Rust before accepting
        the detour. An unsafe detour retains the existing direct fallback for
        downstream admission/execution checks.
        """
        assert self.acs_config is not None, "ACS config must be set to predict slew"
        assert self.constraint is not None, "Constraint must be set to predict slew"
        ephem = self.ephem
        assert ephem is not None, "Ephemeris must be set to predict slew"

        # Determine which constraint to use: ACS slew_constraint or spacecraft constraint
        slew_constraint = (
            self.acs_config.slew_constraint
            if self.acs_config.slew_constraint is not None
            else self.constraint.constraint
        )

        if slew_constraint is None:
            self._predict_slew_quaternion(steps)
            return

        # rust-ephem checks the combined constraint and validates both waypoint
        # legs at the ephemeris sample at or before the slew start time.
        waypoint = slew_constraint.route_around(
            ephemeris=ephem,
            start_ra=self.startra,
            start_dec=self.startdec,
            end_ra=self.endra,
            end_dec=self.enddec,
            time=self.slewstart,
            target_roll=None,
        ).waypoint

        if waypoint is None:
            # No detour available (clear arc or no validated waypoint).
            self._predict_slew_quaternion(steps)
            return

        w_ra, w_dec = waypoint

        # Estimate roll at waypoint by linear interpolation of the total arc fraction

        pointing_dist1 = (
            separation(
                [self.startra * DTOR, self.startdec * DTOR],
                [w_ra * DTOR, w_dec * DTOR],
            )
            / DTOR
        )
        pointing_dist2 = (
            separation(
                [w_ra * DTOR, w_dec * DTOR],
                [self.endra * DTOR, self.enddec * DTOR],
            )
            / DTOR
        )
        pointing_total = pointing_dist1 + pointing_dist2
        pointing_frac = pointing_dist1 / pointing_total if pointing_total > 0 else 0.5
        roll_diff = self._shortest_roll_diff(self.startroll, self.endroll)
        w_roll = (self.startroll + pointing_frac * roll_diff) % 360

        segment_dist1, segment_axis1 = quaternion_attitude_delta(
            self.startra,
            self.startdec,
            self.startroll,
            w_ra,
            w_dec,
            w_roll,
        )
        segment_dist2, segment_axis2 = quaternion_attitude_delta(
            w_ra,
            w_dec,
            w_roll,
            self.endra,
            self.enddec,
            self.endroll,
        )
        attitude_total = segment_dist1 + segment_dist2
        attitude_frac = segment_dist1 / attitude_total if attitude_total > 0 else 0.5

        # Build two SLERP segments; split samples by full attitude distance so
        # progress remains uniform when a segment also contains roll motion.
        steps1 = max(1, round(steps * attitude_frac))
        steps2 = max(1, steps - steps1)

        ras1, decs1, rolls1 = quaternion_slew_path(
            self.startra,
            self.startdec,
            self.startroll,
            w_ra,
            w_dec,
            w_roll,
            steps=steps1,
        )
        ras2, decs2, rolls2 = quaternion_slew_path(
            w_ra,
            w_dec,
            w_roll,
            self.endra,
            self.enddec,
            self.endroll,
            steps=steps2,
        )

        # Concatenate (drop duplicate waypoint at segment junction)
        all_ras = ras1 + ras2[1:]
        all_decs = decs1 + decs2[1:]
        all_rolls = rolls1 + rolls2[1:]

        # Pointing arcs validated by route_around can differ from these full-
        # attitude SLERPs. Check their actual rolls at the same frozen sample.
        sample_index = (
            bisect_right(ephem.timestamp, self.slewstart, key=lambda t: t.timestamp())
            - 1
        )
        if np.any(
            slew_constraint.in_constraint_batch(
                ephemeris=ephem,
                target_ras=all_ras,
                target_decs=all_decs,
                target_rolls=all_rolls,
                indices=sample_index,
            )
        ):
            self._predict_slew_quaternion(steps)
            return

        self.slewpath = (all_ras, all_decs)
        self._quat_roll_path = all_rolls
        self._rotation_axis_body = None
        self._slew_segments = [
            _SlewSegment(
                distance_deg=segment_dist1,
                axis_body=segment_axis1,
                start_quat=attitude_to_quat(
                    self.startra, self.startdec, self.startroll
                ),
                end_quat=attitude_to_quat(w_ra, w_dec, w_roll),
            ),
            _SlewSegment(
                distance_deg=segment_dist2,
                axis_body=segment_axis2,
                start_quat=attitude_to_quat(w_ra, w_dec, w_roll),
                end_quat=attitude_to_quat(self.endra, self.enddec, self.endroll),
            ),
        ]
        self.slewdist = attitude_total
