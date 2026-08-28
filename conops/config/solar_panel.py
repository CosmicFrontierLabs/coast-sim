from datetime import datetime
from typing import cast

import numpy as np
import numpy.typing as npt
import rust_ephem
from pydantic import Field, PrivateAttr, field_validator, model_validator

from ..common import dtutcfromtimestamp
from ..common.enums import ACSMode
from ..common.vector import vecnorm
from ._base import ConfigModel
from .geometry import PanelGeometry


def _unit_vector(
    vector: tuple[float, float, float] | npt.NDArray[np.float64],
    *,
    field_name: str,
) -> npt.NDArray[np.float64]:
    """Return a finite three-component unit vector."""
    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain three finite components")
    magnitude = float(np.linalg.norm(array))
    if magnitude <= 0.0:
        raise ValueError(f"{field_name} must be non-zero")
    return array / magnitude


class IncidenceLossPoint(ConfigModel):
    """Additional panel power factor at a solar-incidence angle."""

    incidence_angle_deg: float = Field(
        ge=0.0,
        le=90.0,
        description="Solar-incidence angle in degrees (zero is panel-normal)",
    )
    power_factor: float = Field(
        ge=0.0,
        le=1.0,
        description="Multiplicative power factor in addition to cosine incidence",
    )


class SolarArrayDriveControl(ConfigModel):
    """Operational policy for a finite solar-array drive.

    Drive kinematics are configured separately by
    :class:`SingleAxisSolarArrayDrive`.  An empty ``sun_tracking_modes`` list is
    the conservative default: the drive holds its current angle in every mode.
    """

    sun_tracking_modes: list[ACSMode] = Field(
        default_factory=list,
        description=(
            "ACS modes in which the drive autonomously tracks the best "
            "reachable Sun-facing angle; unlisted modes hold the current angle"
        ),
    )
    track_in_eclipse: bool = Field(
        default=False,
        description=(
            "Continue Sun tracking in eclipse when the current ACS mode is "
            "listed in sun_tracking_modes"
        ),
    )

    def tracks_sun(self, acs_mode: ACSMode | None, *, in_eclipse: bool) -> bool:
        """Return whether the controller commands Sun tracking for this sample."""
        return (
            acs_mode is not None
            and acs_mode in self.sun_tracking_modes
            and (not in_eclipse or self.track_in_eclipse)
        )


class SingleAxisSolarArrayDrive(ConfigModel):
    """Finite, rate-limited rotation of a panel about one body-frame axis.

    The panel's configured ``normal`` is its zero-angle reference normal.
    Positive angles follow the right-hand rule about ``rotation_axis``.
    """

    rotation_axis: tuple[float, float, float] = Field(
        description="SADA rotation axis in the spacecraft body frame"
    )
    min_angle_deg: float = Field(description="Minimum permitted physical angle")
    max_angle_deg: float = Field(description="Maximum permitted physical angle")
    max_rate_deg_per_s: float = Field(
        gt=0.0, description="Maximum absolute articulation rate"
    )
    initial_angle_deg: float = Field(
        default=0.0, description="Drive angle at the start of each simulation run"
    )
    rotation_origin_m: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "Point on the body-frame rotation axis in metres. Required when "
            "the panel has 3D geometry."
        ),
    )

    @field_validator("rotation_axis")
    @classmethod
    def _normalize_rotation_axis(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        axis = _unit_vector(value, field_name="rotation_axis")
        return (float(axis[0]), float(axis[1]), float(axis[2]))

    @field_validator("rotation_origin_m")
    @classmethod
    def _validate_rotation_origin(
        cls, value: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        if value is None:
            return None
        origin = np.asarray(value, dtype=np.float64)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            raise ValueError("rotation_origin_m must contain three finite components")
        return (float(origin[0]), float(origin[1]), float(origin[2]))

    @model_validator(mode="after")
    def _validate_travel(self) -> "SingleAxisSolarArrayDrive":
        values = (
            self.min_angle_deg,
            self.max_angle_deg,
            self.max_rate_deg_per_s,
            self.initial_angle_deg,
        )
        if not all(np.isfinite(value) for value in values):
            raise ValueError("SADA angles and rate must be finite")
        if self.min_angle_deg >= self.max_angle_deg:
            raise ValueError("min_angle_deg must be less than max_angle_deg")
        if self.max_angle_deg - self.min_angle_deg > 360.0:
            raise ValueError("single-axis SADA travel cannot exceed 360 degrees")
        if not self.min_angle_deg <= self.initial_angle_deg <= self.max_angle_deg:
            raise ValueError("initial_angle_deg must lie within the drive travel")
        return self

    def normal_at_angle(
        self,
        reference_normal: tuple[float, float, float] | npt.NDArray[np.float64],
        angle_deg: float,
    ) -> npt.NDArray[np.float64]:
        """Rotate a reference normal to a physical drive angle."""
        return cast(
            npt.NDArray[np.float64],
            self.normals_at_angles(reference_normal, np.asarray([angle_deg]))[0],
        )

    def rotate_vectors_at_angle(
        self,
        vectors: npt.ArrayLike,
        angle_deg: float,
    ) -> npt.NDArray[np.float64]:
        """Rotate one or more body-frame vectors about the drive axis."""
        array = np.asarray(vectors, dtype=np.float64)
        scalar = array.ndim == 1
        if scalar:
            array = array[None, :]
        if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
            raise ValueError("vectors must have shape (3,) or (N, 3) and be finite")
        axis = np.asarray(self.rotation_axis, dtype=np.float64)
        theta = np.deg2rad(angle_deg)
        rotated = (
            array * np.cos(theta)
            + np.cross(axis, array) * np.sin(theta)
            + np.outer(array @ axis, axis) * (1.0 - np.cos(theta))
        )
        return cast(npt.NDArray[np.float64], rotated[0] if scalar else rotated)

    def normals_at_angles(
        self,
        reference_normal: tuple[float, float, float] | npt.NDArray[np.float64],
        angles_deg: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """Rotate a reference normal to one or more physical drive angles."""
        normal = _unit_vector(reference_normal, field_name="panel normal")
        axis = np.asarray(self.rotation_axis, dtype=np.float64)
        theta = np.deg2rad(np.asarray(angles_deg, dtype=np.float64))
        rotated = (
            normal[None, :] * np.cos(theta)[:, None]
            + np.cross(axis, normal)[None, :] * np.sin(theta)[:, None]
            + axis[None, :] * np.dot(axis, normal) * (1.0 - np.cos(theta))[:, None]
        )
        magnitudes = np.linalg.norm(rotated, axis=1)
        if not np.all(np.isfinite(rotated)) or np.any(magnitudes <= 0.0):
            raise ValueError("rotated panel normals must be finite and non-zero")
        return cast(npt.NDArray[np.float64], rotated / magnitudes[:, None])

    def optimal_angle(
        self,
        reference_normal: tuple[float, float, float] | npt.NDArray[np.float64],
        sun_body: tuple[float, float, float] | npt.NDArray[np.float64],
        reference_angle_deg: float | None = None,
    ) -> float:
        """Return the in-travel angle with the greatest Sun-normal dot product."""
        angles = self.optimal_angles(
            reference_normal,
            np.asarray([sun_body], dtype=np.float64),
            reference_angle_deg=reference_angle_deg,
        )
        return float(angles[0])

    def optimal_angles(
        self,
        reference_normal: tuple[float, float, float] | npt.NDArray[np.float64],
        sun_body: npt.NDArray[np.float64],
        reference_angle_deg: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """Vectorized form of :meth:`optimal_angle` for Sun-vector candidates."""
        normal = _unit_vector(reference_normal, field_name="panel normal")
        sun = np.asarray(sun_body, dtype=np.float64)
        if sun.ndim != 2 or sun.shape[1] != 3 or not np.all(np.isfinite(sun)):
            raise ValueError("Sun body vectors must have shape (N, 3) and be finite")
        sun_magnitudes = np.linalg.norm(sun, axis=1)
        if np.any(sun_magnitudes <= 0.0):
            raise ValueError("Sun body vectors must be non-zero")
        sun = sun / sun_magnitudes[:, None]
        axis = np.asarray(self.rotation_axis, dtype=np.float64)

        reference = (
            self.initial_angle_deg
            if reference_angle_deg is None
            else float(
                np.clip(reference_angle_deg, self.min_angle_deg, self.max_angle_deg)
            )
        )

        cosine_coefficient = sun @ normal - np.dot(axis, normal) * (sun @ axis)
        sine_coefficient = sun @ np.cross(axis, normal)
        unconstrained_deg = np.rad2deg(np.arctan2(sine_coefficient, cosine_coefficient))

        travel_center = 0.5 * (self.min_angle_deg + self.max_angle_deg)
        equivalent = unconstrained_deg + 360.0 * np.rint(
            (travel_center - unconstrained_deg) / 360.0
        )
        equivalent_valid = (equivalent >= self.min_angle_deg) & (
            equivalent <= self.max_angle_deg
        )
        candidates = np.column_stack(
            (
                np.full(len(sun), self.min_angle_deg),
                np.full(len(sun), self.max_angle_deg),
                equivalent,
            )
        )

        candidate_normals = self.normals_at_angles(normal, candidates.ravel()).reshape(
            len(sun), 3, 3
        )
        scores = np.einsum("nkj,nj->nk", candidate_normals, sun)
        scores[:, 2] = np.where(equivalent_valid, scores[:, 2], -np.inf)

        best_scores = np.max(scores, axis=1, keepdims=True)
        tied = np.isclose(scores, best_scores, rtol=0.0, atol=1e-12)
        distances = np.where(tied, np.abs(candidates - reference), np.inf)
        best_indices = np.argmin(distances, axis=1)
        best_angles = cast(
            npt.NDArray[np.float64],
            candidates[np.arange(len(sun)), best_indices],
        )
        # When the Sun is parallel to the drive axis, or the reference normal
        # is parallel to it, rotation cannot change illumination. Hold the
        # current physical angle instead of commanding an arbitrary endpoint.
        flat_objective = np.hypot(cosine_coefficient, sine_coefficient) <= 1e-12
        return np.where(flat_objective, reference, best_angles)

    def step_toward(
        self, current_angle_deg: float, target_angle_deg: float, elapsed_seconds: float
    ) -> float:
        """Advance toward a target without exceeding travel or rate limits."""
        if not np.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        current = float(
            np.clip(current_angle_deg, self.min_angle_deg, self.max_angle_deg)
        )
        target = float(
            np.clip(target_angle_deg, self.min_angle_deg, self.max_angle_deg)
        )
        max_step = self.max_rate_deg_per_s * elapsed_seconds
        delta = float(np.clip(target - current, -max_step, max_step))
        return float(np.clip(current + delta, self.min_angle_deg, self.max_angle_deg))


def get_ephemeris_indices(
    time: datetime | list[datetime], ephemeris: rust_ephem.Ephemeris
) -> np.ndarray:
    """
    Find indices in ephemeris that match the given times.

    Args:
        time: Python datetime object or list of datetime objects
        ephemeris: Ephemeris adapter object with index method

    Returns:
        Array of indices into ephemeris
    """
    if isinstance(time, datetime):
        # Single time - find closest match
        idx = ephemeris.index(time)
        return np.array([idx])
    else:
        # Multiple times - find closest match for each
        indices = []
        for t in time:
            indices.append(ephemeris.index(t))
        return np.array(indices)


class SolarPanel(ConfigModel):
    """
    Configuration for a single solar panel element.

    Attributes:
        name (str): Name/identifier for the panel.
        gimbled (bool): Whether this panel is gimbled.
        normal (tuple[float, float, float]): Panel normal vector in spacecraft body frame.
            Defined as (x, y, z) where:
            - +x is the spacecraft pointing direction (boresight)
            - +y is the spacecraft "up" direction
            - +z completes the right-handed coordinate system
            Should be a unit vector for proper illumination calculations.
            Use create_solar_panel_vector() to generate vectors for common mount types.
        max_power (float): Maximum electrical power output at full illumination (W).
        conversion_efficiency (Optional[float]): Optional per-panel efficiency.
            If not provided, array-level efficiency is used.
        single_axis_drive (SingleAxisSolarArrayDrive | None): Optional finite,
            rate-limited articulation model. This is distinct from the legacy
            ideal ``gimbled`` behavior.
        incidence_loss_curve (list[IncidenceLossPoint] | None): Optional
            additional power-factor curve versus incidence angle.
    """

    # Class-level eclipse constraint (stateless, shared across all instances)
    _eclipse_constraint = rust_ephem.EclipseConstraint()

    name: str = Field(
        default="Panel", description="Name/identifier for the solar panel"
    )
    gimbled: bool = Field(default=False, description="Whether the panel is gimbled")
    normal: tuple[float, float, float] = Field(
        default=(0.0, 1.0, 0.0),
        description="Panel normal vector in spacecraft body frame",
    )
    max_power: float = Field(
        default=800.0, description="Maximum power output at full illumination in Watts"
    )
    conversion_efficiency: float | None = Field(
        default=None,
        description="Optional per-panel efficiency (uses array-level if not specified)",
    )
    geometry: PanelGeometry | None = Field(
        default=None,
        description=(
            "Optional 3D geometry for shadow computation. "
            "When set, this panel can cast shadows onto radiators that list its name in shadowed_by."
        ),
    )
    single_axis_drive: SingleAxisSolarArrayDrive | None = Field(
        default=None,
        description=(
            "Optional finite single-axis drive. The configured normal is the "
            "zero-angle reference normal."
        ),
    )
    drive_control: SolarArrayDriveControl = Field(
        default_factory=SolarArrayDriveControl,
        description=(
            "Explicit operational policy for a finite drive. The default holds "
            "the configured initial angle in every ACS mode."
        ),
    )
    incidence_loss_curve: list[IncidenceLossPoint] | None = Field(
        default=None,
        description=(
            "Optional additional power factor versus solar-incidence angle. "
            "Values are linearly interpolated and endpoint-clamped."
        ),
    )

    _drive_angle_deg: float | None = PrivateAttr(default=None)
    _drive_time_s: float | None = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_articulation(self) -> "SolarPanel":
        if self.gimbled and self.single_axis_drive is not None:
            raise ValueError(
                "gimbled and single_axis_drive are mutually exclusive; "
                "gimbled is the legacy ideal Sun-tracking model"
            )
        if self.single_axis_drive is None and (
            self.drive_control.sun_tracking_modes or self.drive_control.track_in_eclipse
        ):
            raise ValueError("drive_control requires single_axis_drive")
        if (
            self.geometry is not None
            and self.single_axis_drive is not None
            and self.single_axis_drive.rotation_origin_m is None
        ):
            raise ValueError(
                "single_axis_drive.rotation_origin_m is required for articulated geometry"
            )
        if self.incidence_loss_curve:
            angles = [point.incidence_angle_deg for point in self.incidence_loss_curve]
            if any(right <= left for left, right in zip(angles, angles[1:])):
                raise ValueError(
                    "incidence_loss_curve angles must be strictly increasing"
                )
            factors = [point.power_factor for point in self.incidence_loss_curve]
            if any(right > left for left, right in zip(factors, factors[1:])):
                raise ValueError(
                    "incidence_loss_curve power factors must be non-increasing"
                )
        return self

    def reset_drive_state(self) -> None:
        """Reset runtime articulation state to the configured initial angle."""
        self._drive_angle_deg = (
            self.single_axis_drive.initial_angle_deg
            if self.single_axis_drive is not None
            else None
        )
        self._drive_time_s = None

    @property
    def drive_angle_deg(self) -> float | None:
        """Return the current runtime drive angle, if this panel is driven."""
        if self.single_axis_drive is None:
            return None
        if self._drive_angle_deg is None:
            return self.single_axis_drive.initial_angle_deg
        return self._drive_angle_deg

    def tracks_sun(self, acs_mode: ACSMode | None, *, in_eclipse: bool) -> bool:
        """Return the explicit control decision for this panel and sample."""
        return self.drive_control.tracks_sun(acs_mode, in_eclipse=in_eclipse)

    def geometry_at_drive_angle(
        self, angle_deg: float | None = None
    ) -> PanelGeometry | None:
        """Return panel geometry rotated to a physical drive angle.

        Undriven geometry is returned unchanged. Driven geometry rotates its
        centre and spanning vectors about the configured body-frame axis line.
        """
        geometry = self.geometry
        drive = self.single_axis_drive
        if geometry is None or drive is None:
            return geometry
        origin_value = drive.rotation_origin_m
        assert origin_value is not None
        angle = self.drive_angle_deg if angle_deg is None else angle_deg
        assert angle is not None
        origin = np.asarray(origin_value, dtype=np.float64)
        center_offset = np.asarray(geometry.center_m, dtype=np.float64) - origin
        center = origin + drive.rotate_vectors_at_angle(center_offset, angle)
        u = drive.rotate_vectors_at_angle(geometry.u, angle)
        v = drive.rotate_vectors_at_angle(geometry.v, angle)
        return PanelGeometry(
            center_m=(float(center[0]), float(center[1]), float(center[2])),
            u=(float(u[0]), float(u[1]), float(u[2])),
            v=(float(v[0]), float(v[1]), float(v[2])),
            width_m=geometry.width_m,
            height_m=geometry.height_m,
        )

    def incidence_power_factor(self, illumination_fraction: float) -> float:
        """Return the extra incidence-dependent multiplier for panel power."""
        return float(self._incidence_power_factors(illumination_fraction))

    def _incidence_power_factors(
        self, illumination_fraction: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Vectorized implementation of the incidence-loss curve."""
        illumination = np.asarray(illumination_fraction, dtype=np.float64)
        if not self.incidence_loss_curve:
            return np.ones_like(illumination)
        incidence_angle_deg = np.rad2deg(np.arccos(np.clip(illumination, 0.0, 1.0)))
        angles = np.asarray(
            [point.incidence_angle_deg for point in self.incidence_loss_curve],
            dtype=np.float64,
        )
        factors = np.asarray(
            [point.power_factor for point in self.incidence_loss_curve],
            dtype=np.float64,
        )
        return np.asarray(np.interp(incidence_angle_deg, angles, factors))

    def _project_drive_angle(
        self,
        time_s: float,
        sun_body: npt.NDArray[np.float64],
        *,
        track_sun: bool,
        advance_drive_state: bool,
    ) -> float | None:
        drive = self.single_axis_drive
        if drive is None:
            return None

        current = self.drive_angle_deg
        assert current is not None
        if self._drive_time_s is None:
            elapsed_seconds = 0.0
        else:
            elapsed_seconds = time_s - self._drive_time_s
            if elapsed_seconds < 0.0:
                if advance_drive_state:
                    raise ValueError(
                        "cannot advance single-axis drive state backward in time"
                    )
                elapsed_seconds = 0.0

        target = (
            drive.optimal_angle(self.normal, sun_body, reference_angle_deg=current)
            if track_sun
            else current
        )
        projected = drive.step_toward(current, target, elapsed_seconds)
        if advance_drive_state:
            self._drive_angle_deg = projected
            self._drive_time_s = time_s
        return projected

    def illumination_from_sun_body(
        self,
        time_s: float,
        sun_body: tuple[float, float, float] | npt.NDArray[np.float64],
        *,
        track_sun: bool = False,
        advance_drive_state: bool = False,
    ) -> tuple[float, float]:
        """Return geometric illumination and incidence-adjusted power factor.

        Eclipse is intentionally not handled here. The second return value is
        the geometric cosine illumination multiplied by the optional
        incidence-loss curve.
        """
        sun = _unit_vector(sun_body, field_name="Sun body vector")
        if self.gimbled:
            illumination = 1.0
        else:
            angle = self._project_drive_angle(
                time_s,
                sun,
                track_sun=track_sun,
                advance_drive_state=advance_drive_state,
            )
            drive = self.single_axis_drive
            if angle is None:
                normal = _unit_vector(self.normal, field_name="panel normal")
            else:
                assert drive is not None
                normal = drive.normal_at_angle(self.normal, angle)
            illumination = float(np.clip(np.dot(normal, sun), 0.0, 1.0))

        power_factor = illumination * self.incidence_power_factor(illumination)
        return illumination, power_factor

    def preview_power_factors_from_sun_body(
        self,
        sun_body: npt.NDArray[np.float64],
        *,
        track_sun: bool = False,
        elapsed_seconds: float = 0.0,
    ) -> npt.NDArray[np.float64]:
        """Preview candidate power without mutating drive state.

        Drive motion is disabled unless the caller explicitly supplies a
        positive control interval through ``elapsed_seconds``.
        """
        if not np.isfinite(elapsed_seconds) or elapsed_seconds < 0.0:
            raise ValueError("elapsed_seconds must be finite and non-negative")
        sun = np.asarray(sun_body, dtype=np.float64)
        if sun.ndim != 2 or sun.shape[1] != 3 or not np.all(np.isfinite(sun)):
            raise ValueError("Sun body vectors must have shape (N, 3) and be finite")
        magnitudes = np.linalg.norm(sun, axis=1)
        if np.any(magnitudes <= 0.0):
            raise ValueError("Sun body vectors must be non-zero")
        sun = sun / magnitudes[:, None]

        if self.gimbled:
            illumination = np.ones(len(sun), dtype=np.float64)
        elif self.single_axis_drive is None:
            normal = _unit_vector(self.normal, field_name="panel normal")
            illumination = np.clip(sun @ normal, 0.0, 1.0)
        else:
            drive = self.single_axis_drive
            current = self.drive_angle_deg
            assert current is not None
            targets = (
                drive.optimal_angles(self.normal, sun, reference_angle_deg=current)
                if track_sun
                else np.full(len(sun), current, dtype=np.float64)
            )
            max_step = drive.max_rate_deg_per_s * elapsed_seconds
            angles = current + np.clip(targets - current, -max_step, max_step)
            angles = np.clip(angles, drive.min_angle_deg, drive.max_angle_deg)
            normals = drive.normals_at_angles(self.normal, angles)
            illumination = np.clip(np.einsum("nj,nj->n", normals, sun), 0.0, 1.0)

        return illumination * self._incidence_power_factors(illumination)

    def panel_illumination_fraction(
        self,
        time: datetime | list[datetime] | float,
        ephem: rust_ephem.Ephemeris,
        ra: float,
        dec: float,
        roll: float = 0.0,
        acs_mode: ACSMode | None = None,
        advance_drive_state: bool = False,
    ) -> float | npt.NDArray[np.float64]:
        """Calculate the fraction of sunlight on this solar panel.

        Args:
            time: Unix timestamp, datetime object, or list of datetime objects
            ephem: Ephemeris object
            ra: Current spacecraft RA in degrees
            dec: Current spacecraft Dec in degrees
            roll: Spacecraft roll angle in degrees (rotation about boresight axis)
            acs_mode: Executed or candidate ACS mode used by the explicit drive
                control policy. ``None`` conservatively holds finite drives.
            advance_drive_state: Commit rate-limited drive motion. Candidate
                calculations should leave this false; DITL execution sets it true.

        Returns:
            float or np.ndarray: Fraction of panel illumination (0.0 to 1.0)
        """
        from ..common import scbodyvector

        if self.single_axis_drive is not None and not isinstance(
            time, (int, float, datetime)
        ):
            previous_state = (self._drive_angle_deg, self._drive_time_s)
            try:
                values = [
                    cast(
                        float,
                        self.panel_illumination_fraction(
                            time=item,
                            ephem=ephem,
                            ra=ra,
                            dec=dec,
                            roll=roll,
                            acs_mode=acs_mode,
                            advance_drive_state=True,
                        ),
                    )
                    for item in time
                ]
            finally:
                if not advance_drive_state:
                    self._drive_angle_deg, self._drive_time_s = previous_state
            return np.asarray(values, dtype=np.float64)

        # Convert unix time to datetime if needed
        if isinstance(time, (int, float)):
            time = [dtutcfromtimestamp(time)]
            scalar = True
        elif isinstance(time, datetime):
            time = [time]
            scalar = True
        else:
            scalar = False

        # Get the array index of the ephemeris for this time
        try:
            indices = get_ephemeris_indices(
                time=time[0] if scalar else time, ephemeris=ephem
            )
        except Exception as e:
            print(f"Error getting slice for time={time}, ephem={ephem}: {e}")
            raise

        # Use EclipseConstraint to determine if spacecraft is in eclipse
        # EclipseConstraint returns True when IN eclipse, so we need to invert it
        if scalar:
            in_eclipse = self._eclipse_constraint.in_constraint(
                ephemeris=ephem, target_ra=0.0, target_dec=0.0, time=time[0]
            )
            eclipse_flags = np.array([in_eclipse], dtype=bool)
        else:
            result = self._eclipse_constraint.evaluate(
                ephemeris=ephem, target_ra=0.0, target_dec=0.0, times=time
            )
            eclipse_flags = np.array(result.constraint_array, dtype=bool)
        not_in_eclipse = ~eclipse_flags

        # Preserve the legacy ideal-gimbal shortcut, including support for
        # callers whose ephemeris provides eclipse state but no position vectors.
        if self.gimbled:
            fraction = not_in_eclipse.astype(float)
            if scalar:
                return float(fraction[0])
            return fraction

        # Compute illumination with ideal, fixed, or rate-limited articulation.
        illum = np.zeros(len(indices))
        for idx, time_idx in enumerate(indices):
            # Get sun position vector from ephemeris
            sunvec = ephem.sun_pv.position[time_idx] - ephem.gcrs_pv.position[time_idx]

            # Convert sun vector to body frame
            sun_body = scbodyvector(
                np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(roll), sunvec
            )

            sun_mag = np.linalg.norm(sun_body)
            if sun_mag > 0:
                sun_normalized = sun_body / sun_mag
                sample_time = time[idx]
                sample_time_s = (
                    sample_time.timestamp()
                    if isinstance(sample_time, datetime)
                    else float(sample_time)
                )
                illum[idx], _ = self.illumination_from_sun_body(
                    sample_time_s,
                    sun_normalized,
                    track_sun=self.tracks_sun(
                        acs_mode, in_eclipse=bool(eclipse_flags[idx])
                    ),
                    advance_drive_state=advance_drive_state,
                )
            else:
                illum[idx] = 0.0

        # Clip negative illumination to zero and apply eclipse constraint
        illum = np.clip(illum * not_in_eclipse, a_min=0, a_max=1)

        if scalar:
            return float(illum[0])
        # Return with added fudge for mypy type checker
        return cast(npt.NDArray[np.float64], illum)


# Cached SolarPanel instance for accessing eclipse constraint
_ECLIPSE_PANEL_CACHE: SolarPanel | None = None


def _get_eclipse_constraint() -> rust_ephem.EclipseConstraint:
    """Get the eclipse constraint, using SolarPanel's for test compatibility."""
    global _ECLIPSE_PANEL_CACHE
    # Access via an instance to work around Pydantic class attribute interception
    # This allows tests to patch SolarPanel._eclipse_constraint
    if _ECLIPSE_PANEL_CACHE is None:
        _ECLIPSE_PANEL_CACHE = SolarPanel()
    return _ECLIPSE_PANEL_CACHE._eclipse_constraint


def create_solar_panel_vector(
    mount: str | None = None,
    cant_z: float = 0.0,
    cant_perp: float = 0.0,
    cant_x: float | None = None,
    cant_y: float | None = None,
    azimuth_deg: float | None = None,
) -> tuple[float, float, float]:
    """
    Create a unit normal vector for a solar panel based on mount type and cant angles.

    Supports both new and old style parameter configurations for backward compatibility.
    Only one parameter style may be used at a time - either old style OR new style, not both.

    Args:
        mount: Mount type (new style): 'sidemount', 'aftmount', or 'boresight'.
        cant_z: Cant angle around the spacecraft Z-axis in degrees (new style yaw-like rotation).
        cant_perp: Cant angle in degrees around the axis perpendicular to the panel mounting
            direction (new style pitch-like rotation):
            - For 'sidemount': rotates around X-axis.
            - For 'aftmount': rotates around Y-axis.
            - For 'boresight': rotates around Y-axis.
        cant_x: Cant angle around X-axis in degrees (old style), one of two orthogonal tilts.
        cant_y: Cant angle around Y-axis in degrees (old style), one of two orthogonal tilts.
        azimuth_deg: Structural placement angle around boresight/X in degrees (old style).
            0° = +Y (side), 90° = +Z, 180° = -Y, 270° = -Z. This places the
            panel around the spacecraft circumference; roll adds on top of this.

    Returns:
        Unit normal vector (x, y, z) in the spacecraft body frame.

    Mount types:
        - 'sidemount': Panel nominally faces +Y (spacecraft "up").
        - 'aftmount': Panel nominally faces -X (spacecraft "back").
        - 'boresight': Panel nominally faces +X (spacecraft forward/pointing).
    Examples:
        # New style - Sidemount panel with 30° yaw and 15° pitch
        normal = create_solar_panel_vector('sidemount', cant_z=30.0, cant_perp=15.0)

        # New style - Boresight panel tilted backward 45°
        normal = create_solar_panel_vector('boresight', cant_perp=-45.0)

        # Old style - Panel at 0° azimuth (+Y) with 30° cant around X and 15° cant around Y
        normal = create_solar_panel_vector(cant_x=30.0, cant_y=15.0, azimuth_deg=0.0)
    """

    # Validate that only one parameter style is used
    old_style_provided = (
        cant_x is not None or cant_y is not None or azimuth_deg is not None
    )
    new_style_provided = mount is not None

    if old_style_provided and new_style_provided:
        raise ValueError(
            "Cannot mix old style parameters (cant_x, cant_y, azimuth_deg) "
            "with new style parameters (mount, cant_z, cant_perp). "
            "Use either old style OR new style, not both."
        )

    # Check if old style parameters are provided
    if old_style_provided:
        # Use old style parameters
        if cant_x is None:
            cant_x = 0.0
        if cant_y is None:
            cant_y = 0.0
        if azimuth_deg is None:
            azimuth_deg = 0.0

        # Convert old style to rotation matrix approach
        # azimuth_deg determines the base orientation around the boresight
        # cant_x and cant_y are additional tilts

        theta_x = np.radians(cant_x)
        theta_y = np.radians(cant_y)
        azimuth_rad = np.radians(azimuth_deg)

        # Compute base vector continuously around the boresight (X-axis)
        # azimuth_deg: 0° = +Y, 90° = +Z, 180° = -Y, 270° = -Z
        base_x = 0.0
        base_y = np.cos(azimuth_rad)
        base_z = np.sin(azimuth_rad)

        # Apply cant angles
        # First cant around X-axis (theta_x)
        y_after_x = base_y * np.cos(theta_x) - base_z * np.sin(theta_x)
        z_after_x = base_y * np.sin(theta_x) + base_z * np.cos(theta_x)

        # Then cant around Y-axis (theta_y)
        x_final = base_x * np.cos(theta_y) + z_after_x * np.sin(theta_y)
        y_final = y_after_x
        z_final = -base_x * np.sin(theta_y) + z_after_x * np.cos(theta_y)

        return (x_final, y_final, z_final)

    else:
        # Use new style parameters
        if mount is None:
            mount = "sidemount"

        theta_z = np.radians(cant_z)
        theta_perp = np.radians(cant_perp)

        if mount == "sidemount":
            # Start with +Y (0, 1, 0)
            # First rotate around Z axis
            x_after_z = -np.sin(theta_z)
            y_after_z = np.cos(theta_z)

            # Then rotate around X axis (pitch)
            x = x_after_z
            y = y_after_z * np.cos(theta_perp)
            z = y_after_z * np.sin(theta_perp)

        elif mount == "aftmount":
            # Start with -X (-1, 0, 0)
            # First rotate around Z axis
            x_after_z = -np.cos(theta_z)
            y_after_z = -np.sin(theta_z)

            # Then rotate around Y axis (pitch)
            x = x_after_z * np.cos(theta_perp)
            y = y_after_z
            z = -x_after_z * np.sin(theta_perp)

        elif mount == "boresight":
            # Start with +X (1, 0, 0)
            # First rotate around Z axis
            x_after_z = np.cos(theta_z)
            y_after_z = np.sin(theta_z)

            # Then rotate around Y axis (pitch)
            x = x_after_z * np.cos(theta_perp)
            y = y_after_z
            z = x_after_z * np.sin(theta_perp)

        else:
            raise ValueError(f"Unknown mount type: {mount}")

        return (x, y, z)


class _PanelGeometry:
    """Pre-computed panel geometry arrays for vectorized calculations."""

    __slots__ = (
        "gimbled",
        "normal",
        "max_power",
        "efficiency",
        "weights",
    )

    def __init__(
        self,
        gimbled: npt.NDArray[np.bool_],
        normal: npt.NDArray[np.float64],
        max_power: npt.NDArray[np.float64],
        efficiency: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
    ) -> None:
        self.gimbled = gimbled
        self.normal = vecnorm(normal)  # shape (P, 3)
        self.max_power = max_power
        self.efficiency = efficiency
        self.weights = weights


class SolarPanelSet(ConfigModel):
    """
    Model that describes the solar panel configuration and power generation

    Represents the spacecraft solar panel set (array) and power generation model.

    Attributes:
        name (str): Name for the solar panel array.
        panels (list[SolarPanel]): List of panel elements, each with its own config.
        conversion_efficiency (float): Default array-level efficiency if a panel
            does not override it.
    """

    name: str = Field(
        default="Default Solar Panel", description="Name for the solar panel array"
    )
    panels: list[SolarPanel] = Field(
        default_factory=lambda: [SolarPanel()],
        description="List of individual solar panel configurations",
    )

    # Array-level default efficiency
    conversion_efficiency: float = Field(
        default=0.95,
        description="Default array-level conversion efficiency if panel does not override",
    )

    # Cached panel geometry for vectorized calculations
    _geometry_cache: _PanelGeometry | None = PrivateAttr(default=None)

    def reset_drive_state(self) -> None:
        """Reset every rate-limited panel drive for a new simulation run."""
        for panel in self.panels:
            panel.reset_drive_state()

    @property
    def drive_angles_deg(self) -> list[float] | None:
        """Return current driven-panel angles in panel-list order."""
        angles = [
            angle
            for panel in self.panels
            if (angle := panel.drive_angle_deg) is not None
        ]
        return angles or None

    def shadow_geometries(
        self,
        *,
        time_s: float,
        ra: float,
        dec: float,
        roll: float,
        ephem: rust_ephem.Ephemeris,
        acs_mode: ACSMode | None,
        in_eclipse: bool,
    ) -> dict[str, PanelGeometry]:
        """Return static or projected articulated geometry for shadow modelling."""
        from ..common import scbodyvector

        idx = ephem.index(dtutcfromtimestamp(time_s))
        sunvec = ephem.sun_pv.position[idx] - ephem.gcrs_pv.position[idx]
        sun_body = scbodyvector(
            np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(roll), sunvec
        )
        sun_magnitude = float(np.linalg.norm(sun_body))
        sun_normalized = (
            np.asarray(sun_body, dtype=np.float64) / sun_magnitude
            if sun_magnitude > 0.0
            else None
        )

        geometries: dict[str, PanelGeometry] = {}
        for panel in self.panels:
            if panel.geometry is None:
                continue
            angle = panel.drive_angle_deg
            if panel.single_axis_drive is not None and sun_normalized is not None:
                angle = panel._project_drive_angle(
                    time_s,
                    sun_normalized,
                    track_sun=panel.tracks_sun(acs_mode, in_eclipse=in_eclipse),
                    advance_drive_state=False,
                )
            geometry = panel.geometry_at_drive_angle(angle)
            assert geometry is not None
            geometries[panel.name] = geometry
        return geometries

    @property
    def sidemount(self) -> bool:
        """DEPRECATED: Return True if any panel is primarily side-mounted (y-component dominant).

        This is kept for backwards compatibility. With the new normal vector approach,
        panels can have arbitrary orientations.
        """
        for p in self.panels:
            n = p.normal
            # Check if y-component is dominant (side-mounted characteristic)
            if abs(n[1]) > abs(n[0]) and abs(n[1]) > abs(n[2]):
                return True
        return False

    def _get_geometry(self) -> _PanelGeometry:
        """Get or compute cached panel geometry arrays."""
        if self._geometry_cache is not None:
            return self._geometry_cache

        panels = self.panels
        n = len(panels)

        gimbled = np.array([p.gimbled for p in panels], dtype=bool)
        normal = np.array([p.normal for p in panels], dtype=np.float64)  # shape (P, 3)
        max_power = np.array([p.max_power for p in panels], dtype=np.float64)
        efficiency = np.array(
            [
                p.conversion_efficiency
                if p.conversion_efficiency is not None
                else self.conversion_efficiency
                for p in panels
            ],
            dtype=np.float64,
        )

        total_max = max_power.sum()
        weights = max_power / total_max if total_max > 0 else np.zeros(n)

        self._geometry_cache = _PanelGeometry(
            gimbled=gimbled,
            normal=normal,
            max_power=max_power,
            efficiency=efficiency,
            weights=weights,
        )
        return self._geometry_cache

    def panel_illumination_fraction(
        self,
        time: datetime | list[datetime] | float,
        ephem: rust_ephem.Ephemeris,
        ra: float,
        dec: float,
        roll: float = 0.0,
        acs_mode: ACSMode | None = None,
        advance_drive_state: bool = False,
    ) -> float | np.ndarray:
        """Calculate the weighted average fraction of sunlight on the solar panel set.

        Combines illumination from all panels weighted by their max_power.

        Args:
            time: Unix timestamp, datetime, or list of datetimes
            ephem: Ephemeris object
            ra: Current spacecraft RA in degrees
            dec: Current spacecraft Dec in degrees
            roll: Spacecraft roll angle in degrees (rotation about boresight axis)
            acs_mode: Operational mode used by finite-drive control policies.
            advance_drive_state: Commit rate-limited drive motion.

        Returns:
            float or np.ndarray: Weighted average fraction of panel illumination (0.0 to 1.0)
        """
        illumination, _ = self.illumination_and_power(
            time=time,
            ra=ra,
            dec=dec,
            ephem=ephem,
            roll=roll,
            acs_mode=acs_mode,
            advance_drive_state=advance_drive_state,
        )
        return illumination

    def power(
        self,
        time: datetime | list[datetime] | float,
        ra: float,
        dec: float,
        ephem: rust_ephem.Ephemeris,
        roll: float = 0.0,
        acs_mode: ACSMode | None = None,
        advance_drive_state: bool = False,
    ) -> float | np.ndarray:
        """Calculate the power generated by the solar panel set.

        Sums power from all panels, each weighted by illumination, max_power, and efficiency.

        Args:
            time: Unix timestamp, datetime, or list of datetimes
            ra: Current spacecraft RA in degrees
            dec: Current spacecraft Dec in degrees
            ephem: Ephemeris object
            roll: Spacecraft roll angle in degrees (rotation about boresight axis)
            acs_mode: Operational mode used by finite-drive control policies.
            advance_drive_state: Commit rate-limited drive motion.

        Returns:
            float or np.ndarray: Power generated by the solar panels in Watts
        """
        _, power = self.illumination_and_power(
            time=time,
            ra=ra,
            dec=dec,
            ephem=ephem,
            roll=roll,
            acs_mode=acs_mode,
            advance_drive_state=advance_drive_state,
        )
        return power

    def illumination_and_power(
        self,
        time: datetime | list[datetime] | float,
        ra: float,
        dec: float,
        ephem: rust_ephem.Ephemeris,
        roll: float = 0.0,
        acs_mode: ACSMode | None = None,
        advance_drive_state: bool = False,
    ) -> tuple[float | np.ndarray, float | np.ndarray]:
        """Calculate both illumination fraction and power in a single call.

        This is a vectorized implementation that computes all panels efficiently
        by looking up sun position and eclipse state only once per call.

        Args:
            time: Unix timestamp, datetime, or list of datetimes
            ra: Current spacecraft RA in degrees
            dec: Current spacecraft Dec in degrees
            ephem: Ephemeris object
            roll: Spacecraft roll angle in degrees (rotation about boresight axis)
            acs_mode: Operational mode used by finite-drive control policies.
            advance_drive_state: Commit rate-limited drive motion. This should
                be true only for executed simulation samples.

        Returns:
            tuple: (illumination_fraction, power_watts)
        """
        from ..common import scbodyvector

        panels = self.panels
        if not panels or sum(panel.max_power for panel in panels) <= 0.0:
            if isinstance(time, (float, int, datetime)):
                return 0.0, 0.0
            return np.zeros(len(time)), np.zeros(len(time))

        # Get cached panel geometry
        geom = self._get_geometry()

        # Handle time conversion - we only need scalar case for DITL
        if isinstance(time, (int, float)):
            dt = dtutcfromtimestamp(time)
            scalar = True
        elif isinstance(time, datetime):
            dt = time
            scalar = True
        else:
            # List of times - fall back to per-panel loop for now
            # (vectorizing across both panels AND times is more complex)
            return self._illumination_and_power_loop(
                time,
                ra,
                dec,
                ephem,
                roll=roll,
                acs_mode=acs_mode,
                advance_drive_state=advance_drive_state,
            )

        # Get ephemeris index ONCE
        idx = ephem.index(dt)

        # Check eclipse ONCE (use SolarPanel's constraint for test compatibility)
        in_eclipse = _get_eclipse_constraint().in_constraint(
            ephemeris=ephem, target_ra=0.0, target_dec=0.0, time=dt
        )
        has_finite_drive = any(panel.single_axis_drive is not None for panel in panels)
        if in_eclipse and not has_finite_drive:
            return (0.0, 0.0) if scalar else (np.array([0.0]), np.array([0.0]))

        # Get sun vector in body frame
        sunvec = ephem.sun_pv.position[idx] - ephem.gcrs_pv.position[idx]  # km
        sun_body = scbodyvector(
            np.deg2rad(ra), np.deg2rad(dec), np.deg2rad(roll), sunvec
        )

        # Normalize sun vector
        sun_mag = np.linalg.norm(sun_body)
        if sun_mag > 0:
            sun_normalized = sun_body / sun_mag
        else:
            # No sun direction - return zero illumination
            return (0.0, 0.0) if scalar else (np.array([0.0]), np.array([0.0]))

        uses_finite_drive_or_loss = any(
            panel.single_axis_drive is not None or bool(panel.incidence_loss_curve)
            for panel in panels
        )
        if not uses_finite_drive_or_loss:
            # Preserve the established vectorized fixed/ideal-gimbal behavior
            # exactly for existing configurations.
            panel_illum = np.dot(geom.normal, sun_normalized)
            panel_illum = np.where(geom.gimbled, 1.0, panel_illum)
            panel_illum = np.maximum(panel_illum, 0.0)
            panel_power_factor = panel_illum.copy()
        else:
            panel_illum = np.zeros(len(panels), dtype=np.float64)
            panel_power_factor = np.zeros(len(panels), dtype=np.float64)
            for panel_index, panel in enumerate(panels):
                illumination, power_factor = panel.illumination_from_sun_body(
                    dt.timestamp(),
                    sun_normalized,
                    track_sun=panel.tracks_sun(acs_mode, in_eclipse=bool(in_eclipse)),
                    advance_drive_state=advance_drive_state,
                )
                panel_illum[panel_index] = illumination
                panel_power_factor[panel_index] = power_factor

        if in_eclipse:
            panel_illum.fill(0.0)
            panel_power_factor.fill(0.0)

        # Compute weighted illumination and power
        weighted_illum = float(np.sum(panel_illum * geom.weights))
        total_power = float(
            np.sum(panel_power_factor * geom.max_power * geom.efficiency)
        )

        return weighted_illum, total_power

    def _illumination_and_power_loop(
        self,
        time: list[datetime],
        ra: float,
        dec: float,
        ephem: rust_ephem.Ephemeris,
        roll: float = 0.0,
        acs_mode: ACSMode | None = None,
        advance_drive_state: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Fallback loop-based implementation for list of times."""
        panels = self.panels
        total_max = sum(p.max_power for p in panels)

        illum_accum = np.zeros(len(time))
        power_accum = np.zeros(len(time))

        for p in panels:
            eff = (
                p.conversion_efficiency
                if p.conversion_efficiency is not None
                else self.conversion_efficiency
            )
            panel_illum = p.panel_illumination_fraction(
                time=time,
                ephem=ephem,
                ra=ra,
                dec=dec,
                roll=roll,
                acs_mode=acs_mode,
                advance_drive_state=advance_drive_state,
            )
            assert isinstance(panel_illum, np.ndarray)
            weight = p.max_power / total_max
            loss_factor = p._incidence_power_factors(panel_illum)
            panel_power = panel_illum * loss_factor * p.max_power * eff

            illum_accum = illum_accum + (panel_illum * weight)
            power_accum = power_accum + panel_power

        return illum_accum, power_accum

    def optimal_charging_pointing(
        self, time: float, ephem: rust_ephem.Ephemeris
    ) -> tuple[float, float]:
        """Find optimal RA/Dec pointing for maximum solar panel illumination.

        Analyzes panel normal vectors to determine optimal pointing:
        - Panels with dominant Y component (side-mounted): point perpendicular to sun
        - Panels with dominant Z component (body-mounted): point directly at sun
        - Mixed arrays: uses weighted average approach

        Args:
            time: Unix timestamp
            ephem: Ephemeris object

        Returns:
            tuple: (ra, dec) in degrees for optimal charging pointing
        """
        # Get sun position from pre-computed arrays
        index = ephem.index(dtutcfromtimestamp(time))
        sun_ra = ephem.sun_ra_deg[index]
        sun_dec = ephem.sun_dec_deg[index]

        # Analyze panel normal vectors to determine optimal pointing
        panels = self.panels

        # Check dominant axis of panel normals (weighted by max_power)
        total_power = sum(p.max_power for p in panels)
        if total_power <= 0:
            # No physical panels - default to pointing at sun
            return sun_ra, sun_dec

        # Compute weighted average normal vector
        avg_normal = np.zeros(3)
        for p in panels:
            weight = p.max_power / total_power
            avg_normal += np.array(p.normal) * weight

        # Determine dominant axis
        abs_normal = np.abs(avg_normal)
        dominant_axis = np.argmax(abs_normal)

        if dominant_axis == 1:  # Y is dominant (side-mounted-like)
            # Point perpendicular to sun (90 degrees away in RA)
            optimal_ra = (sun_ra + 90.0) % 360.0
            optimal_dec = sun_dec
        else:  # X (boresight) or Z is dominant (body-mounted-like)
            # Point directly at sun
            optimal_ra = sun_ra
            optimal_dec = sun_dec

        return optimal_ra, optimal_dec
