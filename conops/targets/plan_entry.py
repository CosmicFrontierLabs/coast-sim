from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from enum import Enum
from typing import ClassVar, Literal

import numpy as np
import rust_ephem
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from ..common import givename, unixtime2date
from ..common.enums import ACSMode, ObsType
from ..common.vector import attitude_to_quat, quaternion_attitude_delta
from ..config import AttitudeControlSystem, Constraint, MissionConfig, Telescope
from ..config.constraint import (
    attitude_constraint_names_for_scopes,
    mounted_science_attitude_constraint_names,
)
from ..simulation.saa import SAA

BodyAxis = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
RollSource = Literal["planned", "defaulted_from_unconstrained_sentinel"]


def _cardinal_body_axis(vector: tuple[float, float, float]) -> BodyAxis | None:
    axes: dict[BodyAxis, tuple[float, float, float]] = {
        "+X": (1.0, 0.0, 0.0),
        "-X": (-1.0, 0.0, 0.0),
        "+Y": (0.0, 1.0, 0.0),
        "-Y": (0.0, -1.0, 0.0),
        "+Z": (0.0, 0.0, 1.0),
        "-Z": (0.0, 0.0, -1.0),
    }
    for name, axis in axes.items():
        if np.allclose(vector, axis, atol=1e-10):
            return name
    return None


class AttitudeRotationConventionSchema(BaseModel):
    """Machine-readable attitude rotation convention."""

    representation: Literal["quaternion"] = "quaternion"
    direction: Literal["inertial_to_body"] = "inertial_to_body"
    order: Literal["wxyz"] = "wxyz"
    quaternion_product: Literal["hamilton"] = "hamilton"
    vector_action: Literal["q_v_q_conjugate"] = "q_v_q_conjugate"


class AttitudeRotationSchema(AttitudeRotationConventionSchema):
    """Generic attitude rotation representation."""

    values: tuple[float, float, float, float]


class AttitudePointingSchema(BaseModel):
    """Pointing parameters used to generate a target attitude."""

    ra_deg: float
    dec_deg: float
    roll_deg: float
    instrument_name: str | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    boresight_axis: BodyAxis | None = "+X"
    boresight_body: tuple[float, float, float] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    roll_axis: BodyAxis | None = "+X"
    roll_convention: Literal["right_handed_body_rotation"] = (
        "right_handed_body_rotation"
    )
    roll_reference_axis: BodyAxis | None = "+Z"
    roll_reference_body: tuple[float, float, float] | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    roll_reference: Literal["projected_celestial_north"] = "projected_celestial_north"
    roll_source: RollSource = "planned"


class TargetAttitudeSchema(BaseModel):
    """Commanded target attitude for a fixed-attitude plan entry."""

    frame: Literal["GCRS"] = "GCRS"
    body_frame: Literal["COAST_BODY"] = "COAST_BODY"
    rotation: AttitudeRotationSchema
    pointing: AttitudePointingSchema


class PlanEntry(BaseModel):
    """Class to define a entry in the Plan"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _STATIC_TARGET_OBSTYPES: ClassVar[frozenset[ObsType]] = frozenset(
        {ObsType.PPT, ObsType.AT, ObsType.TOO}
    )

    config: MissionConfig | None = Field(default=None, exclude=True)
    constraint: Constraint | None = Field(default=None, exclude=True)
    acs_config: AttitudeControlSystem | None = Field(default=None, exclude=True)
    ephem: rust_ephem.Ephemeris | None = Field(default=None, exclude=True)
    name: str = ""
    instrument_name: str | None = None
    ra: float = 0.0
    dec: float = 0.0
    roll: float = -1.0
    spacecraft_attitude: tuple[float, float, float] | None = None
    begin: float = 0  # start of window, not observation
    slewtime: int = 0
    insaa: int = 0
    end: float = 0
    obsid: int = 0
    station: str | None = None
    station_lat_deg: float | None = None
    station_lon_deg: float | None = None
    station_alt_m: float | None = None
    contact_begin: float | None = None
    contact_end: float | None = None
    track_start_ra: float | None = None
    track_start_dec: float | None = None
    track_start_roll: float | None = None
    track_end_ra: float | None = None
    track_end_dec: float | None = None
    track_end_roll: float | None = None
    saa: SAA | None = Field(default=None, exclude=True)
    merit: float = 101
    windows: list[list[float]] = Field(default_factory=list, exclude=True)
    obstype: ObsType = ObsType.PPT
    slewpath: tuple[list[float], list[float]] = Field(
        default_factory=lambda: ([], []), exclude=True
    )
    slewdist: float = 0.0
    ss_min: float = 1000
    ss_max: float = 1e6
    _exptime: int | None = PrivateAttr(default=None)
    _exporig: int | None = PrivateAttr(default=None)

    @model_validator(mode="wrap")
    @classmethod
    def _set_exptime_exporig_from_input(
        cls, data: object, handler: ModelWrapValidatorHandler[PlanEntry]
    ) -> PlanEntry:
        """Set _exptime/_exporig private attrs from raw input dict keys.

        exptime/exporig are @computed_field properties backed by private
        attrs, not real pydantic fields, so they serialize out via
        model_dump() but are otherwise silently dropped as unrecognized
        input during model_validate()/JSON load. This reads them from the
        raw input before that happens and assigns them directly to the
        private attrs on the constructed instance.
        """
        exptime = data.get("exptime") if isinstance(data, dict) else None
        exporig = data.get("exporig") if isinstance(data, dict) else None
        instance = handler(data)
        if exptime is not None:
            instance._exptime = exptime
        if exporig is not None:
            instance._exporig = exporig
        return instance

    @model_validator(mode="after")
    def _derive_from_config(self) -> PlanEntry:
        """Populate constraint/ephem/acs_config from config, when not already set."""
        if self.config is None:
            return self
        if self.constraint is None:
            self.constraint = self.config.constraint
        assert self.constraint is not None, "Constraint must be set for PlanEntry class"
        if self.ephem is None:
            self.ephem = self.constraint.ephem
        assert self.ephem is not None, "Ephemeris must be set for PlanEntry class"
        if self.acs_config is None:
            self.acs_config = self.config.spacecraft_bus.attitude_control
        assert self.acs_config is not None, "ACS config must be set for PlanEntry class"
        return self

    @model_validator(mode="after")
    def _validate_time_ordering(self) -> PlanEntry:
        """Check begin/end and contact_begin/contact_end ordering.

        Only runs at construction: validate_assignment is disabled on this
        model (see model_config above), so begin/end set individually via
        post-construction attribute assignment - the common pattern used
        throughout the scheduler/DITL code - are not re-checked here.
        """
        if self.begin > self.end:
            raise ValueError(f"begin ({self.begin}) must be <= end ({self.end})")
        if (
            self.contact_begin is not None
            and self.contact_end is not None
            and self.contact_begin > self.contact_end
        ):
            raise ValueError(
                f"contact_begin ({self.contact_begin}) must be <= "
                f"contact_end ({self.contact_end})"
            )
        return self

    @field_validator("begin", "end", "contact_begin", "contact_end", mode="before")
    @classmethod
    def _coerce_time(cls, v: float | int | str | None) -> float | None:
        """Accept Unix timestamps (float/int) or ISO-8601 strings."""
        if v is None:
            return None
        if isinstance(v, str):
            return datetime.fromisoformat(v).timestamp()
        return float(v)

    @field_serializer("begin", "end", "contact_begin", "contact_end")
    def _serialize_time(self, v: float | None) -> str | None:
        if v is None:
            return None
        return datetime.fromtimestamp(v, tz=timezone.utc).isoformat()

    @computed_field  # type: ignore[prop-decorator]
    @property
    def exptime(self) -> int | None:
        return self._exptime

    @exptime.setter
    def exptime(self, t: int) -> None:
        if self._exptime is None:
            self._exporig = t
        self._exptime = t

    @computed_field  # type: ignore[prop-decorator]
    @property
    def exporig(self) -> int | None:
        return self._exporig

    def __str__(self) -> str:
        return f"{unixtime2date(self.begin)} Target: {self.name} ({self.obsid}) Exp: {self.exposure}s "

    @computed_field  # type: ignore[prop-decorator]
    @property
    def exposure(self) -> int:
        if (
            self.obstype == ObsType.GSP
            and self.contact_begin is not None
            and self.contact_end is not None
        ):
            contact_start = max(float(self.contact_begin), float(self.begin))
            return max(0, int(self.contact_end - contact_start))
        exposure = self.end - self.begin - self.slewtime - self.insaa
        return max(0, int(exposure))  # always an integer number of seconds

    @exposure.setter
    def exposure(self, value: int) -> None:
        """Setter for exposure - accepts but ignores the value since exposure is computed."""
        pass

    @computed_field  # type: ignore[prop-decorator]
    @property
    def target_attitude(self) -> TargetAttitudeSchema | None:
        """Physical body attitude generated for a fixed science target."""
        if self.obstype not in self._STATIC_TARGET_OBSTYPES:
            return None

        roll_deg = float(self.roll)
        roll_source: RollSource = "planned"
        if roll_deg == -1.0:
            roll_deg = 0.0
            roll_source = "defaulted_from_unconstrained_sentinel"

        telescope = self.science_telescope()
        if self.spacecraft_attitude is not None:
            body_attitude = self.spacecraft_attitude
        elif telescope is not None:
            body_attitude = telescope.target_body_attitude(self.ra, self.dec, roll_deg)
        else:
            body_attitude = (self.ra, self.dec, roll_deg)

        quat = attitude_to_quat(*body_attitude)
        boresight_body = telescope.boresight if telescope is not None else None
        roll_reference_body = (
            telescope.mounting.roll_reference_body if telescope is not None else None
        )
        return TargetAttitudeSchema(
            rotation=AttitudeRotationSchema(
                values=(
                    float(quat[0]),
                    float(quat[1]),
                    float(quat[2]),
                    float(quat[3]),
                )
            ),
            pointing=AttitudePointingSchema(
                ra_deg=float(self.ra),
                dec_deg=float(self.dec),
                roll_deg=roll_deg,
                instrument_name=(
                    telescope.name if telescope is not None else self.instrument_name
                ),
                boresight_axis=(
                    _cardinal_body_axis(boresight_body)
                    if boresight_body is not None
                    else "+X"
                ),
                boresight_body=boresight_body,
                roll_axis=(
                    _cardinal_body_axis(boresight_body)
                    if boresight_body is not None
                    else "+X"
                ),
                roll_reference_axis=(
                    _cardinal_body_axis(roll_reference_body)
                    if roll_reference_body is not None
                    else "+Z"
                ),
                roll_reference_body=roll_reference_body,
                roll_source=roll_source,
            ),
        )

    def science_telescope(self) -> Telescope | None:
        """Return the telescope assigned to this target, if configured."""
        if self.config is None or self.obstype not in self._STATIC_TARGET_OBSTYPES:
            return None
        telescope = self.config.payload.telescope_for_target(self.instrument_name)
        return telescope if issubclass(type(telescope), Telescope) else None

    def uses_mounted_attitude(self) -> bool:
        """Return whether science coordinates differ from the body attitude."""
        telescope = self.science_telescope()
        return bool(telescope is not None and not telescope.mounting.is_identity)

    def target_body_attitude(
        self, instrument_roll_deg: float | None = None
    ) -> tuple[float, float, float]:
        """Return physical body +X RA/Dec/roll for this science target."""
        roll = self.roll if instrument_roll_deg is None else instrument_roll_deg
        if roll == -1.0:
            roll = 0.0
        telescope = self.science_telescope()
        if telescope is None:
            return self.ra, self.dec, float(roll)
        return telescope.target_body_attitude(self.ra, self.dec, float(roll))

    def attitude_constraint_names(
        self,
        scopes: Sequence[str | Enum],
        spacecraft_attitude: tuple[float, float, float],
        utime: float,
        acs_mode: ACSMode | int | None = None,
    ) -> list[str]:
        """Return violations using science and body attitudes in native frames."""
        if self.constraint is None:
            return []
        if self.uses_mounted_attitude():
            return mounted_science_attitude_constraint_names(
                self.constraint,
                scopes,
                (self.ra, self.dec, self.roll),
                spacecraft_attitude,
                utime,
                acs_mode,
            )
        return attitude_constraint_names_for_scopes(
            self.constraint,
            scopes,
            spacecraft_attitude[0],
            spacecraft_attitude[1],
            utime,
            target_roll=spacecraft_attitude[2],
            acs_mode=acs_mode,
        )

    def givename(self, stem: str = "") -> None:
        self.name = givename(self.ra, self.dec, stem=stem)

    def visibility(
        self,
    ) -> int:
        """Calculate the visibility windows for a target for a given day(s).

        Note: year, day, length, and hires parameters are kept for backwards
        compatibility but are no longer used. The visibility is calculated over
        the entire ephemeris time range.
        """

        assert self.config is not None, "Config must be set to calculate visibility"
        assert self.config.constraint is not None, (
            "Constraint must be set to calculate visibility"
        )
        assert self.ephem is not None, "Ephemeris must be set to calculate visibility"

        # Calculate the visibility of this target
        assert self.constraint is not None, (
            "Constraint must be set to calculate visibility"
        )
        telescope = self.science_telescope()
        if self.uses_mounted_attitude():
            assert telescope is not None
            # Visibility belongs to the selected science line of sight. Physical
            # bus hardware constraints are checked after a concrete instrument
            # roll has been converted to a body attitude.
            combined_constraint = self.constraint.science_line_of_sight_constraint
            telescope_constraint = (
                telescope.constraint.roll_independent_constraint
                if telescope.constraint is not None
                else None
            )
            if combined_constraint is None:
                combined_constraint = telescope_constraint
            elif telescope_constraint is not None:
                combined_constraint = combined_constraint | telescope_constraint
            effective_roll = None
        elif self.constraint.ignore_roll:
            # ignore_roll=True → field-of-regard scheduling.
            #
            # The combined constraint may include star-tracker components wrapped in
            # BoresightOffsetConstraint, which are roll-dependent.  Calling
            # evaluate(target_roll=None) on a roll-dependent constraint uses
            # "visible only if visible at ALL rolls" semantics, which means nearly
            # every target appears unschedulable — the opposite of what we want.
            #
            # For FOR scheduling we want "schedulable if visible at SOME roll".
            # rust_ephem's evaluate() API cannot express that semantics directly for
            # roll-dependent constraints without sweeping all roll angles.  Instead
            # we compute windows using only the roll-independent components (sun,
            # earth, moon, panel) and rely on the runtime in_constraint() checks —
            # which DO use the correct FOR semantics (violated only if violated at
            # ALL rolls) — to reject any observation that has no valid roll at all.
            combined_constraint = self.constraint.roll_independent_constraint
            effective_roll = None
        else:
            combined_constraint = self.constraint.constraint
            effective_roll = self.roll

        if combined_constraint is None:
            self.windows = [
                [
                    float(self.ephem.begin.timestamp()),
                    float(self.ephem.end.timestamp()),
                ]
            ]
            return 0

        in_constraint = combined_constraint.evaluate(
            ephemeris=self.ephem,
            target_ra=self.ra,  # already in degrees
            target_dec=self.dec,
            target_roll=effective_roll,
        )
        # Construct the visibility windows

        self.windows = [
            [v.start_time.timestamp(), v.end_time.timestamp()]
            for v in in_constraint.visibility
        ]

        return 0

    def visible(self, begin: float, end: float) -> list[float] | Literal[False]:
        """Is the target visible between these two times, if yes, return the visibility window"""
        for window in self.windows:
            if begin >= window[0] and end <= window[1]:
                return window
        return False

    def next_vis(self, utime: float) -> float | Literal[False]:
        """Return the current or next visibility-window start."""
        if self.visible(utime, utime):
            return utime
        return next(
            (float(window[0]) for window in self.windows if window[0] > utime),
            False,
        )

    def ra_dec(self, utime: float) -> tuple[float, float] | list[int]:
        """Return Spacecraft RA/Dec for any time during the current PPT"""
        if utime >= self.begin and utime <= self.end:
            return self.ra, self.dec
        else:
            return [-1, -1]

    def calc_slewtime(
        self,
        lastra: float,
        lastdec: float,
        lastroll: float | None = None,
    ) -> int:
        """Calculate time to slew from the prior pointing or attitude.

        Scalar slew limits support legacy RA/Dec-only callers. Direction-dependent
        limits require the complete starting RA/Dec/roll attitude.
        """

        # Slew endpoints are physical body attitudes even when the science
        # coordinates are defined in an off-axis instrument frame.
        endra, enddec, endroll = self.target_body_attitude()
        self.predict_slew(lastra, lastdec, endra=endra, enddec=enddec)

        assert self.acs_config is not None, (
            "ACS config must be set to calculate slew time"
        )
        if self.acs_config.direction_dependent_slew:
            if lastroll is None or lastroll == -1.0:
                raise ValueError(
                    "a planned starting roll is required to estimate a slew with "
                    "body-axis limits"
                )
            if self.roll == -1.0:
                raise ValueError(
                    "a planned target roll is required to estimate a slew with "
                    "body-axis limits"
                )
            self.slewdist, rotation_axis_body = quaternion_attitude_delta(
                lastra,
                lastdec,
                lastroll,
                endra,
                enddec,
                endroll,
            )
            slewtime = round(
                self.acs_config.slew_time(self.slewdist, rotation_axis_body)
            )
        else:
            slewtime = round(self.acs_config.slew_time(self.slewdist))

        return slewtime

    def predict_slew(
        self,
        lastra: float,
        lastdec: float,
        *,
        endra: float | None = None,
        enddec: float | None = None,
    ) -> None:
        """Calculate great circle slew distance and path using ACS configuration."""
        assert self.acs_config is not None, "ACS config must be set to predict slew"
        if endra is None or enddec is None:
            endra, enddec, _ = self.target_body_attitude()
        self.slewdist, self.slewpath = self.acs_config.predict_slew(
            lastra, lastdec, endra, enddec
        )
