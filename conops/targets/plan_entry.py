from __future__ import annotations

from datetime import datetime, timezone
from typing import ClassVar, Literal

import rust_ephem
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    PrivateAttr,
    computed_field,
    field_serializer,
    field_validator,
    model_validator,
)

from ..common import givename, unixtime2date
from ..common.enums import ObsType
from ..common.vector import attitude_to_quat
from ..config import AttitudeControlSystem, Constraint, MissionConfig
from ..simulation.saa import SAA

BodyAxis = Literal["+X", "-X", "+Y", "-Y", "+Z", "-Z"]
RollSource = Literal["planned", "defaulted_from_unconstrained_sentinel"]


class AttitudeRotationSchema(BaseModel):
    """Generic attitude rotation representation."""

    representation: Literal["quaternion"] = "quaternion"
    direction: Literal["inertial_to_body"] = "inertial_to_body"
    order: Literal["wxyz"] = "wxyz"
    values: tuple[float, float, float, float]


class AttitudePointingSchema(BaseModel):
    """Pointing parameters used to generate a target attitude."""

    ra_deg: float
    dec_deg: float
    roll_deg: float
    boresight_axis: BodyAxis = "+X"
    roll_axis: BodyAxis = "+X"
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
    ra: float = 0.0
    dec: float = 0.0
    roll: float = -1.0
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
    _exptime: int = PrivateAttr()
    _exporig: int = PrivateAttr()

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

    @classmethod
    def from_config(
        cls, config: MissionConfig | None = None, exptime: int = 1000
    ) -> PlanEntry:
        """Build a PlanEntry from a mission config, deriving constraint/acs_config/ephem."""
        if config is None:
            raise ValueError("Config must be provided to PlanEntry")
        entry = cls(config=config)
        entry._exptime = exptime
        entry._exporig = exptime
        return entry

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
    def exptime(self) -> int:
        return self._exptime

    @exptime.setter
    def exptime(self, t: int) -> None:
        if self._exptime is None:
            self._exporig = t
        self._exptime = t

    @computed_field  # type: ignore[prop-decorator]
    @property
    def exporig(self) -> int:
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
        """Fixed target attitude generated from COAST's RA/Dec/Roll convention."""
        if self.obstype not in self._STATIC_TARGET_OBSTYPES:
            return None

        roll_deg = float(self.roll)
        roll_source: RollSource = "planned"
        if roll_deg == -1.0:
            roll_deg = 0.0
            roll_source = "defaulted_from_unconstrained_sentinel"

        quat = attitude_to_quat(self.ra, self.dec, roll_deg)
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
                roll_source=roll_source,
            ),
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
        if self.constraint.ignore_roll:
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
    ) -> int:
        """Calculate time to slew between 2 coordinates, given in degrees.

        Uses the AttitudeControlSystem configuration for accurate slew time
        calculation with bang-bang control profile.
        """

        # Use the more accurate slew distance instead of angular distance
        self.predict_slew(lastra, lastdec)

        assert self.acs_config is not None, (
            "ACS config must be set to calculate slew time"
        )
        # Calculate slew time using AttitudeControlSystem
        slewtime = round(self.acs_config.slew_time(self.slewdist))

        return slewtime

    def predict_slew(self, lastra: float, lastdec: float) -> None:
        """Calculate great circle slew distance and path using ACS configuration."""
        assert self.acs_config is not None, "ACS config must be set to predict slew"
        self.slewdist, self.slewpath = self.acs_config.predict_slew(
            lastra, lastdec, self.ra, self.dec
        )
