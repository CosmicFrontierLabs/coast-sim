from __future__ import annotations

from typing import Literal

import rust_ephem
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr

from ..common import givename, unixtime2date
from ..common.enums import ObsType
from ..config import AttitudeControlSystem, Constraint, MissionConfig
from ..simulation.saa import SAA


class PlanEntry(BaseModel):
    """Class to define a entry in the Plan"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: MissionConfig
    constraint: Constraint
    acs_config: AttitudeControlSystem
    ephem: rust_ephem.Ephemeris | None
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
    saa: SAA | None = None
    merit: float = 101
    windows: list[list[float]] = Field(default_factory=list)
    obstype: ObsType = ObsType.PPT
    slewpath: tuple[list[float], list[float]] = Field(default_factory=lambda: ([], []))
    slewdist: float = 0.0
    ss_min: float = 1000
    ss_max: float = 1e6
    _exptime: int = PrivateAttr()
    _exporig: int = PrivateAttr()

    @classmethod
    def from_config(
        cls, config: MissionConfig | None = None, exptime: int = 1000
    ) -> PlanEntry:
        """Build a PlanEntry from a mission config, deriving constraint/acs_config/ephem."""
        if config is None:
            raise ValueError("Config must be provided to PlanEntry")
        constraint = config.constraint
        assert constraint is not None, "Constraint must be set for PlanEntry class"
        ephem = constraint.ephem
        assert ephem is not None, "Ephemeris must be set for PlanEntry class"
        acs_config = config.spacecraft_bus.attitude_control
        assert acs_config is not None, "ACS config must be set for PlanEntry class"
        entry = cls(
            config=config,
            constraint=constraint,
            acs_config=acs_config,
            ephem=ephem,
        )
        entry._exptime = exptime
        entry._exporig = exptime
        return entry

    @property
    def exptime(self) -> int:
        return self._exptime

    @exptime.setter
    def exptime(self, t: int) -> None:
        if self._exptime is None:
            self._exporig = t
        self._exptime = t

    def __str__(self) -> str:
        return f"{unixtime2date(self.begin)} Target: {self.name} ({self.obsid}) Exp: {self.exposure}s "

    @property
    def exposure(self) -> int:  # (),excludesaa=False):
        if (
            self.obstype == ObsType.GSP
            and self.contact_begin is not None
            and self.contact_end is not None
        ):
            contact_start = max(float(self.contact_begin), float(self.begin))
            return max(0, int(self.contact_end - contact_start))
        self.insaa = 0
        exposure = self.end - self.begin - self.slewtime - self.insaa
        return max(0, int(exposure))  # always an integer number of seconds

    @exposure.setter
    def exposure(self, value: int) -> None:
        """Setter for exposure - accepts but ignores the value since exposure is computed."""
        pass

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

        assert self.config.constraint is not None, (
            "Constraint must be set to calculate visibility"
        )
        assert self.ephem is not None, "Ephemeris must be set to calculate visibility"

        # Calculate the visibility of this target
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

        # Calculate slew time using AttitudeControlSystem
        slewtime = round(self.acs_config.slew_time(self.slewdist))

        return slewtime

    def predict_slew(self, lastra: float, lastdec: float) -> None:
        """Calculate great circle slew distance and path using ACS configuration."""
        self.slewdist, self.slewpath = self.acs_config.predict_slew(
            lastra, lastdec, self.ra, self.dec
        )
