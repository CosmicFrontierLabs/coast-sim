from __future__ import annotations

from typing import Literal

import rust_ephem
from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..common import givename, unixtime2date
from ..config import AttitudeControlSystem, Constraint, MissionConfig
from ..simulation.saa import SAA


class PlanEntry(BaseModel):
    """Class to define a entry in the Plan"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    config: MissionConfig = Field(default_factory=MissionConfig, exclude=True)
    constraint: Constraint | None = Field(default=None, exclude=True)
    ephem: rust_ephem.Ephemeris | None = Field(default=None, exclude=True)
    acs_config: AttitudeControlSystem | None = Field(default=None, exclude=True)
    saa: SAA | None = Field(default=None, exclude=True)

    name: str = ""
    ra: float = 0.0
    dec: float = 0.0
    roll: float = -1.0
    begin: float = 0.0
    end: float = 0.0
    windows: list[list[float]] = Field(default_factory=list, exclude=True)
    merit: float = 101.0
    slewpath: tuple[list[float], list[float]] = Field(
        default_factory=lambda: ([], []), exclude=True
    )
    slewtime: int = 0
    insaa: int = 0
    obsid: int = 0
    obstype: str = "PPT"
    slewdist: float = 0.0
    ss_min: float = 1000.0
    ss_max: float = 1e6
    exptime: int = 1000
    exporig: int = 1000

    @model_validator(mode="after")
    def _initialize_from_config(self) -> PlanEntry:
        config = self.config
        if config is None:
            raise ValueError("Config must be provided to PlanEntry")
        self.constraint = config.constraint
        self.acs_config = config.spacecraft_bus.attitude_control

        assert self.constraint is not None, "Constraint must be set for PlanEntry class"
        self.ephem = self.constraint.ephem
        assert self.ephem is not None, "Ephemeris must be set for PlanEntry class"
        assert self.acs_config is not None, "ACS config must be set for PlanEntry class"

        if self.exporig == 1000 and self.exptime != 1000:
            self.exporig = self.exptime
        return self

    def __str__(self) -> str:
        return f"{unixtime2date(self.begin)} Target: {self.name} ({self.obsid}) Exp: {self.exposure}s "

    @property
    def exposure(self) -> int:  # (),excludesaa=False):
        self.insaa = 0
        return int(
            self.end - self.begin - self.slewtime - self.insaa
        )  # always an integer number of seconds

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

        config = self.config
        assert config is not None, "Config must be set to calculate visibility"
        assert config.constraint is not None, (
            "Constraint must be set to calculate visibility"
        )
        ephem = self.ephem
        assert ephem is not None, "Ephemeris must be set to calculate visibility"
        constraint = self.constraint
        assert constraint is not None, "Constraint must be set to calculate visibility"

        # Calculate the visibility of this target
        in_constraint = constraint.constraint.evaluate(
            ephemeris=ephem,
            target_ra=self.ra,  # already in degrees
            target_dec=self.dec,
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
        acs_config = self.acs_config
        assert acs_config is not None, "ACS config must be set for PlanEntry class"

        # Calculate slew time using AttitudeControlSystem
        slewtime = int(round(acs_config.slew_time(self.slewdist)))

        return slewtime

    def predict_slew(self, lastra: float, lastdec: float) -> None:
        """Calculate great circle slew distance and path using ACS configuration."""
        acs_config = self.acs_config
        assert acs_config is not None, "ACS config must be set for PlanEntry class"
        self.slewdist, self.slewpath = acs_config.predict_slew(
            lastra, lastdec, self.ra, self.dec, steps=20
        )
