from typing import cast

import numpy as np
import rust_ephem
from pydantic import BaseModel, ConfigDict, Field
from rust_ephem.constraints import ConstraintConfig

from ..common import dtutcfromtimestamp
from .constants import (
    ANTISUN_OCCULT,
    EARTH_OCCULT,
    MOON_OCCULT,
    PANEL_CONSTRAINT,
    SUN_OCCULT,
)


class Constraint(BaseModel):
    """
    Class to calculate Spacecraft constraints.
    """

    # FIXME: Constraint types should be more general
    sun_constraint: ConstraintConfig = Field(
        default_factory=lambda: rust_ephem.SunConstraint(min_angle=SUN_OCCULT)
    )
    anti_sun_constraint: ConstraintConfig = Field(
        default_factory=lambda: rust_ephem.SunConstraint(
            min_angle=0, max_angle=ANTISUN_OCCULT
        )
    )
    moon_constraint: ConstraintConfig = Field(
        default_factory=lambda: rust_ephem.MoonConstraint(min_angle=MOON_OCCULT)
    )
    earth_constraint: ConstraintConfig = Field(
        default_factory=lambda: rust_ephem.EarthLimbConstraint(min_angle=EARTH_OCCULT)
    )
    # FIXME: For now solar panel constraint is just constraining the spacecraft
    # to be within >45 degrees of the sun and < 45 degrees from anti-sun,
    # except in eclipse
    panel_constraint: ConstraintConfig = (
        rust_ephem.SunConstraint(
            min_angle=PANEL_CONSTRAINT, max_angle=180 - PANEL_CONSTRAINT
        )
        & ~rust_ephem.EclipseConstraint()
    )

    ephem: rust_ephem.Ephemeris | None = Field(default=None, exclude=True)

    bestroll: float = Field(default=0.0, exclude=True)
    bestpointing: np.ndarray = Field(
        default_factory=lambda: np.array([-1, -1, -1]), exclude=True
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @property
    def constraint(self) -> ConstraintConfig:
        """Combined constraint from all individual constraints"""
        if not hasattr(self, "_constraint_cache"):
            self._constraint_cache = (
                self.sun_constraint
                | self.moon_constraint
                | self.earth_constraint
                | self.panel_constraint
            )
        return self._constraint_cache

    def in_sun(self, ra: float, dec: float, time: float) -> bool:
        assert self.ephem is not None, "Ephemeris must be set to use in_sun method"

        dt = dtutcfromtimestamp(time)
        return cast(
            bool,
            self.sun_constraint.in_constraint(
                ephemeris=self.ephem, target_ra=ra, target_dec=dec, time=dt
            ),
        )

    def in_panel(self, ra: float, dec: float, time: float) -> bool:
        assert self.ephem is not None, "Ephemeris must be set to use in_panel method"

        dt = dtutcfromtimestamp(time)
        return cast(
            bool,
            self.panel_constraint.in_constraint(
                ephemeris=self.ephem, target_ra=ra, target_dec=dec, time=dt
            ),
        )

    def in_anti_sun(self, ra: float, dec: float, time: float) -> bool:
        assert self.ephem is not None, "Ephemeris must be set to use in_anti_sun method"

        # Convert time to datetime for rust-ephem
        dt = dtutcfromtimestamp(time)
        return cast(
            bool,
            self.anti_sun_constraint.in_constraint(
                ephemeris=self.ephem, target_ra=ra, target_dec=dec, time=dt
            ),
        )
        # Assume it's an iterable of datetime objects

    def in_earth(self, ra: float, dec: float, time: float) -> bool:
        assert self.ephem is not None, "Ephemeris must be set to use in_earth method"

        # Convert time to datetime for rust-ephem
        dt = dtutcfromtimestamp(time)
        return cast(
            bool,
            self.earth_constraint.in_constraint(
                ephemeris=self.ephem, target_ra=ra, target_dec=dec, time=dt
            ),
        )

    def in_eclipse(self, ra: float, dec: float, time: float) -> bool:
        assert self.ephem is not None, "Ephemeris must be set to use in_eclipse method"

        # Convert time to datetime for rust-ephem

        dt = dtutcfromtimestamp(time)
        return cast(
            bool,
            rust_ephem.EclipseConstraint().in_constraint(
                ephemeris=self.ephem, target_ra=ra, target_dec=dec, time=dt
            ),
        )

    def in_moon(self, ra: float, dec: float, time: float) -> bool:
        assert self.ephem is not None, "Ephemeris must be set to use in_moon method"

        # Convert time to datetime for rust-ephem

        dt = dtutcfromtimestamp(time)
        return cast(
            bool,
            self.moon_constraint.in_constraint(
                ephemeris=self.ephem, target_ra=ra, target_dec=dec, time=dt
            ),
        )

    def in_constraint(self, ra: float, dec: float, utime: float) -> bool:
        """For a given time is a RA/Dec in occult?"""
        # Short-circuit evaluation for scalar times (most common case)
        # For array times, we need to compute all to properly OR the arrays

        # Check constraints in order of likelihood and return early if violated
        if self.in_sun(ra, dec, utime):
            return True
        if self.in_earth(ra, dec, utime):
            return True
        if self.in_panel(ra, dec, utime):
            return True
        if self.in_moon(ra, dec, utime):
            return True
        if self.in_anti_sun(ra, dec, utime):
            return True
        return False

    def in_constraint_count(self, ra: float, dec: float, utime: float) -> int:
        count = 0
        if self.in_sun(ra, dec, utime):
            count += 2
        if self.in_moon(ra, dec, utime):
            count += 2
        if self.in_anti_sun(ra, dec, utime):
            count += 2
        if self.in_earth(ra, dec, utime):
            count += 2
        return count
