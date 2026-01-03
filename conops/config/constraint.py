from functools import lru_cache
from typing import cast

import numpy as np
import rust_ephem
from pydantic import BaseModel, ConfigDict, Field, PrivateAttr
from rust_ephem.constraints import ConstraintConfig

from ..common import dtutcfromtimestamp
from .constants import (
    ANTISUN_OCCULT,
    EARTH_OCCULT,
    MOON_OCCULT,
    PANEL_CONSTRAINT,
    SUN_OCCULT,
)

# Cache precision for constraint results.
# RA/Dec rounded to 0.01 deg (~36 arcsec) - small vs typical constraint zones (10+ deg)
# Time rounded to 1 second - constraint geometry changes slowly
_RA_DEC_PRECISION = 2  # decimal places
_TIME_PRECISION = 0  # round to nearest second

# Integer-based rounding multipliers for faster key generation
_RA_DEC_ROUNDER = 10**_RA_DEC_PRECISION
_TIME_ROUNDER = 10**_TIME_PRECISION


@lru_cache(maxsize=65536)
def _round_constraint_key(
    constraint_type: str, ra: float, dec: float, utime: float
) -> tuple[str, float, float, float]:
    """Generate a cache key with rounded values using integer math.

    This module-level function is memoized with lru_cache to avoid redundant
    rounding calculations. Integer multiplication/division is faster than
    Python's round() function.
    """
    return (
        constraint_type,
        int(ra * _RA_DEC_ROUNDER) / _RA_DEC_ROUNDER,
        int(dec * _RA_DEC_ROUNDER) / _RA_DEC_ROUNDER,
        int(utime * _TIME_ROUNDER) / _TIME_ROUNDER if _TIME_ROUNDER > 1 else int(utime),
    )


class Constraint(BaseModel):
    """Class to calculate Spacecraft constraints.

    Constraint checks are cached to avoid redundant computations when the same
    (ra, dec, time) is checked multiple times. Cache keys are rounded to avoid
    floating-point mismatches (RA/Dec to 0.01 deg, time to 1 second).

    The cache grows during a simulation run. For a 12-hour DITL with 50 targets:
    - ~60K cache entries
    - ~5MB memory (rough estimate)

    Call clear_cache() to reset between independent runs if memory is a concern.
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

    # Per-timestep constraint result cache: {(constraint_type, ra, dec, time): bool}
    _cache: dict[tuple[str, float, float, float], bool] = PrivateAttr(
        default_factory=dict
    )
    _cache_hits: int = PrivateAttr(default=0)
    _cache_misses: int = PrivateAttr(default=0)

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _cache_key(
        self, constraint_type: str, ra: float, dec: float, utime: float
    ) -> tuple[str, float, float, float]:
        """Generate a cache key with rounded values to avoid floating-point mismatches."""
        return _round_constraint_key(constraint_type, ra, dec, utime)

    def _cached_check(
        self,
        constraint_type: str,
        ra: float,
        dec: float,
        utime: float,
        check_fn: ConstraintConfig,
    ) -> bool:
        """Check cache first, compute and store if miss.

        Args:
            constraint_type: Key prefix for cache (e.g., "sun", "earth")
            ra: Right ascension in degrees
            dec: Declination in degrees
            utime: Unix timestamp
            check_fn: Constraint object with in_constraint() method

        Returns:
            True if constraint is violated, False otherwise
        """
        key = self._cache_key(constraint_type, ra, dec, utime)
        if key in self._cache:
            self._cache_hits += 1
            return self._cache[key]

        # Cache miss - compute result (caller has already asserted ephem is not None)
        self._cache_misses += 1
        dt = dtutcfromtimestamp(utime)
        result = cast(
            bool,
            check_fn.in_constraint(
                ephemeris=cast(rust_ephem.Ephemeris, self.ephem),
                target_ra=ra,
                target_dec=dec,
                time=dt,
            ),
        )
        self._cache[key] = result
        return result

    def clear_cache(self) -> None:
        """Clear the constraint result cache. Call at step boundaries."""
        self._cache.clear()

    def cache_stats(self) -> tuple[int, int]:
        """Return (hits, misses) for cache performance monitoring."""
        return (self._cache_hits, self._cache_misses)

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
        return self._cached_check("sun", ra, dec, time, self.sun_constraint)

    def in_panel(self, ra: float, dec: float, time: float) -> bool:
        assert self.ephem is not None, "Ephemeris must be set to use in_panel method"
        return self._cached_check("panel", ra, dec, time, self.panel_constraint)

    def in_anti_sun(self, ra: float, dec: float, time: float) -> bool:
        assert self.ephem is not None, "Ephemeris must be set to use in_anti_sun method"
        return self._cached_check("anti_sun", ra, dec, time, self.anti_sun_constraint)

    def in_earth(self, ra: float, dec: float, time: float) -> bool:
        assert self.ephem is not None, "Ephemeris must be set to use in_earth method"
        return self._cached_check("earth", ra, dec, time, self.earth_constraint)

    def in_eclipse(self, ra: float, dec: float, time: float) -> bool:
        assert self.ephem is not None, "Ephemeris must be set to use in_eclipse method"
        # Eclipse constraint is special - create once and cache
        if not hasattr(self, "_eclipse_constraint"):
            self._eclipse_constraint = rust_ephem.EclipseConstraint()
        return self._cached_check("eclipse", ra, dec, time, self._eclipse_constraint)

    def in_moon(self, ra: float, dec: float, time: float) -> bool:
        assert self.ephem is not None, "Ephemeris must be set to use in_moon method"
        return self._cached_check("moon", ra, dec, time, self.moon_constraint)

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

    def in_constraint_batch(
        self,
        ras: list[float],
        decs: list[float],
        utime: float,
    ) -> np.ndarray:
        """Check constraints for multiple pointings at a single time.

        Uses batch evaluation via rust_ephem's in_constraint_batch() API,
        which is much faster than repeated scalar calls for many candidates.

        Args:
            ras: List of right ascension values in degrees
            decs: List of declination values in degrees
            utime: Unix timestamp

        Returns:
            Boolean array where True means constraint is violated
        """
        if not ras:
            return np.zeros(0, dtype=bool)

        assert self.ephem is not None, (
            "Ephemeris must be set to use in_constraint_batch"
        )

        dt = dtutcfromtimestamp(utime)

        # Check all constraint types and OR the results
        violations: np.ndarray | None = None
        constraint_types = [
            self.sun_constraint,
            self.earth_constraint,
            self.panel_constraint,
            self.moon_constraint,
            self.anti_sun_constraint,
        ]
        for constraint_func in constraint_types:
            result = constraint_func.in_constraint_batch(
                ephemeris=self.ephem,
                target_ras=ras,
                target_decs=decs,
                times=[dt],
            )
            # Result shape is (n_candidates, 1), flatten to (n_candidates,)
            result_flat = np.asarray(result).flatten()
            if violations is None:
                violations = result_flat
            else:
                violations = violations | result_flat

        return violations if violations is not None else np.zeros(len(ras), dtype=bool)

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
