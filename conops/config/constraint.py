from enum import Enum
from functools import cached_property, lru_cache
from typing import cast

import numpy as np
import rust_ephem
from pydantic import ConfigDict, Field, PrivateAttr
from rust_ephem.constraints import ConstraintConfig

from ..common import dtutcfromtimestamp
from ..common.enums import ACSMode
from ._base import ConfigModel
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

_POLICY_SCIENCE_PLUS_SAFETY = "science_plus_safety"
_POLICY_SCIENCE = "science"
_POLICY_SAFETY = "safety"
_POLICY_FULL_MISSION = "full_mission"
_POLICY_HARD_KEEPOUT = "hard_keepout"
_POLICY_NONE = "none"


def _default_sun_constraint() -> ConstraintConfig:
    return rust_ephem.SunConstraint(min_angle=SUN_OCCULT)


def _default_anti_sun_constraint() -> ConstraintConfig:
    return rust_ephem.SunConstraint(min_angle=0, max_angle=ANTISUN_OCCULT)


def _default_moon_constraint() -> ConstraintConfig:
    return rust_ephem.MoonConstraint(min_angle=MOON_OCCULT)


def _default_earth_constraint() -> ConstraintConfig:
    return rust_ephem.EarthLimbConstraint(min_angle=EARTH_OCCULT)


def _default_panel_constraint() -> ConstraintConfig:
    return (
        rust_ephem.SunConstraint(
            min_angle=PANEL_CONSTRAINT,
            max_angle=180 - PANEL_CONSTRAINT,
        )
        & ~rust_ephem.EclipseConstraint()
    )


@lru_cache(maxsize=65536)
def _round_constraint_key(
    constraint_type: str,
    ra: float,
    dec: float,
    utime: float,
    target_roll: float | None,
) -> tuple[str, float, float, float, float | None]:
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
        None
        if target_roll is None
        else int(target_roll * _RA_DEC_ROUNDER) / _RA_DEC_ROUNDER,
    )


class Constraint(ConfigModel):
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
    sun_constraint: ConstraintConfig | None = Field(
        default=None,
        description="Sun constraint configuration",
    )
    anti_sun_constraint: ConstraintConfig | None = Field(
        default=None,
        description="Anti-sun constraint configuration",
    )
    moon_constraint: ConstraintConfig | None = Field(
        default=None,
        description="Moon constraint configuration",
    )
    earth_constraint: ConstraintConfig | None = Field(
        default=None,
        description="Earth constraint configuration",
    )
    orbit_constraint: ConstraintConfig | None = Field(
        default=None,
        description="Orbit constraint configuration",
    )
    panel_constraint: ConstraintConfig | None = Field(
        default=None,
        description="Solar panel constraint configuration",
    )
    science_constraint: ConstraintConfig | None = Field(
        default=None,
        description=(
            "Additional science-quality / scheduling constraint. Legacy top-level "
            "sun, earth, moon, orbit, panel, anti-sun, and star-tracker soft "
            "constraints are also treated as science constraints."
        ),
    )
    safety_constraint: ConstraintConfig | None = Field(
        default=None,
        description=(
            "Additional hardware-safety constraint. Star-tracker hard, radiator "
            "hard, and telescope hard constraints are also treated as safety "
            "constraints."
        ),
    )
    star_tracker_hard_constraint: ConstraintConfig | None = Field(
        default=None,
        description="Star tracker hard exclusion constraint",
    )
    star_tracker_soft_constraint: ConstraintConfig | None = Field(
        default=None,
        description="Star tracker soft exclusion constraint for scheduling",
    )
    star_tracker_enforce_modes: list[ACSMode] | None = Field(
        default=None,
        description=(
            "ACS modes in which star tracker constraints are enforced. "
            "None means enforce in all modes (conservative default). "
            "E.g. [ACSMode.SCIENCE, ACSMode.CHARGING] to skip ST checks during "
            "slews, passes, SAA, and safe mode."
        ),
    )
    ignore_roll: bool = Field(
        default=False,
        description=(
            "When True, roll is intended to be treated as a free parameter for "
            "constraint checks (e.g. callers may pass target_roll=None so that "
            "rust-ephem returns False (visible) for any pointing that is accessible "
            "at some roll angle). This field does not itself modify or override "
            "the target_roll values passed by callers; it is a configuration flag "
            "interpreted by higher-level logic."
        ),
    )
    radiator_hard_constraint: ConstraintConfig | None = Field(
        default=None,
        description="Radiator hard exclusion constraint",
    )
    telescope_hard_constraint: ConstraintConfig | None = Field(
        default=None,
        description="Telescope boresight-offset hard exclusion constraint",
    )
    ephem: rust_ephem.Ephemeris | None = Field(
        default=None,
        exclude=True,
        description="Ephemeris object for constraint calculations",
    )

    bestroll: float = Field(
        default=0.0, exclude=True, description="Best roll angle for optimization"
    )
    bestpointing: np.ndarray = Field(
        default_factory=lambda: np.array([-1, -1, -1]),
        exclude=True,
        description="Best pointing vector for optimization",
    )

    # Per-timestep constraint result cache: {(constraint_type, ra, dec, time): bool}
    _cache: dict[tuple[str, float, float, float, float | None], bool] = PrivateAttr(
        default_factory=dict
    )
    _cache_hits: int = PrivateAttr(default=0)
    _cache_misses: int = PrivateAttr(default=0)

    # validate_assignment=False: this class holds external rust_ephem types
    # (Ephemeris, ConstraintConfig) that tests legitimately mock via assignment.
    model_config = ConfigDict(arbitrary_types_allowed=True, validate_assignment=False)

    def _cache_key(
        self,
        constraint_type: str,
        ra: float,
        dec: float,
        utime: float,
        target_roll: float | None = None,
    ) -> tuple[str, float, float, float, float | None]:
        """Generate a cache key with rounded values to avoid floating-point mismatches."""
        return _round_constraint_key(constraint_type, ra, dec, utime, target_roll)

    def _cached_check(
        self,
        constraint_type: str,
        ra: float,
        dec: float,
        utime: float,
        check_fn: ConstraintConfig,
        target_roll: float | None = None,
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
        key = self._cache_key(constraint_type, ra, dec, utime, target_roll)
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
                target_roll=target_roll,
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

    def invalidate_combined_constraint_cache(self) -> None:
        """Invalidate cached combined constraint after component updates."""
        self.__dict__.pop("constraint", None)
        self.__dict__.pop("science_constraint_config", None)
        self.__dict__.pop("safety_constraint_config", None)
        self.__dict__.pop("roll_independent_constraint", None)
        self.__dict__.pop("roll_dependent_constraint", None)

    @staticmethod
    def _combine_constraints(
        constraint_components: list[ConstraintConfig | None],
    ) -> ConstraintConfig | None:
        active_constraints = [
            component for component in constraint_components if component is not None
        ]

        if not active_constraints:
            return None

        combined = active_constraints[0]
        for component in active_constraints[1:]:
            combined = combined | component
        return combined

    @cached_property
    def science_constraint_config(self) -> ConstraintConfig | None:
        """Combined science-quality / scheduling constraints."""
        return self._combine_constraints(
            [
                self.sun_constraint,
                self.moon_constraint,
                self.earth_constraint,
                self.orbit_constraint,
                self.panel_constraint,
                self.anti_sun_constraint,
                self.star_tracker_soft_constraint,
                self.science_constraint,
            ]
        )

    @cached_property
    def safety_constraint_config(self) -> ConstraintConfig | None:
        """Combined hardware-safety constraints."""
        return self._combine_constraints(
            [
                self.safety_constraint,
                self.star_tracker_hard_constraint,
                self.radiator_hard_constraint,
                self.telescope_hard_constraint,
            ]
        )

    @cached_property
    def constraint(self) -> ConstraintConfig | None:
        """Combined science-plus-safety constraint preserving legacy behavior."""
        return self._combine_constraints(
            [self.science_constraint_config, self.safety_constraint_config]
        )

    @cached_property
    def roll_independent_constraint(self) -> ConstraintConfig | None:
        """Combined constraint from roll-independent components only.

        Excludes star-tracker constraints (which are roll-dependent via
        BoresightOffsetConstraint) so that the result can be passed to
        ``ConstraintConfig.evaluate()`` without triggering the "visible only
        if visible at ALL rolls" semantics that ``evaluate(target_roll=None)``
        applies to roll-dependent constraints.

        Used by ``PlanEntry.visibility()`` when ``ignore_roll=True`` to build
        field-of-regard scheduling windows.  Note that this may over-accept
        targets that genuinely have no valid roll under the star-tracker
        constraints; those cases are caught correctly at observation time because
        ``in_constraint(target_roll=None)`` uses the opposite (permissive) FOR
        semantics — it returns True (violated) only when the constraint is
        violated at every possible roll.
        """
        components = [
            self.sun_constraint,
            self.moon_constraint,
            self.earth_constraint,
            self.orbit_constraint,
            self.panel_constraint,
            self.anti_sun_constraint,
        ]
        components.extend(
            constraint
            for constraint in (self.science_constraint, self.safety_constraint)
            if constraint is not None
            and not self._boresight_offset_constraints(constraint)
        )
        return self._combine_constraints(components)

    @staticmethod
    def _boresight_offset_constraints(
        constraint: ConstraintConfig,
    ) -> list[ConstraintConfig]:
        """Return roll-dependent boresight-offset leaves from a constraint tree."""
        from rust_ephem.constraints import BoresightOffsetConstraint

        if isinstance(constraint, BoresightOffsetConstraint):
            return [constraint]
        leaves: list[ConstraintConfig] = []
        if hasattr(constraint, "constraints"):
            for subconstraint in constraint.constraints:
                leaves.extend(Constraint._boresight_offset_constraints(subconstraint))
        return leaves

    @cached_property
    def roll_dependent_constraint(self) -> ConstraintConfig | None:
        """Combined constraint from roll-dependent components only.

        Walks the constraint trees of the boresight-offset-capable fields and
        collects every ``BoresightOffsetConstraint`` leaf node, then OR-combines
        them.  Only ``BoresightOffsetConstraint`` implements ``roll_range()``
        correctly; ``AtLeastConstraint`` returns ``True`` for ``_is_roll_dependent``
        but has an unimplemented ``roll_range()``, so we filter by concrete type
        rather than the flag.

        Roll-independent constraints (sun/earth/moon/panel on the main boresight)
        contain no ``BoresightOffsetConstraint`` nodes and are therefore excluded
        automatically.
        """
        leaves: list[ConstraintConfig] = []
        for field in (
            self.star_tracker_hard_constraint,
            self.star_tracker_soft_constraint,
            self.radiator_hard_constraint,
            self.telescope_hard_constraint,
            self.science_constraint,
            self.safety_constraint,
        ):
            if field is not None:
                leaves.extend(self._boresight_offset_constraints(field))

        if not leaves:
            return None
        combined: ConstraintConfig = leaves[0]
        for c in leaves[1:]:
            combined = combined | c
        return combined

    def in_sun(
        self, ra: float, dec: float, time: float, target_roll: float | None = None
    ) -> bool:
        if self.sun_constraint is None:
            return False
        assert self.ephem is not None, "Ephemeris must be set to use in_sun method"
        return self._cached_check(
            "sun", ra, dec, time, self.sun_constraint, target_roll=target_roll
        )

    def in_panel(
        self, ra: float, dec: float, time: float, target_roll: float | None = None
    ) -> bool:
        if self.panel_constraint is None:
            return False
        assert self.ephem is not None, "Ephemeris must be set to use in_panel method"
        return self._cached_check(
            "panel", ra, dec, time, self.panel_constraint, target_roll=target_roll
        )

    def in_anti_sun(
        self, ra: float, dec: float, time: float, target_roll: float | None = None
    ) -> bool:
        if self.anti_sun_constraint is None:
            return False
        assert self.ephem is not None, "Ephemeris must be set to use in_anti_sun method"
        return self._cached_check(
            "anti_sun",
            ra,
            dec,
            time,
            self.anti_sun_constraint,
            target_roll=target_roll,
        )

    def in_earth(
        self, ra: float, dec: float, time: float, target_roll: float | None = None
    ) -> bool:
        if self.earth_constraint is None:
            return False
        assert self.ephem is not None, "Ephemeris must be set to use in_earth method"
        return self._cached_check(
            "earth", ra, dec, time, self.earth_constraint, target_roll=target_roll
        )

    def in_eclipse(self, ra: float, dec: float, time: float) -> bool:
        assert self.ephem is not None, "Ephemeris must be set to use in_eclipse method"
        # Eclipse constraint is special - create once and cache
        if not hasattr(self, "_eclipse_constraint"):
            self._eclipse_constraint = rust_ephem.EclipseConstraint()
        return self._cached_check("eclipse", ra, dec, time, self._eclipse_constraint)

    def in_moon(
        self, ra: float, dec: float, time: float, target_roll: float | None = None
    ) -> bool:
        if self.moon_constraint is None:
            return False
        assert self.ephem is not None, "Ephemeris must be set to use in_moon method"
        return self._cached_check(
            "moon", ra, dec, time, self.moon_constraint, target_roll=target_roll
        )

    def in_orbit(
        self, ra: float, dec: float, time: float, target_roll: float | None = None
    ) -> bool:
        if self.orbit_constraint is None:
            return False
        assert self.ephem is not None, "Ephemeris must be set to use in_orbit method"
        return self._cached_check(
            "orbit", ra, dec, time, self.orbit_constraint, target_roll=target_roll
        )

    def in_explicit_science(
        self, ra: float, dec: float, time: float, target_roll: float | None = None
    ) -> bool:
        if self.science_constraint is None:
            return False
        assert self.ephem is not None, (
            "Ephemeris must be set to use in_explicit_science"
        )
        return self._cached_check(
            "explicit_science",
            ra,
            dec,
            time,
            self.science_constraint,
            target_roll=target_roll,
        )

    def in_explicit_safety(
        self, ra: float, dec: float, time: float, target_roll: float | None = None
    ) -> bool:
        if self.safety_constraint is None:
            return False
        assert self.ephem is not None, "Ephemeris must be set to use in_explicit_safety"
        return self._cached_check(
            "explicit_safety",
            ra,
            dec,
            time,
            self.safety_constraint,
            target_roll=target_roll,
        )

    def in_star_tracker_hard(
        self,
        ra: float,
        dec: float,
        time: float,
        target_roll: float | None = None,
        acs_mode: ACSMode | int | None = None,
    ) -> bool:
        """Check if pointing violates a star tracker hard constraint.

        Hard constraints are absolute health-and-safety keep-outs (e.g. blinding
        the sensor with the Sun) and are **always** enforced regardless of
        ``acs_mode`` or ``star_tracker_enforce_modes``.  The ``acs_mode``
        parameter is accepted for API compatibility but has no effect here.
        Use :meth:`in_star_tracker_soft` for the science-quality soft constraint
        that is mode-gated.
        """
        if self.star_tracker_hard_constraint is None:
            return False
        assert self.ephem is not None, (
            "Ephemeris must be set to use in_star_tracker_hard method"
        )
        return self._cached_check(
            "star_tracker_hard",
            ra,
            dec,
            time,
            self.star_tracker_hard_constraint,
            target_roll=target_roll,
        )

    def in_star_tracker_soft(
        self,
        ra: float,
        dec: float,
        time: float,
        target_roll: float | None = None,
        acs_mode: ACSMode | int | None = None,
    ) -> bool:
        if self.star_tracker_soft_constraint is None:
            return False
        if acs_mode is not None and self.star_tracker_enforce_modes is not None:
            if int(acs_mode) not in self.star_tracker_enforce_modes:
                return False
        assert self.ephem is not None, (
            "Ephemeris must be set to use in_star_tracker_soft method"
        )
        return self._cached_check(
            "star_tracker_soft",
            ra,
            dec,
            time,
            self.star_tracker_soft_constraint,
            target_roll=target_roll,
        )

    def in_radiator_hard(
        self, ra: float, dec: float, time: float, target_roll: float | None = None
    ) -> bool:
        if self.radiator_hard_constraint is None:
            return False
        assert self.ephem is not None, "Ephemeris must be set to use in_radiator_hard"
        return self._cached_check(
            "radiator_hard",
            ra,
            dec,
            time,
            self.radiator_hard_constraint,
            target_roll=target_roll,
        )

    def in_telescope_hard(
        self, ra: float, dec: float, time: float, target_roll: float | None = None
    ) -> bool:
        if self.telescope_hard_constraint is None:
            return False
        assert self.ephem is not None, "Ephemeris must be set to use in_telescope_hard"
        return self._cached_check(
            "telescope_hard",
            ra,
            dec,
            time,
            self.telescope_hard_constraint,
            target_roll=target_roll,
        )

    def in_science_constraint(
        self,
        ra: float,
        dec: float,
        utime: float,
        target_roll: float | None = None,
        acs_mode: ACSMode | int | None = None,
    ) -> bool:
        """Check image-quality / scheduling constraints only."""
        return in_science_attitude_constraint(
            self, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
        )

    def in_safety_constraint(
        self,
        ra: float,
        dec: float,
        utime: float,
        target_roll: float | None = None,
        acs_mode: ACSMode | int | None = None,
    ) -> bool:
        """Check hardware-safety constraints only."""
        return in_safety_attitude_constraint(
            self, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
        )

    def in_science_plus_safety_constraint(
        self,
        ra: float,
        dec: float,
        utime: float,
        target_roll: float | None = None,
        acs_mode: ACSMode | int | None = None,
    ) -> bool:
        """Check both scheduling and hardware-safety constraints."""
        return in_science_plus_safety_attitude_constraint(
            self, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
        )

    def in_constraint(
        self,
        ra: float,
        dec: float,
        utime: float,
        target_roll: float | None = None,
        acs_mode: ACSMode | int | None = None,
    ) -> bool:
        """For a given time is a RA/Dec in any science or safety constraint?"""
        return self.in_science_plus_safety_constraint(
            ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
        )

    def science_constraint_name(
        self,
        ra: float,
        dec: float,
        utime: float,
        target_roll: float | None = None,
        acs_mode: ACSMode | int | None = None,
    ) -> str | None:
        """Return the first violated science/scheduling constraint name."""
        return science_attitude_constraint_name(
            self, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
        )

    def safety_constraint_name(
        self,
        ra: float,
        dec: float,
        utime: float,
        target_roll: float | None = None,
        acs_mode: ACSMode | int | None = None,
    ) -> str | None:
        """Return the first violated hardware-safety constraint name."""
        return safety_attitude_constraint_name(
            self, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
        )

    def science_plus_safety_constraint_name(
        self,
        ra: float,
        dec: float,
        utime: float,
        target_roll: float | None = None,
        acs_mode: ACSMode | int | None = None,
    ) -> str | None:
        """Return the first violated science or safety constraint name."""
        return science_plus_safety_attitude_constraint_name(
            self, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
        )

    def instantaneous_field_of_regard(self, utime: float) -> float:
        """Calculate the instantaneous field of regard (FOR) solid angle at a given time.

        This is a simplified calculation based on the current Sun angle and panel constraint.
        A more accurate calculation would consider the actual geometry of the constraints
        and the spacecraft's orientation.

        Returns:
            FOR solid angle in steradians
        """
        assert self.ephem is not None, "Ephemeris must be set to calculate FOR"

        if self.constraint is None:
            return float(4.0 * np.pi)

        field_of_regard = self.constraint.instantaneous_field_of_regard(
            ephemeris=self.ephem,
            time=dtutcfromtimestamp(utime),
        )
        return field_of_regard

    def _in_constraint_batch_for_components(
        self,
        ras: list[float],
        decs: list[float],
        utime: float,
        constraint_types: list[ConstraintConfig | None],
        target_rolls: list[float] | None = None,
    ) -> np.ndarray:
        if not ras:
            return np.zeros(0, dtype=bool)

        if len(ras) != len(decs):
            raise ValueError("ras and decs must have the same length")

        if target_rolls is not None and len(target_rolls) != len(ras):
            raise ValueError(
                "target_rolls must be None or have the same length as ras and decs"
            )

        assert self.ephem is not None, (
            "Ephemeris must be set to use in_constraint_batch"
        )

        dt = dtutcfromtimestamp(utime)

        # Check all constraint types and OR the results
        violations: np.ndarray | None = None
        for constraint_func in constraint_types:
            if constraint_func is None:
                continue
            result = constraint_func.in_constraint_batch(
                ephemeris=self.ephem,
                target_ras=ras,
                target_decs=decs,
                times=[dt],
                target_rolls=target_rolls,
            )
            # Result shape is (n_candidates, 1), flatten to (n_candidates,)
            result_flat = np.asarray(result).flatten()
            if violations is None:
                violations = result_flat
            else:
                violations = violations | result_flat

        return violations if violations is not None else np.zeros(len(ras), dtype=bool)

    def in_science_constraint_batch(
        self,
        ras: list[float],
        decs: list[float],
        utime: float,
        target_rolls: list[float] | None = None,
    ) -> np.ndarray:
        """Check science/scheduling constraints for multiple pointings."""
        return self._in_constraint_batch_for_components(
            ras,
            decs,
            utime,
            [
                self.sun_constraint,
                self.earth_constraint,
                self.panel_constraint,
                self.moon_constraint,
                self.anti_sun_constraint,
                self.orbit_constraint,
                self.star_tracker_soft_constraint,
                self.science_constraint,
            ],
            target_rolls=target_rolls,
        )

    def in_safety_constraint_batch(
        self,
        ras: list[float],
        decs: list[float],
        utime: float,
        target_rolls: list[float] | None = None,
    ) -> np.ndarray:
        """Check hardware-safety constraints for multiple pointings."""
        return self._in_constraint_batch_for_components(
            ras,
            decs,
            utime,
            [
                self.safety_constraint,
                self.star_tracker_hard_constraint,
                self.radiator_hard_constraint,
                self.telescope_hard_constraint,
            ],
            target_rolls=target_rolls,
        )

    def in_constraint_batch(
        self,
        ras: list[float],
        decs: list[float],
        utime: float,
        target_rolls: list[float] | None = None,
    ) -> np.ndarray:
        """Check science-plus-safety constraints for multiple pointings.

        Uses batch evaluation via rust_ephem's in_constraint_batch() API,
        which is much faster than repeated scalar calls for many candidates.
        """
        return self._in_constraint_batch_for_components(
            ras,
            decs,
            utime,
            [
                self.sun_constraint,
                self.earth_constraint,
                self.panel_constraint,
                self.moon_constraint,
                self.anti_sun_constraint,
                self.orbit_constraint,
                self.safety_constraint,
                self.star_tracker_hard_constraint,
                self.star_tracker_soft_constraint,
                self.radiator_hard_constraint,
                self.telescope_hard_constraint,
                self.science_constraint,
            ],
            target_rolls=target_rolls,
        )


def _attitude_policy_value(policy: str | Enum) -> str:
    if isinstance(policy, Enum):
        value = policy.value
        return value if isinstance(value, str) else str(value)
    return policy


def _has_real_constraint_method(constraint: Constraint, method_name: str) -> bool:
    """Return True for real Constraint methods without triggering Mock attributes."""
    return hasattr(type(constraint), method_name)


def in_science_attitude_constraint(
    constraint: Constraint,
    ra: float,
    dec: float,
    utime: float,
    target_roll: float | None = None,
    acs_mode: ACSMode | int | None = None,
) -> bool:
    """Return True if an attitude violates science/scheduling constraints."""
    if constraint.in_sun(ra, dec, utime, target_roll=target_roll):
        return True
    if constraint.in_earth(ra, dec, utime, target_roll=target_roll):
        return True
    if constraint.in_panel(ra, dec, utime, target_roll=target_roll):
        return True
    if constraint.in_moon(ra, dec, utime, target_roll=target_roll):
        return True
    if constraint.in_anti_sun(ra, dec, utime, target_roll=target_roll):
        return True
    if constraint.in_orbit(ra, dec, utime, target_roll=target_roll):
        return True
    if constraint.in_star_tracker_soft(
        ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
    ):
        return True
    if _has_real_constraint_method(constraint, "in_explicit_science"):
        return bool(
            constraint.in_explicit_science(ra, dec, utime, target_roll=target_roll)
        )
    return False


def in_safety_attitude_constraint(
    constraint: Constraint,
    ra: float,
    dec: float,
    utime: float,
    target_roll: float | None = None,
    acs_mode: ACSMode | int | None = None,
) -> bool:
    """Return True if an attitude violates hardware-safety constraints."""
    if _has_real_constraint_method(constraint, "in_explicit_safety"):
        if constraint.in_explicit_safety(ra, dec, utime, target_roll=target_roll):
            return True
    if constraint.in_star_tracker_hard(
        ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
    ):
        return True
    if constraint.in_radiator_hard(ra, dec, utime, target_roll=target_roll):
        return True
    if constraint.in_telescope_hard(ra, dec, utime, target_roll=target_roll):
        return True
    return False


def in_science_plus_safety_attitude_constraint(
    constraint: Constraint,
    ra: float,
    dec: float,
    utime: float,
    target_roll: float | None = None,
    acs_mode: ACSMode | int | None = None,
) -> bool:
    """Return True if either science or safety constraints are violated."""
    return in_science_attitude_constraint(
        constraint, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
    ) or in_safety_attitude_constraint(
        constraint, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
    )


def in_attitude_constraint_policy(
    constraint: Constraint,
    policy: str | Enum,
    ra: float,
    dec: float,
    utime: float,
    target_roll: float | None = None,
    acs_mode: ACSMode | int | None = None,
) -> bool:
    """Return True if an attitude violates the configured policy scope."""
    policy_value = _attitude_policy_value(policy)
    if policy_value == _POLICY_NONE:
        return False
    if policy_value == _POLICY_FULL_MISSION:
        return constraint.in_constraint(
            ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
        )
    if policy_value == _POLICY_SCIENCE_PLUS_SAFETY:
        return in_science_plus_safety_attitude_constraint(
            constraint, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
        )
    if policy_value == _POLICY_SCIENCE:
        return in_science_attitude_constraint(
            constraint, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
        )
    if policy_value in {_POLICY_HARD_KEEPOUT, _POLICY_SAFETY}:
        return in_safety_attitude_constraint(
            constraint, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
        )
    raise ValueError(f"Unknown attitude constraint policy: {policy_value!r}")


def science_attitude_constraint_name(
    constraint: Constraint,
    ra: float,
    dec: float,
    utime: float,
    target_roll: float | None = None,
    acs_mode: ACSMode | int | None = None,
) -> str | None:
    """Return the first violated science/scheduling constraint name."""
    if constraint.in_sun(ra, dec, utime, target_roll=target_roll):
        return "Sun"
    if constraint.in_earth(ra, dec, utime, target_roll=target_roll):
        return "Earth Limb"
    if constraint.in_panel(ra, dec, utime, target_roll=target_roll):
        return "Panel"
    if constraint.in_moon(ra, dec, utime, target_roll=target_roll):
        return "Moon"
    if constraint.in_anti_sun(ra, dec, utime, target_roll=target_roll):
        return "Anti-Sun"
    if constraint.in_orbit(ra, dec, utime, target_roll=target_roll):
        return "Orbit"
    if constraint.in_star_tracker_soft(
        ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
    ):
        return "ST Soft"
    if _has_real_constraint_method(constraint, "in_explicit_science"):
        if constraint.in_explicit_science(ra, dec, utime, target_roll=target_roll):
            return "Science"
    return None


def safety_attitude_constraint_name(
    constraint: Constraint,
    ra: float,
    dec: float,
    utime: float,
    target_roll: float | None = None,
    acs_mode: ACSMode | int | None = None,
) -> str | None:
    """Return the first violated hardware-safety constraint name."""
    if _has_real_constraint_method(constraint, "in_explicit_safety"):
        if constraint.in_explicit_safety(ra, dec, utime, target_roll=target_roll):
            return "Safety"
    if constraint.in_star_tracker_hard(
        ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
    ):
        return "ST Hard"
    if constraint.in_radiator_hard(ra, dec, utime, target_roll=target_roll):
        return "Radiator Hard"
    if constraint.in_telescope_hard(ra, dec, utime, target_roll=target_roll):
        return "Telescope Hard"
    return None


def science_plus_safety_attitude_constraint_name(
    constraint: Constraint,
    ra: float,
    dec: float,
    utime: float,
    target_roll: float | None = None,
    acs_mode: ACSMode | int | None = None,
) -> str | None:
    """Return the first violated science or safety constraint name."""
    return science_attitude_constraint_name(
        constraint, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
    ) or safety_attitude_constraint_name(
        constraint, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
    )


def attitude_constraint_name_for_policy(
    constraint: Constraint,
    policy: str | Enum,
    ra: float,
    dec: float,
    utime: float,
    target_roll: float | None = None,
    acs_mode: ACSMode | int | None = None,
) -> str | None:
    """Return the violated constraint name for the configured policy scope."""
    policy_value = _attitude_policy_value(policy)
    if policy_value == _POLICY_NONE:
        return None
    if policy_value == _POLICY_FULL_MISSION:
        if not constraint.in_constraint(
            ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
        ):
            return None
        return (
            science_plus_safety_attitude_constraint_name(
                constraint,
                ra,
                dec,
                utime,
                target_roll=target_roll,
                acs_mode=acs_mode,
            )
            or "Unknown"
        )
    if policy_value == _POLICY_SCIENCE_PLUS_SAFETY:
        return science_plus_safety_attitude_constraint_name(
            constraint, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
        )
    if policy_value == _POLICY_SCIENCE:
        return science_attitude_constraint_name(
            constraint, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
        )
    if policy_value in {_POLICY_HARD_KEEPOUT, _POLICY_SAFETY}:
        return safety_attitude_constraint_name(
            constraint, ra, dec, utime, target_roll=target_roll, acs_mode=acs_mode
        )
    raise ValueError(f"Unknown attitude constraint policy: {policy_value!r}")


def in_attitude_constraint_policy_batch(
    constraint: Constraint,
    policy: str | Enum,
    ras: list[float],
    decs: list[float],
    utime: float,
    target_rolls: list[float] | None = None,
) -> np.ndarray:
    """Check multiple pointings against the configured policy scope."""
    policy_value = _attitude_policy_value(policy)
    if policy_value == _POLICY_NONE:
        return np.zeros(len(ras), dtype=bool)
    if policy_value in {_POLICY_FULL_MISSION, _POLICY_SCIENCE_PLUS_SAFETY}:
        if target_rolls is None:
            return constraint.in_constraint_batch(ras, decs, utime)
        return constraint.in_constraint_batch(ras, decs, utime, target_rolls)
    if policy_value == _POLICY_SCIENCE:
        if target_rolls is None:
            return constraint.in_science_constraint_batch(ras, decs, utime)
        return constraint.in_science_constraint_batch(ras, decs, utime, target_rolls)
    if policy_value in {_POLICY_HARD_KEEPOUT, _POLICY_SAFETY}:
        if target_rolls is None:
            return constraint.in_safety_constraint_batch(ras, decs, utime)
        return constraint.in_safety_constraint_batch(ras, decs, utime, target_rolls)
    raise ValueError(f"Unknown attitude constraint policy: {policy_value!r}")


class DefaultConstraint(Constraint):
    """Default mission constraint set preserving legacy COAST-Sim behavior."""

    sun_constraint: ConstraintConfig | None = Field(
        default_factory=_default_sun_constraint,
        description="Sun constraint configuration",
    )
    anti_sun_constraint: ConstraintConfig | None = Field(
        default_factory=_default_anti_sun_constraint,
        description="Anti-sun constraint configuration",
    )
    moon_constraint: ConstraintConfig | None = Field(
        default_factory=_default_moon_constraint,
        description="Moon constraint configuration",
    )
    earth_constraint: ConstraintConfig | None = Field(
        default_factory=_default_earth_constraint,
        description="Earth constraint configuration",
    )
    panel_constraint: ConstraintConfig | None = Field(
        default_factory=_default_panel_constraint,
        description="Solar panel constraint configuration",
    )

    def in_constraint_count(
        self,
        ra: float,
        dec: float,
        time: float,
        target_roll: float | None = None,
        acs_mode: ACSMode | int | None = None,
    ) -> int:
        count = 0
        if self.in_sun(ra=ra, dec=dec, time=time, target_roll=target_roll):
            count += 2
        if self.in_moon(ra=ra, dec=dec, time=time, target_roll=target_roll):
            count += 2
        if self.in_anti_sun(ra=ra, dec=dec, time=time, target_roll=target_roll):
            count += 2
        if self.in_orbit(ra=ra, dec=dec, time=time, target_roll=target_roll):
            count += 2
        if self.in_earth(ra=ra, dec=dec, time=time, target_roll=target_roll):
            count += 2
        if self.in_panel(ra=ra, dec=dec, time=time, target_roll=target_roll):
            count += 2
        if self.in_explicit_science(ra=ra, dec=dec, time=time, target_roll=target_roll):
            count += 2
        if self.in_explicit_safety(ra=ra, dec=dec, time=time, target_roll=target_roll):
            count += 2
        if self.in_star_tracker_hard(
            ra=ra, dec=dec, time=time, target_roll=target_roll, acs_mode=acs_mode
        ):
            count += 2
        if self.in_star_tracker_soft(
            ra=ra, dec=dec, time=time, target_roll=target_roll, acs_mode=acs_mode
        ):
            count += 2
        if self.in_radiator_hard(ra=ra, dec=dec, time=time, target_roll=target_roll):
            count += 2
        if self.in_telescope_hard(ra=ra, dec=dec, time=time, target_roll=target_roll):
            count += 2
        return count
