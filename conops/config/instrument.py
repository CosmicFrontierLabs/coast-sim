from enum import Enum
from functools import cached_property
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import AliasChoices, Field, field_validator, model_validator
from rust_ephem.constraints import ConstraintConfig

from ..common.vector import normal_to_euler_deg
from ._base import ConfigModel
from .constraint import Constraint
from .data_generator import DataGeneration
from .power import PowerDraw
from .thermal import Heater


class TelescopeType(str, Enum):
    """Optical design type of a telescope."""

    NEWTONIAN = "Newtonian"
    CASSEGRAIN = "Cassegrain"
    RITCHEY_CHRETIEN = "Ritchey-Chrétien"
    SCHMIDT_CASSEGRAIN = "Schmidt-Cassegrain"
    MAKSUTOV_CASSEGRAIN = "Maksutov-Cassegrain"
    KORSCH = "Korsch"
    GREGORIAN = "Gregorian"
    DALL_KIRKHAM = "Dall-Kirkham"
    TMA = "Three Mirror Anastigmat"
    REFRACTOR = "Refractor"
    OTHER = "Other"


class TelescopeConfig(ConfigModel):
    """Optical configuration for a telescope.

    Groups the key optical parameters that characterise a telescope's
    collecting power and imaging geometry.  The f-number is validated
    against aperture and focal length when all three are provided.

    Attributes:
        aperture_m: Clear aperture diameter in metres.
        focal_length_m: Effective focal length in metres.
        f_number: Focal ratio (focal_length / aperture).  Auto-derived when omitted.
        telescope_type: Optical design family (e.g. Ritchey-Chrétien).
        tube_length_m: Physical tube length in metres.  For folded or
            catadioptric designs this is shorter than the focal length.
    """

    aperture_m: float | None = Field(
        default=None,
        gt=0,
        description="Clear aperture diameter in metres",
    )
    focal_length_m: float | None = Field(
        default=None,
        gt=0,
        description="Effective focal length in metres",
    )
    f_number: float | None = Field(
        default=None,
        gt=0,
        description="Focal ratio (focal_length / aperture).  Auto-derived when omitted.",
    )
    telescope_type: TelescopeType = Field(
        default=TelescopeType.OTHER,
        description="Optical design type of the telescope",
    )
    tube_length_m: float | None = Field(
        default=None,
        gt=0,
        description="Physical tube length in metres",
    )

    @model_validator(mode="after")
    def _derive_or_validate_f_number(self) -> "TelescopeConfig":
        """Derive f_number from aperture and focal_length, or validate consistency."""
        a = self.aperture_m
        fl = self.focal_length_m
        fn = self.f_number

        if a is not None and fl is not None:
            computed = fl / a
            if fn is None:
                object.__setattr__(self, "f_number", round(computed, 6))
            else:
                tol = 1e-3
                if abs(fn - computed) > tol:
                    raise ValueError(
                        f"f_number {fn} is inconsistent with focal_length_m {fl} "
                        f"and aperture_m {a} (expected {computed:.4f})"
                    )
        return self

    @property
    def plate_scale_arcsec_per_um(self) -> float | None:
        """Plate scale in arcseconds per micrometre at the focal plane.

        Returns None if focal_length_m is not set.
        """
        if self.focal_length_m is None:
            return None
        return 206265.0 / (self.focal_length_m * 1e6)


class Instrument(ConfigModel):
    """A spacecraft instrument with power consumption and data generation characteristics.

    Attributes:
        instrument_type: Discriminator tag for polymorphic payload deserialization.
        name: Instrument name/identifier.
        power_draw: Power draw characteristics (nominal, peak, mode-specific).
        heater: Optional heater system.
        data_generation: Data generation characteristics.
    """

    instrument_type: Literal["Instrument"] = Field(
        default="Instrument",
        description="Discriminator tag for polymorphic payload deserialization",
    )
    name: str = Field(
        default="Default Instrument", description="Instrument name/identifier"
    )
    power_draw: PowerDraw = Field(
        default_factory=lambda: PowerDraw(
            nominal_power=50, peak_power=100, power_mode={}
        ),
        description="Power draw characteristics for the instrument",
    )
    heater: Heater | None = Field(
        default=None, description="Optional heater system for the instrument"
    )
    data_generation: DataGeneration = Field(
        default_factory=DataGeneration, description="Data generation characteristics"
    )

    def power(self, mode: int | None = None, in_eclipse: bool = False) -> float:
        """Get the power draw for the instrument in the given mode.

        Args:
            mode: Operational mode (None for nominal)
            in_eclipse: Whether spacecraft is in eclipse

        Returns:
            Total power draw in watts
        """
        base_power = self.power_draw.power(mode, in_eclipse=in_eclipse)
        heater_power = (
            self.heater.power(mode, in_eclipse=in_eclipse) if self.heater else 0.0
        )
        return base_power + heater_power


class Telescope(Instrument):
    """A telescope instrument with optical configuration and pointing direction.

    Extends Instrument with the optical parameters of a telescope and the
    direction its boresight points in the spacecraft body frame.
    Telescope instances can be placed in Payload.instruments alongside
    any other Instrument subclass.

    Attributes:
        instrument_type: Discriminator tag for polymorphic payload deserialization.
        name: Instrument name/identifier.
        optics: Optical configuration (aperture, focal length, f-number,
            design type, tube length).
        boresight: Unit vector in spacecraft body frame giving the telescope
            pointing direction.  Body-frame axes are:

            - +x — spacecraft forward / primary pointing direction
            - +y — spacecraft "up"
            - +z — completes the right-handed frame

            Defaults to ``(1, 0, 0)`` (aligned with spacecraft boresight).
        constraint: Optional pointing constraint in the telescope frame that is
            transformed to the spacecraft body frame when evaluated.


    Example:
        >>> scope = Telescope(
        ...     name="Primary Telescope",
        ...     boresight=(1.0, 0.0, 0.0),
        ...     optics=TelescopeConfig(
        ...         aperture_m=0.5,
        ...         focal_length_m=3.5,              # f/7 telescope
        ...         telescope_type=TelescopeType.RITCHEY_CHRETIEN,
        ...         tube_length_m=1.2,
        ...     ),
        ... )
        >>> scope.optics.f_number
        7.0
    """

    instrument_type: Literal["Telescope"] = Field(  # type: ignore[assignment]
        default="Telescope",
        description="Discriminator tag for polymorphic payload deserialization",
    )
    name: str = Field(default="Telescope", description="Instrument name/identifier")
    boresight: tuple[float, float, float] = Field(
        default=(1.0, 0.0, 0.0),
        description=(
            "Telescope boresight direction as a unit vector in spacecraft body frame. "
            "+x is spacecraft forward, +y is up, +z completes the right-hand frame."
        ),
    )
    optics: TelescopeConfig = Field(
        default_factory=TelescopeConfig,
        validation_alias=AliasChoices("optics", "config"),
        description="Optical configuration of the telescope",
    )

    constraint: Constraint | None = Field(
        default=None,
        description=(
            "Optional pointing constraint defined relative to the telescope boresight. "
            "Automatically transformed to spacecraft frame via boresight_offset."
        ),
    )

    @field_validator("boresight")
    @classmethod
    def _validate_unit_vector(
        cls, v: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        magnitude = float(np.sqrt(sum(x**2 for x in v)))
        if magnitude < 0.99 or magnitude > 1.01:
            raise ValueError(
                f"Telescope boresight must be a unit vector (magnitude {magnitude:.4f})"
            )
        return v

    @cached_property
    def spacecraft_constraint(self) -> ConstraintConfig | None:
        """Telescope constraint transformed to spacecraft body frame.

        Applies boresight_offset so the constraint is evaluated relative to
        the spacecraft pointing direction rather than the telescope-local frame.
        Returns None if no constraint is set or the constraint has no active components.
        """
        if self.constraint is None:
            return None
        base = self.constraint.constraint
        if base is None:
            return None
        if self.boresight == (1.0, 0.0, 0.0):
            return base
        roll_deg, pitch_deg, yaw_deg = normal_to_euler_deg(
            np.asarray(self.boresight, dtype=np.float64)
        )
        return base.boresight_offset(
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
        )

    def invalidate_spacecraft_constraint_cache(self) -> None:
        """Invalidate cached spacecraft_constraint after boresight or constraint updates.

        Call this whenever ``boresight`` or ``constraint`` is reassigned after
        construction so that :attr:`spacecraft_constraint` is recomputed.
        """
        self.__dict__.pop("spacecraft_constraint", None)


# Discriminated union used by Payload.instruments for round-trip JSON serialization.
# Telescope must come first so Pydantic tries the more specific type first when
# the discriminator tag is present.
_AnyInstrument = Annotated[
    Telescope | Instrument,
    Field(discriminator="instrument_type"),
]


class Payload(ConfigModel):
    """A collection of instruments operated together as the spacecraft payload.

    Attributes:
        instruments: List of Instrument (or subclass) instances.
    """

    instruments: list[_AnyInstrument] = [Instrument()]

    @field_validator("instruments", mode="before")
    @classmethod
    def _inject_instrument_type(cls, v: Any) -> Any:
        """Inject instrument_type tag for backwards-compatible JSON loading.

        Old config files don't have the discriminator field. Detect Telescope
        instances by the presence of their specific keys and tag accordingly.
        """
        if not isinstance(v, list):
            return v
        result = []
        for item in v:
            if isinstance(item, dict) and "instrument_type" not in item:
                has_telescope_keys = (
                    "boresight" in item or "optics" in item or "config" in item
                )
                item = {
                    **item,
                    "instrument_type": "Telescope"
                    if has_telescope_keys
                    else "Instrument",
                }
            result.append(item)
        return result

    def power(self, mode: int | None = None, in_eclipse: bool = False) -> float:
        """Get the total power draw for all instruments in the payload in the given mode.

        Args:
            mode: Operational mode (None for nominal)
            in_eclipse: Whether spacecraft is in eclipse

        Returns:
            Total power draw in watts
        """
        return sum(
            instrument.power(mode, in_eclipse=in_eclipse)
            for instrument in self.instruments
        )

    def total_data_rate_gbps(self) -> float:
        """Get the total data generation rate across all instruments.

        Returns:
            float: Total data generation rate in Gigabits per second.
        """
        return sum(
            instrument.data_generation.rate_gbps for instrument in self.instruments
        )

    def combined_telescope_spacecraft_constraint(self) -> ConstraintConfig | None:
        """OR-combine the spacecraft-frame constraints from all Telescope instruments.

        Returns None if no telescopes have constraints defined.
        """
        active: list[ConstraintConfig] = []
        for instrument in self.instruments:
            if isinstance(instrument, Telescope):
                sc = instrument.spacecraft_constraint
                if sc is not None:
                    active.append(sc)
        if not active:
            return None
        combined = active[0]
        for c in active[1:]:
            combined = combined | c
        return combined

    def data_generated(self, duration_seconds: float) -> float:
        """Calculate total data generated by all instruments over a duration.

        Args:
            duration_seconds: Duration of data generation in seconds.

        Returns:
            float: Total amount of data generated in Gigabits.

        Note:
            For instruments with per_observation_gb set, that amount is added once.
            For instruments with rate_gbps, the amount is calculated based on duration.
        """
        total_data = 0.0
        for instrument in self.instruments:
            total_data += instrument.data_generation.data_generated(duration_seconds)
        return total_data
