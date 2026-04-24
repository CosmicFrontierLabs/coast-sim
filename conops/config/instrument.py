from enum import Enum

from pydantic import Field, model_validator

from ._base import ConfigModel
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
        name: Instrument name/identifier.
        power_draw: Power draw characteristics (nominal, peak, mode-specific).
        heater: Optional heater system.
        data_generation: Data generation characteristics.
    """

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
    """A telescope instrument with optical configuration.

    Extends Instrument with the optical parameters of a telescope.
    Telescope instances can be placed in Payload.instruments alongside
    any other Instrument subclass.

    Attributes:
        optics: Optical configuration (aperture, focal length, f-number,
            design type, tube length).

    Example:
        >>> scope = Telescope(
        ...     name="Primary Telescope",
        ...     optics=TelescopeConfig(
        ...         aperture_m=0.6,
        ...         focal_length_m=6.0,
        ...         telescope_type=TelescopeType.RITCHEY_CHRETIEN,
        ...         tube_length_m=1.2,
        ...     ),
        ... )
        >>> scope.optics.f_number
        10.0
    """

    name: str = Field(default="Telescope", description="Instrument name/identifier")
    optics: TelescopeConfig = Field(
        default_factory=TelescopeConfig,
        description="Optical configuration of the telescope",
    )


class Payload(ConfigModel):
    """A collection of instruments operated together as the spacecraft payload.

    Attributes:
        instruments: List of Instrument (or subclass) instances.
    """

    instruments: list[Instrument] = [Instrument()]

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
