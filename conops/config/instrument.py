from collections.abc import Mapping
from enum import Enum
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import (
    AliasChoices,
    ConfigDict,
    Field,
    computed_field,
    field_validator,
    model_validator,
)
from rust_ephem.constraints import ConstraintConfig

from ..common.enums import ACSMode
from ..common.vector import (
    mounted_attitude_to_quat,
    mounting_quaternion_from_boresight,
    normal_to_boresight_offset_euler_deg,
    quat_to_attitude,
    quaternion_rotate_vector,
)
from ._base import ConfigModel
from .constraint import Constraint
from .data_generator import DataGeneration, ScienceData
from .observation_timing import ObservationTiming
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


IDLE_MODE = "IDLE"
"""The mode an instrument with modes is in when no observation uses it."""
SCIENCE_MODE = "SCIENCE"
"""The mode an instrument with modes observes in unless a request names another."""


class InstrumentMode(ConfigModel):
    """One operating mode of an instrument, with its power draw."""

    name: str = Field(description="Mode name, such as IDLE or SCIENCE")
    power_w: float = Field(ge=0.0, description="Power draw in the mode, in watts")
    eclipse_power_w: float | None = Field(
        default=None,
        ge=0.0,
        description="Power draw in the mode during eclipse, in watts; None: power_w",
    )
    science_data: ScienceData | None = Field(
        default=None,
        description=(
            "Science data the instrument generates while collecting in this "
            "mode; None: its data_generation's science data"
        ),
    )
    observing: bool = Field(
        default=True,
        description=(
            "Whether the instrument counts as observing in this mode for the "
            "payload's concurrency rules; False for a standby mode that draws "
            "its power without observing. IDLE never observes"
        ),
    )

    def power(self, in_eclipse: bool = False) -> float:
        """The mode's draw in watts."""
        if in_eclipse and self.eclipse_power_w is not None:
            return self.eclipse_power_w
        return self.power_w


class Instrument(ConfigModel):
    """A spacecraft instrument with power consumption and data generation characteristics.

    Attributes:
        instrument_type: Discriminator tag for polymorphic payload deserialization.
        name: Instrument name/identifier.
        power_draw: Power draw characteristics (nominal, peak, mode-specific).
        modes: Operating modes with their power draws. With modes, the
            instrument draws its mode's power: the mode an observation names
            for it, ``SCIENCE`` when the observation uses it without naming
            one, a mode named like the ACS mode if the instrument defines
            one, and ``IDLE`` otherwise; ``power_draw`` is then unused.
            Without modes it draws ``power_draw`` by ACS mode, as before.
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
    modes: list[InstrumentMode] = Field(
        default_factory=list,
        description=(
            "Operating modes with their power draws; at least IDLE and SCIENCE "
            "when any are given. Empty: power_draw by ACS mode applies"
        ),
    )
    heater: Heater | None = Field(
        default=None, description="Optional heater system for the instrument"
    )
    data_generation: DataGeneration = Field(
        default_factory=DataGeneration, description="Data generation characteristics"
    )

    @field_validator("modes")
    @classmethod
    def _modes_are_named_once_with_idle_and_science(
        cls, modes: list[InstrumentMode]
    ) -> list[InstrumentMode]:
        if not modes:
            return modes
        names = [mode.name for mode in modes]
        if len(set(names)) != len(names):
            raise ValueError("instrument modes must have distinct names")
        for required in (IDLE_MODE, SCIENCE_MODE):
            if required not in names:
                raise ValueError(f"instrument modes must include {required}")
        for mode in modes:
            if mode.name == IDLE_MODE and mode.observing:
                # Idle is, by definition, not observing.
                mode.observing = False
        return modes

    def mode(self, name: str) -> InstrumentMode | None:
        """The mode called ``name``, if the instrument defines it."""
        return next((mode for mode in self.modes if mode.name == name), None)

    def mode_in(self, acs_mode: int | None, instrument_mode: str | None) -> str:
        """The mode the instrument is in: ``instrument_mode``, named for it
        by the observation; else, outside science, a mode named like the ACS
        mode, if the instrument defines one; else ``IDLE``. During science an
        instrument the observation does not name idles."""
        if instrument_mode is not None:
            return instrument_mode
        if acs_mode is not None:
            try:
                named = ACSMode(acs_mode).name
            except ValueError:
                named = None
            if (
                named is not None
                and named != SCIENCE_MODE
                and self.mode(named) is not None
            ):
                return named
        return IDLE_MODE

    def power(
        self,
        mode: int | None = None,
        in_eclipse: bool = False,
        instrument_mode: str | None = None,
    ) -> float:
        """Get the power draw for the instrument in the given mode.

        Args:
            mode: ACS mode (None for nominal)
            in_eclipse: Whether spacecraft is in eclipse
            instrument_mode: The instrument's own mode, named for it by the
                observation in progress; None when no observation uses it

        Returns:
            Total power draw in watts
        """
        if self.modes:
            name = self.mode_in(mode, instrument_mode)
            own = self.mode(name)
            if own is None:
                raise ValueError(f"instrument {self.name!r} has no mode {name!r}")
            base_power = own.power(in_eclipse)
        else:
            base_power = self.power_draw.power(mode, in_eclipse=in_eclipse)
        heater_power = (
            self.heater.power(mode, in_eclipse=in_eclipse) if self.heater else 0.0
        )
        return base_power + heater_power


class InstrumentMounting(ConfigModel):
    """Rigid transform from an instrument frame into spacecraft body axes.

    Instrument ``+X`` is its science boresight and instrument ``+Z`` defines
    zero-roll clocking. The quaternion uses scalar-first Hamilton convention
    and rotates instrument-frame vectors into the physical body frame.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    body_from_instrument_quaternion_wxyz: tuple[float, float, float, float] = Field(
        default=(1.0, 0.0, 0.0, 0.0),
        description=(
            "Unit Hamilton quaternion [w, x, y, z] mapping instrument-frame "
            "vectors into the physical spacecraft body frame"
        ),
    )

    @field_validator("body_from_instrument_quaternion_wxyz")
    @classmethod
    def _normalize_quaternion(
        cls, value: tuple[float, float, float, float]
    ) -> tuple[float, float, float, float]:
        quaternion = np.asarray(value, dtype=np.float64)
        if quaternion.shape != (4,) or not np.all(np.isfinite(quaternion)):
            raise ValueError("mounting quaternion must contain four finite values")
        norm = float(np.linalg.norm(quaternion))
        if norm < 1e-12:
            raise ValueError("mounting quaternion must have nonzero norm")
        quaternion /= norm
        if quaternion[0] < 0.0:
            quaternion *= -1.0
        return (
            float(quaternion[0]),
            float(quaternion[1]),
            float(quaternion[2]),
            float(quaternion[3]),
        )

    def body_vector(
        self, instrument_vector: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Express an instrument-frame vector in physical body coordinates."""
        vector = quaternion_rotate_vector(
            self.body_from_instrument_quaternion_wxyz,
            instrument_vector,
        )
        return float(vector[0]), float(vector[1]), float(vector[2])

    def instrument_vector(
        self, body_vector: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        """Express a body-frame vector in instrument coordinates."""
        quaternion = np.asarray(
            self.body_from_instrument_quaternion_wxyz, dtype=np.float64
        ).copy()
        quaternion[1:] *= -1.0
        vector = quaternion_rotate_vector(quaternion, body_vector)
        return float(vector[0]), float(vector[1]), float(vector[2])

    @property
    def boresight_body(self) -> tuple[float, float, float]:
        vector = np.asarray(self.body_vector((1.0, 0.0, 0.0)))
        vector[np.abs(vector) < 1e-15] = 0.0
        return float(vector[0]), float(vector[1]), float(vector[2])

    @property
    def roll_reference_body(self) -> tuple[float, float, float]:
        return self.body_vector((0.0, 0.0, 1.0))

    @property
    def is_identity(self) -> bool:
        return bool(
            np.allclose(
                self.body_from_instrument_quaternion_wxyz,
                (1.0, 0.0, 0.0, 0.0),
                atol=1e-12,
            )
        )

    def body_attitude(
        self, ra_deg: float, dec_deg: float, roll_deg: float
    ) -> tuple[float, float, float]:
        """Return physical body +X RA/Dec/roll for a target instrument attitude."""
        return quat_to_attitude(
            mounted_attitude_to_quat(
                ra_deg,
                dec_deg,
                roll_deg,
                self.body_from_instrument_quaternion_wxyz,
            )
        )


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
        mounting: Authoritative instrument-to-body rotation. The derived
            ``boresight`` property is its instrument ``+X`` axis expressed in
            spacecraft body coordinates. Body-frame axes are:

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
    mounting: InstrumentMounting = Field(
        default_factory=InstrumentMounting,
        description=(
            "Rigid instrument-to-body mounting. When omitted, a deterministic "
            "mount is derived from the legacy boresight field."
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

    @model_validator(mode="before")
    @classmethod
    def _resolve_mounting_and_legacy_boresight(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value

        data = dict(value)
        raw_mounting = data.get("mounting")
        raw_boresight = data.pop("boresight", None)
        if raw_mounting is None:
            if raw_boresight is not None:
                data["mounting"] = cls._mounting_from_boresight(raw_boresight)
            return data

        mounting = (
            raw_mounting
            if isinstance(raw_mounting, InstrumentMounting)
            else InstrumentMounting.model_validate(raw_mounting)
        )
        derived_boresight = mounting.boresight_body
        if raw_boresight is not None:
            legacy_mounting = cls._mounting_from_boresight(raw_boresight)
            if not np.allclose(
                legacy_mounting.boresight_body, derived_boresight, atol=1e-10
            ):
                raise ValueError(
                    "telescope boresight is inconsistent with its mounting quaternion"
                )
        data["mounting"] = mounting
        return data

    @classmethod
    def _mounting_from_boresight(cls, value: Any) -> InstrumentMounting:
        boresight = np.asarray(value, dtype=np.float64)
        if boresight.shape != (3,) or not np.all(np.isfinite(boresight)):
            raise ValueError("Telescope boresight must contain three finite values")
        magnitude = float(np.linalg.norm(boresight))
        if magnitude < 0.99 or magnitude > 1.01:
            raise ValueError(
                f"Telescope boresight must be a unit vector (magnitude {magnitude:.4f})"
            )
        normalized = boresight / magnitude
        return InstrumentMounting(
            body_from_instrument_quaternion_wxyz=tuple(
                mounting_quaternion_from_boresight(normalized)
            )
        )

    @computed_field(  # type: ignore[prop-decorator]
        description=(
            "Telescope boresight unit vector in spacecraft body coordinates, "
            "derived from mounting"
        )
    )
    @property
    def boresight(self) -> tuple[float, float, float]:
        """Return instrument +X expressed in spacecraft body coordinates."""
        return self.mounting.boresight_body

    @boresight.setter
    def boresight(self, value: tuple[float, float, float]) -> None:
        """Update the mount from a legacy boresight-only specification."""
        self.mounting = self._mounting_from_boresight(value)

    @field_validator("constraint")
    @classmethod
    def _require_roll_independent_local_constraint(
        cls, value: Constraint | None
    ) -> Constraint | None:
        if (
            value is not None
            and value.constraint is not None
            and cls._contains_local_roll_semantics(value.constraint)
        ):
            raise ValueError(
                "telescope-local constraints must be roll-independent; define "
                "body-mounted roll-dependent constraints on the spacecraft instead"
            )
        return value

    @staticmethod
    def _contains_local_roll_semantics(constraint: ConstraintConfig) -> bool:
        """Return whether a local constraint needs instrument-frame roll."""
        data = constraint.model_dump(mode="json")

        def contains(node: Any) -> bool:
            if not isinstance(node, dict):
                return False
            node_type = node.get("type")
            if node_type in {"boresight_offset", "solar_roll"}:
                return True
            if node_type in {"bright_star", "body"}:
                return node.get("fov_polygon") is not None
            if node_type in {"and", "or", "xor", "at_least"}:
                return any(contains(child) for child in node.get("constraints", []))
            if node_type == "not":
                return contains(node.get("constraint"))
            return False

        return contains(data)

    def target_body_attitude(
        self, ra_deg: float, dec_deg: float, roll_deg: float
    ) -> tuple[float, float, float]:
        """Return physical body +X RA/Dec/roll for a science target."""
        return self.mounting.body_attitude(ra_deg, dec_deg, roll_deg)

    @property
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
        if np.allclose(self.boresight, (1.0, 0.0, 0.0), atol=1e-12):
            return base
        roll_deg, pitch_deg, yaw_deg = normal_to_boresight_offset_euler_deg(
            self.boresight
        )
        return base.boresight_offset(
            roll_deg=roll_deg,
            pitch_deg=pitch_deg,
            yaw_deg=yaw_deg,
        )


# Discriminated union used by Payload.instruments for round-trip JSON serialization.
# Telescope must come first so Pydantic tries the more specific type first when
# the discriminator tag is present.
_AnyInstrument = Annotated[
    Telescope | Instrument,
    Field(discriminator="instrument_type"),
]


class ConcurrencyRules(ConfigModel):
    """What may observe at once: rules on the instruments an observation
    uses, checked against each request's instrument modes. An instrument
    observes when a request names it in a mode that observes: any mode but
    ``IDLE``, unless the mode's ``observing`` flag says otherwise."""

    max_observing: int | None = Field(
        default=None,
        ge=1,
        description="The most instruments one observation may use; None: any number",
    )
    required_guider: str | None = Field(
        default=None,
        description=(
            "An instrument every observation that uses any instrument must "
            "also use, such as a guider; None: none required"
        ),
    )
    exclusive: list[tuple[str, str]] = Field(
        default_factory=list,
        description="Pairs of instruments one observation may not use together",
    )

    @property
    def active(self) -> bool:
        """Whether any rule is set."""
        return (
            self.max_observing is not None
            or self.required_guider is not None
            or bool(self.exclusive)
        )

    def names(self) -> set[str]:
        """Every instrument the rules name."""
        named = {a for pair in self.exclusive for a in pair}
        if self.required_guider is not None:
            named.add(self.required_guider)
        return named


class Payload(ConfigModel):
    """A collection of instruments operated together as the spacecraft payload.

    Attributes:
        instruments: List of Instrument (or subclass) instances.
        concurrency: Rules on what may observe at once.
    """

    instruments: list[_AnyInstrument] = [Instrument()]
    observation_timing: ObservationTiming = Field(
        default_factory=ObservationTiming,
        description="Non-collection setup, cleanup, and handoff budgets for science tasks",
    )
    concurrency: ConcurrencyRules = Field(
        default_factory=ConcurrencyRules,
        description="Rules on the instruments one observation may use at once",
    )

    @model_validator(mode="after")
    def _rules_name_instruments(self) -> "Payload":
        instruments = getattr(self, "instruments", None)
        rules = getattr(self, "concurrency", None)
        # A test double standing in for a payload has neither.
        if not isinstance(instruments, list) or not isinstance(rules, ConcurrencyRules):
            return self
        names = {instrument.name for instrument in instruments}
        unknown = sorted(rules.names() - names)
        if unknown:
            raise ValueError(
                f"concurrency rules name instruments the payload lacks: {unknown}"
            )
        return self

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
                    "boresight" in item
                    or "mounting" in item
                    or "optics" in item
                    or "config" in item
                )
                item = {
                    **item,
                    "instrument_type": "Telescope"
                    if has_telescope_keys
                    else "Instrument",
                }
            result.append(item)
        return result

    def power(
        self,
        mode: int | None = None,
        in_eclipse: bool = False,
        instrument_modes: Mapping[str, str] | None = None,
    ) -> float:
        """Get the total power draw for all instruments in the payload in the given mode.

        Args:
            mode: ACS mode (None for nominal)
            in_eclipse: Whether spacecraft is in eclipse
            instrument_modes: The mode each instrument is in during the
                observation in progress, by name; instruments not named,
                or all of them when None, are in no observation

        Returns:
            Total power draw in watts
        """
        return sum(
            instrument.power(
                mode,
                in_eclipse=in_eclipse,
                instrument_mode=(
                    instrument_modes.get(instrument.name)
                    if instrument_modes is not None
                    else None
                ),
            )
            for instrument in self.instruments
        )

    def check_instrument_modes(self, instrument_modes: Mapping[str, str]) -> None:
        """Raise if a mode names an instrument the payload lacks, or a mode
        the instrument does not define; an instrument without modes may be
        named in ``IDLE`` or ``SCIENCE`` only."""
        by_name = {instrument.name: instrument for instrument in self.instruments}
        for name, mode in instrument_modes.items():
            instrument = by_name.get(name)
            if instrument is None:
                raise ValueError(f"no instrument {name!r} in the payload")
            if instrument.modes:
                if instrument.mode(mode) is None:
                    raise ValueError(f"instrument {name!r} has no mode {mode!r}")
            elif mode not in (IDLE_MODE, SCIENCE_MODE):
                raise ValueError(
                    f"instrument {name!r} has no modes; {mode!r} is not one of "
                    f"{IDLE_MODE} and {SCIENCE_MODE}"
                )

    def instruments_with_modes(self) -> list[str]:
        """The names of the instruments that define modes."""
        return [instrument.name for instrument in self.instruments if instrument.modes]

    def default_instrument_modes(self) -> dict[str, str]:
        """What a science observation naming no modes uses: every instrument
        with modes, in ``SCIENCE``."""
        return dict.fromkeys(self.instruments_with_modes(), SCIENCE_MODE)

    def observes_in(self, instrument: str, mode: str) -> bool:
        """Whether ``instrument`` counts as observing in ``mode``: any mode
        but ``IDLE`` whose ``observing`` flag is set, which it is unless the
        mode says otherwise; for an instrument without modes, any mode but
        ``IDLE``."""
        if mode == IDLE_MODE:
            return False
        for candidate in self.instruments:
            if candidate.name == instrument:
                own = candidate.mode(mode)
                return own.observing if own is not None else True
        return True

    def check_concurrency(
        self, instrument_modes: Mapping[str, str], *, pointing: bool = True
    ) -> None:
        """Raise if the instruments ``instrument_modes`` has observing break
        the payload's concurrency rules. The required guider applies only to
        an observation that points (``pointing``), not to a scheduled
        activity, which holds the attitude it has."""
        rules = self.concurrency
        if not rules.active:
            return
        observing = sorted(
            name
            for name, mode in instrument_modes.items()
            if self.observes_in(name, mode)
        )
        if rules.max_observing is not None and len(observing) > rules.max_observing:
            raise ValueError(
                f"{len(observing)} instruments observe at once ({observing}); "
                f"the payload allows {rules.max_observing}"
            )
        if (
            pointing
            and rules.required_guider is not None
            and observing
            and rules.required_guider not in observing
        ):
            raise ValueError(
                f"an observation using {observing} must also use "
                f"{rules.required_guider!r}"
            )
        for first, second in rules.exclusive:
            if first in observing and second in observing:
                raise ValueError(f"{first!r} and {second!r} cannot observe together")

    def check_request(self, instrument_modes: Mapping[str, str]) -> None:
        """Check an observation's instrument modes: the instruments and
        modes exist, and what they have observing is allowed at once. An
        empty mapping is checked as the default, every instrument with
        modes in ``SCIENCE``."""
        self.check_instrument_modes(instrument_modes)
        self.check_concurrency(
            instrument_modes if instrument_modes else self.default_instrument_modes()
        )

    def total_data_rate_gbps(self) -> float:
        """Get the total data generation rate across all instruments.

        Returns:
            float: Total data generation rate in Gigabits per second.
        """
        return sum(
            instrument.data_generation.science.rate_gbps
            for instrument in self.instruments
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

    def telescope_for_target(
        self, instrument_name: str | None = None
    ) -> Instrument | None:
        """Resolve the instrument that observes one target.

        A name identifies exactly one configured instrument, a telescope or
        not; a name that does not is an error. Without a name, the payload's
        single telescope, or None where it has none or several, which keeps
        the body-+X target behaviour. A plain instrument has no mounting of
        its own, so a target naming one points the body boresight at it.
        """
        if instrument_name is not None:
            matches = [
                instrument
                for instrument in self.instruments
                if instrument.name == instrument_name
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"target instrument {instrument_name!r} does not identify "
                    "exactly one configured instrument"
                )
            return matches[0]
        telescopes = [
            instrument
            for instrument in self.instruments
            if isinstance(instrument, Telescope)
        ]
        if len(telescopes) > 1:
            return None
        return telescopes[0] if telescopes else None

    def housekeeping_generated(
        self, duration_seconds: float, *, mode: ACSMode | None = None
    ) -> float:
        """Housekeeping data generated by all instruments over an interval.

        Args:
            duration_seconds: Length of the interval in seconds.
            mode: ACS mode during the interval; None for nominal rates.

        Returns:
            float: Data generated in Gigabits.
        """
        return sum(
            instrument.data_generation.housekeeping.data_generated(
                duration_seconds, mode=mode
            )
            for instrument in self.instruments
        )

    def science_generated(
        self,
        collection_seconds: float,
        science: Mapping[str, ScienceData] | None = None,
        instrument_modes: Mapping[str, str] | None = None,
    ) -> float:
        """Science data an observation generates in all instruments.

        Each instrument's rate for the seconds collected, plus its fixed
        amount; see :meth:`ScienceData.data_generated`.

        Args:
            collection_seconds: Seconds of science the observation collected.
            science: The observed target's science data, keyed by instrument
                name; instruments not named generate their own.
            instrument_modes: The mode each instrument is in during the
                observation, by name. An instrument with modes that is not
                named generates nothing, and one that is generates its
                mode's science data, or its own where the mode sets none.
                None: every instrument generates its own, as an instrument
                without modes always does.

        Returns:
            float: Data generated in Gigabits.
        """
        science = science or {}
        return sum(
            data.data_generated(collection_seconds)
            for instrument in self.instruments
            if (data := self._science_of(instrument, science, instrument_modes))
            is not None
        )

    @staticmethod
    def _science_of(
        instrument: Instrument,
        science: Mapping[str, ScienceData],
        instrument_modes: Mapping[str, str] | None,
    ) -> ScienceData | None:
        """What ``instrument`` generates while the observation collects:
        the target's own for it, else its mode's, else its default; None
        when it has modes and the observation does not use it."""
        own = science.get(instrument.name)
        if own is not None:
            return own
        if instrument.modes and instrument_modes is not None:
            name = instrument_modes.get(instrument.name)
            if name is None:
                return None
            mode = instrument.mode(name)
            if mode is not None and mode.science_data is not None:
                return mode.science_data
        return instrument.data_generation.science

    def data_generated(
        self,
        duration_seconds: float,
        *,
        mode: ACSMode | None = None,
        collection_seconds: float | None = None,
        science: Mapping[str, ScienceData] | None = None,
        instrument_modes: Mapping[str, str] | None = None,
    ) -> float:
        """Data an observation generates in all instruments over an interval.

        Housekeeping data for the interval, plus science data for the seconds
        collected.

        Args:
            duration_seconds: Length of the interval in seconds.
            mode: ACS mode during the interval; None for nominal rates.
            collection_seconds: Seconds of science collected in the interval;
                None for all of it.
            science: The observed target's science data, keyed by instrument
                name; instruments not named generate their own.
            instrument_modes: The mode each instrument is in during the
                observation, by name; see :meth:`science_generated`.

        Returns:
            float: Total amount of data generated in Gigabits.
        """
        if collection_seconds is None:
            collection_seconds = duration_seconds
        return self.housekeeping_generated(
            duration_seconds, mode=mode
        ) + self.science_generated(collection_seconds, science, instrument_modes)
