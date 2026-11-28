from collections.abc import Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt
from pydantic import Field, PrivateAttr, field_validator, model_validator

from ._base import ConfigModel
from .activity_attitude import ActivityAttitude

_AXIS_TOLERANCE = 1e-9
_DIMENSIONS = 3
_RANK_TOLERANCE = 1e-9


class ThrusterDumpConfig(ConfigModel):
    """A thruster momentum dump: a hold during which the wheels are unloaded."""

    attitude: ActivityAttitude = Field(default_factory=ActivityAttitude)

    @field_validator("attitude")
    @classmethod
    def _firing_attitude(cls, value: ActivityAttitude) -> ActivityAttitude:
        if value.kind == "power_positive":
            raise ValueError("a firing attitude must be inherited or fixed")
        return value

    duration_seconds: float = Field(
        default=120.0,
        gt=0.0,
        allow_inf_nan=False,
        description="How long a dump holds the spacecraft, in seconds",
    )
    residual_fraction: float = Field(
        default=0.0,
        ge=0.0,
        lt=1.0,
        allow_inf_nan=False,
        description=(
            "The fraction of the stored momentum a dump leaves in the wheels; "
            "0 unloads them fully"
        ),
    )


class MomentumWheel(ConfigModel):
    """A momentum (reaction) wheel: where it sits, the axis it spins about and
    the momentum it can hold along that axis."""

    name: str = Field(default="Wheel", description="The wheel's name")
    spin_axis: tuple[float, float, float] = Field(
        description=(
            "The wheel's spin axis in spacecraft body coordinates; normalised "
            "to a unit vector"
        ),
    )
    capacity_n_m_s: float = Field(
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "The greatest momentum the wheel holds along its spin axis, in "
            "either sense, in N m s"
        ),
    )
    position_m: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "The wheel's centre in spacecraft body coordinates, in m, for the "
            "3-D spacecraft plot; None places it inside the bus along its "
            "spin axis"
        ),
    )
    radius_m: float = Field(
        default=0.15,
        gt=0.0,
        allow_inf_nan=False,
        description="The wheel's radius, in m, for the 3-D spacecraft plot",
    )
    thickness_m: float = Field(
        default=0.06,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "The wheel's thickness along its spin axis, in m, for the 3-D "
            "spacecraft plot"
        ),
    )

    @field_validator("spin_axis")
    @classmethod
    def _normalise_axis(
        cls, value: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        axis = np.asarray(value, dtype=float)
        norm = float(np.linalg.norm(axis))
        if not np.isfinite(norm) or norm < _AXIS_TOLERANCE:
            raise ValueError("spin_axis must be a finite, non-zero vector")
        axis = axis / norm
        return float(axis[0]), float(axis[1]), float(axis[2])

    @field_validator("position_m")
    @classmethod
    def _validate_position(
        cls, value: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        if value is not None and not all(np.isfinite(x) for x in value):
            raise ValueError("position_m must contain finite values")
        return value


class WheelArray:
    """How a set of wheels shares the momentum they hold, and how much of
    their capacity it takes.

    The momentum the wheels hold together, in body coordinates, is shared
    among them by the capacity-weighted minimum-norm (pseudo-inverse) law:
    of all the ways to make it up along the spin axes, the one with the
    least sum of squared fractions of each wheel's capacity. With three
    wheels there is only one way; with more, the larger wheels take more.
    The set is at its capacity where any one wheel is at its own.
    """

    def __init__(self, wheels: Sequence[MomentumWheel]) -> None:
        axes = np.array([wheel.spin_axis for wheel in wheels], dtype=float)
        capacities = np.array([wheel.capacity_n_m_s for wheel in wheels], dtype=float)
        if axes.shape[0] < _DIMENSIONS:
            raise ValueError("the wheels' spin axes do not span three dimensions")
        weighted = axes.T * capacities**2
        gram = weighted @ axes
        rank = np.linalg.matrix_rank(gram, tol=_RANK_TOLERANCE * float(np.trace(gram)))
        if rank < _DIMENSIONS:
            raise ValueError("the wheels' spin axes do not span three dimensions")
        self.names: tuple[str, ...] = tuple(wheel.name for wheel in wheels)
        self.capacities_n_m_s: npt.NDArray[np.float64] = capacities
        self.distribution: npt.NDArray[np.float64] = weighted.T @ np.linalg.inv(gram)
        """Each wheel's momentum per unit of body momentum: one row a wheel."""
        self.fractions: npt.NDArray[np.float64] = (
            self.distribution / capacities[:, None]
        )
        """Each wheel's fraction of its capacity per unit of body momentum."""
        self.capacity_n_m_s: float = float(
            1.0 / np.linalg.norm(self.fractions, axis=1).max()
        )
        """The momentum the wheels hold whichever way it points: the least,
        over every direction, at which the first wheel reaches its capacity."""

    def wheel_momenta_n_m_s(
        self, momentum_body_n_m_s: npt.ArrayLike
    ) -> npt.NDArray[np.float64]:
        """Each wheel's momentum along its spin axis, in N m s."""
        return np.asarray(
            self.distribution @ np.asarray(momentum_body_n_m_s, dtype=float),
            dtype=np.float64,
        )

    def utilisation(self, momentum_body_n_m_s: npt.ArrayLike) -> float:
        """The fullest wheel's fraction of its capacity: 1 at the limit."""
        return float(
            np.abs(self.fractions @ np.asarray(momentum_body_n_m_s, dtype=float)).max()
        )

    def load_n_m_s(self, momentum_body_n_m_s: npt.ArrayLike) -> float:
        """The momentum to check against ``capacity_n_m_s``: the fullest
        wheel's fraction of its capacity, as that fraction of the set's.

        It is the momentum's norm along the wheels' weakest direction and
        less along any other, never more, so a level or capacity set on the
        norm is reached exactly when a wheel reaches that fraction of its
        own. It never changes by more than the momentum does.
        """
        return self.capacity_n_m_s * self.utilisation(momentum_body_n_m_s)


def wheel_array_capacity_n_m_s(wheels: Sequence[MomentumWheel]) -> float:
    """The momentum a set of wheels holds about every body axis, in N m s:
    the least, over every direction, at which the first wheel reaches its
    capacity as the wheels share it (see :class:`WheelArray`). Zero where
    the spin axes do not span three dimensions."""
    try:
        return WheelArray(wheels).capacity_n_m_s
    except ValueError:
        return 0.0


class AtmosphereConfig(ConfigModel):
    """The atmosphere aerodynamic drag is computed in."""

    model: Literal["table", "msis"] = Field(
        default="table",
        description=(
            "The density model: 'table', the built-in piecewise exponential "
            "profile by height for moderate solar activity; or 'msis', "
            "NRLMSIS 2.1 through the optional pymsis package, at the solar "
            "flux and geomagnetic activity below"
        ),
    )
    f107: float = Field(
        default=150.0,
        ge=0.0,
        allow_inf_nan=False,
        description="The daily 10.7 cm solar flux the 'msis' model assumes, in sfu",
    )
    f107a: float = Field(
        default=150.0,
        ge=0.0,
        allow_inf_nan=False,
        description="The 81-day average 10.7 cm solar flux the 'msis' model assumes",
    )
    ap: float = Field(
        default=4.0,
        ge=0.0,
        allow_inf_nan=False,
        description="The geomagnetic activity index the 'msis' model assumes",
    )
    density_scale: float = Field(
        default=1.0,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "A factor on the density, for sensitivity studies: the density "
            "varies by a factor of twenty between solar minimum and maximum"
        ),
    )


class StoredMomentumConfig(ConfigModel):
    """Configuration for planning-level stored-momentum tracking."""

    _wheel_array_cache: (
        tuple[
            tuple[tuple[str, tuple[float, float, float], float], ...], WheelArray | None
        ]
        | None
    ) = PrivateAttr(default=None)

    @property
    def capacity_enabled(self) -> bool:
        """Whether the capacity is enforced as a hard limit on the executed
        motion (see ``enforce_capacity``)."""
        return self.total_capacity_n_m_s is not None and self.enforce_capacity

    @property
    def wheel_capacity_n_m_s(self) -> float | None:
        """The momentum the configured wheels hold about every axis (see
        ``wheel_array_capacity_n_m_s``); None where no wheels are configured."""
        if not self.wheels:
            return None
        array = self.wheel_array
        return array.capacity_n_m_s if array is not None else 0.0

    @property
    def wheel_array(self) -> WheelArray | None:
        """How the configured wheels share momentum; None where there are no
        wheels or their spin axes do not span three dimensions."""
        if not self.wheels:
            return None
        # Nested wheel edits and in-place list changes bypass this model's
        # assignment validators. Key the cache by values to notice them too.
        key = tuple((w.name, w.spin_axis, w.capacity_n_m_s) for w in self.wheels)
        cached = self._wheel_array_cache
        if cached is not None and cached[0] == key:
            return cached[1]
        try:
            array = WheelArray(self.wheels)
        except ValueError:
            array = None
        self._wheel_array_cache = (key, array)
        return array

    @property
    def wheel_limits(self) -> WheelArray | None:
        """The wheels, where each one's own capacity is the limit: where
        there are wheels and ``capacity_n_m_s`` does not override them."""
        return self.wheel_array if self.capacity_n_m_s is None else None

    @property
    def total_capacity_n_m_s(self) -> float | None:
        """The total momentum capacity that tracking, dumps and enforcement
        use: ``capacity_n_m_s`` where it is given, otherwise what the
        configured wheels hold about every axis; None sets no capacity.
        Where it comes from the wheels, the momentum checked against it is
        the fullest wheel's share (``wheel_limits``), so a direction the
        wheels reach further in holds more."""
        if self.capacity_n_m_s is not None:
            return self.capacity_n_m_s
        return self.wheel_capacity_n_m_s

    enforce_capacity: bool = Field(
        default=False,
        description=(
            "Whether capacity_n_m_s is enforced as a hard limit on the executed "
            "motion: the momentum of every slew and hold, the body's own "
            "included, is certified against it before it is flown, an "
            "observation is admitted only at a roll and rate at which it fits "
            "with a return to the departure attitude (slew_rate_fractions, "
            "roll_step_deg), and a run that cannot continue within it fails. "
            "Off, the capacity is the red line thruster dumps keep the wheels "
            "under and fault management records. With thrusters too, the dumps "
            "empty the wheels at the planning level and enforcement forecasts "
            "on from what they hold"
        ),
    )
    slew_rate_fractions: tuple[float, ...] = Field(
        default=(1.0, 0.75, 0.5, 0.25, 0.125),
        description="Descending speed caps tried after a momentum violation; acceleration and settling are unchanged.",
    )
    roll_step_deg: int = Field(
        default=30,
        gt=0,
        le=360,
        description="Bounded alternative-roll search increment when momentum admission is enabled.",
    )

    @field_validator("slew_rate_fractions")
    @classmethod
    def _validate_speed_search(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        if (
            not values
            or values[0] != 1.0
            or not all(np.isfinite(x) and 0 < x <= 1 for x in values)
            or any(a <= b for a, b in zip(values, values[1:]))
        ):
            raise ValueError(
                "slew_rate_fractions must start at 1 and strictly descend within (0, 1]"
            )
        return values

    @field_validator("roll_step_deg")
    @classmethod
    def _validate_roll_step(cls, value: int) -> int:
        if 360 % value:
            raise ValueError("roll_step_deg must divide 360")
        return value

    gravity_gradient_enabled: bool = Field(
        default=False,
        description=(
            "Accumulate stored momentum required to reject central-body "
            "gravity-gradient torque."
        ),
    )
    solar_radiation_pressure_enabled: bool = Field(
        default=False,
        description=(
            "Accumulate stored momentum required to reject the torque of "
            "sunlight on the spacecraft's surfaces (spacecraft_bus.surfaces, "
            "or a placeholder model built from the bus, panels and telescopes)"
        ),
    )
    aerodynamic_drag_enabled: bool = Field(
        default=False,
        description=(
            "Accumulate stored momentum required to reject the torque of the "
            "atmosphere on the spacecraft's surfaces (spacecraft_bus.surfaces, "
            "or a placeholder model built from the bus, panels and telescopes), "
            "at the density the atmosphere model gives"
        ),
    )
    atmosphere: AtmosphereConfig = Field(
        default_factory=AtmosphereConfig,
        description="The atmosphere drag is computed in",
    )
    geomagnetic_field: Literal["dipole", "igrf"] = Field(
        default="dipole",
        description=(
            "The field the magnetorquers unload against and the residual dipole "
            "is torqued by: 'dipole', the built-in tilted dipole; or 'igrf', the "
            "IGRF through the optional ppigrf package"
        ),
    )
    max_sample_interval_s: float = Field(
        default=10.0,
        gt=0.0,
        le=10.0,
        allow_inf_nan=False,
        description=(
            "Where the capacity is enforced, the longest interval the executor's "
            "budget integrates over, in seconds, further limited by "
            "max_attitude_step_deg at the fastest configured slew rate; the "
            "ephemeris must be at least as fine. The run's tracker samples at "
            "the ephemeris resolution regardless"
        ),
    )
    max_attitude_step_deg: float = Field(
        default=5.0,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "How far the attitude may move between momentum samples along a "
            "slew, in degrees. The simulations and the planners' forecast sample "
            "each slew at this step over the fastest configured slew rate, "
            "whatever the ephemeris resolution; a held attitude is sampled at "
            "the ephemeris resolution. Lower for convergence studies. This is "
            "not an integration-error guarantee."
        ),
    )
    initial_momentum_body_n_m_s: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description=(
            "Initial stored-momentum vector in spacecraft body coordinates, in N m s."
        ),
    )

    magnetorquer_max_dipole_a_m2: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "The magnetorquers' greatest dipole, in A m²; they unload the wheels "
            "against the geomagnetic field (a tilted dipole) continuously, taking "
            "out the stored momentum across the field. None: no magnetorquers"
        ),
    )
    magnetorquer_time_constant_s: float = Field(
        default=1800.0,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "How quickly the magnetorquers aim to unload the stored momentum, "
            "in seconds, within their dipole"
        ),
    )
    residual_dipole_body_a_m2: tuple[float, float, float] | None = Field(
        default=None,
        description=(
            "The spacecraft's residual magnetic dipole in body coordinates, in "
            "A m², torqued by the geomagnetic field; None: none"
        ),
    )

    capacity_n_m_s: float | None = Field(
        default=None,
        gt=0.0,
        allow_inf_nan=False,
        description=(
            "The wheels' total momentum capacity, in N m s, taken as the same "
            "about every axis. None takes the limit from the configured "
            "wheels, each held within its own capacity, and sets no capacity "
            "where there are none. Given alongside wheels, it replaces their "
            "limits with this one figure"
        ),
    )
    wheels: list[MomentumWheel] = Field(
        default_factory=list,
        description=(
            "The momentum wheels: each one's spin axis, capacity and place in "
            "the body. Where capacity_n_m_s is None the momentum is shared "
            "among them and each is held within its own capacity, so the "
            "limit depends on the momentum's direction. Each wheel's momentum "
            "is reported in the telemetry, and they are drawn on the 3-D "
            "spacecraft plot"
        ),
    )
    planning_level: float = Field(
        default=0.8,
        gt=0.0,
        le=1.0,
        allow_inf_nan=False,
        description=(
            "The fraction of the capacity at which a thruster dump is called "
            "for, leaving the rest as margin"
        ),
    )
    thruster_dump: ThrusterDumpConfig | None = Field(
        default=None,
        description=(
            "Thruster dumps: dispatch begins one when the stored momentum reaches "
            "the planning level of the capacity, holding the attitude it has for "
            "the dump's length and starting no observation meanwhile; None: no "
            "thrusters, so the wheels fill until a run reports it"
        ),
    )

    slew_momentum_enabled: bool = Field(
        default=False,
        description=(
            "Whether the momentum the body carries while slewing, I·ω, counts "
            "against the capacity. On, capacity_n_m_s is the wheels' whole "
            "capacity: a slew is admitted against the room the stored momentum "
            "leaves, flown slower (slew_rate_fractions) where it does not fit, "
            "and the wheels are emptied first where no rate fits. Off, the "
            "capacity is the room left for stored momentum at the configured "
            "slew rates. Needs capacity_n_m_s"
        ),
    )

    @model_validator(mode="after")
    def _dumps_need_a_capacity(self) -> "StoredMomentumConfig":
        if self.wheels and self.capacity_n_m_s is None and self.wheel_array is None:
            raise ValueError(
                "the wheels' spin axes do not span three dimensions, so they "
                "hold no momentum about some axis: add wheels or set "
                "capacity_n_m_s"
            )
        capacity = self.total_capacity_n_m_s
        if self.thruster_dump is not None and capacity is None:
            raise ValueError(
                "thruster dumps need capacity_n_m_s or wheels to be triggered at"
            )
        if self.slew_momentum_enabled and capacity is None:
            raise ValueError(
                "slew momentum needs capacity_n_m_s or wheels to be admitted against"
            )
        return self

    @property
    def dump_level_n_m_s(self) -> float | None:
        """The stored momentum at which a thruster dump is called for."""
        capacity = self.total_capacity_n_m_s
        if capacity is None or self.thruster_dump is None:
            return None
        return self.planning_level * capacity

    @property
    def enabled(self) -> bool:
        """Whether momentum is tracked: any torque source, or the momentum of
        slews alone."""
        return (
            self.gravity_gradient_enabled
            or self.solar_radiation_pressure_enabled
            or self.aerodynamic_drag_enabled
            or self.magnetorquer_max_dipole_a_m2 is not None
            or self.residual_dipole_body_a_m2 is not None
            or self.slew_momentum_enabled
            or self.capacity_enabled
        )

    @field_validator("initial_momentum_body_n_m_s", "residual_dipole_body_a_m2")
    @classmethod
    def _validate_vector(
        cls, value: tuple[float, float, float] | None
    ) -> tuple[float, float, float] | None:
        if value is None:
            return None
        if not all(np.isfinite(component) for component in value):
            raise ValueError("a body vector must contain finite values")
        return float(value[0]), float(value[1]), float(value[2])
