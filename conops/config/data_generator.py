from collections.abc import Mapping

from pydantic import Field, field_validator, model_validator

from ..common.enums import ACSMode
from ._base import ConfigModel


class ScienceData(ConfigModel):
    """
    Science data: what an instrument generates while it collects science.

    It depends on what is observed, so an instrument's
    :class:`DataGeneration` sets what it generates by default, and a target
    can set its own for each instrument that observes it.

    Attributes:
        rate_gbps (float): Data generation rate in Gigabits per second while
            collecting. Defaults to 0.0.
        per_observation_gb (float): Fixed data an observation generates in
            Gb, such as a frame readout or headers, whatever its length.
            Defaults to 0.0.

    Example:
        >>> # 0.1 Gbps while collecting, plus 2 Gb for each observation
        >>> ScienceData(rate_gbps=0.1, per_observation_gb=2.0).data_generated(60)
        8.0
    """

    rate_gbps: float = Field(
        default=0.0, ge=0.0, description="Data generation rate in Gbps while collecting"
    )
    per_observation_gb: float = Field(
        default=0.0,
        ge=0.0,
        description="Fixed data generated once per observation in Gb",
    )

    def data_generated(self, collection_seconds: float) -> float:
        """Science data an observation generates, having collected for some seconds.

        The rate for the seconds collected, plus the fixed amount.

        Args:
            collection_seconds: Seconds of science the observation collected.

        Returns:
            float: Data generated in Gigabits.
        """
        return self.rate_gbps * collection_seconds + self.per_observation_gb


class HousekeepingData(ConfigModel):
    """
    Housekeeping data: engineering telemetry such as temperatures, voltages,
    attitude and status, generated at all times whatever is observed.

    The spacecraft bus generates only this; an instrument generates it as
    well as science data (see :class:`DataGeneration`). The rate can depend on
    the ACS mode, the way :class:`~conops.config.PowerDraw` does for power.

    Attributes:
        rate_gbps (float): Data generation rate in Gigabits per second.
            Defaults to 0.0.
        rate_mode (dict[ACSMode, float]): Rate in Gbps in particular ACS modes,
            replacing ``rate_gbps`` there. Modes may be given by name or number.

    Example:
        >>> # 1 Mbps of housekeeping, 4 Mbps of diagnostics in safe mode
        >>> housekeeping = HousekeepingData(rate_gbps=0.001, rate_mode={"SAFE": 0.004})
        >>> housekeeping.data_generated(60, mode=ACSMode.SAFE)
        0.24
    """

    rate_gbps: float = Field(
        default=0.0, ge=0.0, description="Data generation rate in Gbps at all times"
    )
    rate_mode: dict[ACSMode, float] = Field(
        default_factory=dict,
        description="Data generation rate in Gbps in particular ACS modes",
    )

    @field_validator("rate_mode", mode="before")
    @classmethod
    def _mode_names(cls, value: object) -> object:
        """Accept ACS modes by name as well as number."""
        if not isinstance(value, Mapping):
            return value
        return {
            ACSMode[key]
            if isinstance(key, str) and key in ACSMode.__members__
            else key: rate
            for key, rate in value.items()
        }

    @field_validator("rate_mode")
    @classmethod
    def _rates_not_negative(cls, value: dict[ACSMode, float]) -> dict[ACSMode, float]:
        if any(rate < 0 for rate in value.values()):
            raise ValueError("rates must not be negative")
        return value

    def rate(self, mode: ACSMode | None = None) -> float:
        """Data generation rate in Gbps in an ACS mode.

        Args:
            mode: ACS mode, or None for the nominal rate.
        """
        if mode is None:
            return self.rate_gbps
        return self.rate_mode.get(mode, self.rate_gbps)

    def data_generated(
        self, duration_seconds: float, *, mode: ACSMode | None = None
    ) -> float:
        """Housekeeping data generated over an interval.

        Args:
            duration_seconds: Length of the interval in seconds.
            mode: ACS mode during the interval; None for the nominal rate.

        Returns:
            float: Data generated in Gigabits.
        """
        return self.rate(mode) * duration_seconds


class DataGeneration(ConfigModel):
    """
    The data an instrument generates: science data while collecting, and
    housekeeping data at all times.

    Attributes:
        science (ScienceData): What the instrument generates while collecting
            science, unless the observed target sets its own.
        housekeeping (HousekeepingData): What it generates whatever it does,
            by ACS mode.

    Example:
        >>> # 0.5 Gbps while collecting; 1 Mbps of housekeeping, 10 Mbps in safe mode
        >>> data = DataGeneration(
        ...     science=ScienceData(rate_gbps=0.5),
        ...     housekeeping=HousekeepingData(rate_gbps=0.001, rate_mode={"SAFE": 0.01}),
        ... )
        >>> data.data_generated(60)  # an observation collecting for 60 s
        30.06

    As a shorthand, the science fields may be given directly:
    ``DataGeneration(rate_gbps=0.5)`` is ``DataGeneration(science=ScienceData(rate_gbps=0.5))``.
    """

    science: ScienceData = Field(
        default_factory=ScienceData,
        description="Data generated while collecting science",
    )
    housekeeping: HousekeepingData = Field(
        default_factory=HousekeepingData,
        description="Data generated at all times, by ACS mode",
    )

    @model_validator(mode="before")
    @classmethod
    def _science_fields_given_directly(cls, data: object) -> object:
        """Take science fields given directly as the science part."""
        if not isinstance(data, Mapping):
            return data
        direct = {k: v for k, v in data.items() if k in ScienceData.model_fields}
        if not direct:
            return data
        if "science" in data:
            raise ValueError(
                f"give {sorted(direct)} inside 'science', not alongside it"
            )
        rest = {k: v for k, v in data.items() if k not in direct}
        return {**rest, "science": direct}

    def data_generated(
        self,
        duration_seconds: float,
        *,
        mode: ACSMode | None = None,
        collection_seconds: float | None = None,
        science: ScienceData | None = None,
    ) -> float:
        """Data an observation generates over an interval.

        Housekeeping data for the interval, plus the observation's science
        data for the seconds it collected.

        Args:
            duration_seconds: Length of the interval in seconds.
            mode: ACS mode during the interval, which sets the housekeeping
                rate; None for the nominal rate.
            collection_seconds: Seconds of science collected in the interval;
                None for all of it.
            science: The observed target's science data for this instrument,
                in place of the instrument's own.

        Returns:
            float: Data generated in Gigabits.
        """
        if collection_seconds is None:
            collection_seconds = duration_seconds
        if science is None:
            science = self.science
        return self.housekeeping.data_generated(
            duration_seconds, mode=mode
        ) + science.data_generated(collection_seconds)
