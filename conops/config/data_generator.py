from pydantic import BaseModel, Field


class DataGeneration(BaseModel):
    """
    A model representing data generation characteristics for an instrument.

    This class defines how an instrument generates data during observations,
    either at a constant rate or per observation.

    Attributes:
        rate_gbps (float): Data generation rate in Gigabits per second when active.
            Defaults to 0.0 (no data generation).
        per_observation_gb (float): Fixed amount of data generated per observation in Gb.
            If non-zero, this takes precedence over rate_gbps. Defaults to 0.0.

    Example:
        >>> # Instrument that generates 0.1 Gbps continuously
        >>> data_gen = DataGeneration(rate_gbps=0.1)
        >>> # Instrument that generates 5 Gb per observation
        >>> data_gen2 = DataGeneration(per_observation_gb=5.0)
    """

    rate_gbps: float = Field(
        default=0.0, ge=0.0, description="Data generation rate in Gbps when active"
    )
    per_observation_gb: float = Field(
        default=0.0,
        ge=0.0,
        description="Fixed data generated per observation in Gb",
    )

    def data_generated(self, duration_seconds: float) -> float:
        """Calculate data generated over a given duration.

        Args:
            duration_seconds: Duration of data generation in seconds.

        Returns:
            float: Amount of data generated in Gigabits.

        Note:
            If per_observation_gb is set, it returns that value regardless of duration.
            Otherwise, returns rate_gbps * duration_seconds.
        """
        if self.per_observation_gb > 0:
            return self.per_observation_gb
        return self.rate_gbps * duration_seconds
