"""Non-collection time reserved within a science observation task."""

from pydantic import Field

from ._base import ConfigModel

OBSERVATION_TIME_FIELDS = (
    "observation_slew_seconds",
    "setup_seconds",
    "collection_seconds",
    "cleanup_seconds",
    "handoff_seconds",
)


class ObservationTiming(ConfigModel):
    """Payload timing budgets, separate from ACS motion and settling."""

    setup_seconds: float = Field(
        default=0.0,
        ge=0,
        allow_inf_nan=False,
        description="Generic observation preparation after slew and ACS settling, in seconds",
    )
    cleanup_seconds: float = Field(
        default=0.0,
        ge=0,
        allow_inf_nan=False,
        description="Cleanup after science collection, in seconds",
    )
    handoff_seconds: float = Field(
        default=0.0,
        ge=0,
        allow_inf_nan=False,
        description="Reserved margin after cleanup before the next task, in seconds",
    )

    @property
    def post_collection_seconds(self) -> float:
        return self.cleanup_seconds + self.handoff_seconds

    @property
    def total_seconds(self) -> float:
        return self.setup_seconds + self.post_collection_seconds

    def collection_window(
        self, begin: float, slewtime: float, end: float
    ) -> tuple[float, float]:
        """Return the half-open collection interval inside a scheduled task."""
        start = min(end, begin + max(0.0, slewtime) + self.setup_seconds)
        return start, max(start, end - self.post_collection_seconds)
