"""Non-collection time reserved within a science observation task."""

from collections.abc import Iterable

from pydantic import Field

from ._base import ConfigModel


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

    @classmethod
    def bound(cls, timings: Iterable["ObservationTiming"]) -> "ObservationTiming":
        """The largest of each budget over ``timings``: an upper bound on any
        observation's overheads where the observation is not known."""
        setup = cleanup = handoff = 0.0
        for timing in timings:
            setup = max(setup, timing.setup_seconds)
            cleanup = max(cleanup, timing.cleanup_seconds)
            handoff = max(handoff, timing.handoff_seconds)
        return cls(
            setup_seconds=setup, cleanup_seconds=cleanup, handoff_seconds=handoff
        )

    def collection_window(
        self, begin: float, slewtime: float, end: float
    ) -> tuple[float, float]:
        """Return the half-open collection interval inside a scheduled task."""
        start = min(end, begin + max(0.0, slewtime) + self.setup_seconds)
        return start, max(start, end - self.post_collection_seconds)


def timing_for(config: object, obsid: int) -> ObservationTiming:
    """The timing budgets of the observation ``obsid`` under ``config``.

    The category's own timing where its category sets one, else the
    payload's. A configuration that is a test double, or has neither, gives
    the default budgets.
    """
    lookup = getattr(config, "observation_timing_for", None)
    if callable(lookup):
        timing = lookup(int(obsid))
        if isinstance(timing, ObservationTiming):
            return timing
    payload = getattr(config, "payload", None)
    timing = getattr(payload, "observation_timing", None)
    return timing if isinstance(timing, ObservationTiming) else ObservationTiming()
