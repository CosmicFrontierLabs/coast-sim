"""Non-collection time reserved within a science observation task."""

from collections.abc import Iterable

from pydantic import Field, model_validator

from ._base import ConfigModel


class BudgetStep(ConfigModel):
    """One calibration step inside an observation's setup or cleanup budget:
    a target acquisition, a characterisation frame, a dark. It takes its
    duration, puts the instruments it names in its modes, and puts its own
    data on the recorder."""

    name: str
    duration_seconds: float = Field(gt=0.0, allow_inf_nan=False)
    instrument_modes: dict[str, str] = Field(
        default_factory=dict,
        description=(
            "The instruments the step uses and the mode each is in, by name; "
            "empty: the observation's own modes apply through the step"
        ),
    )
    data_gb: float = Field(
        default=0.0,
        ge=0.0,
        description="Data the step puts on the recorder, in Gb, such as its frames",
    )


class ObservationTiming(ConfigModel):
    """Payload timing budgets, separate from ACS motion and settling.

    A budget is one span of non-collection time unless it is given as
    ``setup_steps`` or ``cleanup_steps``, in order: the budget's seconds are
    then the steps' sum, each step puts the instruments it names in its
    modes while it runs, and the steps' data goes on the recorder with the
    observation's.
    """

    setup_steps: list[BudgetStep] = Field(
        default_factory=list,
        description="The calibration steps the setup budget consists of, in order",
    )
    cleanup_steps: list[BudgetStep] = Field(
        default_factory=list,
        description="The calibration steps the cleanup budget consists of, in order",
    )
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

    @model_validator(mode="after")
    def _steps_set_their_budget(self) -> "ObservationTiming":
        """A budget given as steps is their sum."""
        # Assigned only when different: assignment validates again.
        if self.setup_steps:
            setup = sum(step.duration_seconds for step in self.setup_steps)
            if self.setup_seconds != setup:
                self.setup_seconds = setup
        if self.cleanup_steps:
            cleanup = sum(step.duration_seconds for step in self.cleanup_steps)
            if self.cleanup_seconds != cleanup:
                self.cleanup_seconds = cleanup
        return self

    @property
    def budget_data_gb(self) -> float:
        """Data the setup and cleanup steps put on the recorder, in Gb."""
        return sum(step.data_gb for step in (*self.setup_steps, *self.cleanup_steps))

    @staticmethod
    def _step_at(steps: list[BudgetStep], offset: float) -> BudgetStep | None:
        """The step running ``offset`` seconds into its budget, if any."""
        if offset < 0.0:
            return None
        for step in steps:
            if offset < step.duration_seconds:
                return step
            offset -= step.duration_seconds
        return None

    def setup_step_at(self, offset: float) -> BudgetStep | None:
        """The setup step running ``offset`` seconds after setup begins."""
        return self._step_at(self.setup_steps, offset)

    def cleanup_step_at(self, offset: float) -> BudgetStep | None:
        """The cleanup step running ``offset`` seconds after collection ends."""
        return self._step_at(self.cleanup_steps, offset)

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
