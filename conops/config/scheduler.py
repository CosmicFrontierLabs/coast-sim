"""Configuration for how a simulation schedules its observations."""

from __future__ import annotations

from datetime import timedelta
from enum import Enum
from typing import Annotated, Literal

from pydantic import Field, model_validator

from ._base import ConfigModel


class SchedulerMode(str, Enum):
    """How a simulation decides what to observe."""

    DISPATCH = "dispatch"
    """QueueDITL picks each next target as the simulation runs."""
    PLANNED = "planned"
    """A scheduler builds one plan for the whole run, which DITL executes."""
    ROLLING = "rolling"
    """RollingHorizonDITL rebuilds the plan as the simulation runs."""


class SchedulerKind(str, Enum):
    """Which scheduler builds plans in planned and rolling modes."""

    PRIORITY = "priority"
    """PriorityScheduler: priority order, earliest fit."""
    LOCAL_SEARCH = "local_search"
    """LocalSearchScheduler: priority-first, then improved by local search."""
    CP_SAT = "cp_sat"
    """CpSatScheduler: OR-Tools CP-SAT, chunk by chunk (needs coast-sim[cpsat])."""


AllocationStrictness = Literal["strict", "tier", "weighted"]
"""How closely short-term scheduling follows a long-range allocation.

* ``strict``: a request's allocated time, in the bin it is allocated to, comes
  ahead of all unallocated work, whatever its tier; working ahead uses only the
  time left.
* ``tier``: allocated time comes ahead of the unallocated work in its own tier
  only, so a higher tier's work can take a lower tier's allocated time.
* ``weighted``: allocated time stays in its tier and gains ``bonus`` of its
  value, so scheduling trades keeping to the allocation against merit.
"""


_SEARCH_FIELDS = (
    "seed",
    "time_limit",
    "max_iterations",
    "earliness_weight",
    "neighborhood",
    "history_length",
)
"""Settings of the local search, which the CP-SAT scheduler also runs."""
_SOLVER_FIELDS = ("solver_time_limit", "chunk_seconds", "workers", "max_candidates")
"""Settings of the CP-SAT scheduler only."""


class ReplanSettings(ConfigModel):
    """When a rolling-horizon simulation replans."""

    horizon_seconds: float = Field(
        default=86400.0, gt=0, description="How far ahead each plan looks"
    )
    replan_interval_seconds: float = Field(
        default=43200.0, gt=0, description="Time between scheduled replans"
    )
    commit_lead_time_seconds: float = Field(
        default=0.0,
        ge=0,
        description="Time between building a plan and it taking effect",
    )
    rapid_replans: bool = Field(
        default=True,
        description="Whether a ToO with a close deadline triggers an immediate replan",
    )
    allow_interrupts: bool = Field(
        default=True,
        description="Whether a rapid replan may cut short the observation in progress",
    )


class AllocationSettings(ConfigModel):
    """A long-range allocator for dispatch and rolling simulations.

    See :class:`~conops.schedulers.LongRangeAllocator`.
    """

    bin_seconds: float = Field(
        default=86400.0, gt=0, description="Length of each allocation bin"
    )
    efficiency: float = Field(
        default=0.75,
        gt=0,
        le=1,
        description="Fraction of each bin expected to hold science",
    )
    reserve: float = Field(
        default=0.1,
        ge=0,
        lt=1,
        description="Fraction of that capacity held back for unplanned work, such as ToOs",
    )
    solver: Literal["milp", "greedy"] = Field(
        default="milp",
        description=(
            "milp solves the allocation optimally and needs OR-Tools "
            "(coast-sim[cpsat]); greedy is a quick heuristic"
        ),
    )
    time_limit_seconds: float = Field(
        default=10.0,
        gt=0,
        description="Seconds the MILP solver may search per allocation",
    )
    strictness: AllocationStrictness = Field(
        default="strict",
        description=(
            "How closely scheduling follows the allocation: strict, tier or "
            "weighted (see AllocationStrictness)"
        ),
    )
    bonus: float = Field(
        default=0.5,
        ge=0,
        description=(
            "With weighted strictness, the fraction of a request's value its "
            "allocated time gains"
        ),
    )


class SchedulerConfig(ConfigModel):
    """How a simulation schedules its observations.

    Used by :func:`conops.ditl.create_ditl` to build the matching simulation.
    The default, ``dispatch``, is queue scheduling with
    :class:`~conops.ditl.QueueDITL`, which uses only ``allocation``.
    ``replanning`` applies to ``rolling`` mode only, and ``allocation`` to
    ``dispatch`` and ``rolling`` modes.

    In ``planned`` and ``rolling`` modes, ``kind`` picks the scheduler that
    builds plans. Its settings left unset take the scheduler's defaults.
    Search settings apply to the local-search and CP-SAT schedulers, and
    solver settings to the CP-SAT scheduler only; setting one for a
    scheduler that does not use it is an error.
    """

    mode: SchedulerMode = Field(
        default=SchedulerMode.DISPATCH,
        description="dispatch, planned or rolling",
    )
    kind: SchedulerKind = Field(
        default=SchedulerKind.PRIORITY,
        description="Scheduler that builds plans in planned and rolling modes",
    )
    include_passes: bool = Field(
        default=True, description="Whether plans reserve ground-station passes"
    )
    # "unlimited" rather than null, which the YAML writer leaves out like
    # every other unset setting, so a saved configuration would come back
    # with the default.
    successor_retries: Annotated[int, Field(ge=0)] | Literal["unlimited"] = Field(
        default=3,
        description=(
            "Later starts to try in a gap once a snapshot fits but the activity "
            "after it cannot be reached; unlimited tries every start"
        ),
    )
    seed: int | None = Field(
        default=None,
        description=(
            "Random seed for local search and CP-SAT; defaults to random_seed. "
            "The priority scheduler breaks ties with random_seed"
        ),
    )
    time_limit: float | None = Field(
        default=None,
        ge=0,
        description="Seconds of local search per plan",
    )
    max_iterations: int | None = Field(
        default=None,
        ge=0,
        description="Local-search changes to try at most, for reproducible plans",
    )
    earliness_weight: float | None = Field(
        default=None,
        ge=0,
        le=1,
        description="Fraction of a deadline request's value lost by starting at its deadline",
    )
    neighborhood: int | None = Field(
        default=None, ge=1, description="Positions apart a local-search move can be"
    )
    history_length: int | None = Field(
        default=None, ge=1, description="Late-acceptance history length"
    )
    solver_time_limit: float | None = Field(
        default=None, gt=0, description="Seconds CP-SAT may search per chunk"
    )
    chunk_seconds: float | None = Field(
        default=None, gt=0, description="Length of each chunk CP-SAT solves"
    )
    workers: int | None = Field(default=None, ge=1, description="CP-SAT search workers")
    max_candidates: int | None = Field(
        default=None, ge=1, description="Candidate snapshots per CP-SAT chunk"
    )
    replanning: ReplanSettings = Field(
        default_factory=ReplanSettings,
        description="Replanning schedule for rolling mode",
    )
    allocation: AllocationSettings | None = Field(
        default=None,
        description=(
            "Long-range allocation for dispatch and rolling modes; unset for none"
        ),
    )

    @model_validator(mode="after")
    def _settings_fit_the_scheduler(self) -> SchedulerConfig:
        unused: list[str] = []
        if self.kind is SchedulerKind.PRIORITY:
            unused += [f for f in _SEARCH_FIELDS if getattr(self, f) is not None]
        if self.kind is not SchedulerKind.CP_SAT:
            unused += [f for f in _SOLVER_FIELDS if getattr(self, f) is not None]
        if unused:
            raise ValueError(
                f"{', '.join(unused)} do not apply to the {self.kind.value} scheduler"
            )
        if self.allocation is not None and self.mode is SchedulerMode.PLANNED:
            raise ValueError(
                "allocation applies to dispatch and rolling modes; a planned "
                "run already plans the whole run at once"
            )
        return self

    def scheduler_options(self) -> dict[str, object]:
        """Return the scheduler's settings as keyword arguments for its class.

        ``include_passes`` is left out: :class:`~conops.ditl.RollingHorizonDITL`
        takes it as an argument of its own.
        """
        options: dict[str, object] = {
            name: getattr(self, name)
            for name in (*_SEARCH_FIELDS, *_SOLVER_FIELDS)
            if getattr(self, name) is not None and name != "chunk_seconds"
        }
        # The schedulers take None for trying every start.
        options["successor_retries"] = (
            None if self.successor_retries == "unlimited" else self.successor_retries
        )
        if self.chunk_seconds is not None:
            options["chunk"] = timedelta(seconds=self.chunk_seconds)
        return options
