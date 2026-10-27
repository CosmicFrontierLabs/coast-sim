"""Figure of merit used to choose between observation targets.

A target's merit is a set of named terms. Targets are compared by **score**,
the sum of two groups of terms:

- **Value** terms say how much the observation is worth now: the target's base
  merit plus urgency, cadence and completion-deficit terms. The value at
  selection is frozen onto the plan entry and is what a Target of Opportunity
  must beat to interrupt the observation.
- **Cost** terms say what the observation consumes from the current state: slew
  distance and time, useful collection time and radiator exposure.

Every dynamic value term is a weight times a factor bounded to [0, 1] (or
[-1, 1] for the completion deficit), so no term can exceed its weight.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import TYPE_CHECKING

from pydantic import BaseModel, ConfigDict

from ..config import MissionConfig
from ..config.observation_categories import ObservationCategory
from ..config.observation_groups import ObservationGroup, ObservationGroups

if TYPE_CHECKING:
    from .pointing import Pointing


class MeritBreakdown(BaseModel):
    """The terms of one target's merit at one decision.

    Cost terms are stored as signed contributions, so a slew penalty is
    negative.
    """

    model_config = ConfigDict(frozen=True)

    base: float
    urgency: float = 0.0
    cadence: float = 0.0
    completion_deficit: float = 0.0
    allocation: float = 0.0
    """Value added for being in the time a long-range allocation gives it."""
    slew_distance: float = 0.0
    slew_time: float = 0.0
    collection: float = 0.0
    radiator: float = 0.0
    data: float = 0.0
    """The price of the data a snapshot puts on the recorder; zero or negative."""
    group: float = 0.0
    """Merit added for following another target of the same group."""

    @property
    def value(self) -> float:
        """Science value: base merit plus the dynamic value terms."""
        return (
            self.base
            + self.urgency
            + self.cadence
            + self.completion_deficit
            + self.allocation
        )

    @property
    def score(self) -> float:
        """Value plus the cost terms; what targets are ranked by."""
        return (
            self.value
            + self.slew_distance
            + self.slew_time
            + self.collection
            + self.radiator
            + self.data
            + self.group
        )

    def describe(self) -> str:
        """Return a one-line breakdown of every nonzero term."""
        terms = [f"base={self.base:g}"]
        for name in (
            "urgency",
            "cadence",
            "completion_deficit",
            "allocation",
            "slew_distance",
            "slew_time",
            "collection",
            "radiator",
            "data",
            "group",
        ):
            amount = getattr(self, name)
            if amount != 0.0:
                terms.append(f"{name}={amount:+.3f}")
        terms.append(f"value={self.value:.3f}")
        terms.append(f"score={self.score:.3f}")
        return " ".join(terms)


class MeritModel:
    """Evaluate the value terms of a target's merit.

    Weights come from ``config.targets`` and per-category settings (program,
    cadence and time share) from ``config.observation_categories``. With every
    weight at zero, a target's merit is its base merit.
    """

    def __init__(self, config: MissionConfig) -> None:
        self.categories = config.observation_categories
        groups = getattr(config, "observation_groups", None)
        # A test double's configuration has no groups.
        self.groups = (
            groups if isinstance(groups, ObservationGroups) else ObservationGroups()
        )
        targets = config.targets
        weight = getattr(targets, "group_contiguity_weight", 0.0)
        self.group_contiguity_weight = (
            float(weight) if isinstance(weight, (int, float)) else 0.0
        )
        self.urgency_weight = targets.urgency_weight
        self.urgency_timescale_seconds = targets.urgency_timescale_seconds
        self.cadence_weight = targets.cadence_weight
        self.completion_deficit_weight = targets.completion_deficit_weight

    @property
    def is_dynamic(self) -> bool:
        """Whether any value term besides base merit can be nonzero."""
        return (
            self.urgency_weight > 0.0
            or self.cadence_weight > 0.0
            or self.completion_deficit_weight > 0.0
        )

    @property
    def max_dynamic_value(self) -> float:
        """Upper bound on the dynamic value terms of any target."""
        return (
            self.urgency_weight + self.cadence_weight + self.completion_deficit_weight
        )

    def category(self, target: "Pointing") -> ObservationCategory:
        """Return the observation category a target belongs to."""
        return self.categories.get_category(int(target.obsid))

    def group(self, target: "Pointing") -> ObservationGroup | None:
        """The observation group a target belongs to, if any."""
        return self.groups.group_of(int(target.obsid))

    def group_bounds(self, target: "Pointing") -> tuple[float | None, float | None]:
        """The start and deadline of the target's group window, each None
        where the target has no group or the group no bound."""
        group = self.group(target)
        return group.bounds if group is not None else (None, None)

    def collected_by_group(self, targets: Iterable["Pointing"]) -> dict[str, float]:
        """Science collected so far by each group's targets, by group name."""
        collected: dict[str, float] = defaultdict(float)
        for target in targets:
            seconds = getattr(target, "collected_seconds", 0.0)
            if not isinstance(seconds, (int, float)) or seconds <= 0.0:
                continue
            group = self.group(target)
            if group is not None:
                collected[group.name] += float(seconds)
        return dict(collected)

    def group_budget_left(
        self, target: "Pointing", collected: Mapping[str, float]
    ) -> float | None:
        """Science the target's group may still collect within its time
        budget, given ``collected`` by group; None without a budget."""
        group = self.group(target)
        if group is None or group.time_budget_seconds is None:
            return None
        return max(0.0, group.time_budget_seconds - collected.get(group.name, 0.0))

    def contiguity_bonus(self, target: "Pointing", previous_obsid: int | None) -> float:
        """Merit added to ``target`` for following the target ``previous_obsid``
        where both are in one group that prefers contiguity."""
        if self.group_contiguity_weight <= 0.0 or previous_obsid is None:
            return 0.0
        if int(target.obsid) == previous_obsid:
            return 0.0
        group = self.group(target)
        if group is None or not group.prefer_contiguous:
            return 0.0
        return self.group_contiguity_weight if previous_obsid in group.members else 0.0

    def collected_by_category(self, targets: Iterable["Pointing"]) -> dict[str, float]:
        """Science collected so far by each category's targets, by category
        name, for the categories' time budgets."""
        collected: dict[str, float] = defaultdict(float)
        for target in targets:
            seconds = getattr(target, "collected_seconds", 0.0)
            if isinstance(seconds, (int, float)) and seconds > 0.0:
                collected[self.category(target).name] += float(seconds)
        return dict(collected)

    def budget_left(
        self, target: "Pointing", collected: Mapping[str, float]
    ) -> float | None:
        """Science the target's category may still collect within its time
        budget, given ``collected`` by category; None without a budget."""
        category = self.category(target)
        if category.time_budget_seconds is None:
            return None
        return max(
            0.0, category.time_budget_seconds - collected.get(category.name, 0.0)
        )

    def delivered_shares(self, targets: Iterable["Pointing"]) -> dict[str, float]:
        """Return each program's fraction of the science collected so far."""
        collected: dict[str, float] = defaultdict(float)
        for target in targets:
            if target.collected_seconds > 0.0:
                collected[self.category(target).program_name] += (
                    target.collected_seconds
                )
        total = sum(collected.values())
        if total <= 0.0:
            return {}
        return {program: seconds / total for program, seconds in collected.items()}

    def value_terms(
        self,
        target: "Pointing",
        utime: float,
        *,
        base: float | None = None,
        visibility_window: Sequence[float] | None = None,
        delivered_shares: Mapping[str, float] | None = None,
    ) -> MeritBreakdown:
        """Evaluate a target's value terms at ``utime``.

        Args:
            target: Target to evaluate.
            utime: Decision time in Unix seconds.
            base: Base merit; defaults to ``target.merit``.
            visibility_window: The visibility window the observation would use,
                so urgency can tell when it is the last one before the deadline.
            delivered_shares: Programs' delivered fractions of science time,
                from :meth:`delivered_shares`.
        """
        category = self.category(target)
        urgency = (
            self.urgency_weight * self._urgency(target, utime, visibility_window)
            if self.urgency_weight > 0.0
            else 0.0
        )
        cadence = (
            self.cadence_weight * self._cadence(target, utime, category)
            if self.cadence_weight > 0.0
            else 0.0
        )
        completion_deficit = (
            self.completion_deficit_weight
            * self.completion_deficit(category, delivered_shares or {})
            if self.completion_deficit_weight > 0.0
            else 0.0
        )
        return MeritBreakdown(
            base=float(target.merit) if base is None else base,
            urgency=urgency,
            cadence=cadence,
            completion_deficit=completion_deficit,
        )

    def _urgency(
        self,
        target: "Pointing",
        utime: float,
        visibility_window: Sequence[float] | None,
    ) -> float:
        """Return urgency in [0, 1], rising as the target's last chance nears.

        The closing time is the deadline, or the end of the current visibility
        window when no later window opens before the deadline. For a target
        with visit windows the deadline is its next visit window's. Targets
        without a deadline have no urgency.
        """
        deadline = target.deadline
        windows = getattr(target, "visit_windows", None)
        # A test double's target has no windows.
        if isinstance(windows, list) and windows:
            deadline = target.window_bounds(utime)[1]
        if deadline is None:
            return 0.0
        close = deadline
        if visibility_window is not None:
            window_end = float(visibility_window[1])
            later_window = any(
                window_end < float(window[0]) < deadline for window in target.windows
            )
            if window_end < deadline and not later_window:
                close = window_end
        remaining = close - utime
        if remaining <= self.urgency_timescale_seconds:
            return 1.0
        return self.urgency_timescale_seconds / remaining

    @staticmethod
    def _cadence(
        target: "Pointing", utime: float, category: ObservationCategory
    ) -> float:
        """Return cadence pressure in [0, 1]: elapsed fraction of the interval."""
        interval = category.cadence_seconds
        if interval is None:
            return 0.0
        last = target.last_collection_time
        if last is None:
            return 1.0
        return min(1.0, max(0.0, (utime - last) / interval))

    @staticmethod
    def completion_deficit(
        category: ObservationCategory, delivered_shares: Mapping[str, float]
    ) -> float:
        """Return allocated minus delivered share for the program, in [-1, 1].

        This is the completion-deficit factor before its weight; schedulers use
        it with the shares their plan would deliver.
        """
        if category.time_share is None:
            return 0.0
        delivered = delivered_shares.get(category.program_name, 0.0)
        return max(-1.0, min(1.0, category.time_share - delivered))
