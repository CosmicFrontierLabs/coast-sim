"""Figure of merit used to choose between observation targets.

A target's merit is a set of named terms. Its **tier** comes from its
observation category and is absolute: a target in a higher tier is always
preferred, whatever the other terms say. Within a tier, targets are compared
by **score**, the sum of two groups of terms:

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

if TYPE_CHECKING:
    from .pointing import Pointing


class MeritBreakdown(BaseModel):
    """The terms of one target's merit at one decision.

    Cost terms are stored as signed contributions, so a slew penalty is
    negative.
    """

    model_config = ConfigDict(frozen=True)

    tier: int = 0
    base: float
    urgency: float = 0.0
    cadence: float = 0.0
    completion_deficit: float = 0.0
    slew_distance: float = 0.0
    slew_time: float = 0.0
    collection: float = 0.0
    radiator: float = 0.0

    @property
    def value(self) -> float:
        """Science value: base merit plus the dynamic value terms."""
        return self.base + self.urgency + self.cadence + self.completion_deficit

    @property
    def score(self) -> float:
        """Value plus the cost terms; what targets in one tier are ranked by."""
        return (
            self.value
            + self.slew_distance
            + self.slew_time
            + self.collection
            + self.radiator
        )

    @property
    def rank(self) -> tuple[int, float]:
        """Sort key for selection: tier first, then score."""
        return self.tier, self.score

    @property
    def value_rank(self) -> tuple[int, float]:
        """Sort key for interrupt decisions: tier first, then value."""
        return self.tier, self.value

    def describe(self) -> str:
        """Return a one-line breakdown of every nonzero term."""
        terms = [f"tier={self.tier}", f"base={self.base:g}"]
        for name in (
            "urgency",
            "cadence",
            "completion_deficit",
            "slew_distance",
            "slew_time",
            "collection",
            "radiator",
        ):
            amount = getattr(self, name)
            if amount != 0.0:
                terms.append(f"{name}={amount:+.3f}")
        terms.append(f"value={self.value:.3f}")
        terms.append(f"score={self.score:.3f}")
        return " ".join(terms)


class MeritModel:
    """Evaluate the tier and value terms of a target's merit.

    Weights come from ``config.targets`` and per-category settings (tier,
    program, cadence and time share) from ``config.observation_categories``.
    With every weight at zero and every category in tier 0, a target's merit is
    its base merit.
    """

    def __init__(self, config: MissionConfig) -> None:
        self.categories = config.observation_categories
        targets = config.targets
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

    def tier(self, target: "Pointing") -> int:
        """Return a target's scheduling tier."""
        return self.category(target).tier

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
        """Evaluate a target's tier and value terms at ``utime``.

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
            * self._completion_deficit(category, delivered_shares or {})
            if self.completion_deficit_weight > 0.0
            else 0.0
        )
        return MeritBreakdown(
            tier=category.tier,
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
        window when no later window opens before the deadline. Targets without
        a deadline have no urgency.
        """
        deadline = target.deadline
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
    def _completion_deficit(
        category: ObservationCategory, delivered_shares: Mapping[str, float]
    ) -> float:
        """Return allocated minus delivered share for the program, in [-1, 1]."""
        if category.time_share is None:
            return 0.0
        delivered = delivered_shares.get(category.program_name, 0.0)
        return max(-1.0, min(1.0, category.time_share - delivered))
