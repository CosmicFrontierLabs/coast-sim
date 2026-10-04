"""Contracts that scheduler implementations satisfy.

COASTSim runs a scheduler in one of two ways:

- **Dispatch** (closed loop): :class:`~conops.ditl.QueueDITL` asks a
  :class:`DispatchPolicy` for the next target each time the spacecraft is
  free, so every decision sees the simulated state at that moment.
- **Planning** (open loop): a :class:`Planner` builds a whole
  :class:`~conops.targets.Plan` up front, and :class:`~conops.ditl.DITL`
  executes it. ``DITL.validate_plan_matches_execution`` then reports where
  the flown timeline departed from the plan.
"""

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from ..targets import Plan, Pointing, TargetSlewEstimate

if TYPE_CHECKING:
    from ..ditl.ditl_log import DITLLog


class DispatchPolicy(Protocol):
    """Selects the next science target for a closed-loop simulation.

    :class:`~conops.targets.TargetQueue` is the reference implementation.
    """

    targets: list[Pointing]
    """Every target the policy can select from, observed or not."""

    log: "DITLLog | None"
    """Event log; QueueDITL attaches its own when this is None."""

    def get(
        self,
        ra: float,
        dec: float,
        utime: float,
        collection_deadline: Callable[[Pointing, float], float | None] | None = None,
        slew_estimator: Callable[[Pointing], TargetSlewEstimate] | None = None,
        roll: float | None = None,
    ) -> Pointing | None:
        """Return the target to observe next, or None if nothing can be observed.

        Args:
            ra, dec, roll: Current spacecraft attitude in degrees.
            utime: Current time in Unix seconds.
            collection_deadline: Returns the latest time science may be
                collected for a candidate, given when its slew ends.
            slew_estimator: Returns the attitude-aware slew cost of a candidate.

        Returns:
            The selected target, with its visibility windows populated, or None.
        """
        ...

    def add(
        self,
        ra: float = 0.0,
        dec: float = 0.0,
        obsid: int = 0,
        name: str = "FakeTarget",
        merit: float = 100.0,
        exptime: int = 1000,
        ss_min: int = 300,
        ss_max: int = 86400,
        instrument_name: str | None = None,
        deadline: float | None = None,
    ) -> Pointing:
        """Add a target and return it, as QueueDITL does for a Target of Opportunity.

        Args:
            deadline: Latest time (Unix seconds) science collection may begin.
        """
        ...


class Planner(Protocol):
    """Builds an observing plan for :class:`~conops.ditl.DITL` to execute."""

    def schedule(self) -> Plan:
        """Build and return the plan."""
        ...
