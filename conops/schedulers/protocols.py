"""Protocols for scheduler implementations.

COASTSim runs a scheduler in one of two ways:

- **Dispatch** (closed loop): :class:`~conops.ditl.QueueDITL` asks a
  :class:`DispatchPolicy` for the next target each time the spacecraft is
  free, so every decision sees the simulated state at that moment.
- **Planning** (open loop): a :class:`Scheduler` builds a whole
  :class:`~conops.targets.Plan` up front, and :class:`~conops.ditl.DITL`
  executes it. ``DITL.validate_plan_matches_execution`` then lists where
  execution differed from the plan.
"""

from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Protocol

from ..config import ScienceData
from ..targets import ObservationSequence, Plan, Pointing, TargetSlewEstimate

if TYPE_CHECKING:
    from ..ditl.ditl_log import DITLLog
    from ..simulation.coverage import StationCoverage


class DispatchPolicy(Protocol):
    """Selects the next science target for a closed-loop simulation.

    :class:`~conops.targets.TargetQueue` is the reference implementation.
    """

    targets: list[Pointing]
    """Every target the policy can select from, observed or not."""

    log: "DITLLog | None"
    """Event log; QueueDITL attaches its own when this is None."""

    coverage: "StationCoverage | None"
    """The run's link coverage for passes without spacecraft tracking;
    QueueDITL attaches its own."""

    def get(
        self,
        ra: float,
        dec: float,
        utime: float,
        collection_deadline: Callable[[Pointing, float], float | None] | None = None,
        slew_estimator: Callable[[Pointing], TargetSlewEstimate] | None = None,
        roll: float | None = None,
        recorder_room_gb: float | None = None,
    ) -> Pointing | None:
        """Return the target to observe next, or None if nothing can be observed.

        Args:
            ra, dec, roll: Current spacecraft attitude in degrees.
            recorder_room_gb: Gb the recorder can take before its next
                contact, under its planning limit, beyond the housekeeping
                of the time spent waiting; a target collects no more than
                fits, with the housekeeping its visit generates beyond
                that, and one that cannot collect its minimum snapshot is
                passed over. None checks nothing. A policy may offer
                ``collection_within(target, room_gb)``, the seconds of
                collection whose data fits, which admission asks again
                with the slew it flies.
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
        earliest_start: float | None = None,
        science_data: Mapping[str, ScienceData] | None = None,
        sequence: ObservationSequence | None = None,
        visits: int = 1,
        instrument_modes: Mapping[str, str] | None = None,
    ) -> Pointing:
        """Add a target and return it, as QueueDITL does for a Target of Opportunity.

        Args:
            deadline: Latest time (Unix seconds) science collection may begin.
            earliest_start: Earliest time (Unix seconds) science collection
                may begin.
            science_data: Science data observing the target generates, keyed
                by instrument name.
        """
        ...


class Scheduler(Protocol):
    """Builds an observing plan for :class:`~conops.ditl.DITL` to execute."""

    def schedule(self) -> Plan:
        """Build and return the plan."""
        ...
