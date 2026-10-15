"""Offline queue planner.

:class:`QueuePlanner` builds a plan by running a dispatch policy (normally a
:class:`~conops.targets.TargetQueue`) forward over the horizon without
simulating the spacecraft: each selection starts where the previous one ended,
from its attitude. It is as fast as the queue, and its plans follow the queue's
merit ordering, but timing comes from the queue's own estimates rather than
the models :class:`~conops.ditl.DITL` executes with.
"""

from datetime import datetime

from ..targets import Plan, Pointing
from .protocols import DispatchPolicy

Attitude = tuple[float, float, float]


class QueuePlanner:
    """Build a plan by asking a dispatch policy for one target after another.

    Steps:

    1. Starting at ``begin`` from attitude (0, 0, 0), ask the queue for the
       best target given the current time and attitude.
    2. Append the returned entry to the plan, mark it done, and advance the
       clock by its duration and the attitude to its pointing.
    3. Stop at ``end``, when the queue has nothing to offer, or when it returns
       an entry with no duration.

    Args:
        queue: Dispatch policy that selects each target.
        begin: Start of the planning horizon.
        end: End of the planning horizon.
    """

    def __init__(self, queue: DispatchPolicy, begin: datetime, end: datetime) -> None:
        self.queue = queue
        self.begin = begin
        self.end = end
        self.ustart = begin.timestamp()
        self.plan = Plan()

    # ── Public API ────────────────────────────────────────────────────────

    def schedule(self) -> Plan:
        """Build and return the plan."""
        self.ustart = self.begin.timestamp()
        uend = self.end.timestamp()
        self.plan = Plan(metadata={"planner": "queue"})

        utime = self.ustart
        attitude: Attitude = (0.0, 0.0, 0.0)
        while utime < uend:
            item = self.queue.get(*attitude[:2], utime, roll=attitude[2])
            if item is None:
                break
            duration: float = item.end - item.begin
            # Sanity check: avoid infinite loops on zero/negative-duration items
            if duration <= 0:
                break
            utime += duration
            attitude = self._attitude(item)
            item.done = True
            self.plan.extend([item])

        return self.plan

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _attitude(item: Pointing) -> Attitude:
        """Return the attitude the slew after ``item`` leaves from."""
        if isinstance(item.spacecraft_attitude, tuple):
            return item.spacecraft_attitude
        return (item.ra, item.dec, item.roll)
