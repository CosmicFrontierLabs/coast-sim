"""Back-compatible name for :class:`~conops.schedulers.QueuePlanner`."""

from datetime import datetime

from ..targets import Plan
from .protocols import DispatchPolicy
from .queue_planner import QueuePlanner


class DumbQueueScheduler(QueuePlanner):
    """Deprecated name for :class:`~conops.schedulers.QueuePlanner`.

    Unlike :class:`QueuePlanner`, ``begin`` and ``end`` may be set after
    construction, and a starting ``plan`` may be given; it is replaced when
    :meth:`schedule` runs.
    """

    def __init__(
        self,
        queue: DispatchPolicy,
        begin: datetime | None = None,
        end: datetime | None = None,
        plan: Plan | None = None,
    ) -> None:
        self.queue = queue
        self.begin = begin  # type: ignore[assignment]
        self.end = end  # type: ignore[assignment]
        self.ustart = begin.timestamp() if begin is not None else None  # type: ignore[assignment]
        self.plan = plan if plan is not None else Plan()

    def schedule(self) -> Plan:
        assert self.begin is not None and self.end is not None, (
            "Begin and end times must be set for scheduling."
        )
        return super().schedule()
