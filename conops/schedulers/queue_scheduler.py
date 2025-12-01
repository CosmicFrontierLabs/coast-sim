from datetime import datetime

from ..targets import Plan, Queue


class DumbQueueScheduler:
    """A simple Plan generator based on merit-driven Queue Scheduling."""

    def __init__(
        self,
        queue: Queue,
        begin: datetime | None = None,
        end: datetime | None = None,
        plan: Plan | None = None,
    ):
        self.queue = queue
        self.plan = plan if plan is not None else Plan()
        self.begin = begin
        self.end = end
        self.ustart = begin.timestamp() if begin is not None else None

    def schedule(self) -> Plan:
        """Generate a Plan over the configured start/time window.

        Returns:
            Plan: the scheduled plan
        """

        assert self.begin is not None and self.end is not None, (
            "Begin and end times must be set for scheduling."
        )

        # Reset plan for this scheduling run
        self.plan = Plan()

        elapsed = 0.0
        last_ra = 0.0
        last_dec = 0.0

        self.ustart = self.begin.timestamp()
        end_time = self.end.timestamp()

        while True:
            utime = self.ustart + elapsed
            if utime >= end_time:
                break

            item = self.queue.get(last_ra, last_dec, utime)
            if item is None:
                break

            duration: float = item.end - item.begin
            # Sanity check: avoid infinite loops on zero/negative-duration items
            if duration <= 0:
                break

            elapsed += duration
            last_ra = item.ra
            last_dec = item.dec
            item.done = True
            self.plan.extend([item])

        return self.plan
