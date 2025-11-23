from ..common import ics_date_conv
from ..config import DAY_SECONDS
from ..targets import Plan, Queue


class DumbQueueScheduler:
    """A simple Plan generator based on merit-driven Queue Scheduling."""

    def __init__(
        self,
        queue: Queue | None = None,
        plan: Plan | None = None,
        year: int = 2021,
        day: int = 4,
        length: int = 1,
    ):
        self.queue = queue if queue is not None else Queue()
        self.ppst = plan if plan is not None else Plan()
        self.year = year
        self.day = day
        self.length = length
        self.ustart = 0

    def schedule(self) -> Plan:
        """Generate a Plan over the configured start/time window.

        Returns:
            Plan: the scheduled plan
        """
        # Reset plan for this scheduling run
        self.ppst = Plan()

        elapsed = 0.0
        last_ra = 0.0
        last_dec = 0.0

        self.ustart = ics_date_conv(f"{self.year}-{self.day:03}-00:00:00")
        end_time = self.ustart + self.length * DAY_SECONDS

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
            self.ppst.extend([item])

        return self.ppst
