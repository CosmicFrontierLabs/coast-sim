from datetime import datetime

from ..targets import Plan
from .protocols import DispatchPolicy


class GreedyDispatchPlanner:
    """Plan by asking a dispatch policy for each next target, back to back.

    From the start of the window, the planner asks the policy (normally a
    :class:`~conops.targets.TargetQueue`, which picks by merit) for the next
    target given the current attitude and time, appends it to the plan, and
    continues from where it ends, until the window ends or the policy has
    nothing left. It records greedy dispatch as a plan, without the slew and
    constraint checks :class:`~conops.schedulers.PriorityPlanner` makes.

    Formerly ``DumbQueueScheduler``, a name still importable with a deprecation
    warning.
    """

    def __init__(
        self,
        queue: DispatchPolicy,
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
        last_roll = 0.0

        self.ustart = self.begin.timestamp()
        end_time = self.end.timestamp()

        while True:
            utime = self.ustart + elapsed
            if utime >= end_time:
                break

            item = self.queue.get(last_ra, last_dec, utime, roll=last_roll)
            if item is None:
                break

            duration: float = item.end - item.begin
            # Sanity check: avoid infinite loops on zero/negative-duration items
            if duration <= 0:
                break

            elapsed += duration
            spacecraft_attitude = item.spacecraft_attitude
            last_ra, last_dec, last_roll = (
                spacecraft_attitude
                if isinstance(spacecraft_attitude, tuple)
                else (item.ra, item.dec, item.roll)
            )
            item.done = True
            self.plan.extend([item])

        return self.plan
