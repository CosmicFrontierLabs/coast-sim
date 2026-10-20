"""A consumable over a planning horizon: something that builds up and is removed.

Several things a spacecraft carries accumulate between the events that remove
them, and must stay within a limit meanwhile: data on the recorder until a
ground contact downlinks it, stored momentum until a dump unloads it. Their
level rises between removals and falls at them, so its highest points are just
before each removal and at the end of the horizon.

The removals cut the horizon into *segments*; a segment runs from the start of
one removal to the start of the next. What a plan adds in a segment is checked
against the limit at that segment's peak, and at every later peak, since what a
removal cannot take off carries over. Schedulers keep a plan's additions by
segment and ask the ledger whether an addition fits where they mean to make it.
Nothing is ever shortened to fit: an activity that does not fit is placed
elsewhere or not at all.

:class:`ConsumableLedger` holds this arithmetic. A subclass, such as
:class:`~conops.schedulers.recorder.RecorderLedger`, maps a particular
consumable onto it: what accrues on its own, what each removal takes off, and
what an activity adds.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Callable, Sequence
from typing import NamedTuple

_TOLERANCE = 1e-9
"""Slack for rounding in sums."""


class Removal(NamedTuple):
    """An event that takes some of the consumable off, such as a contact or a dump."""

    begin: float
    end: float
    amount: float
    """The most it removes; it never removes more than is there."""
    accrual: float = 0.0
    """What accrues during the event itself, before its removal."""


class ConsumableLedger:
    """A consumable's level over a horizon, by the segments its removals cut it into.

    Args:
        begin: Start of the horizon (Unix seconds).
        end: End of the horizon.
        removals: The events that remove the consumable. Those outside the
            horizon are ignored, and one that straddles its start or end counts
            for the part inside it, with its amount and accrual scaled to it.
        limit: The level a plan keeps the consumable within; None for no limit.
        start: The level at ``begin``.
        accrual: What accrues on its own, apart from any plan, over a number of
            seconds outside removals.
    """

    def __init__(
        self,
        begin: float,
        end: float,
        removals: Sequence[Removal],
        *,
        limit: float | None,
        start: float,
        accrual: Callable[[float], float],
    ) -> None:
        self.limit = limit
        self.start = float(start)
        inside: list[Removal] = []
        for removal in sorted(removals, key=lambda r: r.begin):
            clipped_begin = max(removal.begin, begin)
            clipped_end = min(removal.end, end)
            if clipped_end <= clipped_begin:
                continue
            share = (clipped_end - clipped_begin) / (removal.end - removal.begin)
            inside.append(
                Removal(
                    clipped_begin,
                    clipped_end,
                    removal.amount * share,
                    removal.accrual * share,
                )
            )
        # Segment k starts at bounds[k]; removal k - 1 is at the start of segment k.
        self.bounds: list[float] = [begin, *(r.begin for r in inside), end]
        self.removal: list[float] = [0.0, *(r.amount for r in inside)]
        """What the removal at the start of each segment takes off."""
        self.removal_accrual: list[float] = [0.0, *(r.accrual for r in inside)]
        """What accrues during the removal at the start of each segment."""
        self.accrual: list[float] = []
        """What accrues on its own in each segment, outside its removal."""
        for k in range(self.segments):
            seconds = self.bounds[k + 1] - self.bounds[k]
            if k > 0:
                seconds -= inside[k - 1].end - inside[k - 1].begin
            self.accrual.append(accrual(max(0.0, seconds)))

    @property
    def active(self) -> bool:
        """Whether the consumable limits what may be planned."""
        return self.limit is not None

    @property
    def segments(self) -> int:
        return len(self.bounds) - 1

    def segment(self, time: float) -> int:
        """The segment a time falls in."""
        return min(max(bisect_right(self.bounds, time) - 1, 0), self.segments - 1)

    def levels(self, planned: Sequence[float]) -> list[float]:
        """The level at the end of each segment, given the plan's additions in each.

        A removal takes off what it can of what is there when it starts, and
        no more than that.
        """
        level = self.start
        levels = []
        for k in range(self.segments):
            if k > 0:
                level = max(0.0, level + self.removal_accrual[k] - self.removal[k])
            level += self.accrual[k] + planned[k]
            levels.append(level)
        return levels

    def fits(self, planned: Sequence[float], segment: int, extra: float) -> bool:
        """Whether ``extra`` more in ``segment`` keeps every level within the limit."""
        if self.limit is None:
            return True
        trial = list(planned)
        trial[segment] += extra
        return all(level <= self.limit + _TOLERANCE for level in self.levels(trial))

    @staticmethod
    def add(planned: Sequence[float], segment: int, extra: float) -> tuple[float, ...]:
        """The plan's additions with ``extra`` more in ``segment``."""
        added = list(planned)
        added[segment] += extra
        return tuple(added)
