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
from typing import Generic, NamedTuple, TypeVar, cast

_TOLERANCE = 1e-9
"""How far below the limit a plan keeps the level: the limit is where an
alert fires, so reaching it is not within it, and sums round."""

S = TypeVar("S")
"""What a plan has added in a segment: a number, unless a subclass orders its
additions in time."""
E = TypeVar("E")
"""One more addition to a segment: a number, or one ordered addition."""


class Removal(NamedTuple):
    """An event that takes some of the consumable off, such as a contact or a dump."""

    begin: float
    end: float
    amount: float
    """The most it removes; it never removes more than is there."""
    accrual: float = 0.0
    """What accrues during the event itself, before its removal."""
    lead: float = 0.0
    """Seconds before ``begin`` the event occupies without removing anything,
    such as the slew into a contact, which accrue at the event's own rate."""
    lead_accrual: float = 0.0
    """What accrues during the lead."""


class ConsumableLedger(Generic[S, E]):
    """A consumable's level over a horizon, by the segments its removals cut it into.

    Args:
        begin: Start of the horizon (Unix seconds).
        end: End of the horizon.
        removals: The events that remove the consumable. Those outside the
            horizon are ignored, and one that straddles its start or end counts
            for the part inside it, with its amount and accrual scaled to it.
        limit: The level a plan keeps the consumable below; None for no limit.
        start: The level at ``begin``.
        accrual: What accrues on its own, apart from any plan, over a span of
            time (its begin and end, Unix seconds) outside removals and their
            leads.
    """

    def __init__(
        self,
        begin: float,
        end: float,
        removals: Sequence[Removal],
        *,
        limit: float | None,
        start: float,
        accrual: Callable[[float, float], float],
    ) -> None:
        self.limit = limit
        self.start = float(start)
        inside: list[Removal] = []
        leads: list[tuple[float, float, float]] = []
        """Each lead inside the horizon: when it starts, when it ends, and
        what accrues in it. A lead belongs to the segment it starts in, that
        of the removal before it, whether or not its own removal is inside."""
        for removal in sorted(removals, key=lambda r: r.begin):
            lead_begin = max(removal.begin - removal.lead, begin)
            lead_end = min(removal.begin, end)
            if removal.lead > 0 and lead_end > lead_begin:
                share = (lead_end - lead_begin) / removal.lead
                leads.append((lead_begin, lead_end, removal.lead_accrual * share))
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
        self.removal_ends: list[float] = [begin, *(r.end for r in inside)]
        """When the removal at the start of each segment ends; the start of
        the horizon for the first segment, which has none."""
        self.accrual: list[float] = []
        """What accrues on its own in each segment, outside its removal: at
        the plain rate, and at their own rate in the leads of the removals
        that start in it."""
        for k in range(self.segments):
            # The removal at the start of a segment accrues on its own terms,
            # and so does each lead that starts in the segment.
            cursor = self.removal_ends[k]
            span_end = max(cursor, self.bounds[k + 1])
            accrued = 0.0
            for lead_begin, lead_end, lead_accrual in leads:
                if self.bounds[k] <= lead_begin < self.bounds[k + 1]:
                    plain_end = min(max(lead_begin, cursor), span_end)
                    accrued += accrual(cursor, plain_end) + lead_accrual
                    cursor = min(max(lead_end, cursor), span_end)
            self.accrual.append(accrued + accrual(cursor, span_end))

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

    def _run(self, level: float, segment: int, planned: S) -> tuple[float, float]:
        """Run a segment from the level it starts at: its peak and its end level.

        The plan's additions in a segment are a number here, added with what
        accrues; the level only rises within a segment, so its peak is its
        end. A subclass whose additions are ordered in time, and can take the
        consumable off as well as put it on, walks them instead.
        """
        assert isinstance(planned, (int, float))
        end = level + self.accrual[segment] + float(planned)
        return end, end

    def _peaks_and_levels(
        self, planned: Sequence[S]
    ) -> tuple[list[float], list[float]]:
        level = self.start
        peaks = []
        levels = []
        for k in range(self.segments):
            if k > 0:
                level = max(0.0, level + self.removal_accrual[k] - self.removal[k])
            peak, level = self._run(level, k, planned[k])
            peaks.append(peak)
            levels.append(level)
        return peaks, levels

    def levels(self, planned: Sequence[S]) -> list[float]:
        """The level at the end of each segment, given the plan's additions in each.

        A removal takes off what it can of what is there when it starts, and
        no more than that.
        """
        return self._peaks_and_levels(planned)[1]

    def peaks(self, planned: Sequence[S]) -> list[float]:
        """The highest level reached in each segment."""
        return self._peaks_and_levels(planned)[0]

    def fits(self, planned: Sequence[S], segment: int, extra: E) -> bool:
        """Whether ``extra`` more in ``segment`` keeps every peak below the limit.

        Below, not at it: the limit is where an alert fires.
        """
        if self.limit is None:
            return True
        return all(
            peak + _TOLERANCE < self.limit
            for peak in self.peaks(self.add(planned, segment, extra))
        )

    def fits_all(
        self, planned: Sequence[S], additions: Sequence[tuple[int, E]]
    ) -> bool:
        """Whether ``additions``, each some more in a segment, together keep
        every peak below the limit."""
        if self.limit is None:
            return True
        return all(
            peak + _TOLERANCE < self.limit
            for peak in self.peaks(self.add_all(planned, additions))
        )

    def add(self, planned: Sequence[S], segment: int, extra: E) -> tuple[S, ...]:
        """The plan's additions with ``extra`` more in ``segment``."""
        current = planned[segment]
        assert isinstance(current, (int, float)) and isinstance(extra, (int, float))
        added = list(planned)
        added[segment] = cast(S, float(current) + float(extra))
        return tuple(added)

    def add_all(
        self, planned: Sequence[S], additions: Sequence[tuple[int, E]]
    ) -> tuple[S, ...]:
        """The plan's additions with each of ``additions`` more in its segment."""
        added = tuple(planned)
        for segment, extra in additions:
            added = self.add(added, segment, extra)
        return added
