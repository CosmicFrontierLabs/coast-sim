"""The onboard recorder as a consumable: data builds up until a contact downlinks it.

Science and housekeeping data go onto the recorder all the time and come off
it only during ground contacts, so the volume rises between contacts and falls
during them. :class:`RecorderLedger` maps the recorder onto
:class:`~conops.schedulers.consumable.ConsumableLedger`: housekeeping is what
accrues on its own, a contact is a removal of what its rate and length allow,
and a science snapshot adds what the observed target generates in each
instrument. The limit is the recorder's planning limit.
"""

from __future__ import annotations

import math
from bisect import bisect_right
from collections.abc import Mapping, Sequence
from typing import NamedTuple

from ..common.enums import ACSMode, ObsType
from ..config import MissionConfig
from ..config.data_storage import RecorderView
from ..ditl.plan_validator import entry_obstype
from ..simulation.coverage import StationCoverage
from ..targets import PlanEntry
from .consumable import ConsumableLedger, Removal


class DataFlow(NamedTuple):
    """Gb produced uniformly over an interval, or at an instantaneous boundary."""

    begin: float
    end: float
    volume: float


RecorderAmount = float | tuple[DataFlow, ...]


def amount_total(amount: RecorderAmount) -> float:
    return amount if isinstance(amount, (int, float)) else sum(f.volume for f in amount)


def combine_amounts(
    a: RecorderAmount, b: RecorderAmount, sign: float = 1.0
) -> RecorderAmount:
    """Combine immutable flow histories; scalar stores retain their arithmetic."""
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return a + sign * b
    flows: dict[tuple[float, float], float] = {}
    for value, factor in ((a, 1.0), (b, sign)):
        if isinstance(value, (int, float)):
            if value != 0.0:
                raise ValueError(
                    "Timed recorder additions require their production interval"
                )
            continue
        for flow in value:
            key = (flow.begin, flow.end)
            flows[key] = flows.get(key, 0.0) + factor * flow.volume
    return tuple(
        DataFlow(lo, hi, v) for (lo, hi), v in sorted(flows.items()) if abs(v) > 1e-12
    )


def amounts_equal(a: RecorderAmount, b: RecorderAmount) -> bool:
    d = combine_amounts(a, b, -1.0)
    return (
        abs(d) <= 1e-9
        if isinstance(d, (int, float))
        else all(abs(f.volume) <= 1e-9 for f in d)
    )


def contact_downlink_gb(
    config: MissionConfig | RecorderView, station: str, seconds: float
) -> float:
    """What a contact of ``seconds`` with ``station`` downlinks, in Gb.

    The same arithmetic as the simulation: the effective rate in Mbps, from
    the station's and spacecraft's bands, for the time in contact.
    """
    if isinstance(config, RecorderView):
        return config.downlink_rate(station) * max(0.0, seconds) / 1000.0
    try:
        ground_station = config.ground_stations.get(station)
    except KeyError:
        return 0.0
    rate_mbps = ground_station.effective_downlink_rate_mbps(
        config.spacecraft_bus.communications
    )
    if rate_mbps is None:
        return 0.0
    return rate_mbps * max(0.0, seconds) / 1000.0


class RecorderLedger(ConsumableLedger[RecorderAmount, RecorderAmount]):
    """Recorder volume over a horizon, by the segments its contacts cut it into.

    Args:
        config: Mission configuration: the recorder and its planning limit,
            what the bus and instruments generate, and the ground stations'
            downlink rates.
        begin: Start of the horizon (Unix seconds).
        end: End of the horizon.
        contacts: The plan's ground-contact entries: each has a ``station``
            and the ``contact_begin`` and ``contact_end`` of its link. Those
            outside the horizon are ignored, and one that straddles its start
            or end counts for the part inside it.
        start_volume_gb: What is on the recorder at ``begin``.
        step_size: The simulation step in seconds. The simulation generates
            housekeeping for each whole step at the rate of the mode the step
            starts in, so the ledger counts it the same way: a span counts
            for the steps that start in it. None counts spans exactly.
        origin: A time the step grid passes through; ``begin`` if None.
        coverage: Link coverage for passes without spacecraft tracking.
            Downlink is credited at the end of each simulation step or link
            span, with science split across those boundaries. The peak before each
            credit conservatively counts that step's incoming data before
            its outgoing data. Housekeeping stays at the nominal rate.

    Housekeeping is counted at the nominal rate throughout, and at the
    ``PASS`` rate during spacecraft tracking: the plan does not know when the spacecraft
    will be in other modes. A horizon that ends between steps counts the
    last step whole, as a plan that hands over to another does; the
    simulation's last step, cut at its end, generates a little less. The slew into a contact, which ACS flies in
    ``PASS`` mode too, depends on where the spacecraft is before it, so what
    it adds beyond the nominal rate is not in the ledger's own accrual: the
    schedulers add it with the contact, as they add science, and recompute it
    when the slew changes (:meth:`ingress_volume`). The limit is kept below,
    not reached: the recorder's red alert fires at its threshold.
    """

    def __init__(
        self,
        config: MissionConfig | RecorderView,
        begin: float,
        end: float,
        contacts: Sequence[PlanEntry],
        *,
        start_volume_gb: float,
        step_size: float | None = None,
        origin: float | None = None,
        coverage: StationCoverage | None = None,
    ) -> None:
        self.config = config
        self.step_size = float(step_size) if step_size is not None else None
        self.origin = float(begin if origin is None else origin)
        self._timed_science = coverage is not None
        self._end = float(end)
        removals: list[Removal] = []
        tails: list[tuple[float, float]] = []
        """Where each contact's removal starts, and what the step its link
        ends in generates after the link, at the PASS rate, beyond the
        nominal rate the segment counts it at."""
        self._link_flows: tuple[DataFlow, ...] = ()
        if coverage is not None:
            spans = coverage.rate_spans(
                self.grid_step(begin),
                end,
                config if isinstance(config, RecorderView) else None,
            )
            self._link_flows = tuple(
                DataFlow(p.begin, p.end, -p.rate_mbps * (p.end - p.begin) / 1000.0)
                for p in spans
            )
            # Coarse solver/retry boundaries follow link changes. The exact
            # sweep below also includes every science production boundary.
            amounts: dict[float, float] = {}
            for span in spans:
                amounts[span.end] = (
                    amounts.get(span.end, 0.0)
                    + span.rate_mbps * (span.end - span.begin) / 1000.0
                )
            removals.extend(Removal(t, t, amount) for t, amount in amounts.items())
        for entry in contacts:
            if (
                entry.station is None
                or entry.contact_begin is None
                or entry.contact_end is None
            ):
                continue
            # The part of the link inside the horizon sets what it downlinks.
            link_begin = max(float(entry.contact_begin), float(begin))
            link_end = min(float(entry.contact_end), float(end))
            if link_end <= link_begin:
                continue
            # The link removes data from the step it begins in, and what
            # that step and the ones after generate up to the link's end;
            # what the last step generates after the link stays on.
            first = self.grid_step(link_begin)
            removals.append(
                Removal(
                    first,
                    link_end,
                    contact_downlink_gb(config, entry.station, link_end - link_begin),
                    config.housekeeping_generated(link_end - first, mode=ACSMode.PASS),
                )
            )
            after = self.grid_step(link_end) - link_end
            if after > 0.0 and link_end < float(end):
                tails.append(
                    (
                        first,
                        config.housekeeping_generated(after, mode=ACSMode.PASS)
                        - config.housekeeping_generated(after),
                    )
                )
        super().__init__(
            self.grid_step(begin),
            self.grid_step(end),
            removals,
            limit=config.recorder.planning_limit_gb,
            start=start_volume_gb,
            accrual=lambda span_begin, span_end: config.housekeeping_generated(
                span_end - span_begin
            ),
        )
        for first, excess in tails:
            self.accrual[self.segment(first)] += excess

    def grid_step(self, utime: float) -> float:
        """The first simulation step at or after ``utime``, or ``utime``
        itself without a step grid. A span's steps are those that start in
        it, so a span between two of these is a whole number of steps."""
        if self.step_size is None:
            return float(utime)
        steps = math.ceil((utime - self.origin) / self.step_size - 1e-9)
        return self.origin + max(0, steps) * self.step_size

    # ── In the recorder's terms ───────────────────────────────────────────

    @property
    def limit_gb(self) -> float | None:
        """The planning limit in Gb, or None."""
        return self.limit

    @property
    def start_volume_gb(self) -> float:
        return self.start

    @property
    def downlink_gb(self) -> list[float]:
        """What the contact at the start of each segment downlinks."""
        return self.removal

    @property
    def housekeeping_gb(self) -> list[float]:
        """Housekeeping generated in each segment outside its contact, at the
        nominal rate."""
        return self.accrual

    @property
    def contact_housekeeping_gb(self) -> list[float]:
        """Housekeeping generated during the contact at the start of each segment."""
        return self.removal_accrual

    # ── What a plan adds ──────────────────────────────────────────────────

    def science_volume(self, entry: PlanEntry, collection_seconds: float) -> float:
        """What a snapshot of ``entry`` collecting for some seconds generates."""
        return self.config.science_generated(collection_seconds, entry.science_data)

    def science_additions(
        self,
        entry: PlanEntry,
        seconds: float,
        time: float,
        *,
        include_fixed: bool = True,
    ) -> list[tuple[int, RecorderAmount]]:
        """Production intervals, coalesced between partial execution steps."""
        fixed = self.science_volume(entry, 0.0)
        if not self._timed_science:
            volume = self.science_volume(entry, seconds)
            return [(self.segment(time), volume if include_fixed else volume - fixed)]
        if seconds <= 0.0:
            return []
        rate = self.science_volume(entry, 1.0) - fixed
        flows = []
        if self.step_size is None:
            flows.append(DataFlow(time, time + seconds, rate * seconds))
            if include_fixed and fixed:
                flows.append(DataFlow(time, time, fixed))
        else:
            step = self.step_size
            first = self.origin + math.floor((time - self.origin) / step) * step
            last = (
                self.origin
                + math.floor((time + seconds - self.origin - 1e-6) / step) * step
            )
            edges = sorted(
                {first, min(first + step, self._end), last, min(last + step, self._end)}
            )
            for lo, hi in zip(edges, edges[1:]):
                duration = max(0.0, min(hi, time + seconds) - max(lo, time))
                if hi > lo and duration > 0.0:
                    volume = rate * duration + (
                        fixed if include_fixed and lo == first else 0.0
                    )
                    flows.append(DataFlow(lo, hi, volume))
        additions: dict[int, tuple[DataFlow, ...]] = {}
        for flow in flows:
            if flow.begin == flow.end:
                k = self.segment(flow.begin)
                additions[k] = (*additions.get(k, ()), flow)
                continue
            for k in range(self.segment(flow.begin), self.segments):
                lo, hi = (
                    max(flow.begin, self.bounds[k]),
                    min(flow.end, self.bounds[k + 1]),
                )
                if hi > lo:
                    piece = DataFlow(
                        lo, hi, flow.volume * (hi - lo) / (flow.end - flow.begin)
                    )
                    additions[k] = (*additions.get(k, ()), piece)
                if self.bounds[k + 1] >= flow.end:
                    break
        return list(additions.items())

    def _walk(
        self, planned: Sequence[RecorderAmount], stop: float | None = None
    ) -> tuple[list[float], list[float]]:
        """Sweep rate and production changes in order, reflecting volume at zero."""
        events: dict[float, list[float]] = {t: [0.0, 0.0] for t in self.bounds}
        end = self.bounds[-1] if stop is None else stop
        events.setdefault(end, [0.0, 0.0])
        flows = list(self._link_flows)
        for k, amount in enumerate(planned):
            if isinstance(amount, (int, float)):
                if amount:
                    flows.append(DataFlow(self.bounds[k], self.bounds[k], amount))
            else:
                flows.extend(amount)
        for flow in flows:
            if flow.begin == flow.end:
                events.setdefault(flow.begin, [0.0, 0.0])[1] += flow.volume
            else:
                rate = flow.volume / (flow.end - flow.begin)
                events.setdefault(flow.begin, [0.0, 0.0])[0] += rate
                events.setdefault(flow.end, [0.0, 0.0])[0] -= rate
        level = peak = self.start
        rate = self.config.housekeeping_generated(1.0)
        cursor = self.bounds[0]
        peaks: list[float] = []
        levels: list[float] = []
        boundary = 1
        for time, (change, impulse) in sorted(events.items()):
            if time < cursor:
                continue
            if time > end:
                break
            level = max(0.0, level + rate * (time - cursor))
            peak = max(peak, level)
            while boundary < len(self.bounds) and self.bounds[boundary] <= time:
                peaks.append(peak)
                levels.append(level)
                peak = level
                boundary += 1
            level += impulse
            peak = max(peak, level)
            rate += change
            cursor = time
        return ([max(peak, level)], [level]) if stop is not None else (peaks, levels)

    def _peaks_and_levels(
        self, planned: Sequence[RecorderAmount]
    ) -> tuple[list[float], list[float]]:
        return (
            self._walk(planned)
            if self._timed_science
            else super()._peaks_and_levels(planned)
        )

    def level_at(self, planned: Sequence[RecorderAmount], time: float) -> float:
        if self._timed_science:
            return self._walk(planned, time)[1][-1]
        if time >= self.bounds[-1]:
            return self.levels(planned)[-1]
        first = self.segment(time)
        return (
            self.peaks(planned)[first]
            - self.accrual[first]
            - amount_total(planned[first])
        )

    def add(
        self, planned: Sequence[RecorderAmount], segment: int, extra: RecorderAmount
    ) -> tuple[RecorderAmount, ...]:
        result = list(planned)
        result[segment] = combine_amounts(result[segment], extra)
        return tuple(result)

    def ingress_volume(
        self,
        entry: PlanEntry,
        *,
        begin: float | None = None,
        slewtime: float | None = None,
    ) -> float:
        """What the slew into the contact ``entry`` adds beyond nominal
        housekeeping, in Gb: ACS flies it in ``PASS`` mode, so each step it
        is under way counts at the ``PASS`` rate instead. As it would be
        flown from ``begin`` for ``slewtime`` seconds, if given, else as the
        entry has it; only the part inside the horizon counts. 0 for
        anything but a contact.
        """
        if (
            entry_obstype(entry) != ObsType.GSP
            or entry.station is None
            or entry.contact_begin is None
        ):
            return 0.0
        slew_begin = float(entry.begin if begin is None else begin)
        seconds = float(entry.slewtime or 0 if slewtime is None else slewtime)
        slew_end = min(slew_begin + seconds, float(entry.contact_begin))
        first = max(self.grid_step(slew_begin), self.bounds[0])
        last = min(self.grid_step(slew_end), self.bounds[-1])
        if last <= first:
            return 0.0
        return self.config.housekeeping_generated(
            last - first, mode=ACSMode.PASS
        ) - self.config.housekeeping_generated(last - first)

    def ingress_segment(self, entry: PlanEntry, *, begin: float | None = None) -> int:
        """The segment the slew into the contact ``entry`` starts in, from
        ``begin`` if given."""
        return self.segment(
            self.grid_step(float(entry.begin if begin is None else begin))
        )

    def initial_volumes(
        self, entries: Sequence[PlanEntry]
    ) -> tuple[RecorderAmount, ...]:
        """What entries already on the timeline add, by segment: science
        where it is collected, and the slews into contacts where they start."""
        volumes: list[RecorderAmount] = [0.0] * self.segments
        for entry in entries:
            ingress = self.ingress_volume(entry)
            if ingress != 0.0:
                # Negative where the PASS rate is below the nominal one.
                k = self.ingress_segment(entry)
                volumes[k] = combine_amounts(volumes[k], ingress)
            if entry.collection_begin is None or entry.collection_end is None:
                continue
            seconds = float(entry.collection_end) - float(entry.collection_begin)
            if seconds <= 0.0:
                continue
            for segment, volume in self.science_additions(
                entry, seconds, float(entry.collection_begin)
            ):
                volumes[segment] = combine_amounts(volumes[segment], volume)
        return tuple(volumes)


class RecorderLedgers:
    """Independent scalar ledgers stored in one flat planning-state tuple.

    Offsets identify stores; capacity and downlink credit never cross them.
    """

    def __init__(
        self,
        config: MissionConfig,
        begin: float,
        end: float,
        contacts: Sequence[PlanEntry],
        *,
        start_volume_gb: float | Mapping[str, float] | None = None,
        step_size: float | None = None,
        origin: float | None = None,
        coverage: StationCoverage | None = None,
    ) -> None:
        views = config.recorder_views()
        if isinstance(start_volume_gb, Mapping) and set(start_volume_gb) != set(views):
            raise ValueError("initial volumes must specify every recorder exactly once")
        if (
            isinstance(start_volume_gb, (int, float))
            and config.data_storage is not None
        ):
            raise ValueError("named recorders require initial volumes by store")
        self.members: list[tuple[str, RecorderLedger, int]] = []
        offset = 0
        for name, view in views.items():
            start = (
                start_volume_gb[name]
                if isinstance(start_volume_gb, Mapping)
                else start_volume_gb
                if start_volume_gb is not None
                else view.recorder.current_volume_gb
            )
            ledger = RecorderLedger(
                view,
                begin,
                end,
                contacts,
                start_volume_gb=start,
                step_size=step_size,
                origin=origin,
                coverage=coverage,
            )
            self.members.append((name, ledger, offset))
            offset += ledger.segments
        self.segments = offset

    def next_contact(self, time: float) -> float | None:
        """Next useful link boundary, found by binary lookup."""
        return min(
            (
                ledger.bounds[k]
                for _, ledger, _ in self.members
                if (k := max(1, bisect_right(ledger.bounds, time))) < ledger.segments
            ),
            default=None,
        )

    @property
    def active(self) -> bool:
        return any(ledger.active for _, ledger, _ in self.members)

    def initial_volumes(
        self, entries: Sequence[PlanEntry]
    ) -> tuple[RecorderAmount, ...]:
        return tuple(
            v for _, ledger, _ in self.members for v in ledger.initial_volumes(entries)
        )

    def peaks(self, planned: Sequence[RecorderAmount]) -> list[float]:
        return [
            v
            for _, ledger, offset in self.members
            for v in ledger.peaks(planned[offset : offset + ledger.segments])
        ]

    def fits_all(
        self,
        planned: Sequence[RecorderAmount],
        additions: Sequence[tuple[int, RecorderAmount]],
    ) -> bool:
        trial = self.add_all(planned, additions)
        return all(
            ledger.fits_all(trial[offset : offset + ledger.segments], [])
            for _, ledger, offset in self.members
        )

    @staticmethod
    def add_all(
        planned: Sequence[RecorderAmount],
        additions: Sequence[tuple[int, RecorderAmount]],
    ) -> tuple[RecorderAmount, ...]:
        result = list(planned)
        for segment, value in additions:
            result[segment] = combine_amounts(result[segment], value)
        return tuple(result)

    def science_volume(self, entry: PlanEntry, seconds: float) -> float:
        return sum(
            ledger.science_volume(entry, seconds) for _, ledger, _ in self.members
        )

    def science_additions(
        self, entry: PlanEntry, seconds: float, time: float
    ) -> list[tuple[int, RecorderAmount]]:
        return [
            (offset + segment, volume)
            for _, ledger, offset in self.members
            for segment, volume in ledger.science_additions(entry, seconds, time)
        ]

    def ingress_additions(
        self,
        entry: PlanEntry,
        *,
        begin: float | None = None,
        slewtime: float | None = None,
    ) -> list[tuple[int, float]]:
        return [
            (
                offset + ledger.ingress_segment(entry, begin=begin),
                ledger.ingress_volume(entry, begin=begin, slewtime=slewtime),
            )
            for _, ledger, offset in self.members
        ]
