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
from collections.abc import Sequence

from ..common.enums import ACSMode, ObsType
from ..config import MissionConfig
from ..ditl.plan_validator import entry_obstype
from ..simulation.coverage import StationCoverage
from ..targets import PlanEntry
from ..targets.plan_entry import extra_data_gb, instrument_modes_of
from .consumable import ConsumableLedger, Removal


def contact_downlink_gb(config: MissionConfig, station: str, seconds: float) -> float:
    """What a contact of ``seconds`` with ``station`` downlinks, in Gb.

    The same arithmetic as the simulation: the effective rate in Mbps, from
    the station's and spacecraft's bands, for the time in contact.
    """
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


class RecorderLedger(ConsumableLedger[float, float]):
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
        coverage: Station coverage for an antenna that needs no attitude.
            Its spans of contact in the horizon downlink at their station's
            rate; the spacecraft stays in whatever mode it is in, so
            housekeeping is at the nominal rate throughout. A plan for such
            an antenna has no ground-contact entries.
        step_size: The simulation step in seconds. The simulation generates
            housekeeping for each whole step at the rate of the mode the step
            starts in, so the ledger counts it the same way: a span counts
            for the steps that start in it. None counts spans exactly.
        origin: A time the step grid passes through; ``begin`` if None.

    Housekeeping is counted at the nominal rate throughout, and at the
    ``PASS`` rate during contacts: the plan does not know when the spacecraft
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
        config: MissionConfig,
        begin: float,
        end: float,
        contacts: Sequence[PlanEntry],
        *,
        start_volume_gb: float,
        coverage: StationCoverage | None = None,
        step_size: float | None = None,
        origin: float | None = None,
    ) -> None:
        self.config = config
        self.step_size = float(step_size) if step_size is not None else None
        self.origin = float(begin if origin is None else origin)
        removals = []
        tails: list[tuple[float, float]] = []
        """Where each contact's removal starts, and what the step its link
        ends in generates after the link, at the PASS rate, beyond the
        nominal rate the segment counts it at."""
        if coverage is not None:
            for span in coverage.spans(begin, end):
                seconds = span.end - span.begin
                # The spacecraft stays in its mode, so the steps in contact
                # generate at the nominal rate; the link's own length sets
                # what it downlinks.
                first = self.grid_step(span.begin)
                if span.end <= first:
                    continue
                removals.append(
                    Removal(
                        first,
                        span.end,
                        span.rate_mbps * seconds / 1000.0,
                        config.housekeeping_generated(span.end - first),
                    )
                )
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
        return self.config.science_generated(
            collection_seconds, entry.science_data, instrument_modes_of(entry)
        ) + extra_data_gb(entry)

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

    def initial_volumes(self, entries: Sequence[PlanEntry]) -> tuple[float, ...]:
        """What entries already on the timeline add, by segment: science
        where it is collected, and the slews into contacts where they start."""
        volumes = [0.0] * self.segments
        for entry in entries:
            ingress = self.ingress_volume(entry)
            if ingress != 0.0:
                # Negative where the PASS rate is below the nominal one.
                volumes[self.ingress_segment(entry)] += ingress
            if entry_obstype(entry) == ObsType.ACTIVITY:
                # An activity's frames, when it begins.
                fixed = extra_data_gb(entry)
                if fixed > 0.0:
                    volumes[self.segment(float(entry.begin))] += fixed
            if entry.collection_begin is None or entry.collection_end is None:
                continue
            seconds = float(entry.collection_end) - float(entry.collection_begin)
            if seconds <= 0.0:
                continue
            segment = self.segment(float(entry.collection_begin))
            volumes[segment] += self.science_volume(entry, seconds)
        return tuple(volumes)
