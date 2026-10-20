"""The onboard recorder's volume over a planning horizon.

Data goes onto the recorder all the time and comes off it only during ground
contacts, so the volume rises between contacts and falls during them. Its
highest points are just before each contact and at the end of the horizon.
The contacts cut the horizon into *segments*; a segment runs from the start
of one contact to the start of the next. What a plan puts in a segment is
checked against the recorder's planning limit at that segment's peak, and at
every later peak, since what a contact cannot remove carries over.

Schedulers keep a plan's science volume by segment and ask the ledger whether
a snapshot's volume fits where they mean to put it. Nothing is ever shortened
to fit: a snapshot that does not fit is placed elsewhere or not at all.
"""

from __future__ import annotations

from bisect import bisect_right
from collections.abc import Sequence

from ..common.enums import ACSMode
from ..config import MissionConfig
from ..targets import PlanEntry

_TOLERANCE_GB = 1e-9
"""Slack for rounding in sums of volumes."""


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


class RecorderLedger:
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

    Housekeeping is counted at the nominal rate throughout, and at the
    ``PASS`` rate during contacts: the plan does not know when the
    spacecraft will be in other modes.
    """

    def __init__(
        self,
        config: MissionConfig,
        begin: float,
        end: float,
        contacts: Sequence[PlanEntry],
        *,
        start_volume_gb: float,
    ) -> None:
        self.config = config
        self.limit_gb = config.recorder.planning_limit_gb
        self.start_volume_gb = float(start_volume_gb)
        inside: list[tuple[float, float, str]] = []
        for entry in contacts:
            if (
                entry.station is None
                or entry.contact_begin is None
                or entry.contact_end is None
            ):
                continue
            link_begin = max(float(entry.contact_begin), begin)
            link_end = min(float(entry.contact_end), end)
            if link_end > link_begin:
                inside.append((link_begin, link_end, entry.station))
        inside.sort()
        # Segment k starts at bounds[k]; contact k - 1 is at the start of segment k.
        self.bounds: list[float] = [begin, *(c[0] for c in inside), end]
        self.downlink_gb: list[float] = [
            0.0,
            *(contact_downlink_gb(config, station, e - b) for b, e, station in inside),
        ]
        self.contact_housekeeping_gb: list[float] = [
            0.0,
            *(
                config.housekeeping_generated(e - b, mode=ACSMode.PASS)
                for b, e, _ in inside
            ),
        ]
        self.housekeeping_gb: list[float] = []
        """Housekeeping generated in each segment outside its contact."""
        for k in range(self.segments):
            seconds = self.bounds[k + 1] - self.bounds[k]
            if k > 0:
                seconds -= inside[k - 1][1] - inside[k - 1][0]
            self.housekeeping_gb.append(
                config.housekeeping_generated(max(0.0, seconds))
            )

    @property
    def active(self) -> bool:
        """Whether the recorder limits what may be planned."""
        return self.limit_gb is not None

    @property
    def segments(self) -> int:
        return len(self.bounds) - 1

    def segment(self, time: float) -> int:
        """The segment a time falls in."""
        return min(max(bisect_right(self.bounds, time) - 1, 0), self.segments - 1)

    def science_volume(self, entry: PlanEntry, collection_seconds: float) -> float:
        """What a snapshot of ``entry`` collecting for some seconds generates."""
        return self.config.science_generated(collection_seconds, entry.science_data)

    def initial_volumes(self, entries: Sequence[PlanEntry]) -> tuple[float, ...]:
        """Science volume by segment of entries already on the timeline."""
        volumes = [0.0] * self.segments
        for entry in entries:
            if entry.collection_begin is None or entry.collection_end is None:
                continue
            seconds = float(entry.collection_end) - float(entry.collection_begin)
            if seconds <= 0.0:
                continue
            segment = self.segment(float(entry.collection_begin))
            volumes[segment] += self.science_volume(entry, seconds)
        return tuple(volumes)

    def peaks(self, volumes: Sequence[float]) -> list[float]:
        """The recorder volume at the end of each segment.

        A contact removes what it can of what is there when it starts, and
        no more than that.
        """
        fill = self.start_volume_gb
        peaks = []
        for k in range(self.segments):
            if k > 0:
                fill = max(
                    0.0, fill + self.contact_housekeeping_gb[k] - self.downlink_gb[k]
                )
            fill += self.housekeeping_gb[k] + volumes[k]
            peaks.append(fill)
        return peaks

    def fits(self, volumes: Sequence[float], segment: int, extra_gb: float) -> bool:
        """Whether ``extra_gb`` more in ``segment`` keeps every peak within the limit."""
        if self.limit_gb is None:
            return True
        trial = list(volumes)
        trial[segment] += extra_gb
        return all(peak <= self.limit_gb + _TOLERANCE_GB for peak in self.peaks(trial))

    def add(
        self, volumes: Sequence[float], segment: int, extra_gb: float
    ) -> tuple[float, ...]:
        """The volumes with ``extra_gb`` more in ``segment``."""
        added = list(volumes)
        added[segment] += extra_gb
        return tuple(added)
