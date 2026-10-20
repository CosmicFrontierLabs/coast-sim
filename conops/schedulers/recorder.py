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

from collections.abc import Sequence

from ..common.enums import ACSMode
from ..config import MissionConfig
from ..targets import PlanEntry
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


class RecorderLedger(ConsumableLedger):
    """Recorder volume over a horizon, by the segments its contacts cut it into.

    Args:
        config: Mission configuration: the recorder and its planning limit,
            what the bus and instruments generate, and the ground stations'
            downlink rates.
        begin: Start of the horizon (Unix seconds).
        end: End of the horizon.
        contacts: The plan's ground-contact entries: each has a ``station``
            and the ``contact_begin`` and ``contact_end`` of its link, and its
            ``begin`` and ``slewtime`` give the slew into it. Those outside
            the horizon are ignored, and one that straddles its start or end
            counts for the part inside it.
        start_volume_gb: What is on the recorder at ``begin``.

    Housekeeping is counted at the nominal rate throughout, and at the
    ``PASS`` rate during contacts and the slews into them, which ACS runs in
    ``PASS`` mode: the plan does not know when the spacecraft will be in
    other modes. The limit is kept below, not reached: the recorder's red
    alert fires at its threshold.
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
        removals = []
        for entry in contacts:
            if (
                entry.station is None
                or entry.contact_begin is None
                or entry.contact_end is None
            ):
                continue
            link_begin = float(entry.contact_begin)
            link_end = float(entry.contact_end)
            if link_end <= link_begin:
                continue
            seconds = link_end - link_begin
            # The slew into the contact runs in PASS mode too, from the
            # entry's begin, though it carries no link.
            ingress = max(
                0.0, min(float(entry.slewtime or 0), link_begin - float(entry.begin))
            )
            removals.append(
                Removal(
                    link_begin,
                    link_end,
                    contact_downlink_gb(config, entry.station, seconds),
                    config.housekeeping_generated(seconds, mode=ACSMode.PASS),
                    lead=ingress,
                    lead_accrual=config.housekeeping_generated(
                        ingress, mode=ACSMode.PASS
                    ),
                )
            )
        super().__init__(
            begin,
            end,
            removals,
            limit=config.recorder.planning_limit_gb,
            start=start_volume_gb,
            accrual=config.housekeeping_generated,
        )

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
        nominal rate, and at the ``PASS`` rate during the slew into the next."""
        return self.accrual

    @property
    def contact_housekeeping_gb(self) -> list[float]:
        """Housekeeping generated during the contact at the start of each segment."""
        return self.removal_accrual

    def peaks(self, volumes: Sequence[float]) -> list[float]:
        """The recorder volume at the end of each segment."""
        return self.levels(volumes)

    # ── What a plan adds ──────────────────────────────────────────────────

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
