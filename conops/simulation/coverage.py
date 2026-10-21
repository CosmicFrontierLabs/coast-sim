"""Ground-station contacts for antennas that need no spacecraft attitude.

A body-fixed antenna downlinks only during a ground-station pass, when the
spacecraft points the antenna at the station (:class:`~conops.simulation.passes.PassTimes`).
An omnidirectional or gimballed antenna downlinks whenever it has a link to a
station, in whatever ACS mode the spacecraft is in, so its contacts are a
matter of geometry and of the link: :class:`StationCoverage` finds, at every
step of the ephemeris, which stations are above their elevation masks, and
schedules the contacts from that. One station is tracked at a time: a contact
stays with its station until it sets, then the link is handed over to the
best station then in view. Every contact starts with the communications
system's acquisition latency, and a handover adds its own, during which
nothing is downlinked. Every station with a downlink
carries data, whether or not it may be scheduled for spacecraft-tracking
passes.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import rust_ephem

from ..common import AntennaType, scbodyvector
from ..config import CommunicationsSystem, MissionConfig
from ..config.groundstation import (
    GroundStation,
    effective_downlink_rate_mbps,
    rate_varies_with_range,
)

Attitude = tuple[float, float, float]
"""Spacecraft body attitude: RA, Dec and roll in degrees."""


class ContactSpan(NamedTuple):
    """A span of continuous contact with one station."""

    begin: float
    """Start of the span, in Unix seconds."""
    end: float
    """End of the span."""
    station: str
    """The station's code."""
    rate_mbps: float
    """The mean downlink rate through the span, in Mbps: the span downlinks
    this rate for its length. With a rate that varies with range it is the
    time-weighted mean of the rate at each step."""


class Contact(NamedTuple):
    """The station in contact at one step."""

    station: str
    rate_mbps: float
    """The downlink rate at the step, at its range."""
    range_km: float
    """Distance from the spacecraft to the station."""


class StationCoverage:
    """The contacts of an omni or gimballed antenna over the ephemeris.

    Args:
        config: Mission configuration: the stations, the communications
            system and the ephemeris on its constraint.

    At each step of the ephemeris a station is in view when its elevation is
    above its mask. Contacts are scheduled from the views: the link stays
    with its station until that station sets, so two stations in view at
    once do not share the time; when it sets, or when no station was
    tracked, the link goes to the station in view with the highest downlink
    rate, and of those the one in view for longest. Every contact begins
    the communications system's ``acquisition_seconds`` after its station
    comes into view, and a switch straight from one station to another
    adds ``handover_seconds`` to that; nothing is downlinked meanwhile. A
    station acquired after a gap with none in view pays no handover, since
    the antenna can be pointed at it in advance. A gimballed antenna also has to reach
    the station within its gimbal range from its boresight, which depends
    on the attitude flown and is checked at execution (:meth:`contact_at`);
    planning takes the station as reachable.
    """

    def __init__(self, config: MissionConfig) -> None:
        ephem = config.constraint.ephem
        if ephem is None:
            raise ValueError("config.constraint.ephem must be set for coverage")
        self.config = config
        self.ephem = ephem
        self.times: npt.NDArray[np.float64] = np.array(
            [t.timestamp() for t in ephem.timestamp], dtype=float
        )
        self.step_size = float(ephem.step_size)
        communications = config.spacecraft_bus.communications
        self.acquisition_seconds = (
            float(communications.acquisition_seconds)
            if communications is not None
            else 0.0
        )
        self.handover_seconds = (
            float(communications.handover_seconds)
            if communications is not None
            else 0.0
        )
        self.stations: list[GroundStation] = [
            station
            for station in config.ground_stations.stations
            if (effective_downlink_rate_mbps(station, communications) or 0.0) > 0.0
        ]
        self.rates: npt.NDArray[np.float64] = np.array(
            [
                float(effective_downlink_rate_mbps(s, communications) or 0.0)
                for s in self.stations
            ],
            dtype=float,
        )
        satellite = np.asarray(ephem.gcrs_pv.position, dtype=float)
        count, steps = len(self.stations), len(self.times)
        self.elevation_deg = np.full((count, steps), -90.0)
        self.range_km = np.full((count, steps), np.inf)
        self._to_station = np.zeros((count, steps, 3))
        """Unit vectors from the spacecraft to each station, in GCRS."""
        for k, station in enumerate(self.stations):
            ground = rust_ephem.GroundEphemeris(
                latitude=station.latitude_deg,
                longitude=station.longitude_deg,
                height=station.elevation_m,
                begin=ephem.timestamp[0],
                end=ephem.timestamp[-1],
                step_size=ephem.step_size,
            )
            station_pos = np.asarray(ground.gcrs_pv.position, dtype=float)
            to_satellite = satellite - station_pos
            distance = np.linalg.norm(to_satellite, axis=1)
            to_satellite_unit = to_satellite / distance[:, None]
            up = station_pos / np.linalg.norm(station_pos, axis=1)[:, None]
            cosine = np.sum(up * to_satellite_unit, axis=1)
            self.elevation_deg[k] = np.degrees(np.arcsin(np.clip(cosine, -1.0, 1.0)))
            self.range_km[k] = distance
            self._to_station[k] = -to_satellite_unit
        masks = np.array([s.min_elevation_deg for s in self.stations], dtype=float)
        self.visible: npt.NDArray[np.bool_] = self.elevation_deg > masks[:, None]
        """Whether each station is above its mask at each step."""
        self.rate_steps = np.where(self.visible, self.rates[:, None], 0.0)
        """The downlink rate from each station at each step in view, at the
        step's range where a band's rate varies with it."""
        self._ranged = [
            rate_varies_with_range(s, communications) for s in self.stations
        ]
        """Whether each station's rate varies with range."""
        for k, station in enumerate(self.stations):
            if not self._ranged[k]:
                continue
            for step in np.flatnonzero(self.visible[k]):
                self.rate_steps[k, step] = float(
                    effective_downlink_rate_mbps(
                        station, communications, float(self.range_km[k, step])
                    )
                    or 0.0
                )
        # Gb downlinked from each station by the start of each step, were it
        # in contact throughout: the rate is constant within a step.
        self._cumulative = np.zeros((count, steps + 1))
        self._cumulative[:, 1:] = np.cumsum(
            self.rate_steps * self.step_size / 1000.0, axis=1
        )
        self.contacts: list[ContactSpan] = []
        """The scheduled contacts over the whole ephemeris, in time order."""
        self.handovers = 0
        """How many contacts began with a handover from another station."""
        self._schedule_contacts()
        self._begins = [span.begin for span in self.contacts]

    @classmethod
    def for_config(cls, config: MissionConfig) -> StationCoverage | None:
        """Coverage for a configuration whose antenna needs no attitude, else None.

        None for a body-fixed antenna, which downlinks during passes, for a
        configuration without a communications system, which keeps the
        legacy body-fixed behaviour, and where no station has a downlink.
        """
        # A test double's bus has no communications system.
        communications = getattr(config.spacecraft_bus, "communications", None)
        if not isinstance(communications, CommunicationsSystem):
            return None
        if config.constraint.ephem is None:
            return None
        if communications.antenna_pointing.antenna_type == AntennaType.FIXED:
            return None
        if not any(
            (effective_downlink_rate_mbps(station, communications) or 0.0) > 0.0
            for station in config.ground_stations.stations
        ):
            return None
        return cls(config)

    # ── Scheduling ────────────────────────────────────────────────────────

    def _schedule_contacts(self) -> None:
        """Walk the ephemeris tracking one station at a time (see the class)."""
        count, steps = self.visible.shape
        if count == 0 or steps == 0:
            return
        # How many consecutive steps each station stays in view from each step.
        remaining = np.zeros((count, steps), dtype=int)
        for k in range(steps - 1, -1, -1):
            later = remaining[:, k + 1] if k + 1 < steps else 0
            remaining[:, k] = np.where(self.visible[:, k], later + 1, 0)
        current = -1
        begin = 0.0
        for k in range(steps):
            t = float(self.times[k])
            if current >= 0 and self.visible[current, k]:
                continue
            if current >= 0:
                # The tracked station has set.
                self._add_contact(current, begin, t)
            in_view = np.flatnonzero(self.visible[:, k])
            if in_view.size == 0:
                current = -1
                continue
            chosen = max(
                (int(j) for j in in_view),
                key=lambda j: (self.rate_steps[j, k], remaining[j, k], -j),
            )
            begin = t + self.acquisition_seconds
            if current >= 0:
                begin += self.handover_seconds
                self.handovers += 1
            current = chosen
        if current >= 0:
            self._add_contact(current, begin, float(self.times[-1]) + self.step_size)

    def _add_contact(self, station: int, begin: float, end: float) -> None:
        if end > begin:
            self.contacts.append(self._span(station, begin, end))

    def _span(self, station: int, begin: float, end: float) -> ContactSpan:
        """A span with ``station`` over ``[begin, end)`` at its mean rate."""
        if self._ranged[station]:
            downlinked = self._downlinked(station, end) - self._downlinked(
                station, begin
            )
            rate = downlinked * 1000.0 / (end - begin)
        else:
            rate = float(self.rates[station])
        return ContactSpan(begin, end, self.stations[station].code, rate)

    def _downlinked(self, station: int, utime: float) -> float:
        """Gb from ``station`` by ``utime`` were it in contact throughout."""
        k = self.index(utime)
        into = min(max(utime - float(self.times[k]), 0.0), self.step_size)
        return float(
            self._cumulative[station, k] + self.rate_steps[station, k] * into / 1000.0
        )

    def _station_index(self, code: str) -> int:
        return next(
            n for n, station in enumerate(self.stations) if station.code == code
        )

    # ── Lookup ────────────────────────────────────────────────────────────

    @property
    def gimballed(self) -> bool:
        communications = self.config.spacecraft_bus.communications
        return (
            communications is not None
            and communications.antenna_pointing.antenna_type == AntennaType.GIMBALED
        )

    def index(self, utime: float) -> int:
        """The ephemeris step at or before ``utime``."""
        return min(max(bisect_right(self.times, utime) - 1, 0), len(self.times) - 1)

    def _contact_index(self, utime: float) -> int:
        """The index of the contact in progress at ``utime``, or -1."""
        i = bisect_right(self._begins, utime) - 1
        if i >= 0 and utime < self.contacts[i].end:
            return i
        return -1

    def contact_at(
        self, utime: float, attitude: Attitude | None = None
    ) -> Contact | None:
        """The station in contact at ``utime``, or None.

        With ``attitude`` (RA, Dec, roll in degrees) a gimballed antenna must
        reach the station within its gimbal range from its boresight; an
        omni antenna reaches every station it is linked to.
        """
        i = self._contact_index(utime)
        if i < 0:
            return None
        span = self.contacts[i]
        j = self._station_index(span.station)
        k = self.index(utime)
        if attitude is not None and self.gimballed:
            communications = self.config.spacecraft_bus.communications
            assert communications is not None
            antenna = communications.antenna_pointing
            ra, dec, roll = attitude
            direction = scbodyvector(
                np.deg2rad(ra),
                np.deg2rad(dec),
                np.deg2rad(roll),
                self._to_station[j, k],
            )
            boresight = np.asarray(antenna.fixed_boresight_body, dtype=float)
            boresight /= np.linalg.norm(boresight)
            cosine = float(np.dot(direction, boresight) / np.linalg.norm(direction))
            if (
                np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                > antenna.gimbal_range_deg
            ):
                return None
        return Contact(
            span.station, float(self.rate_steps[j, k]), float(self.range_km[j, k])
        )

    def spans(self, begin: float, end: float) -> list[ContactSpan]:
        """The contacts in ``[begin, end)``, clipped to the window, each at
        its mean rate over the part inside it."""
        spans: list[ContactSpan] = []
        first = max(0, bisect_right(self._begins, begin) - 1)
        for span in self.contacts[first:]:
            if span.begin >= end:
                break
            lo, hi = max(span.begin, begin), min(span.end, end)
            if hi > lo:
                if (lo, hi) == (span.begin, span.end):
                    spans.append(span)
                else:
                    spans.append(self._span(self._station_index(span.station), lo, hi))
        return spans

    def downlink_gb(self, begin: float, end: float) -> float:
        """What the contacts in ``[begin, end)`` downlink, in Gb."""
        return sum(
            span.rate_mbps * (span.end - span.begin) / 1000.0
            for span in self.spans(begin, end)
        )

    def contact_in(
        self, begin: float, end: float, attitude: Attitude | None = None
    ) -> tuple[float, float, float]:
        """The contact in ``[begin, end)``: when it starts, when it stops and
        what it downlinks in Gb, ``(begin, begin, 0.0)`` without one.

        With ``attitude`` a gimballed antenna counts only the spans it can
        reach from it (see :meth:`contact_at`). A window of one simulation
        step holds at most one span of each station, so what is downlinked
        is between the start and the stop.
        """
        start = stop = begin
        downlinked = 0.0
        for span in self.spans(begin, end):
            if attitude is not None and self.contact_at(span.begin, attitude) is None:
                continue
            if downlinked == 0.0:
                start = span.begin
            stop = span.end
            downlinked += span.rate_mbps * (span.end - span.begin) / 1000.0
        return start, stop, downlinked

    def next_contact(self, utime: float) -> float | None:
        """When the next contact begins at or after ``utime``: ``utime`` itself
        while in contact, else the start of the next span, or None."""
        if self._contact_index(utime) >= 0:
            return utime
        i = bisect_right(self._begins, utime)
        return self.contacts[i].begin if i < len(self.contacts) else None

    def fraction(self, begin: float, end: float) -> float:
        """The fraction of ``[begin, end)`` in contact."""
        covered = sum(span.end - span.begin for span in self.spans(begin, end))
        return covered / (end - begin) if end > begin else 0.0
