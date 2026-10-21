"""Ground-station coverage for antennas that need no spacecraft attitude.

A body-fixed antenna downlinks only during a ground-station pass, when the
spacecraft points the antenna at the station (:class:`~conops.simulation.passes.PassTimes`).
An omnidirectional or gimballed antenna downlinks whenever a station is in
view, in whatever ACS mode the spacecraft is in, so the contact is a matter
of geometry alone: :class:`StationCoverage` finds, at every step of the
ephemeris, the station with the best downlink rate above its elevation mask,
and from that the spans of continuous contact and the fraction of time with
a station in view. Every station with a downlink carries data, whether or not
it may be scheduled for spacecraft-tracking passes.
"""

from __future__ import annotations

from bisect import bisect_right
from typing import NamedTuple

import numpy as np
import numpy.typing as npt
import rust_ephem

from ..common import AntennaType, scbodyvector
from ..config import CommunicationsSystem, MissionConfig
from ..config.groundstation import GroundStation, effective_downlink_rate_mbps

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
    """The downlink rate through the span, in Mbps."""


class Contact(NamedTuple):
    """The station in contact at one step."""

    station: str
    rate_mbps: float
    range_km: float
    """Distance from the spacecraft to the station."""


class StationCoverage:
    """Which station an omni or gimballed antenna can downlink to at each step.

    Args:
        config: Mission configuration: the stations, the communications
            system and the ephemeris on its constraint.

    At each step of the ephemeris the station in contact is the one with the
    best downlink rate among those above their elevation mask, by elevation
    where rates tie. A gimballed antenna also has to reach the station
    within its gimbal range from its boresight, which depends on the
    attitude flown and is checked at execution (:meth:`contact_at`); planning
    takes the station as reachable.
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
        visible = self.elevation_deg > masks[:, None]
        # The best station at each step: highest rate, then highest elevation.
        score = np.where(
            visible, self.rates[:, None] * 1e6 + self.elevation_deg, -np.inf
        )
        self.best: npt.NDArray[np.int64] = np.where(
            visible.any(axis=0), np.argmax(score, axis=0), -1
        )
        """Index of the station in contact at each step, -1 for none."""

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

    def contact_at(
        self, utime: float, attitude: Attitude | None = None
    ) -> Contact | None:
        """The station in contact at ``utime``, or None.

        With ``attitude`` (RA, Dec, roll in degrees) a gimballed antenna must
        reach the station within its gimbal range from its boresight; an
        omni antenna reaches every station in view.
        """
        k = self.index(utime)
        best = int(self.best[k])
        if best < 0:
            return None
        if attitude is not None and self.gimballed:
            communications = self.config.spacecraft_bus.communications
            assert communications is not None
            antenna = communications.antenna_pointing
            ra, dec, roll = attitude
            direction = scbodyvector(
                np.deg2rad(ra),
                np.deg2rad(dec),
                np.deg2rad(roll),
                self._to_station[best, k],
            )
            boresight = np.asarray(antenna.fixed_boresight_body, dtype=float)
            boresight /= np.linalg.norm(boresight)
            cosine = float(np.dot(direction, boresight) / np.linalg.norm(direction))
            if (
                np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
                > antenna.gimbal_range_deg
            ):
                return None
        station = self.stations[best]
        return Contact(
            station.code, float(self.rates[best]), float(self.range_km[best, k])
        )

    def spans(self, begin: float, end: float) -> list[ContactSpan]:
        """The spans of continuous contact in ``[begin, end)``, by station.

        Each step is in contact with its best station for the whole step;
        consecutive steps with the same station form one span, clipped to
        the window.
        """
        spans: list[ContactSpan] = []
        first = max(0, self.index(begin))
        for k in range(first, len(self.times)):
            step_begin = float(self.times[k])
            if step_begin >= end:
                break
            best = int(self.best[k])
            if best < 0:
                continue
            step_end = min(step_begin + self.step_size, end)
            step_begin = max(step_begin, begin)
            if step_end <= step_begin:
                continue
            station = self.stations[best].code
            if spans and spans[-1].station == station and spans[-1].end >= step_begin:
                spans[-1] = spans[-1]._replace(end=step_end)
            else:
                spans.append(
                    ContactSpan(step_begin, step_end, station, float(self.rates[best]))
                )
        return spans

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
        k = self.index(utime)
        if self.best[k] >= 0 and self.times[k] + self.step_size > utime:
            return utime
        later = np.flatnonzero(self.best[k + 1 :] >= 0)
        if later.size == 0:
            return None
        return float(self.times[k + 1 + int(later[0])])

    def fraction(self, begin: float, end: float) -> float:
        """The fraction of ``[begin, end)`` with a station in contact."""
        covered = sum(span.end - span.begin for span in self.spans(begin, end))
        return covered / (end - begin) if end > begin else 0.0
