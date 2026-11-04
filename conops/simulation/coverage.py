"""Usable link coverage during scheduled ground-station passes.

:class:`~conops.simulation.passes.PassTimes` predicts and books the passes.
:class:`StationCoverage` models link rates, acquisitions and repoints for
passes that require no spacecraft tracking, allowing downlink during science.
Physical visibility alone does not provide a contact. One station is
booked at a time. Every contact starts with the communications
system's acquisition latency, and a handover adds its own, during which
nothing is downlinked. Stations with a compatible downlink may be booked even
when they are excluded from spacecraft-tracking activities.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections.abc import Sequence
from functools import cached_property
from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from ..common import AntennaType, scbodyvector
from ..config import CommunicationsSystem, MissionConfig
from ..config.communications import interpolate_range_table
from ..config.data_storage import RecorderView
from ..config.groundstation import (
    GroundStation,
    effective_downlink_rate_mbps,
    rate_varies_with_range,
)
from .passes import PassTimes, StationGeometry

Attitude = tuple[float, float, float]
"""Spacecraft body attitude: RA, Dec and roll in degrees."""


class ContactSpan(NamedTuple):
    """A usable link span within a scheduled ground-station pass."""

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


class Repoint(NamedTuple):
    """A repoint of a gimballed antenna at a station.

    The antenna moves from ``time`` to ``end``: for an acquisition or a
    handover, from the moment the link switches to the moment it is up; for
    a tracking repoint, for the communications system's ``repoint_seconds``.
    An instant when both are equal.
    """

    time: float
    end: float
    station: str
    kind: Literal["acquisition", "handover", "tracking"]
    pointing: tuple[float, float, float] | None = None
    """Where the antenna points after the repoint, as a unit vector in GCRS:
    the centre of the cap the station is about to cross."""
    maneuver_deg: float | None = None
    """How far the antenna turned from its previous pointing, where known."""


def cut_windows(
    windows: Sequence[tuple[float, float]], cuts: Sequence[tuple[float, float]]
) -> list[tuple[float, float]]:
    """``windows`` less ``cuts``; a cut of no length splits a window in two."""
    result: list[tuple[float, float]] = []
    ordered = sorted(cuts)
    for w0, w1 in windows:
        start = float(w0)
        for c0, c1 in ordered:
            if c1 < start or c0 >= w1:
                continue
            if c0 > start:
                result.append((start, c0))
            start = max(start, c1)
            if start >= w1:
                break
        if w1 > start:
            result.append((start, float(w1)))
    return result


class Contact(NamedTuple):
    """The station in contact at one step."""

    station: str
    rate_mbps: float
    """The downlink rate at the step, at its range."""
    range_km: float
    """Distance from the spacecraft to the station."""


class StationCoverage:
    """Link coverage for scheduled passes that require no spacecraft tracking.

    Args:
        config: Mission configuration: the stations, the communications
            system and the ephemeris on its constraint.
        scheduled_passes: Existing bookings as (begin, end, station) tuples.
            Reconstruct them without a new scheduling decision; an empty
            sequence reserves no station time.

    At each step of the ephemeris a station is in view when its elevation is
    above its mask. :class:`~conops.simulation.passes.PassTimes` accepts windows at least
    eight minutes long, draws each station's scheduling probability once
    per window, and selects whole non-overlapping passes by downlink volume,
    duration and earlier start. The configured random seed reproduces the
    bookings over the same ephemeris. Every contact begins
    the communications system's ``acquisition_seconds`` after its station's
    booked pass begins, and a switch straight from one station to another
    adds ``handover_seconds`` to that; nothing is downlinked meanwhile. A
    station acquired after a gap with no booking pays no handover, since
    the antenna can be pointed at it in advance. A gimballed antenna also has to reach
    the station within its gimbal range from its boresight, which depends
    on the attitude flown and is checked at execution (:meth:`contact_at`);
    planning takes the station as reachable.

    A gimballed antenna is re-pointed at each acquisition and handover, and
    within a contact as the spacecraft moves. The station's direction
    moves in the body frame as it moves across the sky while an attitude is
    held, so the chain is fixed by the geometry from the contact's start:
    with ``repoint_threshold_deg``, the antenna is pointed at the centre of
    the longest arc of the station's coming track that fits within a cap of
    that radius, and re-pointed where the arc no longer fits, which is the
    fewest repoints for the allowance; with ``repoint_interval_vs_range_km``
    instead, at the table's interval at the spacecraft's range from the
    Earth's centre. A repoint takes ``repoint_seconds``, during which
    nothing is downlinked; the contact is split around it.
    """

    def __init__(
        self,
        config: MissionConfig,
        *,
        scheduled_passes: Sequence[tuple[float, float, str]] | None = None,
        _stations: list[GroundStation] | None = None,
    ) -> None:
        ephem = config.constraint.ephem
        if ephem is None:
            raise ValueError("config.constraint.ephem must be set for coverage")
        self.config = config
        self.ephem = ephem
        self.pass_times = PassTimes(config)
        self.times = self.pass_times.times
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
        self.stations: list[GroundStation] = (
            _stations
            if _stations is not None
            else [
                station
                for station in config.ground_stations.stations
                if (effective_downlink_rate_mbps(station, communications) or 0.0) > 0.0
            ]
        )
        self.rates: npt.NDArray[np.float64] = np.array(
            [
                float(effective_downlink_rate_mbps(s, communications) or 0.0)
                for s in self.stations
            ],
            dtype=float,
        )
        satellite = np.asarray(ephem.gcrs_pv.position, dtype=float)
        self.earth_range_km: npt.NDArray[np.float64] = np.linalg.norm(satellite, axis=1)
        """The spacecraft's distance from the Earth's centre at each step."""
        antenna = (
            communications.antenna_pointing if communications is not None else None
        )
        self.repointing = (
            antenna is not None and antenna.antenna_type == AntennaType.GIMBALED
        )
        """Whether the antenna is re-pointed at all: only a gimballed one is."""
        self.repoint_threshold_deg = antenna.repoint_threshold_deg if antenna else None
        self.repoint_table = antenna.repoint_interval_vs_range_km if antenna else None
        self.repoint_seconds = float(antenna.repoint_seconds) if antenna else 0.0
        count, steps = len(self.stations), len(self.times)
        self.elevation_deg = np.full((count, steps), -90.0)
        self.range_km = np.full((count, steps), np.inf)
        self._to_station = np.zeros((count, steps, 3))
        """Unit vectors from the spacecraft to each station, in GCRS."""
        for k, station in enumerate(self.stations):
            self.pass_times.station_geometry(
                station,
                0,
                steps,
                out=StationGeometry(
                    self.elevation_deg[k], self.range_km[k], self._to_station[k]
                ),
            )
        masks = np.array([s.min_elevation_deg for s in self.stations], dtype=float)
        self.visible: npt.NDArray[np.bool_] = self.elevation_deg > masks[:, None]
        """Whether each station is above its mask at each step."""
        self.rate_steps = np.empty((count, steps))
        """The downlink rate from each station at each step in view, at the
        step's range where a band's rate varies with it."""
        self._ranged = [
            rate_varies_with_range(s, communications) for s in self.stations
        ]
        """Whether each station's rate varies with range."""
        for k, station in enumerate(self.stations):
            self.pass_times.downlink_rates(station, 0, steps, out=self.rate_steps[k])
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
        self.acquisitions = 0
        """How many contacts began after a gap with no station in view."""
        self.repoints: list[Repoint] = []
        """Every repoint of the antenna over the ephemeris, in time order."""
        self._last_pointing: np.ndarray | None = None
        self._schedule_passes(scheduled_passes)
        self._schedule_contacts()
        self._begins = [span.begin for span in self.contacts]

    @classmethod
    def for_config(
        cls,
        config: MissionConfig,
        *,
        scheduled_passes: Sequence[tuple[float, float, str]] | None = None,
    ) -> StationCoverage | None:
        """Link coverage for passes without spacecraft tracking, else None.

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
        stations = [
            s
            for s in config.ground_stations.stations
            if (effective_downlink_rate_mbps(s, communications) or 0.0) > 0.0
        ]
        return (
            cls(config, scheduled_passes=scheduled_passes, _stations=stations)
            if stations
            else None
        )

    # ── Scheduling ────────────────────────────────────────────────────────

    def _schedule_passes(
        self, scheduled_passes: Sequence[tuple[float, float, str]] | None
    ) -> None:
        """Consume the bookings predicted or restored by PassTimes.

        These reservations require station time, but no spacecraft tracking
        activity. Keep physical visibility separate from booked availability.
        """
        if scheduled_passes is None:
            self.pass_times.predict()
        else:
            self.pass_times.restore_bookings(scheduled_passes)
        self.scheduled_passes = [
            self._span(self._station_index(p.station), p.begin, p.end)
            for p in self.pass_times.passes
        ]
        """Booked station passes before acquisition and repoint outages."""
        self.dropped_overlapping_passes = self.pass_times.dropped_overlapping_passes
        self.scheduled = np.zeros_like(self.visible)
        """Whether a station is booked at each ephemeris step."""
        for span in self.scheduled_passes:
            k = self._station_index(span.station)
            start = self.index(span.begin)
            end = int(np.searchsorted(self.times, span.end))
            self.scheduled[k, start:end] = True
        self.scheduled &= self.visible

    def _schedule_contacts(self) -> None:
        """Apply acquisitions, handovers and repoints to the booked passes."""
        count, steps = self.scheduled.shape
        self._repoint_times: list[float] = []
        self._repoint_ends: list[float] = []
        if count == 0 or steps == 0:
            return
        previous_end = float("-inf")
        linked = False
        for booking in self.scheduled_passes:
            station = self._station_index(booking.station)
            first = self.index(booking.begin)
            last = self.index(booking.end - 1e-6) + 1
            # Visibility and rates are constant within each ephemeris step;
            # reservations retain their exact boundaries between those steps.
            visible = self.visible[station, first:last]
            edges = np.diff(np.r_[False, visible, False].astype(int))
            for start, stop in zip(
                np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
            ):
                acquired = max(booking.begin, float(self.times[first + start]))
                end = min(
                    booking.end, float(self.times[first + stop - 1]) + self.step_size
                )
                handover = linked and acquired == previous_end
                kind: Literal["acquisition", "handover"] = (
                    "handover" if handover else "acquisition"
                )
                begin = acquired + self.acquisition_seconds
                if handover:
                    begin += self.handover_seconds
                linked = self._finish_contact(station, acquired, begin, end, kind)
                previous_end = end
        self.repoints.sort(key=lambda r: (r.time, r.end))
        self._repoint_times = [r.time for r in self.repoints]
        self._repoint_ends = list(np.maximum.accumulate([r.end for r in self.repoints]))

    def _finish_contact(
        self,
        station: int,
        acquired: float,
        begin: float,
        end: float,
        kind: Literal["acquisition", "handover"],
    ) -> bool:
        """Commit acquisition metadata only for a link with a usable span."""
        if begin >= end:
            return False
        before, repoints = len(self.contacts), len(self.repoints)
        pointing = self._last_pointing
        if self.repointing:
            self.repoints.append(
                Repoint(acquired, begin, self.stations[station].code, kind)
            )
        self._add_contact(station, begin, end)
        if len(self.contacts) == before:
            del self.repoints[repoints:]
            self._last_pointing = pointing
            return False
        if kind == "handover":
            self.handovers += 1
        else:
            self.acquisitions += 1
        return True

    def _add_contact(self, station: int, begin: float, end: float) -> None:
        """The contact with ``station`` over ``[begin, end)``, split around
        the tracking repoints within it."""
        if end <= begin:
            return
        code = self.stations[station].code
        first, tracking = self._repoint_arcs(station, begin, end)
        if (
            first is not None
            and self.repoints
            and self.repoints[-1].station == code
            and self.repoints[-1].end == begin
        ):
            # The repoint that began this contact now has its pointing.
            self.repoints[-1] = self._pointed(self.repoints[-1], first)
        cursor = begin
        for t, t_end, pointing in tracking:
            self.repoints.append(
                self._pointed(Repoint(t, t_end, code, "tracking"), pointing)
            )
            if t_end <= t:
                # An instant: the link stays up.
                continue
            if t > cursor:
                self.contacts.append(self._span(station, cursor, t))
            cursor = max(cursor, t_end)
        if end > cursor:
            self.contacts.append(self._span(station, cursor, end))

    def _pointed(self, repoint: Repoint, pointing: np.ndarray) -> Repoint:
        """``repoint`` with its pointing and the turn from the previous one."""
        norm = float(np.linalg.norm(pointing))
        if norm == 0.0:
            return repoint
        unit = pointing / norm
        maneuver = None
        if self._last_pointing is not None:
            cosine = float(np.clip(np.dot(unit, self._last_pointing), -1.0, 1.0))
            maneuver = float(np.degrees(np.arccos(cosine)))
        self._last_pointing = unit
        return repoint._replace(
            pointing=(float(unit[0]), float(unit[1]), float(unit[2])),
            maneuver_deg=maneuver,
        )

    def _repoint_arcs(
        self, station: int, begin: float, end: float
    ) -> tuple[np.ndarray | None, list[tuple[float, float, np.ndarray]]]:
        """Where the antenna points from the contact's start, and each tracking
        repoint within the contact: when, until when, and where it points.

        With a threshold, the antenna leads the station: at each repoint it
        points at the centre of the longest arc of the station's track that
        fits within a cap of the threshold's radius, so the station enters
        the cap on one side and leaves on the other, and the next repoint
        comes where the arc no longer fits. Taking each arc as long as it
        can be gives the fewest repoints for the allowance. With a table the
        antenna points at the station at each repoint, at the table's
        interval.
        """
        tracking: list[tuple[float, float, np.ndarray]] = []
        if not self.repointing:
            return None, tracking
        directions = self._to_station[station]
        if self.repoint_table is not None:
            first = directions[self.index(begin)]
            t = begin
            while True:
                t += interpolate_range_table(
                    self.repoint_table, float(self.earth_range_km[self.index(t)])
                )
                if t >= end:
                    break
                t_end = min(t + self.repoint_seconds, end)
                tracking.append((t, t_end, directions[self.index(t)]))
            return first, tracking
        if self.repoint_threshold_deg is None:
            return None, tracking
        threshold = self.repoint_threshold_deg + 1e-9
        last = self.index(end - 1e-6)

        def centre(a: int, b: int) -> np.ndarray:
            chord = directions[a] + directions[b]
            norm = float(np.linalg.norm(chord))
            return np.asarray(chord / norm if norm > 0 else directions[a])

        def fits(a: int, b: int) -> bool:
            cosines = np.clip(directions[a : b + 1] @ centre(a, b), -1.0, 1.0)
            return bool(np.degrees(np.arccos(cosines)).max() <= threshold)

        k = self.index(begin)
        first = None
        while True:
            # The longest arc from k that fits: gallop, then bisect.
            lo, step = k, 1
            while lo + step <= last and fits(k, lo + step):
                lo += step
                step *= 2
            hi = min(lo + step, last)
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if fits(k, mid):
                    lo = mid
                else:
                    hi = mid
            if hi > lo and fits(k, hi):
                lo = hi
            pointing = centre(k, lo)
            if first is None:
                first = pointing
            else:
                t = float(self.times[k])
                tracking.append((t, min(t + self.repoint_seconds, end), pointing))
            if lo >= last:
                break
            k = lo + 1
        return first, tracking

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
        return self.repointing

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
        last = bisect_left(self._begins, end)
        for span in self.contacts[first:last]:
            lo, hi = max(span.begin, begin), min(span.end, end)
            if hi > lo:
                if (lo, hi) == (span.begin, span.end):
                    spans.append(span)
                else:
                    spans.append(self._span(self._station_index(span.station), lo, hi))
        return spans

    def recorder_spans(
        self, recorder: RecorderView, begin: float, end: float
    ) -> list[ContactSpan]:
        """Usable links at one store's reserved rate on its routed band."""
        result = []
        for span in self.spans(begin, end):
            station = self._station_index(span.station)
            key = (recorder.name, span.station)
            if key not in self._recorder_integrals:
                rates = np.fromiter(
                    (
                        recorder.downlink_rate(span.station, float(d))
                        for d in self.range_km[station]
                    ),
                    dtype=float,
                )
                cumulative = np.r_[0.0, np.cumsum(rates * self.step_size / 1000.0)]
                self._recorder_integrals[key] = (rates, cumulative)
            rates, cumulative = self._recorder_integrals[key]

            def at(time: float) -> float:
                k = self.index(time)
                into = min(max(time - float(self.times[k]), 0.0), self.step_size)
                return float(cumulative[k] + rates[k] * into / 1000.0)

            volume = at(span.end) - at(span.begin)
            result.append(
                span._replace(rate_mbps=volume * 1000.0 / (span.end - span.begin))
            )
        return result

    @cached_property
    def _recorder_integrals(
        self,
    ) -> dict[tuple[str, str], tuple[np.ndarray, np.ndarray]]:
        return {}

    def rate_spans(
        self, begin: float, end: float, recorder: RecorderView | None = None
    ) -> list[ContactSpan]:
        """Usable links split only where their effective rate changes."""
        result = []
        for span in self.spans(begin, end):
            j = self._station_index(span.station)
            if recorder is None:
                rates = self.rate_steps[j]
            else:
                self.recorder_spans(recorder, span.begin, span.end)
                rates = self._recorder_integrals[(recorder.name, span.station)][0]
            first, last = self.index(span.begin), self.index(span.end - 1e-6) + 1
            edges = [
                first,
                *(first + 1 + np.flatnonzero(np.diff(rates[first:last]) != 0)),
                last,
            ]
            for a, b in zip(edges, edges[1:]):
                lo, hi = (
                    max(span.begin, float(self.times[a])),
                    min(span.end, float(self.times[b - 1]) + self.step_size),
                )
                if hi > lo:
                    result.append(ContactSpan(lo, hi, span.station, float(rates[a])))
        return result

    def downlink_gb(self, begin: float, end: float) -> float:
        """What the contacts in ``[begin, end)`` downlink, in Gb."""
        return sum(
            span.rate_mbps * (span.end - span.begin) / 1000.0
            for span in self.spans(begin, end)
        )

    def contact_in(
        self,
        begin: float,
        end: float,
        attitude: Attitude | None = None,
        *,
        recorder: RecorderView | None = None,
    ) -> tuple[float, float, float]:
        """The contact in ``[begin, end)``: when it starts, when it stops and
        what it downlinks in Gb, ``(begin, begin, 0.0)`` without one.

        With ``attitude`` a gimballed antenna counts only the spans it can
        reach from it (see :meth:`contact_at`). This is a summary: gaps may
        fall between the start and stop. Use :meth:`rate_spans` when updating
        a recorder so that outages and rate changes are processed in order.
        """
        start = stop = begin
        downlinked = 0.0
        spans = (
            self.recorder_spans(recorder, begin, end)
            if recorder is not None
            else self.spans(begin, end)
        )
        for span in spans:
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

    # ── Repoints ──────────────────────────────────────────────────────────

    def repoints_in(self, begin: float, end: float) -> list[Repoint]:
        """The repoints that begin in ``[begin, end)``."""
        return self.repoints[
            bisect_left(self._repoint_times, begin) : bisect_left(
                self._repoint_times, end
            )
        ]

    def repoint_cuts(self, begin: float, end: float) -> list[tuple[float, float]]:
        """When the antenna is moving in ``[begin, end)``, as ``(time, end)``
        pairs, to cut from the windows of an observation that must not
        contain a repoint (see :func:`cut_windows`)."""
        first = bisect_left(self._repoint_ends, begin)
        last = bisect_left(self._repoint_times, end)
        return [(r.time, r.end) for r in self.repoints[first:last] if r.end >= begin]

    def next_repoint_cut(self, utime: float) -> tuple[float, float] | None:
        """The first repoint still to end after ``utime``: in progress if it
        began at or before ``utime``, else the next one; None without one."""
        k = bisect_right(self._repoint_ends, utime)
        return (
            (self.repoints[k].time, self.repoints[k].end)
            if k < len(self.repoints)
            else None
        )

    def fraction(self, begin: float, end: float) -> float:
        """The fraction of ``[begin, end)`` in contact."""
        covered = sum(span.end - span.begin for span in self.spans(begin, end))
        return covered / (end - begin) if end > begin else 0.0
