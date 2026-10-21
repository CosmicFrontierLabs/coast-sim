"""Contacts of an antenna that needs no attitude, for the timeline plots.

A body-fixed antenna's contacts are its passes, which the timeline shows as
bars. An omni or gimballed antenna's contacts are the spans of station
coverage (:class:`~conops.simulation.StationCoverage`). On a run of a day or
two each span is a bar of its own, coloured by station; on a longer run a
span of minutes is narrower than a pixel, so the row shows each hour's
fraction in contact instead, coloured by the station with most of it and
as dark as that fraction against the run's fullest hour.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import NamedTuple

from ..simulation.coverage import StationCoverage

CONTACT_BAR_MAX_HOURS = 48.0
"""Runs up to this long show each contact as a bar; longer runs show the
fraction of each hour in contact."""

STATION_COLORS = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e", "#e6ab02"]
"""Colours by station, in the order stations first appear."""


class ContactBar(NamedTuple):
    """One bar in the Ground Contact row."""

    start: float
    """Hours from the plot's start."""
    duration: float
    """Hours."""
    station: str
    fraction: float
    """Fraction of the bar in contact: 1 for a span, the hour's share for a bin."""
    rate_mbps: float


def contact_bars(
    coverage: StationCoverage,
    begin: float,
    end: float,
    t_start: float,
    offset_hours: float,
) -> list[ContactBar]:
    """The bars for the coverage in ``[begin, end)`` (Unix seconds), placed in
    hours from ``t_start`` less ``offset_hours``.

    Each span is a bar on a run up to :data:`CONTACT_BAR_MAX_HOURS` long;
    on a longer run each hour with a contact is a bar, with the station
    that has most of the hour and the hour's fraction in contact.
    """
    spans = coverage.spans(begin, end)
    if (end - begin) / 3600.0 <= CONTACT_BAR_MAX_HOURS:
        return [
            ContactBar(
                (span.begin - t_start) / 3600.0 - offset_hours,
                (span.end - span.begin) / 3600.0,
                span.station,
                1.0,
                span.rate_mbps,
            )
            for span in spans
        ]
    bars = []
    hour = begin
    while hour < end:
        hour_end = min(hour + 3600.0, end)
        seconds: dict[str, float] = {}
        rates: dict[str, float] = {}
        for span in spans:
            lo, hi = max(span.begin, hour), min(span.end, hour_end)
            if hi > lo:
                seconds[span.station] = seconds.get(span.station, 0.0) + hi - lo
                rates[span.station] = span.rate_mbps
        if seconds:
            station = max(seconds, key=lambda code: seconds[code])
            bars.append(
                ContactBar(
                    (hour - t_start) / 3600.0 - offset_hours,
                    (hour_end - hour) / 3600.0,
                    station,
                    sum(seconds.values()) / (hour_end - hour),
                    rates[station],
                )
            )
        hour = hour_end
    return bars


def contact_alphas(bars: Sequence[ContactBar], floor: float = 0.3) -> list[float]:
    """How opaque to draw each bar: its fraction in contact against the
    largest in the run, so the fullest hour is solid, and no fainter than
    ``floor`` so every hour with a contact shows."""
    fullest = max((bar.fraction for bar in bars), default=1.0)
    if fullest <= 0.0:
        return [1.0] * len(bars)
    return [max(floor, min(1.0, bar.fraction / fullest)) for bar in bars]


def station_colors(stations: Iterable[str]) -> dict[str, str]:
    """A colour for each station, in the order given."""
    colors: dict[str, str] = {}
    for station in stations:
        if station not in colors:
            colors[station] = STATION_COLORS[len(colors) % len(STATION_COLORS)]
    return colors


def run_window(ditl: object) -> tuple[float, float] | None:
    """The simulated run's ``(begin, end)`` in Unix seconds, from its steps."""
    utime = getattr(ditl, "utime", None)
    if not isinstance(utime, (list, tuple)) or not utime:
        return None
    step = getattr(ditl, "step_size", 0)
    return float(utime[0]), float(utime[-1]) + float(
        step if isinstance(step, (int, float)) else 0
    )
