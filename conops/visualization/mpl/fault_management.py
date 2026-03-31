"""Fault Management Event Timeline Visualization.

Provides functions to plot a timeline of fault management events with
yellow/red threshold states shown as colored bands and event markers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING

import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from ...common import ACSMode
from ...config.visualization import VisualizationConfig

if TYPE_CHECKING:
    from ...config.fault_management import FaultEvent, FaultManagement

# ── Colors ────────────────────────────────────────────────────────────────────
_COLOR_YELLOW = "#FFD700"
_COLOR_RED = "#FF4444"
_COLOR_SAFE = "#AA44FF"  # safe_mode_trigger
_COLOR_NOMINAL = "#44BB66"

_ACS_MODE_NAMES: dict[int, str] = {m.value: m.name for m in ACSMode}

# ── Helpers ────────────────────────────────────────────────────────────────────


def _utime_to_hours(events: list[FaultEvent], t0: float) -> list[float]:
    return [(e.utime - t0) / 3600.0 for e in events]


def _utime_to_datetime(utime: float) -> datetime:
    return datetime.fromtimestamp(utime, tz=timezone.utc)


_THRESHOLD_STATE_COLORS: dict[str, str] = {
    "nominal": _COLOR_NOMINAL,
    "yellow": _COLOR_YELLOW,
    "red": _COLOR_RED,
}


def _event_color(event: FaultEvent) -> str:
    """Return a color string for a single FaultEvent."""
    if event.event_type == "threshold_transition":
        new_state = str((event.metadata or {}).get("new_state", "nominal"))
        return _THRESHOLD_STATE_COLORS.get(new_state, _COLOR_NOMINAL)
    if event.event_type == "constraint_violation":
        return _COLOR_RED
    if event.event_type == "safe_mode_trigger":
        return _COLOR_SAFE
    return "gray"


def _event_marker_label(event: FaultEvent) -> str:
    """Short label for the event marker tooltip."""
    meta = event.metadata or {}
    if event.event_type == "threshold_transition":
        prev = meta.get("previous_state", "?")
        new = meta.get("new_state", "?")
        val = meta.get("value", "")
        val_str = f"  val={val:.3g}" if isinstance(val, (int, float)) else ""
        return f"{prev}→{new}{val_str}"
    if event.event_type == "constraint_violation":
        return str(meta.get("constraint_type", "violation"))
    if event.event_type == "safe_mode_trigger":
        return "SAFE"
    return event.event_type


# ── Marker vertical offsets (traffic-light order within each row) ─────────────
# Red floats to the top, yellow stays centred, nominal/green sinks to the bottom.
_PLOTLY_CAT_YOFFSET: dict[str, float] = {
    "Nominal": -0.18,
    "Yellow": 0.0,
    "Red": 0.18,
    "Safe Mode": 0.22,
    "Constraint Violation": -0.10,
}


def _marker_yoffset(event: FaultEvent) -> float:
    """Vertical offset within a row so marker states don't overlap."""
    if event.event_type == "safe_mode_trigger":
        return _PLOTLY_CAT_YOFFSET["Safe Mode"]
    if event.event_type == "constraint_violation":
        return _PLOTLY_CAT_YOFFSET["Constraint Violation"]
    new_state = str((event.metadata or {}).get("new_state", "nominal"))
    return _PLOTLY_CAT_YOFFSET.get(new_state.capitalize(), 0.0)


# ── Matplotlib ─────────────────────────────────────────────────────────────────


def plot_fault_management_timeline(
    fault_management: FaultManagement,
    t0: float | None = None,
    t_end: float | None = None,
    x_axis: str = "hours",
    figsize: tuple[float, float] | None = None,
    config: VisualizationConfig | None = None,
) -> tuple[Figure, Axes]:
    """Plot a fault management event timeline using matplotlib.

    Creates a scatter / timeline chart with:
    - Y-axis: parameter / constraint name (one row per monitored item)
    - X-axis: elapsed time (hours by default, or absolute UTC datetime)
    - Colour-coded markers: green=nominal, yellow=yellow, red=red, purple=safe_mode

    Each transition event is drawn as a vertical line segment connecting the
    previous and new state on the same row, which makes it easy to see when a
    parameter entered or exited a degraded state.

    Args:
        fault_management: :class:`~conops.config.fault_management.FaultManagement`
            instance after a simulation run (contains a populated ``events`` list).
        t0: Reference epoch in Unix seconds for X=0.  Defaults to the timestamp
            of the first event.
        t_end: End of the simulation in Unix seconds.  When provided the X-axis
            is clamped to ``[t0, t_end]`` so the plot always spans the full DITL
            duration regardless of when the last fault event occurred.  Pass
            ``ditl.end.timestamp()`` to pin the axis to the DITL window.
        x_axis: ``"hours"`` (default) or ``"datetime"`` — controls X-axis units.
        figsize: ``(width, height)`` in inches.  Defaults to ``(12, max(4, n*0.8))``.
        config: :class:`~conops.config.visualization.VisualizationConfig`.  If
            *None* the defaults are used.

    Returns:
        ``(fig, ax)`` — the matplotlib Figure and Axes.

    Example::

        from conops.visualization import plot_fault_management_timeline
        fig, ax = plot_fault_management_timeline(config.fault_management)
        fig.savefig("faults.png", dpi=150, bbox_inches="tight")
    """
    if config is None:
        config = VisualizationConfig()

    events = fault_management.events
    if not events:
        fig, ax = plt.subplots(figsize=figsize or (10, 3))
        ax.text(
            0.5,
            0.5,
            "No fault events recorded",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
            color="gray",
        )
        ax.set_title("Fault Management Event Timeline", fontsize=config.title_font_size)
        if t0 is not None and t_end is not None:
            _t0_ref = t0
            ax.set_xlim(
                _x_value(_t0_ref, _t0_ref, x_axis), _x_value(t_end, _t0_ref, x_axis)
            )
        return fig, ax

    # ── Build Y-axis rows ──────────────────────────────────────────────────────
    # Collect all unique names, ordered: thresholds first (config order), then
    # constraint names, then safe_mode entries.
    threshold_names = [t.name for t in fault_management.thresholds]
    constraint_names = [c.name for c in fault_management.red_limit_constraints]
    extra_names: list[str] = []
    for e in events:
        if (
            e.name not in threshold_names
            and e.name not in constraint_names
            and e.name not in extra_names
        ):
            extra_names.append(e.name)
    all_names = threshold_names + constraint_names + extra_names
    # Remove duplicates while preserving order
    seen: set[str] = set()
    row_names: list[str] = []
    for n in all_names:
        if n not in seen:
            row_names.append(n)
            seen.add(n)
    y_map = {name: i for i, name in enumerate(row_names)}
    n_rows = len(row_names)

    # ── Reference time ────────────────────────────────────────────────────────
    if t0 is None:
        t0 = events[0].utime

    # ── Figure sizing ─────────────────────────────────────────────────────────
    if figsize is None:
        figsize = (12, max(4.0, n_rows * 0.9 + 2.0))

    fig, ax = plt.subplots(figsize=figsize)

    # ── Background rows ───────────────────────────────────────────────────────
    for i in range(n_rows):
        ax.axhspan(
            i - 0.4, i + 0.4, color="whitesmoke" if i % 2 == 0 else "white", zorder=0
        )

    # ── Draw state-span bars ──────────────────────────────────────────────────
    # For each parameter, reconstruct the state timeline from transition events
    # and draw coloured horizontal bars covering each state interval.
    _draw_state_bars_mpl(ax, events, row_names, y_map, t0, x_axis, t_end)

    # ── Event markers ─────────────────────────────────────────────────────────
    for event in events:
        if event.name not in y_map:
            continue
        y = y_map[event.name] + _marker_yoffset(event)
        x = _x_value(event.utime, t0, x_axis)
        color = _event_color(event)
        ax.scatter(
            x, y, color=color, zorder=5, s=60, linewidths=0.8, edgecolors="black"
        )

    # ── Y-axis labels ─────────────────────────────────────────────────────────
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_names, fontsize=config.label_font_size)
    ax.set_ylim(-0.6, n_rows - 0.4)

    # ── X-axis ────────────────────────────────────────────────────────────────
    if t_end is not None:
        ax.set_xlim(_x_value(t0, t0, x_axis), _x_value(t_end, t0, x_axis))
    if x_axis == "datetime":
        ax.xaxis_date()
        fig.autofmt_xdate()
        ax.set_xlabel("UTC Time", fontsize=config.label_font_size)
    else:
        ax.set_xlabel("Elapsed Time (hours)", fontsize=config.label_font_size)

    ax.set_title(
        "Fault Management Event Timeline",
        fontsize=config.title_font_size,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4, zorder=1)

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_handles = [
        mpatches.Patch(facecolor=_COLOR_NOMINAL, edgecolor="black", label="Nominal"),
        mpatches.Patch(facecolor=_COLOR_YELLOW, edgecolor="black", label="Yellow"),
        mpatches.Patch(facecolor=_COLOR_RED, edgecolor="black", label="Red"),
        mpatches.Patch(facecolor=_COLOR_SAFE, edgecolor="black", label="Safe Mode"),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        fontsize=config.label_font_size,
        framealpha=0.8,
    )

    fig.tight_layout()
    return fig, ax


def _x_value(utime: float, t0: float, x_axis: str) -> float:
    """Return the x-axis value for a given utime.

    Always returns a ``float``.  For the ``"hours"`` mode this is elapsed hours
    since ``t0``; for the ``"datetime"`` mode this is a matplotlib date number
    (days since the matplotlib epoch) as returned by ``matplotlib.dates.date2num``,
    which is compatible with ``ax.xaxis_date()``.
    """
    if x_axis == "datetime":
        return float(mdates.date2num(_utime_to_datetime(utime)))  # type: ignore[no-untyped-call]
    return (utime - t0) / 3600.0


def _draw_state_bars_mpl(
    ax: Axes,
    events: list[FaultEvent],
    row_names: list[str],
    y_map: dict[str, int],
    t0: float,
    x_axis: str,
    t_end: float | None = None,
) -> None:
    """Draw horizontal coloured bars for each state interval per parameter."""
    # Group transition events by name
    from collections import defaultdict

    transitions: dict[str, list[FaultEvent]] = defaultdict(list)
    for e in events:
        if e.event_type == "threshold_transition":
            transitions[e.name].append(e)

    state_colors = {
        "nominal": _COLOR_NOMINAL,
        "yellow": _COLOR_YELLOW,
        "red": _COLOR_RED,
    }

    for name, tevents in transitions.items():
        if name not in y_map:
            continue
        y = y_map[name]
        tevents_sorted = sorted(tevents, key=lambda e: e.utime)
        for i, e in enumerate(tevents_sorted):
            new_state = str((e.metadata or {}).get("new_state", "nominal"))
            color = state_colors.get(new_state, "gray")
            x_start = _x_value(e.utime, t0, x_axis)
            # Bar extends to the next transition or to the end edge of the axis
            if i + 1 < len(tevents_sorted):
                x_end = _x_value(tevents_sorted[i + 1].utime, t0, x_axis)
            elif t_end is not None:
                x_end = _x_value(t_end, t0, x_axis)
            else:
                # Extend a short distance past the last event (10 min)
                x_end = _x_value(e.utime + 600, t0, x_axis)
            ax.barh(
                y,
                x_end - x_start,
                left=x_start,
                height=0.7,
                color=color,
                alpha=0.45,
                zorder=2,
            )
