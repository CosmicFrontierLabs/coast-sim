"""Fault Management Event Timeline Visualization.

Provides functions to plot a timeline of fault management events with
yellow/red threshold states shown as colored bands and event markers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

import plotly.graph_objects as go

from ...common import ACSMode
from ...config.visualization import VisualizationConfig

if TYPE_CHECKING:
    from ...config.fault_management import FaultEvent, FaultManagement
    from ...ditl import DITL, QueueDITL

# ── Colors ────────────────────────────────────────────────────────────────────
_COLOR_YELLOW = "#FFD700"
_COLOR_RED = "#FF4444"
_COLOR_SAFE = "#AA44FF"  # safe_mode_trigger
_COLOR_NOMINAL = "#44BB66"

_ACS_MODE_NAMES: dict[int, str] = {m.value: m.name for m in ACSMode}

# ── Helpers ────────────────────────────────────────────────────────────────────


def _utime_to_datetime(utime: float) -> datetime:
    return datetime.fromtimestamp(utime, tz=timezone.utc)


_THRESHOLD_STATE_COLORS: dict[str, str] = {
    "nominal": _COLOR_NOMINAL,
    "yellow": _COLOR_YELLOW,
    "red": _COLOR_RED,
}


# ── Marker vertical offsets (traffic-light order within each row) ─────────────
# Red floats to the top, yellow stays centred, nominal/green sinks to the bottom.
_PLOTLY_CAT_YOFFSET: dict[str, float] = {
    "Nominal": -0.18,
    "Yellow": 0.0,
    "Red": 0.18,
    "Safe Mode": 0.22,
    "Constraint Violation": -0.10,
}


# ── Plotly ─────────────────────────────────────────────────────────────────────


def plot_fault_management_timeline_plotly(
    fault_management: FaultManagement,
    t0: float | None = None,
    t_end: float | None = None,
    x_axis: str = "hours",
    config: VisualizationConfig | None = None,
    ditl: DITL | QueueDITL | None = None,
) -> go.Figure:
    """Plot a fault management event timeline using Plotly (interactive).

    Creates an interactive scatter / timeline chart with:
    - Y-axis: parameter / constraint name (one row per monitored item)
    - X-axis: elapsed time in hours (or absolute UTC if ``x_axis="datetime"``)
    - Hover tooltips showing the transition cause and metadata
    - Colour-coded markers and background bars for each fault state

    Args:
        fault_management: :class:`~conops.config.fault_management.FaultManagement`
            instance after a simulation run.
        t0: Reference epoch (Unix seconds) for X=0.  Defaults to first event.
        t_end: End of the simulation in Unix seconds.  When provided the X-axis
            is clamped to ``[t0, t_end]`` so the plot always spans the full DITL
            duration.  Pass ``ditl.end.timestamp()`` to pin the axis to the
            DITL window.
        x_axis: ``"hours"`` (default) or ``"datetime"``.
        config: :class:`~conops.config.visualization.VisualizationConfig`.

    Returns:
        :class:`plotly.graph_objects.Figure`

    Example::

        from conops.visualization import plot_fault_management_timeline_plotly
        fig = plot_fault_management_timeline_plotly(config.fault_management)
        fig.show()
    """
    if config is None:
        config = VisualizationConfig()

    events = fault_management.events

    if not events:
        fig = go.Figure()
        fig.add_annotation(
            text="No fault events recorded",
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            showarrow=False,
            font={"size": 14, "color": "gray"},
        )
        fig.update_layout(title="Fault Management Event Timeline")
        return fig

    # ── Row ordering ──────────────────────────────────────────────────────────
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
    seen: set[str] = set()
    row_names: list[str] = []
    for n in all_names:
        if n not in seen:
            row_names.append(n)
            seen.add(n)

    if t0 is None:
        t0 = events[0].utime

    y_map = {name: i for i, name in enumerate(row_names)}
    n_rows = len(row_names)

    # ── ACS Mode row check ────────────────────────────────────────────────────
    _has_acs = (
        ditl is not None
        and hasattr(ditl, "telemetry")
        and bool(getattr(getattr(ditl, "telemetry", None), "housekeeping", None))
    )
    _acs_row_y = -1  # placed below all fault rows

    fig = go.Figure()

    # ── Alternating row backgrounds (matches matplotlib whitesmoke bands) ──────
    for i in range(n_rows):
        if i % 2 == 0:
            fig.add_shape(
                type="rect",
                xref="paper",
                yref="y",
                x0=0,
                x1=1,
                y0=i - 0.4,
                y1=i + 0.4,
                fillcolor="rgba(245,245,245,1)",
                line={"width": 0},
                layer="below",
            )
    if _has_acs:
        fig.add_shape(
            type="rect",
            xref="paper",
            yref="y",
            x0=0,
            x1=1,
            y0=_acs_row_y - 0.4,
            y1=_acs_row_y + 0.4,
            fillcolor="rgba(235,235,235,1)",
            line={"width": 0},
            layer="below",
        )

    # ── State-span bars ────────────────────────────────────────────────────────
    for s in _build_state_shapes_plotly(events, row_names, y_map, t0, x_axis, t_end):
        fig.add_shape(**s)

    # ── Per-event-type scatter traces ─────────────────────────────────────────
    # Each category is offset vertically within its row (traffic-light order:
    # red=top, yellow=centre, nominal/green=bottom) so markers don't obscure
    # each other.
    categories = {
        "Nominal": (_COLOR_NOMINAL, "circle"),
        "Yellow": (_COLOR_YELLOW, "diamond"),
        "Red": (_COLOR_RED, "square"),
        "Safe Mode": (_COLOR_SAFE, "star"),
        "Constraint Violation": (_COLOR_RED, "x"),
    }

    # Bin each event into a category
    cat_events: dict[str, list[FaultEvent]] = {k: [] for k in categories}
    for e in events:
        if e.event_type == "safe_mode_trigger":
            cat_events["Safe Mode"].append(e)
        elif e.event_type == "constraint_violation":
            cat_events["Constraint Violation"].append(e)
        elif e.event_type == "threshold_transition":
            new_state = (e.metadata or {}).get("new_state", "nominal")
            if new_state == "red":
                cat_events["Red"].append(e)
            elif new_state == "yellow":
                cat_events["Yellow"].append(e)
            else:
                cat_events["Nominal"].append(e)
        else:
            cat_events.setdefault("Nominal", []).append(e)

    for cat_name, (color, symbol) in categories.items():
        cat_evs = [e for e in cat_events.get(cat_name, []) if e.name in y_map]
        if not cat_evs:
            continue

        yoffset = _PLOTLY_CAT_YOFFSET[cat_name]
        xs = [_x_val_plotly(e.utime, t0, x_axis) for e in cat_evs]
        ys = [y_map[e.name] + yoffset for e in cat_evs]
        texts = [
            f"<b>{e.name}</b><br>"
            f"Type: {e.event_type}<br>"
            f"Time: {_utime_to_datetime(e.utime).strftime('%Y-%m-%d %H:%M:%S UTC')}<br>"
            f"Cause: {e.cause}<br>" + _meta_html(e.metadata)
            for e in cat_evs
        ]

        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                name=cat_name,
                marker={
                    "color": color,
                    "symbol": symbol,
                    "size": 10,
                    "line": {"width": 1, "color": "black"},
                },
                text=texts,
                hoverinfo="text",
            )
        )

    # ── ACS Mode row ──────────────────────────────────────────────────────────
    if _has_acs:
        assert ditl is not None
        acs_shapes, acs_scatter = _build_acs_mode_shapes_plotly(
            ditl, _acs_row_y, t0, x_axis, config, t_end
        )
        for s in acs_shapes:
            fig.add_shape(**s)
        if acs_scatter is not None:
            fig.add_trace(acs_scatter)

    # ── Layout ────────────────────────────────────────────────────────────────
    x_title = "UTC Time" if x_axis == "datetime" else "Elapsed Time (hours)"
    n_display_rows = n_rows + (1 if _has_acs else 0)
    height = max(350, n_display_rows * 70 + 150)

    xaxis_cfg: dict[str, Any] = {
        "title": x_title,
        "showgrid": True,
        "gridcolor": "rgba(200,200,200,0.5)",
        "zeroline": False,
    }
    if t_end is not None:
        xaxis_cfg["range"] = [
            _x_val_plotly(t0, t0, x_axis),
            _x_val_plotly(t_end, t0, x_axis),
        ]

    fig.update_layout(
        title={
            "text": "<b>Fault Management Event Timeline</b>",
            "font": {"size": config.title_font_size},
            "x": 0.5,
            "xanchor": "center",
        },
        yaxis={
            "title": "Parameter / Constraint",
            "tickvals": ([-1] + list(range(n_rows)))
            if _has_acs
            else list(range(n_rows)),
            "ticktext": (["ACS Mode"] + row_names) if _has_acs else row_names,
            "range": [(-1.6 if _has_acs else -0.6), n_rows - 0.4],
            "showgrid": True,
            "gridcolor": "rgba(200,200,200,0.5)",
            "zeroline": False,
        },
        height=height,
        legend={"title": "Event State", "orientation": "v"},
        hovermode="closest",
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=xaxis_cfg,
    )

    return fig


def _x_val_plotly(utime: float, t0: float, x_axis: str) -> float | str:
    if x_axis == "datetime":
        return _utime_to_datetime(utime).strftime("%Y-%m-%dT%H:%M:%S")
    return (utime - t0) / 3600.0


def _meta_html(metadata: dict[str, Any] | None) -> str:
    if not metadata:
        return ""
    lines = []
    for k, v in metadata.items():
        if isinstance(v, float):
            lines.append(f"{k}: {v:.4g}")
        else:
            lines.append(f"{k}: {v}")
    return "<br>".join(lines)


def _build_state_shapes_plotly(
    events: list[FaultEvent],
    row_names: list[str],
    y_map: dict[str, int],
    t0: float,
    x_axis: str,
    t_end: float | None = None,
) -> list[dict[str, Any]]:
    """Build Plotly shape dicts for state-span bars using numeric Y coordinates."""
    from collections import defaultdict

    transitions: dict[str, list[FaultEvent]] = defaultdict(list)
    for e in events:
        if e.event_type == "threshold_transition":
            transitions[e.name].append(e)

    state_colors = {
        "nominal": "rgba(68,187,102,0.35)",
        "yellow": "rgba(255,215,0,0.40)",
        "red": "rgba(255,68,68,0.40)",
    }

    shapes = []
    for name, tevents in transitions.items():
        if name not in y_map:
            continue
        y_idx = y_map[name]
        tevents_sorted = sorted(tevents, key=lambda e: e.utime)
        for i, e in enumerate(tevents_sorted):
            new_state = (e.metadata or {}).get("new_state", "nominal")
            color = state_colors.get(new_state, "rgba(128,128,128,0.2)")
            x0 = _x_val_plotly(e.utime, t0, x_axis)
            if i + 1 < len(tevents_sorted):
                x1 = _x_val_plotly(tevents_sorted[i + 1].utime, t0, x_axis)
            elif t_end is not None:
                x1 = _x_val_plotly(t_end, t0, x_axis)
            else:
                x1 = _x_val_plotly(e.utime + 600, t0, x_axis)

            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "y",
                    "x0": x0,
                    "x1": x1,
                    "y0": y_idx - 0.45,
                    "y1": y_idx + 0.45,
                    "fillcolor": color,
                    "opacity": 1.0,
                    "line": {"width": 0},
                    "layer": "below",
                }
            )

    return shapes


def _build_acs_mode_shapes_plotly(
    ditl: DITL | QueueDITL,
    row_y: int,
    t0: float,
    x_axis: str,
    config: VisualizationConfig,
    t_end: float | None = None,
) -> tuple[list[dict[str, Any]], go.Scatter | None]:
    """Build shapes and a hover scatter trace for the ACS Mode row."""
    hk = getattr(getattr(ditl, "telemetry", None), "housekeeping", None)
    if hk is None:
        return [], None

    timestamps = hk.timestamp
    modes = hk.acs_mode
    if not timestamps or not modes:
        return [], None

    def _to_utime(ts: Any) -> float | None:
        return ts.timestamp() if ts is not None else None

    pairs: list[tuple[float, int]] = [
        (u, m)
        for u, m in (
            (_to_utime(ts), int(mv) if mv is not None else None)
            for ts, mv in zip(timestamps, modes)
        )
        if u is not None and m is not None
    ]
    if not pairs:
        return [], None

    shapes: list[dict[str, Any]] = []
    seg_xs: list[Any] = []
    seg_texts: list[str] = []

    def _close_seg(mode: int, x0_utime: float, x1_utime: float) -> None:
        color = config.mode_colors.get(_ACS_MODE_NAMES.get(mode, ""), "gray")
        shapes.append(
            {
                "type": "rect",
                "xref": "x",
                "yref": "y",
                "x0": _x_val_plotly(x0_utime, t0, x_axis),
                "x1": _x_val_plotly(x1_utime, t0, x_axis),
                "y0": row_y - 0.45,
                "y1": row_y + 0.45,
                "fillcolor": color,
                "opacity": 1.0,
                "line": {"width": 0},
                "layer": "below",
            }
        )
        seg_xs.append(_x_val_plotly(x0_utime, t0, x_axis))
        seg_texts.append(
            f"<b>ACS Mode: {_ACS_MODE_NAMES.get(mode, str(mode))}</b><br>"
            f"Start: {_utime_to_datetime(x0_utime).strftime('%Y-%m-%d %H:%M:%S UTC')}"
        )

    prev_mode: int | None = None
    seg_start: float | None = None

    for utime, mode_int in pairs:
        if prev_mode is None:
            prev_mode = mode_int
            seg_start = utime
        elif mode_int != prev_mode:
            assert seg_start is not None
            _close_seg(prev_mode, seg_start, utime)
            prev_mode = mode_int
            seg_start = utime

    if prev_mode is not None and seg_start is not None:
        end_utime = pairs[-1][0]
        if t_end is not None:
            end_utime = max(end_utime, t_end)
        _close_seg(prev_mode, seg_start, end_utime)

    if not seg_xs:
        return shapes, None

    scatter: go.Scatter = go.Scatter(
        x=seg_xs,
        y=[float(row_y)] * len(seg_xs),
        mode="markers",
        name="ACS Mode",
        marker={"color": "rgba(0,0,0,0)", "size": 12},
        text=seg_texts,
        hoverinfo="text",
        showlegend=False,
    )
    return shapes, scatter
