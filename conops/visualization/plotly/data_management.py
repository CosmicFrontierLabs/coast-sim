"""Data management visualization utilities for CONOPS simulations (Plotly)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...config.visualization import VisualizationConfig

if TYPE_CHECKING:
    from ...ditl.ditl import DITL
    from ...ditl.queue_ditl import QueueDITL


def plot_data_management_telemetry_plotly(
    ditl: QueueDITL | DITL,
    config: VisualizationConfig | None = None,
) -> go.Figure:
    """Plot comprehensive data management telemetry from a DITL simulation.

    Creates a 5-panel interactive figure showing:

    1. Onboard recorder data volume over time (with capacity limit)
    2. Recorder fill fraction with yellow / red alert thresholds
    3. Cumulative data generated
    4. Cumulative data downlinked
    5. Recorder alert timeline (colour-coded by alert level)

    Args:
        ditl: DITL simulation object with data management telemetry.
        config: VisualizationConfig object. If None, uses
            ``ditl.config.visualization`` if available, otherwise defaults.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.

    Example:
        >>> from conops.ditl import QueueDITL
        >>> from conops.visualization import plot_data_management_telemetry_plotly
        >>> ditl = QueueDITL(config=config)
        >>> ditl.calc()
        >>> fig = plot_data_management_telemetry_plotly(ditl)
        >>> fig.show()
    """
    if not isinstance(config, VisualizationConfig):
        if (
            hasattr(ditl, "config")
            and hasattr(ditl.config, "visualization")
            and isinstance(ditl.config.visualization, VisualizationConfig)
        ):
            config = ditl.config.visualization
        else:
            config = VisualizationConfig()

    times = list(ditl.telemetry.housekeeping.timestamp)

    fig = make_subplots(
        rows=5,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.04,
        subplot_titles=[
            "Onboard Recorder Data Volume",
            "Recorder Fill Level",
            "Cumulative Data Generated",
            "Cumulative Data Downlinked",
            "Recorder Alert Timeline",
        ],
    )

    # ------------------------------------------------------------------
    # Panel 1 — Recorder volume
    # ------------------------------------------------------------------
    capacity_gb = ditl.config.recorder.capacity_gb

    fig.add_trace(
        go.Scatter(
            x=times,
            y=np.array(
                ditl.telemetry.housekeeping.recorder_volume_gb, dtype=np.float64
            ),
            mode="lines",
            name="Recorder Volume",
            line=dict(color="royalblue", width=2),
            showlegend=True,
        ),
        row=1,
        col=1,
    )
    fig.add_hline(
        y=capacity_gb,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Capacity ({capacity_gb} Gb)",
        annotation_position="top right",
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Volume (Gb)", row=1, col=1)

    # ------------------------------------------------------------------
    # Panel 2 — Fill fraction
    # ------------------------------------------------------------------
    yellow_threshold = ditl.config.recorder.yellow_threshold
    red_threshold = ditl.config.recorder.red_threshold

    fig.add_trace(
        go.Scatter(
            x=times,
            y=np.array(
                ditl.telemetry.housekeeping.recorder_fill_fraction, dtype=np.float64
            ),
            mode="lines",
            name="Fill Fraction",
            line=dict(color="mediumseagreen", width=2),
            showlegend=True,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=yellow_threshold,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Yellow ({yellow_threshold:.0%})",
        annotation_position="top right",
        row=2,
        col=1,
    )
    fig.add_hline(
        y=red_threshold,
        line_dash="dash",
        line_color="red",
        annotation_text=f"Red ({red_threshold:.0%})",
        annotation_position="top right",
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Fill Fraction", range=[0, 1], row=2, col=1)

    # ------------------------------------------------------------------
    # Panel 3 — Cumulative data generated
    # ------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=times,
            y=list(ditl.data_generated_gb),
            mode="lines",
            name="Data Generated",
            line=dict(color="mediumpurple", width=2),
            showlegend=True,
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="Data Generated (Gb)", row=3, col=1)

    # ------------------------------------------------------------------
    # Panel 4 — Cumulative data downlinked
    # ------------------------------------------------------------------
    fig.add_trace(
        go.Scatter(
            x=times,
            y=list(ditl.data_downlinked_gb),
            mode="lines",
            name="Data Downlinked",
            line=dict(color="darkcyan", width=2),
            showlegend=True,
        ),
        row=4,
        col=1,
    )
    fig.update_yaxes(title_text="Data Downlinked (Gb)", row=4, col=1)

    # ------------------------------------------------------------------
    # Panel 5 — Alert timeline (one trace per alert level for legend)
    # ------------------------------------------------------------------
    alert_levels = list(ditl.telemetry.housekeeping.recorder_alert)
    _alert_cfg: list[tuple[int, str, str]] = [
        (0, "No Alert", "green"),
        (1, "Yellow Alert", "orange"),
        (2, "Red Alert", "red"),
    ]
    _shown_alert_legend: set[str] = set()
    for level, label, color in _alert_cfg:
        mask = [i for i, a in enumerate(alert_levels) if a == level]
        if not mask:
            continue
        _show = label not in _shown_alert_legend
        _shown_alert_legend.add(label)
        fig.add_trace(
            go.Scatter(
                x=[times[i] for i in mask],
                y=[float(level)] * len(mask),
                mode="markers",
                name=label,
                marker=dict(color=color, size=6, opacity=0.7),
                showlegend=_show,
                legendgroup=label,
            ),
            row=5,
            col=1,
        )
    fig.update_yaxes(
        title_text="Alert Level",
        tickvals=[0, 1, 2],
        ticktext=["None", "Yellow", "Red"],
        row=5,
        col=1,
    )
    fig.update_xaxes(title_text="Time (UTC)", row=5, col=1)

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    title = getattr(getattr(ditl, "config", None), "name", "DITL")
    px_per_inch = 90
    fig_width = int(config.data_telemetry_figsize[0] * px_per_inch)
    fig_height = int(config.data_telemetry_figsize[1] * px_per_inch)

    axis_title_font = dict(
        family=config.font_family,
        size=config.label_font_size,
    )
    axis_tick_font = dict(
        family=config.font_family,
        size=config.tick_font_size,
    )

    fig.update_xaxes(title_font=axis_title_font, tickfont=axis_tick_font)
    fig.update_yaxes(title_font=axis_title_font, tickfont=axis_tick_font)

    # Style all subplot and threshold annotations with the configured font.
    fig.update_annotations(
        font=dict(
            family=config.font_family,
            size=config.label_font_size,
        )
    )

    fig.update_layout(
        width=fig_width,
        height=fig_height,
        title=dict(
            text=f"Data Management — {title}",
            font=dict(
                family=config.font_family,
                size=config.title_font_size,
            ),
        ),
        font=dict(
            family=config.font_family,
            size=config.label_font_size,
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(
                family=config.font_family,
                size=config.legend_font_size,
            ),
        ),
    )

    return fig
