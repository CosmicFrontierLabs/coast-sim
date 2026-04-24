"""Radiator telemetry visualization for CONOPS simulations (Plotly)."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...config.visualization import VisualizationConfig

if TYPE_CHECKING:
    from ...ditl.ditl import DITL
    from ...ditl.queue_ditl import QueueDITL


def _none_to_nan(values: Sequence[float | int | None]) -> list[float]:
    return [float(v) if v is not None else float("nan") for v in values]


def plot_radiator_telemetry_plotly(
    ditl: QueueDITL | DITL,
    config: VisualizationConfig | None = None,
) -> go.Figure:
    """Plot radiator thermal exposure and heat dissipation telemetry using Plotly.

    Creates an interactive three-panel figure showing:
    1. Area-weighted Sun and Earth exposure fractions (0–1)
    2. Net heat dissipation in Watts (positive = rejecting heat)
    3. Count of radiators violating hard keep-out constraints

    Args:
        ditl: DITL simulation object after calling calc().
        config: VisualizationConfig. Falls back to ditl.config.visualization or defaults.

    Returns:
        plotly.graph_objects.Figure: The interactive Plotly figure.

    Example:
        >>> fig = plot_radiator_telemetry_plotly(ditl)
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

    hk = ditl.telemetry.housekeeping
    times = hk.timestamp

    sun_exposure = _none_to_nan(hk.radiator_sun_exposure)
    earth_exposure = _none_to_nan(hk.radiator_earth_exposure)
    heat_dissipation = _none_to_nan(hk.radiator_heat_dissipation_w)
    hard_violations = _none_to_nan(hk.radiator_hard_violations)

    fig = make_subplots(
        rows=3,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.06,
        subplot_titles=[
            "Radiator Exposure Fractions",
            "Net Radiator Heat Dissipation",
            "Radiator Hard Constraint Violations",
        ],
    )

    # Panel 1: Sun and Earth exposure
    fig.add_trace(
        go.Scatter(
            x=times,
            y=sun_exposure,
            mode="lines",
            name="Sun Exposure",
            line=dict(color="orange", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=times,
            y=earth_exposure,
            mode="lines",
            name="Earth Exposure",
            line=dict(color="cornflowerblue", width=1.5),
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="Exposure", range=[0, 1.05], row=1, col=1)

    # Panel 2: Heat dissipation
    fig.add_trace(
        go.Scatter(
            x=times,
            y=heat_dissipation,
            mode="lines",
            name="Heat Dissipation",
            line=dict(color="tomato", width=1.5),
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.add_hline(
        y=0.0, line_dash="dash", line_color="black", line_width=0.8, row=2, col=1
    )
    fig.update_yaxes(title_text="Heat Dissipation (W)", row=2, col=1)

    # Panel 3: Hard constraint violations — bars coloured red when > 0
    bar_colors = ["red" if v > 0 else "steelblue" for v in hard_violations]
    fig.add_trace(
        go.Bar(
            x=times,
            y=hard_violations,
            name="Hard Violations",
            marker_color=bar_colors,
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="Violations", dtick=1, row=3, col=1)
    fig.update_xaxes(title_text="Time", row=3, col=1)

    mission_name = getattr(getattr(ditl, "config", None), "name", "")
    fig.update_layout(
        height=700,
        title_text=f"Radiator Telemetry: {mission_name}"
        if mission_name
        else "Radiator Telemetry",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        font=dict(family=config.font_family),
    )

    return fig
