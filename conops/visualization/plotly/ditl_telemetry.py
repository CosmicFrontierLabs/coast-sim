"""Basic DITL timeline visualization with core spacecraft telemetry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ...config.visualization import VisualizationConfig

if TYPE_CHECKING:
    from ...ditl.ditl_mixin import DITLMixin


def plot_ditl_telemetry_plotly(
    ditl: DITLMixin,
    config: VisualizationConfig | None = None,
) -> go.Figure:
    """Plot basic DITL timeline with core spacecraft telemetry using Plotly.

    Creates a 7-panel figure showing:
    - RA (Right Ascension)
    - Dec (Declination)
    - ACS Mode
    - Battery charge level with DoD limit
    - Solar panel illumination
    - Power consumption (with subsystem breakdown if available)
    - Observation ID

    Args:
        ditl: DITLMixin instance containing simulation telemetry data.
        config: VisualizationConfig object. If None, uses ditl.config.visualization if available.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.

    Example:
        >>> from conops.ditl import QueueDITL
        >>> from conops.visualization import plot_ditl_telemetry_plotly
        >>> ditl = QueueDITL(config=config)
        >>> ditl.calc()
        >>> fig = plot_ditl_telemetry_plotly(ditl)
        >>> fig.show()
    """
    # Resolve config: if the provided config is not a VisualizationConfig instance,
    # then try to use ditl.config.visualization if it's a VisualizationConfig, else use defaults.
    if not isinstance(config, VisualizationConfig):
        if (
            hasattr(ditl, "config")
            and hasattr(ditl.config, "visualization")
            and isinstance(ditl.config.visualization, VisualizationConfig)
        ):
            config = ditl.config.visualization
        else:
            config = VisualizationConfig()

    timehours = (np.array(ditl.utime) - ditl.utime[0]) / 3600

    # Create subplots
    fig = make_subplots(
        rows=7,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.02,
    )

    # RA subplot
    fig.add_trace(
        go.Scatter(
            x=timehours,
            y=ditl.ra,
            mode="lines",
            name="RA",
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(title_text="RA", row=1, col=1)

    # Dec subplot
    fig.add_trace(
        go.Scatter(
            x=timehours,
            y=ditl.dec,
            mode="lines",
            name="Dec",
            showlegend=False,
        ),
        row=2,
        col=1,
    )
    fig.update_yaxes(title_text="Dec", row=2, col=1)

    # ACS Mode subplot
    fig.add_trace(
        go.Scatter(
            x=timehours,
            y=ditl.mode,
            mode="lines",
            name="Mode",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    fig.update_yaxes(title_text="Mode", row=3, col=1)

    # Battery charge level subplot
    fig.add_trace(
        go.Scatter(
            x=timehours,
            y=ditl.batterylevel,
            mode="lines",
            name="Battery Level",
            showlegend=False,
        ),
        row=4,
        col=1,
    )
    # Add DoD limit line
    dod_limit = 1.0 - ditl.config.battery.max_depth_of_discharge
    fig.add_hline(
        y=dod_limit,
        line_dash="dash",
        line_color="red",
        row=4,
        col=1,
    )
    fig.update_yaxes(title_text="Battery Charge", range=[0, 1], row=4, col=1)

    # Solar panel illumination subplot
    fig.add_trace(
        go.Scatter(
            x=timehours,
            y=ditl.panel,
            mode="lines",
            name="Panel Illumination",
            showlegend=False,
        ),
        row=5,
        col=1,
    )

    # Add eclipse shading using the DITL in_eclipse array
    if hasattr(ditl, "in_eclipse") and ditl.in_eclipse:
        eclipse_mask = np.array(ditl.in_eclipse)
        if np.any(eclipse_mask):
            # Find eclipse start and end times
            eclipse_starts = []
            eclipse_ends = []
            in_eclipse = False
            for i, is_eclipse in enumerate(eclipse_mask):
                if is_eclipse and not in_eclipse:
                    eclipse_starts.append(timehours[i])
                    in_eclipse = True
                elif not is_eclipse and in_eclipse:
                    eclipse_ends.append(timehours[i - 1])
                    in_eclipse = False
            # Handle case where eclipse continues to end
            if in_eclipse:
                eclipse_ends.append(timehours[-1])

            # Add shaded rectangles for eclipse periods
            for start, end in zip(eclipse_starts, eclipse_ends):
                fig.add_vrect(
                    x0=start,
                    x1=end,
                    fillcolor="gray",
                    opacity=0.3,
                    layer="below",
                    line_width=0,
                    # annotation_text="Eclipse" if start == eclipse_starts[0] else None,
                    # annotation_position="top left",
                    row=5,
                    col=1,
                )

    fig.update_yaxes(title_text="Panel Illumination", range=[0, 1], row=5, col=1)

    # Power consumption subplot
    if (
        hasattr(ditl, "power_bus")
        and hasattr(ditl, "power_payload")
        and ditl.power_bus is not None
        and ditl.power_payload is not None
        and len(ditl.power_bus) > 0
        and len(ditl.power_payload) > 0
    ):
        # Power breakdown available
        fig.add_trace(
            go.Scatter(
                x=timehours,
                y=ditl.power_bus,
                mode="lines",
                name="Bus Power",
                legendgroup="power",
            ),
            row=6,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=timehours,
                y=ditl.power_payload,
                mode="lines",
                name="Payload Power",
                legendgroup="power",
            ),
            row=6,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=timehours,
                y=ditl.power,
                mode="lines",
                name="Total Power",
                line=dict(width=3),
                legendgroup="power",
            ),
            row=6,
            col=1,
        )
    else:
        # Fall back to total power only
        fig.add_trace(
            go.Scatter(
                x=timehours,
                y=ditl.power,
                mode="lines",
                name="Total Power",
                showlegend=False,
            ),
            row=6,
            col=1,
        )
    fig.update_yaxes(title_text="Power (W)", row=6, col=1)

    # Observation ID subplot
    fig.add_trace(
        go.Scatter(
            x=timehours,
            y=ditl.obsid,
            mode="lines",
            name="ObsID",
            showlegend=False,
        ),
        row=7,
        col=1,
    )
    fig.update_yaxes(title_text="ObsID", row=7, col=1)
    fig.update_xaxes(title_text="Time (hours)", row=7, col=1)

    # Update layout
    fig.update_layout(
        height=1000,
        title_text=f"DITL Timeline: {ditl.config.name}",
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
    )

    return fig
