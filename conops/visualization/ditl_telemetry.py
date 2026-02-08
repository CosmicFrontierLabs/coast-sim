"""Basic DITL timeline visualization with core spacecraft telemetry."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from plotly.subplots import make_subplots

from ..common.enums import ACSMode
from ..config.visualization import VisualizationConfig

if TYPE_CHECKING:
    from ..common.enums import ACSMode
    from ..ditl.ditl_mixin import DITLMixin


def plot_ditl_telemetry(
    ditl: DITLMixin,
    figsize: tuple[float, float] = (10, 8),
    config: VisualizationConfig | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot basic DITL timeline with core spacecraft telemetry using matplotlib.

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
        figsize: Tuple of (width, height) for the figure size. Default: (10, 8)
        config: VisualizationConfig object. If None, uses ditl.config.visualization if available.

    Returns:
        tuple: (fig, axes) - The matplotlib figure and list of axes objects.

    Note:
        For an interactive Plotly version, use plot_ditl_telemetry_plotly().

    Example:
        >>> from conops.ditl import QueueDITL
        >>> from conops.visualization import plot_ditl_telemetry
        >>> ditl = QueueDITL(config=config)
        >>> ditl.calc()
        >>> fig, axes = plot_ditl_telemetry(ditl)
        >>> plt.show()
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

    # Set default font settings
    font_family = config.font_family
    title_font_size = config.title_font_size
    label_font_size = config.label_font_size
    tick_font_size = config.tick_font_size
    title_prop = FontProperties(family=font_family, size=title_font_size, weight="bold")

    # Helper function to replace None with NaN for plotting
    def none_to_nan(values: list[float | None]) -> list[float]:
        return [v if v is not None else float("nan") for v in values]

    def none_to_default(
        values: list[ACSMode | int | None], default: int = -1
    ) -> list[int]:
        return [
            v.value if isinstance(v, ACSMode) else (v if v is not None else default)
            for v in values
        ]

    timehours = np.array(
        [
            hk.timestamp.timestamp() if hk.timestamp else 0
            for hk in ditl.telemetry.housekeeping
        ]
    ) - (
        ditl.telemetry.housekeeping[0].timestamp.timestamp()
        if ditl.telemetry.housekeeping[0].timestamp
        else 0
    )
    timehours = timehours / 3600

    fig = plt.figure(figsize=figsize)
    axes = []

    ax = plt.subplot(711)
    axes.append(ax)
    plt.plot(timehours, none_to_nan(ditl.telemetry.housekeeping.ra))
    ax.xaxis.set_visible(False)
    plt.ylabel("RA", fontsize=label_font_size, fontfamily=font_family)
    ax.set_title(
        f"Timeline for DITL Simulation: {ditl.config.name}", fontproperties=title_prop
    )

    ax = plt.subplot(712)
    axes.append(ax)
    ax.plot(timehours, none_to_nan(ditl.telemetry.housekeeping.dec))
    ax.xaxis.set_visible(False)
    plt.ylabel("Dec", fontsize=label_font_size, fontfamily=font_family)

    ax = plt.subplot(713)
    axes.append(ax)
    ax.plot(timehours, none_to_default(ditl.telemetry.housekeeping.acs_mode))
    ax.xaxis.set_visible(False)
    plt.ylabel("ACS Mode", fontsize=label_font_size, fontfamily=font_family)

    ax = plt.subplot(714)
    axes.append(ax)
    ax.plot(timehours, none_to_nan(ditl.telemetry.housekeeping.battery_level))
    ax.axhline(
        y=1.0 - ditl.config.battery.max_depth_of_discharge,
        color="r",
        linestyle="--",
    )
    ax.xaxis.set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Batt. charge", fontsize=label_font_size, fontfamily=font_family)

    ax = plt.subplot(715)
    axes.append(ax)
    ax.plot(timehours, none_to_nan(ditl.telemetry.housekeeping.panel_illumination))

    # Mark eclipse periods using the telemetry in_eclipse data
    # Build a 1:1 boolean mask aligned with housekeeping/timehours.
    # Treat None (or missing) values as not in eclipse (False).
    eclipse_mask = np.array(
        [
            bool(hk.in_eclipse) if hk.in_eclipse is not None else False
            for hk in ditl.telemetry.housekeeping
        ],
        dtype=bool,
    )
    if eclipse_mask.size and np.any(eclipse_mask):
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

        # Shade eclipse regions
        for start, end in zip(eclipse_starts, eclipse_ends):
            ax.axvspan(
                start,
                end,
                alpha=0.3,
                color="gray",
                label="Eclipse" if start == eclipse_starts[0] else "",
            )

    ax.xaxis.set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Panel Ill.", fontsize=label_font_size, fontfamily=font_family)

    ax = plt.subplot(716)
    axes.append(ax)
    # Check if subsystem power data is available
    power_bus_values = none_to_nan(ditl.telemetry.housekeeping.power_bus)
    power_payload_values = none_to_nan(ditl.telemetry.housekeeping.power_payload)
    if (
        power_bus_values
        and power_payload_values
        and any(v is not None for v in power_bus_values)
        and any(v is not None for v in power_payload_values)
    ):
        # Line plot showing power breakdown
        ax.plot(timehours, power_bus_values, label="Bus", alpha=0.8)
        ax.plot(
            timehours,
            power_payload_values,
            label="Payload",
            alpha=0.8,
        )
        ax.plot(
            timehours,
            none_to_nan(ditl.telemetry.housekeeping.power_usage),
            label="Total",
            linewidth=2,
            alpha=0.9,
        )
        ax.legend(
            loc="upper right",
            fontsize=config.legend_font_size,
            prop={"family": font_family},
        )
    else:
        # Fall back to total power only
        ax.plot(
            timehours,
            none_to_nan(ditl.telemetry.housekeeping.power_usage),
            label="Total",
        )
    # Calculate y-limit for power plot safely
    power_values = none_to_nan(ditl.telemetry.housekeeping.power_usage)
    finite_power_values = [x for x in power_values if not np.isnan(x)]
    if finite_power_values:
        max_power = max(finite_power_values) * 1.1
    else:
        max_power = 1.0  # Default when no power data available
    ax.set_ylim(0, max_power)
    ax.set_ylabel("Power (W)", fontsize=label_font_size, fontfamily=font_family)
    ax.xaxis.set_visible(False)
    ax.set_ylabel("Power (W)", fontsize=label_font_size, fontfamily=font_family)
    ax.xaxis.set_visible(False)

    ax = plt.subplot(717)
    axes.append(ax)
    ax.plot(timehours, none_to_default(ditl.telemetry.housekeeping.obsid))
    ax.set_ylabel("ObsID", fontsize=label_font_size, fontfamily=font_family)
    ax.set_xlabel(
        "Time (hour of day)", fontsize=label_font_size, fontfamily=font_family
    )

    # Set tick font sizes for all axes
    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=tick_font_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)

    return fig, axes


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
