"""ACS mode analysis and visualization utilities (Plotly)."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING

import plotly.graph_objects as go

from ...common import normalize_acs_mode
from ...config.visualization import VisualizationConfig

if TYPE_CHECKING:
    from ...ditl.ditl_mixin import DITLMixin


def plot_acs_mode_distribution_plotly(
    ditl: DITLMixin,
    config: VisualizationConfig | None = None,
) -> go.Figure:
    """Plot a pie chart showing the distribution of time spent in each ACS mode.

    Creates an interactive pie chart displaying the percentage of time spent
    in different ACS modes during the simulation.

    Args:
        ditl: DITLMixin instance containing simulation telemetry data.
        config: VisualizationConfig object. If None, uses
            ``ditl.config.visualization`` if available, otherwise defaults.

    Returns:
        plotly.graph_objects.Figure: The Plotly figure object.

    Example:
        >>> from conops.ditl import QueueDITL
        >>> from conops.visualization import plot_acs_mode_distribution_plotly
        >>> ditl = QueueDITL(config=config)
        >>> ditl.calc()
        >>> fig = plot_acs_mode_distribution_plotly(ditl)
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

    modes = []
    for mode_val in ditl.telemetry.housekeeping.acs_mode:
        normalized = normalize_acs_mode(mode_val)
        modes.append(
            normalized.name if normalized is not None else f"UNKNOWN({mode_val})"
        )

    mode_counts = Counter(modes)
    labels = list(mode_counts.keys())
    values = list(mode_counts.values())
    colors = [config.mode_colors.get(label.upper(), "gray") for label in labels]

    title = getattr(getattr(ditl, "config", None), "name", "DITL")
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            marker=dict(colors=colors),
            textinfo="label+percent",
            hovertemplate="%{label}: %{value} steps (%{percent})<extra></extra>",
            sort=False,
        )
    )
    fig.update_layout(
        title_text=f"ACS Mode Distribution — {title}",
        showlegend=True,
        legend=dict(orientation="v"),
    )

    return fig
