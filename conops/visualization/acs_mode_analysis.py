"""ACS mode analysis and visualization utilities."""

from collections import Counter
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from ..common import normalize_acs_mode
from ..config.visualization import VisualizationConfig

if TYPE_CHECKING:
    from ..ditl.ditl_mixin import DITLMixin


def plot_acs_mode_distribution(
    ditl: "DITLMixin",
    figsize: tuple[float, float] = (10, 8),
    config: VisualizationConfig | None = None,
) -> tuple[Figure, Axes]:
    """Plot a pie chart showing the distribution of time spent in each ACS mode.

    Creates a pie chart displaying the percentage of time spent in different
    ACS modes during the simulation. This helps analyze observing efficiency
    and operational patterns.

    Args:
        ditl: DITLMixin instance containing simulation telemetry data.
        figsize: Tuple of (width, height) for the figure size. Default: (10, 8)
        config: VisualizationConfig object. If None, uses ditl.config.visualization if available.

    Returns:
        tuple: (fig, ax) - The matplotlib figure and axes objects.

    Example:
        >>> from conops.ditl import QueueDITL
        >>> from conops.visualization import plot_acs_mode_distribution
        >>> ditl = QueueDITL(config=config)
        >>> ditl.calc()
        >>> fig, ax = plot_acs_mode_distribution(ditl)
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
    title_prop = FontProperties(family=font_family, size=title_font_size, weight="bold")

    # Convert mode values to names

    modes = []
    for mode_val in ditl.telemetry.housekeeping.acs_mode:
        normalized_mode = normalize_acs_mode(mode_val)
        if normalized_mode is not None:
            mode_name = normalized_mode.name
        else:
            mode_name = f"UNKNOWN({mode_val})"
        modes.append(mode_name)

    # Count occurrences of each mode
    mode_counts = Counter(modes)

    # Prepare data for pie chart
    labels = list(mode_counts.keys())
    sizes = list(mode_counts.values())

    # Create pie chart
    fig, ax = plt.subplots(figsize=figsize)
    # Compute colors for each wedge based on config.mode_colors (fallback to Matplotlib defaults)
    mode_colors_map = config.mode_colors
    wedge_colors = [mode_colors_map.get(label.upper(), "gray") for label in labels]

    pie_res = ax.pie(
        sizes,
        labels=labels,
        colors=wedge_colors,
        autopct="%1.1f%%",
        startangle=140,
        textprops={"fontsize": label_font_size, "fontfamily": font_family},
    )
    # matplotlib's pie() can return 2 or 3 values depending on whether autopct is set/used.
    # Be defensive: ensure we always have wedges, texts, and autotexts variables
    autotexts = pie_res[2] if len(pie_res) > 2 else []
    ax.set_title(
        "Percentage of Time Spent in Each ACS Mode", fontproperties=title_prop, pad=20
    )
    ax.axis("equal")  # Equal aspect ratio ensures pie is a circle

    # Set font for autotexts (percentage labels), if present
    for autotext in autotexts:
        autotext.set_fontsize(label_font_size)
        autotext.set_fontfamily(font_family)

    return fig, ax
