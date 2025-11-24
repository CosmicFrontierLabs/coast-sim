"""ACS mode analysis and visualization utilities."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from collections import Counter

if TYPE_CHECKING:
    from ..ditl.ditl_mixin import DITLMixin


def plot_acs_mode_distribution(ditl: "DITLMixin", figsize=(10, 8)):
    """Plot a pie chart showing the distribution of time spent in each ACS mode.

    Creates a pie chart displaying the percentage of time spent in different
    ACS modes during the simulation. This helps analyze observing efficiency
    and operational patterns.

    Args:
        ditl: DITLMixin instance containing simulation telemetry data.
        figsize: Tuple of (width, height) for the figure size. Default: (10, 8)

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
    # Convert mode values to names
    from ..common import ACSMode

    modes = []
    for mode_val in ditl.mode:
        if mode_val in [m.value for m in ACSMode]:
            mode_name = ACSMode(mode_val).name
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
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=140)
    ax.set_title("Percentage of Time Spent in Each ACS Mode")
    ax.axis("equal")  # Equal aspect ratio ensures pie is a circle

    return fig, ax
