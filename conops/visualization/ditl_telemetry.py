"""Basic DITL timeline visualization with core spacecraft telemetry."""

import math
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from ..common.enums import ACSMode
from ..config.visualization import VisualizationConfig

if TYPE_CHECKING:
    from ..common.enums import ACSMode
    from ..ditl.ditl_mixin import DITLMixin


def plot_ditl_telemetry(
    ditl: "DITLMixin",
    figsize: tuple[float, float] = (10, 8),
    config: VisualizationConfig | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot basic DITL timeline with core spacecraft telemetry.

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
    ) -> list[ACSMode | int]:
        return [v if v is not None else default for v in values]

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
    ax.plot(timehours, none_to_default(ditl.telemetry.housekeeping.mode))
    ax.xaxis.set_visible(False)
    plt.ylabel("Mode", fontsize=label_font_size, fontfamily=font_family)

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
    ax.set_ylim(
        0,
        max(
            x
            for x in none_to_nan(ditl.telemetry.housekeeping.power_usage)
            if not math.isnan(x)
        )
        * 1.1,
    )
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
