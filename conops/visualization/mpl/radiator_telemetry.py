"""Radiator telemetry visualization for ConOps simulations."""

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from conops.ditl.ditl import DITL
from conops.ditl.queue_ditl import QueueDITL

from ...config.visualization import VisualizationConfig


def _none_to_nan(values: Sequence[float | int | None]) -> list[float]:
    return [float(v) if v is not None else float("nan") for v in values]


def plot_radiator_telemetry(
    ditl: QueueDITL | DITL,
    figsize: tuple[float, float] = (12, 8),
    config: VisualizationConfig | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot radiator thermal exposure and heat dissipation telemetry.

    Creates a three-panel figure showing:
    1. Area-weighted Sun and Earth exposure fractions (0–1)
    2. Net heat dissipation in Watts (positive = rejecting heat)
    3. Count of radiators violating hard keep-out constraints

    Args:
        ditl: DITL simulation object after calling calc().
        figsize: Figure size as (width, height). Default: (12, 8).
        config: VisualizationConfig. Falls back to ditl.config.visualization or defaults.

    Returns:
        tuple: (fig, axes) — the matplotlib Figure and a list of three Axes.

    Example:
        >>> fig, axes = plot_radiator_telemetry(ditl)
        >>> plt.show()
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

    font_family = config.font_family
    title_prop = FontProperties(
        family=font_family, size=config.title_font_size, weight="bold"
    )
    label_font_size = config.label_font_size
    tick_font_size = config.tick_font_size
    legend_font_size = config.legend_font_size

    hk = ditl.telemetry.housekeeping
    times = hk.timestamp

    sun_exposure = _none_to_nan(hk.radiator_sun_exposure)
    earth_exposure = _none_to_nan(hk.radiator_earth_exposure)
    heat_dissipation = _none_to_nan(hk.radiator_heat_dissipation_w)
    hard_violations = _none_to_nan(hk.radiator_hard_violations)

    fig, axes = plt.subplots(3, 1, figsize=figsize)

    # Panel 1: Sun and Earth exposure
    axes[0].plot(times, sun_exposure, color="orange", linewidth=1.5, label="Sun")
    axes[0].plot(
        times, earth_exposure, color="cornflowerblue", linewidth=1.5, label="Earth"
    )
    axes[0].set_ylabel("Exposure", fontsize=label_font_size, fontfamily=font_family)
    axes[0].set_ylim(0.0, 1.05)
    axes[0].set_title("Radiator Exposure Fractions", fontproperties=title_prop)
    axes[0].legend(prop={"family": font_family, "size": legend_font_size})
    axes[0].grid(True, alpha=0.3)

    # Panel 2: Heat dissipation
    axes[1].plot(times, heat_dissipation, color="tomato", linewidth=1.5)
    axes[1].axhline(y=0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.5)
    axes[1].set_ylabel(
        "Heat Dissipation (W)", fontsize=label_font_size, fontfamily=font_family
    )
    axes[1].set_title("Net Radiator Heat Dissipation", fontproperties=title_prop)
    axes[1].grid(True, alpha=0.3)

    # Panel 3: Hard constraint violations
    violation_array = np.array(hard_violations)
    colors = np.where(violation_array > 0, "red", "steelblue")
    axes[2].bar(times, hard_violations, color=colors, width=0.0005, align="center")
    axes[2].set_ylabel("Violations", fontsize=label_font_size, fontfamily=font_family)
    axes[2].set_xlabel("Time", fontsize=label_font_size, fontfamily=font_family)
    axes[2].set_title("Radiator Hard Constraint Violations", fontproperties=title_prop)
    axes[2].yaxis.get_major_locator().set_params(integer=True)
    axes[2].grid(True, alpha=0.3, axis="y")

    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=tick_font_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)

    plt.tight_layout()
    return fig, list(axes)
