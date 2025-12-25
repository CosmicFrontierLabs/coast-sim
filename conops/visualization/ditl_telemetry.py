"""Basic DITL timeline visualization with core spacecraft telemetry."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from ..common import ACSCommandType, ACSMode
from ..config.visualization import VisualizationConfig

if TYPE_CHECKING:
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

    timehours = (np.array(ditl.utime) - ditl.utime[0]) / 3600

    def _is_sequence(val: object) -> bool:
        return isinstance(val, (list, tuple, np.ndarray))

    wm_raw = getattr(ditl, "wheel_momentum_fraction", None)
    wt_raw = getattr(ditl, "wheel_torque_fraction", None)
    wm_seq = wm_raw if _is_sequence(wm_raw) else None
    wt_seq = wt_raw if _is_sequence(wt_raw) else None
    has_wheel = (
        wm_seq is not None
        and wt_seq is not None
        and len(wm_seq) > 0
        and len(wt_seq) > 0
    )
    n_panels = 8 if has_wheel else 7
    fig = plt.figure(figsize=figsize)
    axes = []

    ax = plt.subplot(n_panels * 100 + 11)
    axes.append(ax)
    plt.plot(timehours, ditl.ra)
    ax.xaxis.set_visible(False)
    plt.ylabel("RA", fontsize=label_font_size, fontfamily=font_family)
    ax.set_title(
        f"Timeline for DITL Simulation: {ditl.config.name}", fontproperties=title_prop
    )

    ax = plt.subplot(n_panels * 100 + 12)
    axes.append(ax)
    ax.plot(timehours, ditl.dec)
    ax.xaxis.set_visible(False)
    plt.ylabel("Dec", fontsize=label_font_size, fontfamily=font_family)

    ax = plt.subplot(n_panels * 100 + 13)
    axes.append(ax)
    ax.plot(timehours, ditl.mode)
    ax.xaxis.set_visible(False)
    plt.ylabel("Mode", fontsize=label_font_size, fontfamily=font_family)
    mode_ticks = [m.value for m in ACSMode]
    ax.set_yticks(mode_ticks)
    ax.set_yticklabels(
        [m.name.title() for m in ACSMode],
        fontsize=tick_font_size,
        fontfamily=font_family,
    )

    ax = plt.subplot(n_panels * 100 + 14)
    axes.append(ax)
    ax.plot(timehours, ditl.batterylevel)
    ax.axhline(
        y=1.0 - ditl.config.battery.max_depth_of_discharge,
        color="r",
        linestyle="--",
    )
    ax.xaxis.set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Batt. charge", fontsize=label_font_size, fontfamily=font_family)

    ax = plt.subplot(n_panels * 100 + 15)
    axes.append(ax)
    ax.plot(timehours, ditl.panel)
    ax.xaxis.set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Panel Ill.", fontsize=label_font_size, fontfamily=font_family)

    ax = plt.subplot(n_panels * 100 + 16)
    axes.append(ax)
    # Check if subsystem power data is available
    if (
        hasattr(ditl, "power_bus")
        and hasattr(ditl, "power_payload")
        and ditl.power_bus
        and ditl.power_payload
    ):
        # Line plot showing power breakdown
        ax.plot(timehours, ditl.power_bus, label="Bus", alpha=0.8)
        ax.plot(timehours, ditl.power_payload, label="Payload", alpha=0.8)
        ax.plot(timehours, ditl.power, label="Total", linewidth=2, alpha=0.9)
        ax.legend(
            loc="upper right",
            fontsize=config.legend_font_size,
            prop={"family": font_family},
        )
    else:
        # Fall back to total power only
        ax.plot(timehours, ditl.power, label="Total")
    ax.set_ylim(0, max(ditl.power) * 1.1)
    ax.set_ylabel("Power (W)", fontsize=label_font_size, fontfamily=font_family)
    ax.xaxis.set_visible(False)

    if has_wheel:
        ax = plt.subplot(n_panels * 100 + 17)
        axes.append(ax)
        momentum_series = wm_seq if wm_seq is not None else ditl.wheel_momentum_fraction
        torque_series = wt_seq if wt_seq is not None else ditl.wheel_torque_fraction
        ax.plot(timehours, momentum_series, label="Momentum", alpha=0.8)
        ax.plot(timehours, torque_series, label="Torque", alpha=0.8)
        # Highlight desat windows (if ACS command history is available)
        desat_spans = []
        if hasattr(ditl, "acs") and hasattr(ditl.acs, "executed_commands"):
            for cmd in getattr(ditl.acs, "executed_commands", []):
                if getattr(cmd, "command_type", None) == ACSCommandType.DESAT:
                    start = float(getattr(cmd, "execution_time", timehours[0] * 3600))
                    duration = float(getattr(cmd, "duration", 0.0) or 0.0)
                    end = start + duration
                    desat_spans.append((start, end))
        # Merge overlapping spans for cleaner visualization
        desat_spans = sorted(desat_spans, key=lambda x: x[0])
        merged_spans: list[list[float]] = []
        for span in desat_spans:
            if not merged_spans or span[0] > merged_spans[-1][1]:
                merged_spans.append(list(span))
            else:
                merged_spans[-1][1] = max(merged_spans[-1][1], span[1])
        for start, end in merged_spans:
            ax.axvspan(
                (start - ditl.utime[0]) / 3600.0,
                (end - ditl.utime[0]) / 3600.0,
                color="gray",
                alpha=0.2,
                label="Desat",
            )
        if hasattr(ditl, "wheel_saturation") and ditl.wheel_saturation:
            sat_times = [t for t, s in zip(timehours, ditl.wheel_saturation) if s]
            sat_vals = [1.0] * len(sat_times)
            if sat_times:
                ax.scatter(
                    sat_times, sat_vals, color="r", marker="x", label="Saturated"
                )
        # Deduplicate legend labels
        handles, labels = ax.get_legend_handles_labels()
        seen = {}
        dedup_handles = []
        dedup_labels = []
        for handle, label in zip(handles, labels):
            if label not in seen:
                seen[label] = True
                dedup_handles.append(handle)
                dedup_labels.append(label)
        if dedup_handles:
            ax.legend(
                dedup_handles,
                dedup_labels,
                loc="upper right",
                fontsize=config.legend_font_size,
                prop={"family": font_family},
            )
        ax.set_ylim(0, 1.05)
        ax.set_ylabel(
            "Wheel\n(resource)", fontsize=label_font_size, fontfamily=font_family
        )
        ax.xaxis.set_visible(False)

    ax = plt.subplot(n_panels * 100 + (17 if not has_wheel else 18))
    axes.append(ax)
    ax.plot(timehours, ditl.obsid)
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
