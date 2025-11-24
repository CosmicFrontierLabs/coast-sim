"""Basic DITL timeline visualization with core spacecraft telemetry."""

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

if TYPE_CHECKING:
    from ..ditl.ditl_mixin import DITLMixin


def plot_ditl_telemetry(ditl: "DITLMixin", figsize=(10, 8)):
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
    timehours = (np.array(ditl.utime) - ditl.utime[0]) / 3600

    fig = plt.figure(figsize=figsize)
    axes = []

    ax = plt.subplot(711)
    axes.append(ax)
    plt.plot(timehours, ditl.ra)
    ax.xaxis.set_visible(False)
    plt.ylabel("RA")
    ax.set_title(f"Timeline for DITL Simulation: {ditl.config.name}")

    ax = plt.subplot(712)
    axes.append(ax)
    ax.plot(timehours, ditl.dec)
    ax.xaxis.set_visible(False)
    plt.ylabel("Dec")

    ax = plt.subplot(713)
    axes.append(ax)
    ax.plot(timehours, ditl.mode)
    ax.xaxis.set_visible(False)
    plt.ylabel("Mode")

    ax = plt.subplot(714)
    axes.append(ax)
    ax.plot(timehours, ditl.batterylevel)
    ax.axhline(
        y=1.0 - ditl.config.battery.max_depth_of_discharge,
        color="r",
        linestyle="--",
    )
    ax.xaxis.set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Batt. charge")

    ax = plt.subplot(715)
    axes.append(ax)
    ax.plot(timehours, ditl.panel)
    ax.xaxis.set_visible(False)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Panel Ill.")

    ax = plt.subplot(716)
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
        ax.legend(loc="upper right", fontsize="small")
    else:
        # Fall back to total power only
        ax.plot(timehours, ditl.power, label="Total")
    ax.set_ylim(0, max(ditl.power) * 1.1)
    ax.set_ylabel("Power (W)")
    ax.xaxis.set_visible(False)

    ax = plt.subplot(717)
    axes.append(ax)
    ax.plot(timehours, ditl.obsid)
    ax.set_ylabel("ObsID")
    ax.set_xlabel("Time (hour of day)")

    return fig, axes
