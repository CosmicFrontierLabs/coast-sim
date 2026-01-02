"""Momentum conservation analysis and visualization for DITL simulations."""

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties

from ..common import ACSMode
from ..config.visualization import VisualizationConfig

if TYPE_CHECKING:
    from ..ditl.ditl_mixin import DITLMixin


def plot_ditl_momentum(
    ditl: "DITLMixin",
    figsize: tuple[float, float] = (14, 14),
    config: VisualizationConfig | None = None,
) -> tuple[Figure, list[Axes]]:
    """Plot momentum conservation analysis for a DITL simulation.

    Creates a 7-panel figure showing:
    - Per-wheel momentum over time
    - Wheel fill fraction (saturation indicator)
    - External disturbance torques breakdown (log scale)
    - Instantaneous torques (disturbance vs MTQ bleed)
    - Cumulative impulse (momentum budget)
    - Conservation check (wheel momentum vs external impulse)
    - Mode timeline (color-coded operational modes)

    Args:
        ditl: DITLMixin instance containing simulation telemetry data.
        figsize: Tuple of (width, height) for the figure size. Default: (14, 14)
        config: VisualizationConfig object. If None, uses ditl.config.visualization
            if available.

    Returns:
        tuple: (fig, axes) - The matplotlib figure and list of axes objects.

    Example:
        >>> from conops.ditl import QueueDITL
        >>> from conops.visualization import plot_ditl_momentum
        >>> ditl = QueueDITL(config=config)
        >>> ditl.calc()
        >>> fig, axes = plot_ditl_momentum(ditl)
        >>> plt.show()
    """
    import matplotlib.pyplot as plt

    # Resolve config
    if not isinstance(config, VisualizationConfig):
        if (
            hasattr(ditl, "config")
            and hasattr(ditl.config, "visualization")
            and isinstance(ditl.config.visualization, VisualizationConfig)
        ):
            config = ditl.config.visualization
        else:
            config = VisualizationConfig()

    # Font settings
    font_family = config.font_family
    title_font_size = config.title_font_size
    label_font_size = config.label_font_size
    tick_font_size = config.tick_font_size
    legend_font_size = config.legend_font_size
    title_prop = FontProperties(family=font_family, size=title_font_size, weight="bold")

    # Time array
    utime = np.array(ditl.utime)
    hours = (utime - utime[0]) / 3600
    dt = ditl.step_size

    # Extract telemetry
    wheel_frac = np.array(getattr(ditl, "wheel_momentum_fraction", []))
    wheel_frac_raw = np.array(getattr(ditl, "wheel_momentum_fraction_raw", []))
    wheel_history = getattr(ditl, "wheel_momentum_history", {})
    mtq_torque = np.array(getattr(ditl, "mtq_torque_mag", []))
    dist_torque = np.array(getattr(ditl, "disturbance_total", []))
    mode = np.array(ditl.mode)

    # Disturbance breakdown
    dist_gg = np.array(getattr(ditl, "disturbance_gg", []), dtype=float)
    dist_drag = np.array(getattr(ditl, "disturbance_drag", []), dtype=float)
    dist_srp = np.array(getattr(ditl, "disturbance_srp", []), dtype=float)
    dist_mag = np.array(getattr(ditl, "disturbance_mag", []), dtype=float)

    # Compute cumulative impulse
    if dist_torque.size == len(hours):
        dist_impulse_cum = np.cumsum(dist_torque) * dt
    else:
        dist_impulse_cum = np.zeros_like(hours)
    if mtq_torque.size == len(hours):
        mtq_impulse_cum = np.cumsum(mtq_torque) * dt
    else:
        mtq_impulse_cum = np.zeros_like(hours)

    # Get momentum warnings count
    n_warnings = 0
    if hasattr(ditl, "acs") and hasattr(ditl.acs, "get_momentum_warnings"):
        n_warnings = len(ditl.acs.get_momentum_warnings())

    # Create figure with 7 panels
    fig, axes_arr = plt.subplots(7, 1, figsize=figsize, sharex=True)
    axes: list[Axes] = list(axes_arr)

    # Panel 1: Per-wheel momentum
    ax = axes[0]
    if wheel_history:
        for name, hist in wheel_history.items():
            if len(hist) == len(hours):
                ax.plot(hours, hist, linewidth=0.5, alpha=0.8, label=name)
        # Get max momentum from config if available
        max_mom = 1.0
        if (
            hasattr(ditl, "acs")
            and hasattr(ditl.acs, "reaction_wheels")
            and ditl.acs.reaction_wheels
        ):
            max_mom = float(getattr(ditl.acs.reaction_wheels[0], "max_momentum", 1.0))
        ax.axhline(
            max_mom, color="red", linestyle="--", alpha=0.5, label="Max capacity"
        )
        ax.axhline(-max_mom, color="red", linestyle="--", alpha=0.5)
        ax.set_ylabel(
            "Per-Wheel\nMomentum (Nms)",
            fontsize=label_font_size,
            fontfamily=font_family,
        )
        ax.legend(loc="upper right", fontsize=legend_font_size, ncol=3)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(
            0.5,
            0.5,
            "No per-wheel momentum history available",
            transform=ax.transAxes,
            ha="center",
            fontsize=label_font_size,
        )
        ax.set_ylabel(
            "Per-Wheel\nMomentum", fontsize=label_font_size, fontfamily=font_family
        )

    # Panel 2: Wheel fill fraction
    ax = axes[1]
    if wheel_frac_raw.size == len(hours):
        ax.fill_between(
            hours, 0, wheel_frac_raw * 100, alpha=0.3, color="blue", label="Raw fill"
        )
        ax.plot(hours, wheel_frac * 100, "b-", linewidth=0.5, label="Effective fill")
        ax.axhline(95, color="orange", linestyle="--", alpha=0.7, label="95% threshold")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper right", fontsize=legend_font_size)
    else:
        ax.text(
            0.5,
            0.5,
            "No wheel fraction data available",
            transform=ax.transAxes,
            ha="center",
            fontsize=label_font_size,
        )
    ax.set_ylabel("Wheel Fill\n(%)", fontsize=label_font_size, fontfamily=font_family)
    ax.grid(True, alpha=0.3)

    # Panel 3: External disturbance torques breakdown (log scale)
    ax = axes[2]
    has_disturbance_data = (
        dist_torque.size == len(hours)
        and dist_gg.size == len(hours)
        and dist_drag.size == len(hours)
        and dist_srp.size == len(hours)
        and dist_mag.size == len(hours)
    )
    if has_disturbance_data:
        ax.plot(hours, dist_torque, label="Total", color="C0", linewidth=1)
        ax.plot(hours, dist_gg, label="Gravity gradient", color="C1", alpha=0.7)
        ax.plot(hours, dist_drag, label="Aero drag", color="C2", alpha=0.7)
        ax.plot(hours, dist_srp, label="Solar pressure", color="C3", alpha=0.7)
        ax.plot(hours, dist_mag, label="Residual magnetic", color="C4", alpha=0.7)
        ax.set_yscale("log")
        ax.set_ylim(1e-10, 1e-4)
        ax.legend(loc="upper right", fontsize=legend_font_size, ncol=2)
    else:
        ax.text(
            0.5,
            0.5,
            "No disturbance breakdown data available",
            transform=ax.transAxes,
            ha="center",
            fontsize=label_font_size,
        )
    ax.set_ylabel(
        "Disturbance\nTorque (N·m)", fontsize=label_font_size, fontfamily=font_family
    )
    ax.grid(True, alpha=0.2)

    # Panel 4: Instantaneous torques (total disturbance vs MTQ)
    ax = axes[3]
    if dist_torque.size == len(hours) and mtq_torque.size == len(hours):
        ax.plot(
            hours,
            dist_torque * 1000,
            "r-",
            linewidth=0.3,
            alpha=0.7,
            label="Disturbance",
        )
        ax.plot(
            hours, mtq_torque * 1000, "b-", linewidth=0.3, alpha=0.7, label="MTQ bleed"
        )
        ax.legend(loc="upper right", fontsize=legend_font_size)
    else:
        ax.text(
            0.5,
            0.5,
            "No torque data available",
            transform=ax.transAxes,
            ha="center",
            fontsize=label_font_size,
        )
    ax.set_ylabel("Torque\n(mNm)", fontsize=label_font_size, fontfamily=font_family)
    ax.grid(True, alpha=0.3)

    # Panel 5: Cumulative impulse (momentum budget)
    ax = axes[4]
    ax.plot(hours, dist_impulse_cum, "r-", linewidth=1, label="Cumulative disturbance")
    ax.plot(hours, mtq_impulse_cum, "b-", linewidth=1, label="Cumulative MTQ bleed")
    ax.plot(
        hours,
        dist_impulse_cum - mtq_impulse_cum,
        "g-",
        linewidth=1,
        label="Net accumulation",
    )
    ax.set_ylabel(
        "Cumulative\nImpulse (Nms)", fontsize=label_font_size, fontfamily=font_family
    )
    ax.legend(loc="upper left", fontsize=legend_font_size)
    ax.grid(True, alpha=0.3)

    # Panel 6: Conservation check
    ax = axes[5]
    if wheel_frac_raw.size == len(hours):
        # Estimate wheel momentum from fill fraction
        max_mom = 1.0
        if (
            hasattr(ditl, "acs")
            and hasattr(ditl.acs, "reaction_wheels")
            and ditl.acs.reaction_wheels
        ):
            max_mom = float(getattr(ditl.acs.reaction_wheels[0], "max_momentum", 1.0))
        est_wheel_h = wheel_frac_raw * max_mom
        net_acc = dist_impulse_cum - mtq_impulse_cum
        ax.plot(hours, est_wheel_h, "b-", linewidth=1, label="Est. wheel H (from fill)")
        ax.plot(
            hours, net_acc, "r--", linewidth=1, alpha=0.7, label="Net external impulse"
        )
        ax.legend(loc="upper left", fontsize=legend_font_size)
    else:
        ax.text(
            0.5,
            0.5,
            "No conservation data available",
            transform=ax.transAxes,
            ha="center",
            fontsize=label_font_size,
        )
    ax.set_ylabel("Momentum\n(Nms)", fontsize=label_font_size, fontfamily=font_family)
    ax.grid(True, alpha=0.3)

    # Panel 7: Mode timeline (color-coded)
    ax = axes[6]
    mode_colors = {
        ACSMode.SCIENCE.value: "green",
        ACSMode.SLEWING.value: "blue",
        ACSMode.PASS.value: "purple",
        ACSMode.CHARGING.value: "orange",
        ACSMode.SAFE.value: "red",
        ACSMode.SAA.value: "gray",
    }
    for m in ACSMode:
        mask = mode == m
        if np.any(mask):
            ax.fill_between(
                hours,
                0,
                1,
                where=mask,
                alpha=0.5,
                color=mode_colors.get(m.value, "gray"),
                label=m.name,
            )
    ax.set_ylabel("Mode", fontsize=label_font_size, fontfamily=font_family)
    ax.set_xlabel("Time (hours)", fontsize=label_font_size, fontfamily=font_family)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.legend(loc="upper right", fontsize=legend_font_size, ncol=4)

    # Title
    config_name = getattr(ditl.config, "name", "DITL Simulation")
    warning_text = (
        f"({n_warnings} momentum warnings)" if n_warnings > 0 else "(0 warnings)"
    )
    fig.suptitle(
        f"Momentum Conservation Analysis: {config_name}\n{warning_text}",
        fontproperties=title_prop,
    )

    # Set tick font sizes for all axes
    for ax in axes:
        ax.tick_params(axis="both", which="major", labelsize=tick_font_size)
        for label in ax.get_xticklabels() + ax.get_yticklabels():
            label.set_fontfamily(font_family)

    plt.tight_layout()
    return fig, axes
