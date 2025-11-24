"""Data management visualization utilities for CONOPS simulations."""

from datetime import datetime

import matplotlib.pyplot as plt


def plot_data_management_telemetry(
    ditl,
    figsize=(12, 10),
    show_summary=True,
):
    """Plot comprehensive data management telemetry from a DITL simulation.

    Creates a multi-panel figure showing:
    1. Onboard recorder data volume over time
    2. Recorder fill fraction with alert thresholds
    3. Cumulative data generated
    4. Cumulative data downlinked
    5. Recorder alert timeline

    Args:
        ditl: DITL simulation object with data management telemetry.
        figsize: Tuple of (width, height) for the figure size. Default: (12, 10)
        show_summary: Whether to print summary statistics. Default: True

    Returns:
        tuple: (fig, axes) - The matplotlib figure and axes objects.

    Example:
        >>> from conops.ditl import QueueDITL
        >>> from conops.visualization import plot_data_management_telemetry
        >>> ditl = QueueDITL(config=config)
        >>> ditl.calc()
        >>> fig, axes = plot_data_management_telemetry(ditl)
        >>> plt.show()
    """
    # Create figure with 5 subplots
    fig, axes = plt.subplots(5, 1, figsize=figsize)

    # Convert unix timestamps to datetime objects
    times = [datetime.fromtimestamp(t) for t in ditl.utime]

    # Plot 1: Recorder Volume (Gb)
    axes[0].plot(times, ditl.recorder_volume_gb, "b-", linewidth=2)
    axes[0].set_ylabel("Volume (Gb)", fontsize=10)
    axes[0].set_title("Onboard Recorder Data Volume", fontsize=12, fontweight="bold")
    axes[0].grid(True, alpha=0.3)
    axes[0].axhline(
        y=ditl.config.recorder.capacity_gb,
        color="r",
        linestyle="--",
        label=f"Capacity ({ditl.config.recorder.capacity_gb} Gb)",
    )
    axes[0].legend()

    # Plot 2: Recorder Fill Fraction
    axes[1].plot(times, ditl.recorder_fill_fraction, "g-", linewidth=2)
    axes[1].axhline(
        y=ditl.config.recorder.yellow_threshold,
        color="orange",
        linestyle="--",
        label=f"Yellow Threshold ({ditl.config.recorder.yellow_threshold:.0%})",
    )
    axes[1].axhline(
        y=ditl.config.recorder.red_threshold,
        color="red",
        linestyle="--",
        label=f"Red Threshold ({ditl.config.recorder.red_threshold:.0%})",
    )
    axes[1].set_ylabel("Fill Fraction", fontsize=10)
    axes[1].set_title("Recorder Fill Level", fontsize=12, fontweight="bold")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    # Plot 3: Cumulative Data Generated (Gb)
    axes[2].plot(times, ditl.data_generated_gb, "m-", linewidth=2)
    axes[2].set_ylabel("Data Generated (Gb)", fontsize=10)
    axes[2].set_title("Cumulative Data Generated", fontsize=12, fontweight="bold")
    axes[2].grid(True, alpha=0.3)

    # Plot 4: Cumulative Data Downlinked (Gb)
    axes[3].plot(times, ditl.data_downlinked_gb, "c-", linewidth=2)
    axes[3].set_ylabel("Data Downlinked (Gb)", fontsize=10)
    axes[3].set_title("Cumulative Data Downlinked", fontsize=12, fontweight="bold")
    axes[3].grid(True, alpha=0.3)

    # Plot 5: Recorder Alert Timeline
    # Map integer alert levels to colors
    alert_colors = {0: "green", 1: "yellow", 2: "red"}
    colors = [alert_colors[alert] for alert in ditl.recorder_alert]
    axes[4].scatter(times, ditl.recorder_alert, c=colors, s=10, alpha=0.6)
    axes[4].set_yticks([0, 1, 2])
    axes[4].set_yticklabels(["None", "Yellow", "Red"])
    axes[4].set_ylabel("Alert Level", fontsize=10)
    axes[4].set_xlabel("Time", fontsize=10)
    axes[4].set_title("Recorder Alert Timeline", fontsize=12, fontweight="bold")
    axes[4].grid(True, alpha=0.3)

    plt.tight_layout()

    # Print summary statistics if requested
    if show_summary:
        _print_data_management_summary(ditl)

    return fig, axes


def _print_data_management_summary(ditl):
    """Print summary statistics for data management.

    Args:
        ditl: DITL simulation object with data management telemetry.
    """
    total_generated = ditl.data_generated_gb[-1] if ditl.data_generated_gb else 0
    total_downlinked = ditl.data_downlinked_gb[-1] if ditl.data_downlinked_gb else 0
    final_volume = ditl.recorder_volume_gb[-1] if ditl.recorder_volume_gb else 0
    max_fill = max(ditl.recorder_fill_fraction) if ditl.recorder_fill_fraction else 0

    print("\n" + "=" * 70)
    print("DATA MANAGEMENT SUMMARY")
    print("=" * 70)
    print(f"Total Data Generated:    {total_generated:.2f} Gb")
    print(f"Total Data Downlinked:   {total_downlinked:.2f} Gb")
    print(f"Final Recorder Volume:   {final_volume:.2f} Gb")
    print(f"Max Fill Fraction:       {max_fill:.1%}")
    print(f"Recorder Capacity:       {ditl.config.recorder.capacity_gb:.2f} Gb")

    if total_generated > 0:
        downlink_efficiency = (total_downlinked / total_generated) * 100
        print(f"Downlink Efficiency:     {downlink_efficiency:.1f}%")

    print("=" * 70)
