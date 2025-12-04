"""DITL Timeline Visualization

Provides functions to create timeline plots similar to proposal figures,
showing spacecraft operations including observations, slews, SAA passages,
eclipses, and ground station passes.
"""

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from conops.ditl.ditl import DITL
from conops.ditl.queue_ditl import QueueDITL

from ..common import ACSMode
from ..config import ObservationCategories
from ..config.visualization import VisualizationConfig


def plot_ditl_timeline(
    ditl: DITL | QueueDITL,
    offset_hours: float = 0.0,
    figsize: tuple[float, float] = (10, 6),
    orbit_period: float = 5762.0,
    show_orbit_numbers: bool = True,
    show_saa: bool = False,
    save_path: str | None = None,
    font_family: str | None = None,
    font_size: int | None = None,
    observation_categories: ObservationCategories | None = None,
    config: VisualizationConfig | None = None,
) -> tuple[Figure, Axes]:
    """Plot a DITL timeline showing spacecraft operations.

    Creates a comprehensive timeline visualization showing:
    - Orbit numbers (optional)
    - Science observations (color-coded by obsid range)
    - Slews and settling time
    - Safe mode periods
    - SAA passages
    - Eclipses
    - Ground station passes

    Parameters
    ----------
    ditl : QueueDITL or DITL
        The DITL simulation object with completed simulation data.
    offset_hours : float, optional
        Time offset in hours to shift the timeline (default: 0).
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (10, 4)).
    orbit_period : float, optional
        Orbital period in seconds for orbit number display (default: 5762.0).
    show_orbit_numbers : bool, optional
        Whether to show orbit numbers at the top (default: True).
    show_saa : bool, optional
        Whether to show SAA passages (default: True).
    save_path : str, optional
        If provided, save the figure to this path (default: None).
    font_family : str, optional
        Font family to use for text. If None, uses config.font_family (default: 'Helvetica').
    font_size : int, optional
        Base font size for labels. If None, uses config.label_font_size (default: 11).
    observation_categories : ObservationCategories, optional
        Configuration for categorizing observations by target ID ranges.
        If None, attempts to use ditl.config.observation_categories, then
        falls back to default categories.
    config : VisualizationConfig, optional
        Visualization configuration settings. If None, uses ditl.config.visualization if available.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axes containing the timeline plot.

    Examples
    --------
    >>> from conops import QueueDITL
    >>> ditl = QueueDITL(config)
    >>> ditl.calc()
    >>> fig, ax = plot_ditl_timeline(ditl, save_path='ditl_timeline.pdf')
    >>> plt.show()
    """
    # Resolve config
    if config is None:
        # Resolve config: if the provided config is not a VisualizationConfig instance,
        # then try to use ditl.config.visualization if it's a VisualizationConfig, else use defaults.
        if (
            hasattr(ditl, "config")
            and hasattr(ditl.config, "visualization")
            and isinstance(ditl.config.visualization, VisualizationConfig)
        ):
            config = ditl.config.visualization
        else:
            config = VisualizationConfig()

    # Set default font settings
    if font_family is None:
        font_family = config.font_family
    if font_size is None:
        font_size = config.label_font_size

    # Extract simulation start time
    if not ditl.plan or len(ditl.plan) == 0:
        raise ValueError("DITL simulation has no pointings. Run calc() first.")

    t_start = ditl.plan[0].begin

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = plt.axes((0.12, 0.3, 0.8, 0.5), frameon=True)

    # Calculate timeline duration in hours
    if hasattr(ditl, "utime") and len(ditl.utime) > 0:
        duration_hours = (ditl.utime[-1] - ditl.utime[0]) / 3600.0
    else:
        duration_hours = 24.0

    # Define timeline rows with labels and spacing
    bar_height = 0.15
    row_spacing = 0.25  # Space between row centers

    # Build the timeline rows dynamically
    timeline_rows = []

    # Always include these rows
    timeline_rows.append(("Observations", None, None))
    timeline_rows.append(("Slewing", None, None))
    timeline_rows.append(("Charging", None, None))
    timeline_rows.append(("Safe Mode", None, None))

    # Conditionally include SAA
    if show_saa:
        timeline_rows.append(("SAA", None, None))

    # Always include these rows
    timeline_rows.append(("Eclipse", None, None))
    timeline_rows.append(("Ground Contact", None, None))

    # Calculate y-positions for data rows (starting from 0 and going down)
    num_data_rows = len(timeline_rows)
    data_y_positions = [-(i * row_spacing) for i in range(num_data_rows)]

    # Create mapping of row names to y-positions
    row_positions = dict(zip([row[0] for row in timeline_rows], data_y_positions))

    # If showing orbit numbers, shift everything down and add orbit at top
    if show_orbit_numbers:
        orbit_y_position = data_y_positions[0] + row_spacing
        data_y_positions = [orbit_y_position] + data_y_positions
        timeline_rows.insert(0, ("Orbit", None, None))

        # Draw orbit numbers if requested
        num_orbits = int(duration_hours * 3600 / orbit_period) + 1
        for i in range(num_orbits):
            barcol = "grey" if i % 2 == 1 else "white"
            orbit_start = i * orbit_period / 3600
            orbit_width = orbit_period / 3600
            ax.broken_barh(
                [(orbit_start, orbit_width)],
                (orbit_y_position, bar_height),
                facecolors=barcol,
                edgecolor="black",
                lw=1,
                linestyle="-",
            )

            ax.text(
                (i + 0.5) * orbit_period / 3600,
                orbit_y_position + bar_height / 2,
                f"{i + 1}",
                horizontalalignment="center",
                verticalalignment="center",
                fontname=font_family,
                fontsize=font_size - 2,
                zorder=2,
            )

    # Extract observation segments from plan by obsid ranges
    # Get observation categories from config if not provided
    if observation_categories is None:
        if hasattr(ditl, "config") and hasattr(ditl.config, "observation_categories"):
            observation_categories = ditl.config.observation_categories
    # If still None, default will be used in _extract_observations

    observations_by_type = _extract_observations(
        ditl, t_start, offset_hours, observation_categories
    )

    # Get color mapping from categories configuration
    if observation_categories is None:
        observation_categories = ObservationCategories.default_categories()

    obs_y_pos = row_positions["Observations"]
    labels_shown = set()
    for obs_type, segments in observations_by_type.items():
        if segments and obs_type != "Charging":
            color = observation_categories.get_category_color(obs_type)
            label = f"{obs_type} Target" if obs_type != "Calibration" else obs_type
            if label not in labels_shown:
                ax.broken_barh(
                    segments,
                    (obs_y_pos, bar_height),
                    facecolors=color,
                    label=label,
                )
                labels_shown.add(label)
            else:
                ax.broken_barh(
                    segments,
                    (obs_y_pos, bar_height),
                    facecolors=color,
                )

    # Extract and plot slews
    slew_segments = _extract_slews(ditl, t_start, offset_hours)
    if slew_segments:
        slew_y_pos = row_positions["Slewing"]
        ax.broken_barh(
            slew_segments,
            (slew_y_pos, bar_height),
            facecolor="tab:grey",
            label="Slew and Settle",
        )

    # Extract and plot charging mode observations
    segments = observations_by_type.get("Charging", [])
    charging_y_pos = row_positions["Charging"]
    charging_color = observation_categories.get_category_color("Charging")
    ax.broken_barh(
        segments,
        (charging_y_pos, bar_height),
        facecolors=charging_color,
        label="Battery Charging",
    )

    # Extract and plot safe mode
    safe_segments = _extract_safe_mode(ditl, t_start, offset_hours)
    if safe_segments:
        safe_y_pos = row_positions["Safe Mode"]
        ax.broken_barh(
            safe_segments,
            (safe_y_pos, bar_height),
            facecolor="tab:red",
            label="Safe Mode",
        )

    # Extract and plot SAA passages (if enabled)
    if show_saa:
        saa_segments = _extract_saa_passages(ditl, t_start, offset_hours)
        if saa_segments:
            saa_y_pos = row_positions["SAA"]
            ax.broken_barh(
                saa_segments,
                (saa_y_pos, bar_height),
                facecolor="tab:orange",
                label="SAA",
            )

    # Extract and plot eclipses
    eclipse_segments = _extract_eclipses(ditl, t_start, offset_hours)
    if eclipse_segments:
        eclipse_y_pos = row_positions["Eclipse"]
        ax.broken_barh(
            eclipse_segments,
            (eclipse_y_pos, bar_height),
            facecolor="black",
            label="Eclipse",
        )

    # Extract and plot ground station passes
    gs_segments = _extract_ground_passes(ditl, t_start, offset_hours)
    if gs_segments:
        gs_y_pos = row_positions["Ground Contact"]
        ax.broken_barh(
            gs_segments,
            (gs_y_pos, bar_height),
            facecolor="white",
            edgecolor="black",
            lw=0.5,
            label="Ground Contact",
        )

    # Set up axes with labels and tick positions
    # Y-ticks should be at the center of each bar for the grid lines to cut through
    y_labels = [row[0] for row in timeline_rows]
    y_ticks = [pos + bar_height / 2 for pos in data_y_positions]

    ax.set_yticks(y_ticks, labels=y_labels, fontname=font_family, fontsize=font_size)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    # Set x-axis limits and labels
    ax.set_xlim(-0.1, duration_hours + 0.1)
    x_ticks = range(0, int(duration_hours) + 1, max(1, int(duration_hours / 6)))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(
        [f"{t}" for t in x_ticks], fontname=font_family, fontsize=font_size
    )
    ax.set_xlabel("Hour", fontname=font_family, fontsize=font_size)

    # Add legend
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=4,
        fancybox=True,
        shadow=False,
        fontsize=config.legend_font_size if config else font_size,
        prop={"family": font_family},
    )

    # Save if requested
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def _extract_observations(
    ditl: QueueDITL | DITL,
    t_start: float,
    offset_hours: float,
    categories: ObservationCategories | None = None,
) -> dict[str, list[tuple[float, float]]]:
    """Extract observation segments grouped by type based on obsid.

    Parameters
    ----------
    ditl : QueueDITL or DITL
        The DITL simulation object.
    t_start : float
        Simulation start time in seconds.
    offset_hours : float
        Time offset in hours.
    categories : ObservationCategories, optional
        Configuration for observation categories. If None, uses default categories.
    """
    # Use provided categories or default
    if categories is None:
        categories = ObservationCategories.default_categories()

    # Initialize observation dict with all category names
    observations: dict[str, list[tuple[float, float]]] = {
        name: [] for name in categories.get_all_category_names()
    }

    for ppt in ditl.plan:
        # Calculate observation start and duration
        obs_start = (ppt.begin + ppt.slewtime - t_start) / 3600 - offset_hours
        obs_duration = (ppt.end - (ppt.begin + ppt.slewtime)) / 3600

        # Skip if no duration or negative duration
        if obs_duration <= 0:
            continue

        # Categorize by obsid using configuration
        category = categories.get_category(ppt.obsid)
        observations[category.name].append((obs_start, obs_duration))

    return observations


def _extract_slews(
    ditl: QueueDITL | DITL, t_start: float, offset_hours: float
) -> list[tuple[float, float]]:
    """Extract slew segments from plan."""
    slew_segments = []
    for ppt in ditl.plan:
        if ppt.slewtime > 0:
            slew_start = (ppt.begin - t_start) / 3600 - offset_hours
            slew_duration = ppt.slewtime / 3600
            slew_segments.append((slew_start, slew_duration))
    return slew_segments


def _extract_safe_mode(
    ditl: QueueDITL | DITL, t_start: float, offset_hours: float
) -> list[tuple[float, float]]:
    """Extract safe mode periods from mode timeline."""
    if not hasattr(ditl, "mode") or not hasattr(ditl, "utime"):
        return []

    safe_segments: list[tuple[float, float]] = []
    in_safe = False
    safe_start = 0.0

    for i, mode_val in enumerate(ditl.mode):
        # Check if in SAFE mode (mode value = 5)
        if isinstance(mode_val, ACSMode):
            is_safe = mode_val == ACSMode.SAFE
        else:
            is_safe = mode_val == ACSMode.SAFE.value

        time_hours = (ditl.utime[i] - t_start) / 3600 - offset_hours

        if is_safe and not in_safe:
            # Entering safe mode
            in_safe = True
            safe_start = time_hours
        elif not is_safe and in_safe:
            # Exiting safe mode
            in_safe = False
            safe_duration = time_hours - safe_start
            safe_segments.append((safe_start, safe_duration))

    # Handle safe mode extending to end of simulation
    if in_safe:
        safe_duration = (ditl.utime[-1] - t_start) / 3600 - offset_hours - safe_start
        safe_segments.append((safe_start, safe_duration))

    return safe_segments


def _extract_saa_passages(
    ditl: QueueDITL | DITL, t_start: float, offset_hours: float
) -> list[tuple[float, float]]:
    """Extract SAA passage times from mode timeline."""
    if not hasattr(ditl, "mode") or not hasattr(ditl, "utime"):
        return []

    saa_segments: list[tuple[float, float]] = []
    in_saa = False
    saa_start = 0.0

    for i, mode_val in enumerate(ditl.mode):
        # Check if in SAA mode
        if isinstance(mode_val, ACSMode):
            is_saa = mode_val == ACSMode.SAA
        else:
            is_saa = mode_val == ACSMode.SAA.value

        time_hours = (ditl.utime[i] - t_start) / 3600 - offset_hours

        if is_saa and not in_saa:
            # Entering SAA
            in_saa = True
            saa_start = time_hours
        elif not is_saa and in_saa:
            # Exiting SAA
            in_saa = False
            saa_duration = time_hours - saa_start
            saa_segments.append((saa_start, saa_duration))

    # Handle SAA extending to end of simulation
    if in_saa:
        saa_duration = (ditl.utime[-1] - t_start) / 3600 - offset_hours - saa_start
        saa_segments.append((saa_start, saa_duration))

    return saa_segments


def _extract_eclipses(
    ditl: QueueDITL | DITL, t_start: float, offset_hours: float
) -> list[tuple[float, float]]:
    """Extract eclipse periods from constraint or mode timeline."""
    eclipse_segments: list[tuple[float, float]] = []

    # Try to get eclipse info from the constraint if available
    if (
        hasattr(ditl, "constraint")
        and ditl.constraint is not None
        and hasattr(ditl, "utime")
    ):
        in_eclipse = False
        eclipse_start = 0.0

        for i, utime in enumerate(ditl.utime):
            time_hours = (utime - t_start) / 3600 - offset_hours

            # Check if in eclipse using constraint
            is_eclipsed = ditl.constraint.in_eclipse(ra=0, dec=0, time=utime)

            if is_eclipsed and not in_eclipse:
                # Entering eclipse
                in_eclipse = True
                eclipse_start = time_hours
            elif not is_eclipsed and in_eclipse:
                # Exiting eclipse
                in_eclipse = False
                eclipse_duration = time_hours - eclipse_start
                eclipse_segments.append((eclipse_start, eclipse_duration))

        # Handle eclipse extending to end of simulation
        if in_eclipse:
            eclipse_duration = (
                (ditl.utime[-1] - t_start) / 3600 - offset_hours - eclipse_start
            )
            eclipse_segments.append((eclipse_start, eclipse_duration))

    return eclipse_segments


def _extract_ground_passes(
    ditl: QueueDITL | DITL, t_start: float, offset_hours: float
) -> list[tuple[float, float]]:
    """Extract ground station pass times from ACS pass list."""
    if not hasattr(ditl, "acs") or ditl.acs is None:
        return []

    gs_segments = []

    # Check if ACS has pass requests (PassTimes object)
    if hasattr(ditl.acs, "passrequests") and ditl.acs.passrequests:
        pass_list = ditl.acs.passrequests
        # PassTimes object has a passes attribute with the list
        if hasattr(pass_list, "passes"):
            for gs_pass in pass_list.passes:
                if gs_pass.length is not None:
                    pass_start = (gs_pass.begin - t_start) / 3600 - offset_hours
                    pass_duration = gs_pass.length / 3600
                    gs_segments.append((pass_start, pass_duration))

    return gs_segments


def annotate_slew_distances(
    ax: Axes,
    ditl: QueueDITL | DITL,
    t_start: float,
    offset_hours: float,
    slew_indices: list[int],
    font_family: str = "Helvetica",
    font_size: int = 9,
) -> Axes:
    """Add annotations showing slew distances for specific slews.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to add annotations to.
    ditl : QueueDITL or DITL
        The DITL simulation object.
    t_start : float
        Simulation start time in Unix seconds.
    offset_hours : float
        Time offset in hours.
    slew_indices : list of int
        Indices in ditl.plan of slews to annotate.
    font_family : str, optional
        Font family for annotation text.
    font_size : int, optional
        Font size for annotation text.
    """
    connectionstyle = "angle,angleA=0,angleB=90,rad=0"

    for idx in slew_indices:
        if idx < len(ditl.plan):
            ppt = ditl.plan[idx]
            if ppt.slewtime > 0 and hasattr(ppt, "slewdist"):
                slew_start = (ppt.begin - t_start) / 3600 - offset_hours

                # Add arrow annotation
                ax.annotate(
                    "",
                    (slew_start, 0.25),
                    xycoords="data",
                    xytext=(slew_start - 0.75, 0.14),
                    textcoords="data",
                    arrowprops=dict(
                        arrowstyle="->",
                        color="blue",
                        shrinkA=5,
                        shrinkB=5,
                        patchA=None,
                        patchB=None,
                        connectionstyle=connectionstyle,
                    ),
                )

                # Add distance text
                ax.text(
                    slew_start - 0.55,
                    0.14,
                    f"{ppt.slewdist:.0f}°",
                    ha="right",
                    va="center",
                    fontsize=font_size,
                    fontname=font_family,
                )

    return ax
