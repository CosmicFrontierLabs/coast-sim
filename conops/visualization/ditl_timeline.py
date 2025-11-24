"""DITL Timeline Visualization

Provides functions to create timeline plots similar to proposal figures,
showing spacecraft operations including observations, slews, SAA passages,
eclipses, and ground station passes.
"""

import matplotlib.pyplot as plt

from ..common import ACSMode


def plot_ditl_timeline(
    ditl,
    offset_hours=0,
    figsize=(10, 6),
    orbit_period=5762.0,
    show_orbit_numbers=True,
    save_path=None,
    font_family="Helvetica",
    font_size=11,
):
    """Plot a DITL timeline showing spacecraft operations.

    Creates a comprehensive timeline visualization showing:
    - Orbit numbers (optional)
    - Science observations (color-coded by obsid range)
    - Slews and settling time
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
    save_path : str, optional
        If provided, save the figure to this path (default: None).
    font_family : str, optional
        Font family to use for text (default: 'Helvetica').
    font_size : int, optional
        Base font size for labels (default: 11).

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
    hfont = {"fontname": font_family, "fontsize": font_size}

    # Extract simulation start time
    if not ditl.ppst or len(ditl.ppst) == 0:
        raise ValueError("DITL simulation has no pointings. Run calc() first.")

    t_start = ditl.ppst[0].begin

    # Create figure
    fig = plt.figure(figsize=figsize)
    ax = plt.axes([0.12, 0.3, 0.8, 0.5], frameon=True)

    # Calculate timeline duration in hours
    if hasattr(ditl, "utime") and len(ditl.utime) > 0:
        duration_hours = (ditl.utime[-1] - ditl.utime[0]) / 3600.0
    else:
        duration_hours = 24.0

    # Draw orbit numbers if requested
    if show_orbit_numbers:
        num_orbits = int(duration_hours * 3600 / orbit_period) + 1
        for i in range(num_orbits):
            if i % 2 == 1:
                barcol = "grey"
            else:
                barcol = "white"
            orbit_start = i * orbit_period / 3600
            orbit_width = orbit_period / 3600
            ax.broken_barh(
                [[orbit_start, orbit_width]],
                (0.6, 0.15),
                facecolors=barcol,
                edgecolor="black",
                lw=1,
                linestyle="-",
            )

            ax.text(
                (i + 0.5) * orbit_period / 3600,
                0.675,
                f"{i + 1}",
                horizontalalignment="center",
                verticalalignment="center",
                fontname=font_family,
                fontsize=font_size - 2,
                zorder=2,
            )

    # Extract observation segments from ppst by obsid ranges
    observations_by_type = _extract_observations(ditl, t_start, offset_hours)

    # Plot observations by type
    colors = {
        "GO": "tab:green",  # 20000-30000
        "Survey": "tab:blue",  # 10000-20000
        "GRB": "tab:orange",  # 1000000-2000000
        "TOO": "tab:red",  # 30000-40000
        "Calibration": "purple",  # others
        "Charging": "yellow",  # 90000-100000
    }

    labels_shown = set()
    for obs_type, segments in observations_by_type.items():
        if segments and obs_type != "Charging":
            label = f"{obs_type} Target" if obs_type != "Calibration" else obs_type
            if label not in labels_shown:
                ax.broken_barh(
                    segments,
                    (0.5, 0.15),
                    facecolors=colors.get(obs_type, "gray"),
                    label=label,
                )
                labels_shown.add(label)
            else:
                ax.broken_barh(
                    segments, (0.5, 0.15), facecolors=colors.get(obs_type, "gray")
                )

    # Extract and plot slews
    slew_segments = _extract_slews(ditl, t_start, offset_hours)
    if slew_segments:
        ax.broken_barh(
            slew_segments, (0.25, 0.15), facecolor="tab:grey", label="Slew and Settle"
        )

    # Extract and plot charging mode
    charging_segments = _extract_charging_mode(ditl, t_start, offset_hours)
    if charging_segments:
        ax.broken_barh(
            charging_segments, (0.1, 0.15), facecolor="gold", label="Battery Charging"
        )

    # Extract and plot SAA passages
    saa_segments = _extract_saa_passages(ditl, t_start, offset_hours)
    if saa_segments:
        ax.broken_barh(saa_segments, (-0.1, 0.15), facecolor="tab:red", label="SAA")

    # Extract and plot eclipses
    eclipse_segments = _extract_eclipses(ditl, t_start, offset_hours)
    if eclipse_segments:
        ax.broken_barh(
            eclipse_segments, (-0.3, 0.15), facecolor="black", label="Eclipse"
        )

    # Extract and plot ground station passes
    gs_segments = _extract_ground_passes(ditl, t_start, offset_hours)
    if gs_segments:
        ax.broken_barh(
            gs_segments,
            (-0.5, 0.15),
            facecolor="white",
            edgecolor="black",
            lw=0.5,
            label="Ground Contact",
        )

    # Set up axes
    y_labels = [
        "Observations",
        "Slewing",
        "Charging",
        "SAA",
        "Eclipse",
        "Ground Contact",
    ]
    y_ticks = [0.575, 0.325, 0.175, -0.025, -0.225, -0.425]
    if show_orbit_numbers:
        y_labels.insert(0, "Orbit")
        y_ticks.insert(0, 0.7)

    ax.set_yticks(y_ticks, labels=y_labels, **hfont)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)

    # Set x-axis limits and labels
    ax.set_xlim(-0.1, duration_hours + 0.1)
    x_ticks = range(0, int(duration_hours) + 1, max(1, int(duration_hours / 6)))
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([f"{t}" for t in x_ticks], **hfont)
    ax.set_xlabel("Hour", **hfont)

    # Add legend
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.2),
        ncol=4,
        fancybox=True,
        shadow=False,
        fontsize=font_size,
        prop={"family": font_family},
    )

    # Save if requested
    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig, ax


def _extract_observations(ditl, t_start, offset_hours):
    """Extract observation segments grouped by type based on obsid."""
    observations = {
        "GO": [],  # 20000-30000
        "Survey": [],  # 10000-20000
        "GRB": [],  # 1000000-2000000
        "TOO": [],  # 30000-40000
        "Calibration": [],
        "Charging": [],  # 90000-100000
    }

    for i, ppt in enumerate(ditl.ppst):
        # Calculate observation start
        obs_start = (ppt.begin + ppt.slewtime - t_start) / 3600 - offset_hours

        # Calculate observation end - need to handle unrealistic end times
        # If end time is way in the future (more than 1 day ahead of begin),
        # use the next PPT's begin time or simulation end
        max_reasonable_duration = 86400  # 1 day in seconds
        if ppt.end and (ppt.end - ppt.begin) < max_reasonable_duration:
            # Normal end time
            obs_end_time = ppt.end
        else:
            # Use next PPT's begin time or a reasonable default
            if i + 1 < len(ditl.ppst):
                obs_end_time = ditl.ppst[i + 1].begin
            else:
                # Last observation - use simulation end
                obs_end_time = (
                    ditl.utime[-1]
                    if hasattr(ditl, "utime") and ditl.utime
                    else ppt.begin + 3600
                )

        obs_duration = (obs_end_time - (ppt.begin + ppt.slewtime)) / 3600

        # Skip if no duration or negative duration
        if obs_duration <= 0:
            continue

        # Categorize by obsid
        if 20000 <= ppt.obsid < 30000:
            observations["GO"].append((obs_start, obs_duration))
        elif 10000 <= ppt.obsid < 20000:
            observations["Survey"].append((obs_start, obs_duration))
        elif 1000000 <= ppt.obsid < 2000000:
            observations["GRB"].append((obs_start, obs_duration))
        elif 30000 <= ppt.obsid < 40000:
            observations["TOO"].append((obs_start, obs_duration))
        elif 90000 <= ppt.obsid < 100000:
            observations["Charging"].append((obs_start, obs_duration))
        else:
            observations["Calibration"].append((obs_start, obs_duration))

    return observations


def _extract_slews(ditl, t_start, offset_hours):
    """Extract slew segments from ppst."""
    slew_segments = []
    for ppt in ditl.ppst:
        if ppt.slewtime > 0:
            slew_start = (ppt.begin - t_start) / 3600 - offset_hours
            slew_duration = ppt.slewtime / 3600
            slew_segments.append((slew_start, slew_duration))
    return slew_segments


def _extract_charging_mode(ditl, t_start, offset_hours):
    """Extract battery charging periods from mode timeline."""
    if not hasattr(ditl, "mode") or not hasattr(ditl, "utime"):
        return []

    charging_segments = []
    in_charging = False
    charging_start = 0

    for i, mode_val in enumerate(ditl.mode):
        # Check if in CHARGING mode (mode value = 2)
        if isinstance(mode_val, ACSMode):
            is_charging = mode_val == ACSMode.CHARGING
        else:
            is_charging = mode_val == ACSMode.CHARGING.value

        time_hours = (ditl.utime[i] - t_start) / 3600 - offset_hours

        if is_charging and not in_charging:
            # Entering charging mode
            in_charging = True
            charging_start = time_hours
        elif not is_charging and in_charging:
            # Exiting charging mode
            in_charging = False
            charging_duration = time_hours - charging_start
            charging_segments.append((charging_start, charging_duration))

    # Handle charging extending to end of simulation
    if in_charging:
        charging_duration = (
            (ditl.utime[-1] - t_start) / 3600 - offset_hours - charging_start
        )
        charging_segments.append((charging_start, charging_duration))

    return charging_segments


def _extract_saa_passages(ditl, t_start, offset_hours):
    """Extract SAA passage times from mode timeline."""
    if not hasattr(ditl, "mode") or not hasattr(ditl, "utime"):
        return []

    saa_segments = []
    in_saa = False
    saa_start = 0

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


def _extract_eclipses(ditl, t_start, offset_hours):
    """Extract eclipse periods from constraint or mode timeline."""
    eclipse_segments = []

    # Try to get eclipse info from the constraint if available
    if (
        hasattr(ditl, "constraint")
        and ditl.constraint is not None
        and hasattr(ditl, "utime")
    ):
        in_eclipse = False
        eclipse_start = 0

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


def _extract_ground_passes(ditl, t_start, offset_hours):
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
    ax, ditl, t_start, offset_hours, slew_indices, font_family="Helvetica", font_size=9
):
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
        Indices in ditl.ppst of slews to annotate.
    font_family : str, optional
        Font family for annotation text.
    font_size : int, optional
        Font size for annotation text.
    """
    connectionstyle = "angle,angleA=0,angleB=90,rad=0"

    for idx in slew_indices:
        if idx < len(ditl.ppst):
            ppt = ditl.ppst[idx]
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
