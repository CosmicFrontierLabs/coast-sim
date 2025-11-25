"""Sky Pointing Visualization

Interactive visualization showing spacecraft pointing on a mollweide projection
of the sky with scheduled observations and constraint regions.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.widgets import Button, Slider

from ..common import dtutcfromtimestamp
from ..config.visualization import VisualizationConfig


def plot_sky_pointing(
    ditl,
    figsize=(14, 8),
    n_grid_points=100,
    show_controls=True,
    time_step_seconds=None,
    constraint_alpha=0.3,
    config=None,
):
    """Plot spacecraft pointing on a mollweide sky map with constraints.

    Creates an interactive visualization showing:
    - All scheduled observations as markers
    - Current spacecraft pointing direction
    - Sun, Moon, and Earth constraint regions (shaded)
    - Time controls to step through or play the DITL

    Parameters
    ----------
    ditl : DITL or QueueDITL
        The DITL simulation object with completed simulation data.
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (14, 8)).
    n_grid_points : int, optional
        Number of grid points per axis for constraint region calculation
        (default: 100). Higher values give smoother regions but slower rendering.
    show_controls : bool, optional
        Whether to show interactive time controls (default: True).
    time_step_seconds : float, optional
        Time step in seconds for controls. If None, uses ditl.step_size (default: None).
    constraint_alpha : float, optional
        Alpha transparency for constraint regions (default: 0.3).
    config : Config, optional
        Configuration object containing visualization settings. If None, uses ditl.config.visualization if available.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axes containing the sky map.
        if not isinstance(config, VisualizationConfig):
            if hasattr(ditl, 'config') and hasattr(ditl.config, 'visualization') and isinstance(ditl.config.visualization, VisualizationConfig):
                config = ditl.config.visualization
            else:
                config = VisualizationConfig()
    >>> from conops import DITL
    >>> ditl = DITL(config)
    >>> ditl.calc()
    >>> fig, ax, ctrl = plot_sky_pointing(ditl)
    >>> plt.show()

    Notes
    -----
    - RA coordinates are in degrees (0-360)
    - Dec coordinates are in degrees (-90 to 90)
    - Constraint regions are computed at each time step based on Sun, Moon,
      and Earth positions from the ephemeris
    """
    # Validate inputs
    if not hasattr(ditl, "plan") or len(ditl.plan) == 0:
        raise ValueError("DITL simulation has no pointings. Run calc() first.")
    if not hasattr(ditl, "utime") or len(ditl.utime) == 0:
        raise ValueError("DITL has no time data. Run calc() first.")
    if ditl.constraint.ephem is None:
        raise ValueError("DITL constraint has no ephemeris set.")

    # Set default time step
    if time_step_seconds is None:
        time_step_seconds = ditl.step_size

    # Get visualization config
    if config is None:
        if (
            hasattr(ditl, "config")
            and hasattr(ditl.config, "visualization")
            and isinstance(ditl.config.visualization, VisualizationConfig)
        ):
            config = ditl.config.visualization
        else:
            config = VisualizationConfig()

    # Create the visualization
    if show_controls:
        fig = plt.figure(figsize=figsize)
        # Main plot takes most of the space
        ax = fig.add_subplot(111, projection="mollweide")
        # Leave space at bottom for controls
        fig.subplots_adjust(bottom=0.25)
    else:
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "mollweide"})

    # Create the controller
    controller = SkyPointingController(
        ditl=ditl,
        fig=fig,
        ax=ax,
        n_grid_points=n_grid_points,
        time_step_seconds=time_step_seconds,
        constraint_alpha=constraint_alpha,
        config=config,
    )

    # Initial plot
    controller.update_plot(ditl.utime[0])

    # Add interactive controls if requested
    if show_controls:
        controller.add_controls()

    return fig, ax, controller if show_controls else None


class SkyPointingController:
    """Controller for interactive sky pointing visualization."""

    def __init__(
        self,
        ditl,
        fig,
        ax,
        n_grid_points=100,
        time_step_seconds=60,
        constraint_alpha=0.3,
        config=None,
    ):
        """Initialize the controller.

        Parameters
        ----------
        ditl : DITL or QueueDITL
            The DITL simulation object.
        fig : matplotlib.figure.Figure
            The figure to draw on.
        ax : matplotlib.axes.Axes
            The axes with mollweide projection.
        n_grid_points : int
            Number of grid points for constraint calculation.
        time_step_seconds : float
            Time step for controls in seconds.
        constraint_alpha : float
            Alpha transparency for constraint regions.
        config : VisualizationConfig, optional
            Visualization configuration settings.
        """
        self.ditl = ditl
        self.fig = fig
        self.ax = ax
        self.n_grid_points = n_grid_points
        self.time_step_seconds = time_step_seconds
        self.constraint_alpha = constraint_alpha
        self.config = config

        # State
        self.current_time_idx = 0
        self.playing = False
        self.timer = None

        # Plot elements (will be created in update_plot)
        self.constraint_patches = {}
        self.current_pointing_marker = None
        self.scheduled_obs_scatter = None
        self.title_text = None

        # Control widgets
        self.slider = None
        self.play_button = None
        self.prev_button = None
        self.next_button = None

    def add_controls(self):
        """Add interactive control widgets to the figure."""
        # Create axes for controls
        ax_slider = plt.axes([0.2, 0.15, 0.6, 0.03])
        ax_play = plt.axes([0.42, 0.05, 0.08, 0.04])
        ax_prev = plt.axes([0.32, 0.05, 0.08, 0.04])
        ax_next = plt.axes([0.52, 0.05, 0.08, 0.04])

        # Time slider
        self.slider = Slider(
            ax_slider,
            "Time",
            0,
            len(self.ditl.utime) - 1,
            valinit=0,
            valstep=1,
            valfmt="%d",
        )
        self.slider.on_changed(self.on_slider_change)

        # Buttons
        self.play_button = Button(ax_play, "Play")
        self.play_button.on_clicked(self.on_play_clicked)

        self.prev_button = Button(ax_prev, "< Prev")
        self.prev_button.on_clicked(self.on_prev_clicked)

        self.next_button = Button(ax_next, "Next >")
        self.next_button.on_clicked(self.on_next_clicked)

    def on_slider_change(self, val):
        """Handle slider value change."""
        idx = int(val)
        if idx != self.current_time_idx:
            self.current_time_idx = idx
            self.update_plot(self.ditl.utime[idx])

    def on_play_clicked(self, event):
        """Handle play button click."""
        if self.playing:
            self.stop_animation()
        else:
            self.start_animation()

    def on_prev_clicked(self, event):
        """Handle previous button click."""
        if self.current_time_idx > 0:
            self.current_time_idx -= 1
            self.slider.set_val(self.current_time_idx)
            self.update_plot(self.ditl.utime[self.current_time_idx])

    def on_next_clicked(self, event):
        """Handle next button click."""
        if self.current_time_idx < len(self.ditl.utime) - 1:
            self.current_time_idx += 1
            self.slider.set_val(self.current_time_idx)
            self.update_plot(self.ditl.utime[self.current_time_idx])

    def start_animation(self):
        """Start playing through time steps."""
        self.playing = True
        self.play_button.label.set_text("Pause")
        # Use matplotlib's animation timer
        self.timer = self.fig.canvas.new_timer(interval=100)  # 100ms between frames
        self.timer.add_callback(self.animation_step)
        self.timer.start()

    def stop_animation(self):
        """Stop playing animation."""
        self.playing = False
        if self.play_button is not None:
            self.play_button.label.set_text("Play")
        if self.timer is not None:
            self.timer.stop()
            self.timer = None

    def animation_step(self):
        """Advance to next time step during animation."""
        if self.current_time_idx < len(self.ditl.utime) - 1:
            self.current_time_idx += 1
            self.slider.set_val(self.current_time_idx)
            self.update_plot(self.ditl.utime[self.current_time_idx])
        else:
            # Reached the end
            self.stop_animation()

    def update_plot(self, utime):
        """Update the plot for a given time.

        Parameters
        ----------
        utime : float
            Unix timestamp to display.
        """
        self.ax.clear()

        # Get current spacecraft pointing
        idx = self._find_time_index(utime)
        current_ra = self.ditl.ra[idx]
        current_dec = self.ditl.dec[idx]
        current_mode = self.ditl.mode[idx]

        # Plot scheduled observations
        self._plot_scheduled_observations()

        # Plot constraint regions
        self._plot_constraint_regions(utime)

        # Plot Earth physical disk
        self._plot_earth_disk(utime)

        # Plot current pointing
        self._plot_current_pointing(current_ra, current_dec, current_mode)

        # Set up the plot
        self._setup_plot_appearance(utime)

        # Redraw
        self.fig.canvas.draw_idle()

    def _find_time_index(self, utime):
        """Find the index in utime array closest to the given time."""
        return np.argmin(np.abs(np.array(self.ditl.utime) - utime))

    def _plot_scheduled_observations(self):
        """Plot all scheduled observations as markers."""
        if len(self.ditl.plan) == 0:
            return

        ras = []
        decs = []
        colors = []
        sizes = []

        for ppt in self.ditl.plan:
            ra = ppt.ra
            dec = ppt.dec

            # Convert RA from 0-360 to -180 to 180 for mollweide, with RA=0 on left
            ra_plot = ra - 180

            ras.append(np.deg2rad(ra_plot))
            decs.append(np.deg2rad(dec))

            # Color by observation type or ID
            if hasattr(ppt, "obsid"):
                # Use obsid to determine color
                if ppt.obsid >= 1000000:
                    colors.append("red")  # TOO/GRB
                    sizes.append(100)
                elif ppt.obsid >= 20000:
                    colors.append("orange")  # High priority
                    sizes.append(60)
                elif ppt.obsid >= 10000:
                    colors.append("yellow")  # Survey
                    sizes.append(40)
                else:
                    colors.append("lightblue")  # Standard
                    sizes.append(40)
            else:
                colors.append("lightblue")
                sizes.append(40)

        # Scatter plot
        self.ax.scatter(
            ras,
            decs,
            s=sizes,
            c=colors,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
            label="Targets",
            zorder=2,
        )

    def _plot_constraint_regions(self, utime):
        """Plot constraint regions for Sun, Moon, and Earth.

        Parameters
        ----------
        utime : float
            Unix timestamp for constraint calculation.
        """
        dt = dtutcfromtimestamp(utime)
        ephem = self.ditl.constraint.ephem
        idx = ephem.index(dt)

        # Get celestial body positions
        sun_ra = ephem.sun[idx].ra.deg
        sun_dec = ephem.sun[idx].dec.deg
        moon_ra = ephem.moon[idx].ra.deg
        moon_dec = ephem.moon[idx].dec.deg
        earth_ra = ephem.earth[idx].ra.deg
        earth_dec = ephem.earth[idx].dec.deg

        # Create a grid of RA/Dec points
        ra_grid = np.linspace(0, 360, self.n_grid_points)
        dec_grid = np.linspace(-90, 90, self.n_grid_points)

        # Check each constraint type and plot regions
        constraint_types = [
            ("Sun", self.ditl.constraint.in_sun, "yellow", sun_ra, sun_dec),
            ("Moon", self.ditl.constraint.in_moon, "gray", moon_ra, moon_dec),
            ("Earth", self.ditl.constraint.in_earth, "blue", earth_ra, earth_dec),
            (
                "Anti-Sun",
                self.ditl.constraint.in_anti_sun,
                "orange",
                (sun_ra + 180) % 360,
                -sun_dec,
            ),
            ("Panel", self.ditl.constraint.in_panel, "green", None, None),
        ]

        for name, constraint_func, color, body_ra, body_dec in constraint_types:
            self._plot_single_constraint(
                name,
                constraint_func,
                color,
                utime,
                ra_grid,
                dec_grid,
                body_ra,
                body_dec,
            )

    def _plot_earth_disk(self, utime):
        """Plot the physical extent of Earth as seen from the spacecraft.

        Parameters
        ----------
        utime : float
            Unix timestamp for Earth position calculation.
        """
        dt = dtutcfromtimestamp(utime)
        ephem = self.ditl.constraint.ephem
        idx = ephem.index(dt)

        # Get Earth position and angular radius
        earth_ra = ephem.earth[idx].ra.deg
        earth_dec = ephem.earth[idx].dec.deg
        earth_angular_radius = ephem.earth_radius_deg[idx]

        # Use the same grid sampling approach as constraints
        # Linear declination sampling
        dec_samples = np.linspace(-90, 90, self.n_grid_points)

        # For each declination, calculate how many RA samples we need
        # based on cos(dec) for even visual density in Mollweide projection
        cos_factors = np.cos(np.radians(dec_samples))
        n_ra_array = np.maximum(8, (self.n_grid_points * 2 * cos_factors).astype(int))

        ra_flat = np.concatenate(
            [np.linspace(0, 360, n, endpoint=False) for n in n_ra_array]
        )
        dec_flat = np.concatenate(
            [np.full(n, dec) for n, dec in zip(n_ra_array, dec_samples)]
        )

        # Check which points are inside the Earth disk
        earth_disk_points = []
        for ra, dec in zip(ra_flat, dec_flat):
            # Calculate angular distance from Earth center
            # Using spherical distance formula
            delta_ra = np.radians(ra - earth_ra)
            dec_rad = np.radians(dec)
            earth_dec_rad = np.radians(earth_dec)

            angular_dist = np.degrees(
                np.arccos(
                    np.sin(earth_dec_rad) * np.sin(dec_rad)
                    + np.cos(earth_dec_rad) * np.cos(dec_rad) * np.cos(delta_ra)
                )
            )

            if angular_dist <= earth_angular_radius:
                earth_disk_points.append((ra, dec))

        # Plot Earth disk points
        if earth_disk_points:
            points = np.array(earth_disk_points)
            ra_vals = points[:, 0]
            dec_vals = points[:, 1]

            # Convert so RA=0 appears on the left
            ra_plot = ra_vals - 180

            self.ax.scatter(
                np.deg2rad(ra_plot),
                np.deg2rad(dec_vals),
                s=20,
                c="darkblue",
                alpha=0.8,
                marker="s",
                edgecolors="none",
                label="Earth Disk",
                zorder=2.5,
            )

    def _plot_single_constraint(
        self, name, constraint_func, color, utime, ra_grid, dec_grid, body_ra, body_dec
    ):
        """Plot a single constraint region.

        Parameters
        ----------
        name : str
            Name of the constraint.
        constraint_func : callable
            Function to check if a point violates the constraint.
        color : str
            Color for the constraint region.
        utime : float
            Unix timestamp.
        ra_grid : array
            RA values to check.
        dec_grid : array
            Dec values to check.
        body_ra : float or None
            RA of celestial body (for marker).
        body_dec : float or None
            Dec of celestial body (for marker).
        """
        # Sample uniformly in Mollweide projection space for even point density
        # The Mollweide projection compresses RA (longitude) near the poles
        # We need fewer RA samples at high declinations to maintain even visual density

        # Linear declination sampling is fine
        dec_samples = np.linspace(-90, 90, self.n_grid_points)

        # For each declination, calculate how many RA samples we need
        # based on cos(dec) - this accounts for the convergence of longitude lines
        # Create the sky grid points, sampling RA density by cos(dec)
        cos_factors = np.cos(np.radians(dec_samples))
        n_ra_array = np.maximum(8, (self.n_grid_points * 2 * cos_factors).astype(int))

        ra_flat = np.concatenate(
            [np.linspace(0, 360, n, endpoint=False) for n in n_ra_array]
        )
        dec_flat = np.concatenate(
            [np.full(n, dec) for n, dec in zip(n_ra_array, dec_samples)]
        )

        constrained_points = []
        for ra, dec in zip(ra_flat, dec_flat):
            try:
                if constraint_func(ra, dec, utime):
                    constrained_points.append((ra, dec))
            except Exception:
                continue

        # Plot constrained region, handling RA wrapping at boundaries
        if constrained_points:
            points = np.array(constrained_points)
            ra_vals = points[:, 0]
            dec_vals = points[:, 1]

            # For Mollweide projection: RA range is -180 to 180
            # Convert so RA=0 appears on the left
            ra_plot = ra_vals - 180

            # Plot main points
            self.ax.scatter(
                np.deg2rad(ra_plot),
                np.deg2rad(dec_vals),
                s=20,
                c=color,
                alpha=self.constraint_alpha,
                marker="s",
                zorder=1,
                edgecolors="none",
                label=f"{name} Cons.",
            )

        # Mark celestial body position
        if body_ra is not None and body_dec is not None:
            ra_plot = body_ra - 180
            self.ax.plot(
                np.deg2rad(ra_plot),
                np.deg2rad(body_dec),
                marker="o",
                markersize=12,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=2,
                label=name,
                zorder=3,
            )

    def _plot_current_pointing(self, ra, dec, mode):
        """Plot the current spacecraft pointing direction.

        Parameters
        ----------
        ra : float
            Right ascension in degrees.
        dec : float
            Declination in degrees.
        mode : ACSMode
            Current ACS mode.
        """
        # Convert RA for plotting (RA=0 on left)
        ra_plot = ra - 180

        # Color based on ACS mode
        mode_name = mode.name if hasattr(mode, "name") else str(mode)
        mode_colors = (
            self.config.mode_colors
            if self.config
            else {
                "SCIENCE": "green",
                "SLEWING": "orange",
                "SAA": "purple",
                "PASS": "cyan",
                "CHARGING": "yellow",
                "SAFE": "red",
            }
        )
        color = mode_colors.get(mode_name, "red")

        # Plot with distinctive marker
        self.ax.plot(
            np.deg2rad(ra_plot),
            np.deg2rad(dec),
            marker="*",
            markersize=25,
            markerfacecolor=color,
            markeredgecolor="white",
            markeredgewidth=2,
            label="Pointing",
            zorder=5,
        )

        # Add a small circle around it for visibility
        circle = plt.Circle(
            (np.deg2rad(ra_plot), np.deg2rad(dec)),
            radius=np.deg2rad(5),
            fill=False,
            edgecolor=color,
            linewidth=2,
            zorder=4,
        )
        self.ax.add_patch(circle)

    def _setup_plot_appearance(self, utime):
        """Set up plot labels, grid, and appearance.

        Parameters
        ----------
        utime : float
            Unix timestamp for the title.
        """
        # Grid
        self.ax.grid(True, alpha=0.3)

        # Labels
        font_family = self.config.font_family if self.config else "Helvetica"
        label_font_size = self.config.label_font_size if self.config else 10
        title_font_size = self.config.title_font_size if self.config else 12
        title_prop = FontProperties(
            family=font_family, size=title_font_size, weight="bold"
        )
        tick_font_size = self.config.tick_font_size if self.config else 9

        self.ax.set_xlabel(
            "Right Ascension (deg)", fontsize=label_font_size, fontfamily=font_family
        )
        self.ax.set_ylabel(
            "Declination (deg)", fontsize=label_font_size, fontfamily=font_family
        )

        # Title with time
        dt = dtutcfromtimestamp(utime)
        time_str = dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        self.ax.set_title(
            f"Spacecraft Pointing at {time_str}", fontproperties=title_prop, pad=20
        )

        # Legend (reduce clutter by only showing unique labels)
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))

        # Add ACS mode color legend entries
        from matplotlib.lines import Line2D

        mode_colors = (
            self.config.mode_colors
            if self.config
            else {
                "SCIENCE": "green",
                "SLEWING": "orange",
                "SAA": "purple",
                "PASS": "cyan",
                "CHARGING": "yellow",
                "SAFE": "red",
            }
        )

        # Add separator and ACS mode entries
        mode_handles = [
            Line2D(
                [0],
                [0],
                marker="*",
                color="w",
                markerfacecolor=color,
                markersize=10,
                markeredgecolor="white",
                markeredgewidth=1,
                label=f"{mode.lower().capitalize()}",
            )
            for mode, color in mode_colors.items()
        ]

        all_handles = list(by_label.values()) + mode_handles
        all_labels = list(by_label.keys()) + [h.get_label() for h in mode_handles]

        self.ax.legend(
            all_handles,
            all_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=self.config.legend_font_size if self.config else 8,
            title="ACS Modes",
            prop={"family": font_family},
        )

        # Set RA tick labels (mollweide uses radians internally)
        # RA=0 is on the left at -180°
        ra_ticks = np.deg2rad(np.array([-180, -120, -60, 0, 60, 120, 180]))
        self.ax.set_xticks(ra_ticks)
        self.ax.set_xticklabels(
            ["0°", "60°", "120°", "180°", "240°", "300°", "360°"],
            fontsize=tick_font_size,
            fontfamily=font_family,
        )
        # Set y-axis tick labels
        self.ax.tick_params(axis="y", labelsize=tick_font_size)
        for label in self.ax.get_yticklabels():
            try:
                label.set_fontfamily(font_family)
            except Exception:
                # Some environments return MagicMock objects during tests; ignore
                pass


def save_sky_pointing_frames(
    ditl, output_dir, figsize=(14, 8), n_grid_points=50, frame_interval=1, config=None
):
    """Save individual frames of the sky pointing visualization.

    Useful for creating animations or reviewing specific time steps.

    Parameters
    ----------
    ditl : DITL or QueueDITL
        The DITL simulation object.
    output_dir : str
        Directory to save frames.
    figsize : tuple
        Figure size.
    n_grid_points : int
        Grid resolution for constraints.
    frame_interval : int
        Save every Nth time step (default: 1 = save all).
    config : Config, optional
        Configuration object containing visualization settings.

    Returns
    -------
    list
        List of saved file paths.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Create controller without controls
    fig, ax, _ = plot_sky_pointing(
        ditl,
        figsize=figsize,
        n_grid_points=n_grid_points,
        show_controls=False,
        config=config,
    )

    controller = SkyPointingController(
        ditl=ditl,
        fig=fig,
        ax=ax,
        n_grid_points=n_grid_points,
        time_step_seconds=ditl.step_size,
        constraint_alpha=0.3,
        config=config,
    )

    saved_files = []
    for idx in range(0, len(ditl.utime), frame_interval):
        utime = ditl.utime[idx]
        controller.update_plot(utime)

        # Save frame
        filename = os.path.join(output_dir, f"sky_pointing_frame_{idx:05d}.png")
        fig.savefig(filename, dpi=150, bbox_inches="tight")
        saved_files.append(filename)

        if (idx + 1) % 100 == 0:
            print(f"Saved {idx + 1}/{len(ditl.utime)} frames")

    plt.close(fig)
    print(f"Saved {len(saved_files)} frames to {output_dir}")
    return saved_files
