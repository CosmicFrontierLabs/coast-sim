"""Sky Pointing Visualization

Interactive visualization showing spacecraft pointing on a mollweide projection
of the sky with scheduled observations and constraint regions.
"""

import os
from typing import TYPE_CHECKING, Any, Optional

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import rust_ephem
import rust_ephem.constraints
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.font_manager import FontProperties
from matplotlib.patches import Circle
from matplotlib.widgets import Button, Slider

from ..common import dtutcfromtimestamp
from ..config.observation_categories import ObservationCategories
from ..config.visualization import VisualizationConfig

if TYPE_CHECKING:
    from ..ditl import DITL, QueueDITL


def _get_visualization_config(
    ditl: "DITL | QueueDITL", config: VisualizationConfig | None = None
) -> VisualizationConfig:
    """Get visualization configuration, with fallback to defaults.

    Parameters
    ----------
    ditl : DITL or QueueDITL
        The DITL simulation object.
    config : VisualizationConfig, optional
        Explicit config to use. If None, tries to get from ditl.config.visualization.

    Returns
    -------
    VisualizationConfig
        The configuration object to use.
    """
    if config is None:
        if (
            hasattr(ditl, "config")
            and hasattr(ditl.config, "visualization")
            and isinstance(ditl.config.visualization, VisualizationConfig)
        ):
            config = ditl.config.visualization
        else:
            config = VisualizationConfig()
    return config


def _lighten_color(color: str, factor: float = 0.5) -> str:
    """Lighten a color by blending with white.

    Parameters
    ----------
    color : str
        The color to lighten (hex, name, or RGB tuple).
    factor : float
        Lightening factor (0.0 = original color, 1.0 = white).

    Returns
    -------
    str
        The lightened color as hex string.
    """
    # Convert to RGB
    rgb = mcolors.to_rgb(color)
    # Blend with white
    r, g, b = rgb
    lightened = (
        (1 - factor) * r + factor * 1.0,
        (1 - factor) * g + factor * 1.0,
        (1 - factor) * b + factor * 1.0,
    )
    return mcolors.to_hex(lightened)


def plot_sky_pointing(
    ditl: "DITL | QueueDITL",
    figsize: tuple[float, float] = (14, 8),
    n_grid_points: int = 100,
    show_controls: bool = True,
    time_step_seconds: float | None = None,
    constraint_alpha: float = 0.3,
    config: VisualizationConfig | None = None,
    observation_categories: ObservationCategories | None = None,
) -> tuple[Figure, Axes, Optional["SkyPointingController"]]:
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
    config : MissionConfig, optional
        Configuration object containing visualization settings. If None, uses ditl.config.visualization if available.
    observation_categories : ObservationCategories, optional
        Configuration for observation category colors. If None, uses ditl.config.observation_categories if available, otherwise defaults.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    ax : matplotlib.axes.Axes
        The axes containing the sky map.
    controller : SkyPointingController or None
        The controller object if show_controls=True, otherwise None.

    Examples
    --------
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
    config = _get_visualization_config(ditl, config)

    # Get observation categories
    if observation_categories is None:
        if hasattr(ditl, "config") and hasattr(ditl.config, "observation_categories"):
            observation_categories = ditl.config.observation_categories
    if observation_categories is None:
        observation_categories = ObservationCategories.default_categories()

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
        observation_categories=observation_categories,
    )

    # Initial plot
    controller.update_plot(ditl.utime[0])

    # Add interactive controls if requested
    if show_controls:
        controller.add_controls()

    return fig, ax, controller if show_controls else None


class SkyPointingController:
    """Controller for interactive sky pointing visualization."""

    _constraint_cache: dict[str, Any]

    def __init__(
        self,
        ditl: "DITL | QueueDITL",
        fig: Figure,
        ax: Axes,
        n_grid_points: int = 100,
        time_step_seconds: float = 60,
        constraint_alpha: float = 0.3,
        config: VisualizationConfig | None = None,
        observation_categories: "ObservationCategories | None" = None,
    ) -> None:
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
        observation_categories : ObservationCategories, optional
            Observation categories for color coding.
        """
        self.ditl: "DITL | QueueDITL" = ditl
        self.fig: Figure = fig
        self.ax: Axes = ax
        self.n_grid_points: int = n_grid_points
        self.time_step_seconds: float = time_step_seconds
        self.constraint_alpha: float = constraint_alpha
        self.config: VisualizationConfig | None = config
        self.observation_categories: "ObservationCategories | None" = (
            observation_categories
        )

        # State
        self.current_time_idx: int = 0
        self.playing: bool = False
        self.timer: Any | None = None

        # Plot elements (will be created in update_plot)
        self.constraint_patches: dict[str, Any] = {}
        self.current_pointing_marker: Any | None = None
        self.scheduled_obs_scatter: Any | None = None
        self.title_text: Any | None = None

        # Track categories present in current plot
        self.current_plot_categories: dict[str, str] = {}  # category_name -> color

        # Control widgets
        self.slider: Slider
        self.play_button: Button
        self.prev_button: Button
        self.next_button: Button

    def add_controls(self) -> None:
        """Add interactive control widgets to the figure."""
        # Create axes for controls
        ax_slider = plt.axes((0.2, 0.15, 0.6, 0.03))
        ax_play = plt.axes((0.42, 0.05, 0.08, 0.04))
        ax_prev = plt.axes((0.32, 0.05, 0.08, 0.04))
        ax_next = plt.axes((0.52, 0.05, 0.08, 0.04))

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

    def on_slider_change(self, val: float) -> None:
        """Handle slider value change."""
        idx = int(val)
        if idx != self.current_time_idx:
            self.current_time_idx = idx
            self.update_plot(self.ditl.utime[idx])

    def on_play_clicked(self, event: Any) -> None:
        """Handle play button click."""
        if self.playing:
            self.stop_animation()
        else:
            self.start_animation()

    def on_prev_clicked(self, event: Any) -> None:
        """Handle previous button click."""
        if self.current_time_idx > 0:
            self.current_time_idx -= 1
            self.slider.set_val(self.current_time_idx)
            self.update_plot(self.ditl.utime[self.current_time_idx])

    def on_next_clicked(self, event: Any) -> None:
        """Handle next button click."""
        if self.current_time_idx < len(self.ditl.utime) - 1:
            self.current_time_idx += 1
            self.slider.set_val(self.current_time_idx)
            self.update_plot(self.ditl.utime[self.current_time_idx])

    def start_animation(self) -> None:
        """Start playing through time steps."""
        self.playing = True
        self.play_button.label.set_text("Pause")
        # Use matplotlib's animation timer
        self.timer = self.fig.canvas.new_timer(interval=100)  # 100ms between frames
        self.timer.add_callback(self.animation_step)
        self.timer.start()

    def stop_animation(self) -> None:
        """Stop playing animation."""
        self.playing = False
        if self.play_button is not None:
            self.play_button.label.set_text("Play")
        if self.timer is not None:
            self.timer.stop()
            self.timer = None

    def animation_step(self) -> None:
        """Advance to next time step during animation."""
        if self.current_time_idx < len(self.ditl.utime) - 1:
            self.current_time_idx += 1
            self.slider.set_val(self.current_time_idx)
            self.update_plot(self.ditl.utime[self.current_time_idx])
        else:
            # Reached the end
            self.stop_animation()

    def update_plot(self, utime: float) -> None:
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

    def _find_time_index(self, utime: float) -> int:
        """Find the index in utime array closest to the given time."""
        idx = int(np.argmin(np.abs(np.array(self.ditl.utime) - utime)))
        # Ensure index is within bounds for all arrays
        max_idx = (
            min(
                len(self.ditl.utime),
                len(self.ditl.ra),
                len(self.ditl.dec),
                len(self.ditl.mode),
            )
            - 1
        )
        return min(idx, max_idx)

    def _plot_scheduled_observations(self) -> None:
        """Plot all scheduled observations as markers."""
        if len(self.ditl.plan) == 0:
            return

        # Cache the observation data since it doesn't change between frames
        if not hasattr(self, "_cached_observations"):
            ras: list[float] = []
            decs: list[float] = []
            sizes: list[int] = []

            for ppt in self.ditl.plan:
                ra = ppt.ra
                dec = ppt.dec

                # Convert RA from 0-360 to -180 to 180 for mollweide, with RA=0 on left
                ra_plot = self._convert_ra_for_plotting(np.array([ra]))[0]

                ras.append(np.deg2rad(ra_plot))
                decs.append(np.deg2rad(dec))

                # Size by observation type or ID
                if hasattr(ppt, "obsid"):
                    # Use obsid to determine size
                    if ppt.obsid >= 1000000:
                        sizes.append(100)
                    elif ppt.obsid >= 20000:
                        sizes.append(60)
                    elif ppt.obsid >= 10000:
                        sizes.append(40)
                    else:
                        sizes.append(40)
                else:
                    sizes.append(40)

            self._cached_observations: dict[str, list[float] | list[int]] = {
                "ras": ras,
                "decs": decs,
                "sizes": sizes,
            }

        # Calculate colors dynamically based on current active target
        colors: list[str] = []
        self.current_plot_categories = {}  # Reset for this plot

        for ppt in self.ditl.plan:
            # Get base color from observation categories
            base_color = "tab:blue"  # Default fallback
            category_name = "Other"  # Default category name
            try:
                if self.observation_categories is not None and hasattr(ppt, "obsid"):
                    category = self.observation_categories.get_category(ppt.obsid)
                    if hasattr(category, "color") and isinstance(category.color, str):
                        base_color = category.color
                        category_name = category.name
                    else:
                        base_color = "tab:blue"  # Simple fallback color
                else:
                    base_color = "tab:blue"  # Simple fallback color
            except (AttributeError, TypeError):
                # If anything goes wrong with observation_categories, use simple fallback
                base_color = "tab:blue"  # Simple fallback color

            # Use full color for active target, lightened color for others
            if (
                hasattr(self.ditl, "ppt")
                and self.ditl.ppt is not None
                and ppt == self.ditl.ppt
            ):
                final_color = base_color  # Full color for active target
                self.current_plot_categories[category_name] = (
                    base_color  # Track active category with full color
                )
            else:
                final_color = _lighten_color(
                    base_color
                )  # Lightened color for inactive targets
                # Track inactive category with lightened color (only if not already tracked with full color)
                if category_name not in self.current_plot_categories:
                    self.current_plot_categories[category_name] = final_color

            colors.append(final_color)

        # Scatter plot using cached data and dynamic colors
        # Don't add a generic "Targets" label since we'll add specific category labels to legend
        self.ax.scatter(
            self._cached_observations["ras"],
            self._cached_observations["decs"],
            s=self._cached_observations["sizes"],
            c=colors,
            alpha=0.6,
            edgecolors="black",
            linewidths=0.5,
            zorder=2,
            rasterized=True,  # Rasterize for faster rendering
        )

    def _precompute_constraints(self, time_indices: np.ndarray | None = None) -> None:
        """Pre-compute constraint masks for all time steps using in_constraint_batch.

        This evaluates all constraints for the entire DITL in a single batch operation
        for much better performance during movie rendering.

        Parameters
        ----------
        time_indices : array-like, optional
            Indices of time steps to pre-compute. If None, uses all time steps.
        """
        if time_indices is None:
            time_indices = np.arange(len(self.ditl.utime))

        # Get sky grid points (same for all times)
        ra_flat, dec_flat = self._create_sky_grid(self.n_grid_points)
        n_points = len(ra_flat)

        # Convert times to datetime objects for rust_ephem
        times = [dtutcfromtimestamp(self.ditl.utime[idx]) for idx in time_indices]
        n_times = len(times)

        print(
            f"Pre-computing constraints for {n_times} time steps with {n_points} grid points..."
        )

        # Pre-compute all constraint types
        constraint_cache: dict[str, Any] = {}

        constraint_types = [
            ("sun", self.ditl.config.constraint.sun_constraint),
            ("moon", self.ditl.config.constraint.moon_constraint),
            ("earth", self.ditl.config.constraint.earth_constraint),
            ("anti_sun", self.ditl.config.constraint.anti_sun_constraint),
            ("panel", self.ditl.config.constraint.panel_constraint),
        ]
        assert self.ditl.ephem is not None, (
            "Ephemeris must be set for constraint calculations."
        )

        for name, constraint_func in constraint_types:
            # Batch evaluation with datetime array
            result = constraint_func.in_constraint_batch(
                ephemeris=self.ditl.ephem,
                target_ras=list(ra_flat),
                target_decs=list(dec_flat),
                times=times,  # Pass entire array of datetime objects
            )
            # Result shape is (n_points, n_times) from rust_ephem
            constraint_cache[name] = result

        # Store cache
        self._constraint_cache = {
            "ra_grid": ra_flat,
            "dec_grid": dec_flat,
            "time_indices": time_indices,
            "constraints": constraint_cache,
        }

        print(
            f"Constraint pre-computation complete. Cached {len(constraint_types)} constraint types."
        )

    def _plot_constraint_regions(self, utime: float) -> None:
        """Plot constraint regions for Sun, Moon, and Earth.

        Parameters
        ----------
        utime : float
            Unix timestamp for constraint calculation.
        """

        dt = dtutcfromtimestamp(utime)
        ephem = self.ditl.constraint.ephem
        assert ephem is not None, "Ephemeris must be set for constraint calculations."
        idx = ephem.index(dt)

        # Get celestial body positions
        sun_ra = ephem.sun[idx].ra.deg
        sun_dec = ephem.sun[idx].dec.deg
        moon_ra = ephem.moon[idx].ra.deg
        moon_dec = ephem.moon[idx].dec.deg
        earth_ra = ephem.earth[idx].ra.deg
        earth_dec = ephem.earth[idx].dec.deg

        # Check each constraint type and plot regions
        constraint_types = [
            (
                "Sun",
                self.ditl.config.constraint.sun_constraint,
                "yellow",
                sun_ra,
                sun_dec,
            ),
            (
                "Moon",
                self.ditl.config.constraint.moon_constraint,
                "gray",
                moon_ra,
                moon_dec,
            ),
            (
                "Earth",
                self.ditl.config.constraint.earth_constraint,
                "blue",
                earth_ra,
                earth_dec,
            ),
            (
                "Anti-Sun",
                self.ditl.config.constraint.anti_sun_constraint,
                "orange",
                (sun_ra + 180) % 360,
                -sun_dec,
            ),
            (
                "Panel",
                self.ditl.config.constraint.panel_constraint,
                "green",
                None,
                None,
            ),
        ]

        for name, constraint_func, color, body_ra, body_dec in constraint_types:
            self._plot_single_constraint(
                name,
                constraint_func,
                color,
                utime,
                body_ra,
                body_dec,
            )

    def _plot_earth_disk(self, utime: float) -> None:
        """Plot the physical extent of Earth as seen from the spacecraft.

        Parameters
        ----------
        utime : float
            Unix timestamp for Earth position calculation.
        """

        dt = dtutcfromtimestamp(utime)
        ephem = self.ditl.ephem
        idx = ephem.index(dt)

        # Get Earth position and angular radius
        earth_ra = ephem.earth[idx].ra.deg
        earth_dec = ephem.earth[idx].dec.deg
        earth_angular_radius = ephem.earth_radius_deg[idx]

        # Get sky grid points (use cached if available)
        if hasattr(self, "_constraint_cache"):
            ra_flat = self._constraint_cache["ra_grid"]
            dec_flat: npt.NDArray[np.float64] = self._constraint_cache["dec_grid"]
        else:
            ra_flat, dec_flat = self._create_sky_grid(self.n_grid_points)

        # Vectorized calculation of angular distances from Earth center
        delta_ra = np.radians(ra_flat - earth_ra)
        dec_rad = np.radians(dec_flat)
        earth_dec_rad = np.radians(earth_dec)

        cos_value = np.sin(earth_dec_rad) * np.sin(dec_rad) + np.cos(
            earth_dec_rad
        ) * np.cos(dec_rad) * np.cos(delta_ra)
        angular_dist = np.degrees(np.arccos(np.clip(cos_value, -1.0, 1.0)))

        # Find points inside the Earth disk
        inside_earth = angular_dist <= earth_angular_radius

        # Plot Earth disk points
        if inside_earth.any():
            ra_vals = ra_flat[inside_earth]
            dec_vals = dec_flat[inside_earth]

            self._plot_points_on_sky(
                ra_vals, dec_vals, "darkblue", alpha=0.8, label="Earth Disk", zorder=2.5
            )

    def _create_sky_grid(
        self, n_points: int
    ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        """Create a grid of RA/Dec points optimized for Mollweide projection.

        Parameters
        ----------
        n_points : int
            Number of points per axis for the grid.

        Returns
        -------
        ra_flat : array
            Flattened RA coordinates (0-360 degrees).
        dec_flat : array
            Flattened Dec coordinates (-90-90 degrees).
        """
        # Linear declination sampling
        dec_samples = np.linspace(-90, 90, n_points)

        # For each declination, calculate how many RA samples we need
        # based on cos(dec) for even visual density in Mollweide projection
        cos_factors = np.cos(np.radians(dec_samples))
        n_ra_array = np.maximum(8, (n_points * 2 * cos_factors).astype(int))

        ra_flat = np.concatenate(
            [np.linspace(0, 360, n, endpoint=False) for n in n_ra_array]
        )
        dec_flat = np.concatenate(
            [np.full(n, dec) for n, dec in zip(n_ra_array, dec_samples)]
        )

        return ra_flat, dec_flat

    def _convert_ra_for_plotting(self, ra_vals: np.ndarray) -> np.ndarray:
        """Convert RA coordinates for Mollweide projection plotting.

        Parameters
        ----------
        ra_vals : array
            RA values in degrees (0-360).

        Returns
        -------
        array
            RA values converted for plotting (-180 to 180).
        """
        return ra_vals - 180

    def _plot_points_on_sky(
        self,
        ra_vals: np.ndarray,
        dec_vals: np.ndarray,
        color: str,
        alpha: float = 0.3,
        size: int = 20,
        marker: str = "s",
        label: str | None = None,
        zorder: float = 1,
    ) -> None:
        """Plot points on the sky map.

        Parameters
        ----------
        ra_vals : array
            RA coordinates in degrees (0-360).
        dec_vals : array
            Dec coordinates in degrees (-90-90).
        color : str
            Color for the points.
        alpha : float, optional
            Transparency (default: 0.3).
        size : int, optional
            Point size (default: 20).
        marker : str, optional
            Marker style (default: "s" for square).
        label : str, optional
            Legend label.
        zorder : float, optional
            Z-order for layering (default: 1).
        """
        ra_plot = self._convert_ra_for_plotting(ra_vals)

        self.ax.scatter(
            np.deg2rad(ra_plot),
            np.deg2rad(dec_vals),
            s=size,
            c=color,
            alpha=alpha,
            marker=marker,
            edgecolors="none",
            label=label,
            zorder=zorder,
            rasterized=True,  # Rasterize for faster rendering
        )

    def _plot_single_constraint(
        self,
        name: str,
        constraint_func: rust_ephem.constraints.ConstraintConfig,
        color: str,
        utime: float,
        body_ra: float | None,
        body_dec: float | None,
    ) -> None:
        """Plot a single constraint region.

        Parameters
        ----------
        name : str
            Name of the constraint.
        constraint_func : Constraint
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
        # Check if we have pre-computed constraints
        if (
            hasattr(self, "_constraint_cache")
            and name.lower() in self._constraint_cache["constraints"]
        ):
            cache = self._constraint_cache

            # Verify cache consistency
            current_ra_flat, current_dec_flat = self._create_sky_grid(
                self.n_grid_points
            )
            if len(cache["ra_grid"]) == len(current_ra_flat) and len(
                cache["dec_grid"]
            ) == len(current_dec_flat):
                # Use pre-computed constraints
                ra_flat = cache["ra_grid"]
                dec_flat = cache["dec_grid"]

                # Find time index in cache
                time_idx = self._find_time_index(utime)
                cache_time_idx = np.where(cache["time_indices"] == time_idx)[0]
                if (
                    len(cache_time_idx) > 0
                    and cache_time_idx[0] < cache["constraints"][name.lower()].shape[1]
                ):
                    cache_time_idx = cache_time_idx[0]
                    # Cache shape is (n_points, n_times), so index with [:, time_idx]
                    constrained_coords = cache["constraints"][name.lower()][
                        :, cache_time_idx
                    ]

                    # Ensure constrained_coords is a boolean array
                    constrained_coords = np.asarray(constrained_coords, dtype=bool)

                    # Plot constrained region
                    if constrained_coords.any():
                        points = np.column_stack(
                            (ra_flat[constrained_coords], dec_flat[constrained_coords])
                        )

                        ra_vals = points[:, 0]
                        dec_vals = points[:, 1]

                        self._plot_points_on_sky(
                            ra_vals,
                            dec_vals,
                            color,
                            alpha=self.constraint_alpha,
                            label=f"{name} Cons.",
                            zorder=1,
                        )

                    # Mark celestial body position
                    if body_ra is not None and body_dec is not None:
                        ra_plot = self._convert_ra_for_plotting(np.array([body_ra]))
                        self.ax.plot(
                            np.deg2rad(ra_plot),
                            np.deg2rad(body_dec),
                            marker="o",
                            markersize=8,
                            markeredgecolor="black",
                            markerfacecolor=color,
                            markeredgewidth=2,
                            zorder=3,
                            label=name,
                        )
                    return

        # Fall back to real-time evaluation if no cache available
        # Get sky grid points
        ra_flat, dec_flat = self._create_sky_grid(self.n_grid_points)

        constrained_coords = constraint_func.in_constraint_batch(
            ephemeris=self.ditl.ephem,
            target_ras=ra_flat.tolist(),
            target_decs=dec_flat.tolist(),
            times=[dtutcfromtimestamp(utime)],
        )[:, 0]

        # Plot constrained region
        if constrained_coords.any():
            points = np.column_stack(
                (ra_flat[constrained_coords], dec_flat[constrained_coords])
            )

            ra_vals = points[:, 0]
            dec_vals = points[:, 1]

            self._plot_points_on_sky(
                ra_vals,
                dec_vals,
                color,
                alpha=self.constraint_alpha,
                label=f"{name} Cons.",
                zorder=1,
            )

        # Mark celestial body position
        if body_ra is not None and body_dec is not None:
            ra_plot = self._convert_ra_for_plotting(np.array([body_ra]))
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

    def _plot_current_pointing(self, ra: float, dec: float, mode: Any) -> None:
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
        ra_plot = self._convert_ra_for_plotting(np.array([ra]))[0]

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
        circle = Circle(
            (np.deg2rad(ra_plot), np.deg2rad(dec)),
            radius=np.deg2rad(5),
            fill=False,
            edgecolor=color,
            linewidth=2,
            zorder=4,
        )
        self.ax.add_patch(circle)

    def _setup_plot_appearance(self, utime: float) -> None:
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

        # Add ACS mode entries
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

        # Add observation category entries for categories present in current plot
        category_handles = []
        for category_name, color in self.current_plot_categories.items():
            category_handles.append(
                Line2D(
                    [0],
                    [0],
                    marker="o",
                    color="w",
                    markerfacecolor=color,
                    markersize=8,
                    markeredgecolor="black",
                    markeredgewidth=0.5,
                    label=category_name,
                )
            )

        all_handles = list(by_label.values()) + mode_handles + category_handles

        # Create legend with both ACS modes and observation categories
        legend_labels = (
            list(by_label.keys())
            + [f"Mode: {h.get_label()}" for h in mode_handles]
            + [h.get_label() for h in category_handles]
        )

        self.ax.legend(
            all_handles,
            legend_labels,
            loc="upper left",
            bbox_to_anchor=(1.02, 1),
            fontsize=self.config.legend_font_size if self.config else 8,
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
    ditl: "DITL | QueueDITL",
    output_dir: str,
    figsize: tuple[float, float] = (14, 8),
    n_grid_points: int = 50,
    frame_interval: int = 1,
    config: VisualizationConfig | None = None,
) -> list[str]:
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
    config : MissionConfig, optional
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

    saved_files: list[str] = []
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


def save_sky_pointing_movie(
    ditl: "DITL | QueueDITL",
    output_file: str,
    fps: float = 10,
    figsize: tuple[float, float] = (14, 8),
    n_grid_points: int = 50,
    frame_interval: int = 1,
    dpi: int = 100,
    codec: str = "h264",
    bitrate: int = 1800,
    config: VisualizationConfig | None = None,
    show_progress: bool = True,
) -> str:
    """Export the entire DITL sky pointing visualization as a movie.

    Creates an animated movie showing how spacecraft pointing and constraints
    evolve throughout the entire DITL simulation.

    Parameters
    ----------
    ditl : DITL or QueueDITL
        The DITL simulation object with completed simulation data.
    output_file : str
        Output filename for the movie. Extension determines format:
        - '.mp4' for MP4 video (requires ffmpeg)
        - '.gif' for animated GIF (requires pillow)
        - '.avi' for AVI video (requires ffmpeg)
    fps : float, optional
        Frames per second in the output movie (default: 10).
        Lower values create slower playback, higher values faster playback.
    figsize : tuple, optional
        Figure size as (width, height) in inches (default: (14, 8)).
    n_grid_points : int, optional
        Number of grid points per axis for constraint region calculation
        (default: 50). Lower values render faster but with less detail.
    frame_interval : int, optional
        Use every Nth time step from the DITL (default: 1 = use all frames).
        Higher values create shorter movies with faster playback.
    dpi : int, optional
        Resolution in dots per inch (default: 100).
        Higher values create larger, higher quality files.
    codec : str, optional
        Video codec for MP4/AVI output (default: 'h264').
        Other options: 'mpeg4', 'libx264', etc.
    bitrate : int, optional
        Video bitrate in kbps (default: 1800).
        Higher values create better quality but larger files.
    config : MissionConfig, optional
        Configuration object containing visualization settings. If None,
        uses ditl.config.visualization if available.
    show_progress : bool, optional
        Whether to show a progress bar using tqdm (default: True).

    Returns
    -------
    str
        Path to the saved movie file.

    Raises
    ------
    ValueError
        If output format is not supported or required codecs are not available.
    RuntimeError
        If movie encoding fails.

    Examples
    --------
    >>> # Create MP4 movie at 15 fps
    >>> save_sky_pointing_movie(ditl, "pointing.mp4", fps=15)

    >>> # Create animated GIF (slower, larger file)
    >>> save_sky_pointing_movie(ditl, "pointing.gif", fps=5)

    >>> # Fast preview with reduced detail
    >>> save_sky_pointing_movie(
    ...     ditl, "preview.mp4",
    ...     fps=20, frame_interval=5, n_grid_points=30
    ... )

    >>> # Disable progress bar for automated scripts
    >>> save_sky_pointing_movie(ditl, "pointing.mp4", show_progress=False)

    Notes
    -----
    - MP4 and AVI formats require ffmpeg to be installed on your system
    - GIF format requires the pillow library
    - Frame rate (fps) controls playback speed, not simulation time
    - frame_interval controls which simulation time steps are included
    - Lower n_grid_points speeds up rendering but reduces visual quality
    - The movie shows the same view as plot_sky_pointing() but automated
    - Progress bar requires tqdm library
    """
    from matplotlib.animation import FFMpegWriter, PillowWriter

    # Try to import tqdm, fall back to no progress bar if unavailable
    if show_progress:
        try:
            from tqdm import tqdm
        except ImportError:
            show_progress = False
            print("Note: tqdm not available, progress bar disabled")

    # Validate inputs
    if not hasattr(ditl, "plan") or len(ditl.plan) == 0:
        raise ValueError("DITL simulation has no pointings. Run calc() first.")
    if not hasattr(ditl, "utime") or len(ditl.utime) == 0:
        raise ValueError("DITL has no time data. Run calc() first.")

    # Determine file format and writer
    file_ext = os.path.splitext(output_file)[1].lower()
    writer_kwargs: dict[str, Any]
    writer_class: Any
    if file_ext == ".gif":
        writer_class = PillowWriter
        writer_kwargs = {"fps": fps}
    elif file_ext in [".mp4", ".avi"]:
        writer_class = FFMpegWriter
        writer_kwargs = {
            "fps": fps,
            "codec": codec,
            "bitrate": bitrate,
        }
    else:
        raise ValueError(
            f"Unsupported output format: {file_ext}. Use .mp4, .avi, or .gif"
        )

    # Get visualization config
    config = _get_visualization_config(ditl, config)

    # Create figure without controls
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={"projection": "mollweide"})

    # Create controller for updates
    controller = SkyPointingController(
        ditl=ditl,
        fig=fig,
        ax=ax,
        n_grid_points=n_grid_points,
        time_step_seconds=ditl.step_size,
        constraint_alpha=0.3,
        config=config,
    )

    # Select time steps to animate
    time_indices = list(range(0, len(ditl.utime), frame_interval))
    total_frames = len(time_indices)

    # Pre-compute constraints for all time steps to be rendered
    controller._precompute_constraints(np.array(time_indices))

    print(f"Creating movie with {total_frames} frames at {fps} fps...")
    print(f"Movie duration: {total_frames / fps:.1f} seconds")
    print(f"Output: {output_file}")

    # Set up the writer
    writer = writer_class(**writer_kwargs)

    try:
        with writer.saving(fig, output_file, dpi=dpi):
            # Create iterator with optional progress bar
            iterator: Any
            if show_progress:
                iterator = tqdm(
                    enumerate(time_indices),
                    total=total_frames,
                    desc="Rendering frames",
                    unit="frame",
                    ncols=80,
                )
            else:
                iterator = enumerate(time_indices)

            for frame_num, idx in iterator:
                utime = ditl.utime[idx]
                controller.update_plot(utime)

                # Save this frame
                writer.grab_frame()

        plt.close(fig)
        print(f"\nSuccessfully saved movie to {output_file}")
        return output_file

    except Exception as e:
        plt.close(fig)
        raise RuntimeError(f"Failed to create movie: {e}") from e
