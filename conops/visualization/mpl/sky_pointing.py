"""Sky Pointing Visualization

Interactive visualization showing spacecraft pointing on a mollweide projection
of the sky with scheduled observations and constraint regions.
"""

# pyright: reportMissingTypeStubs=false

import os
from importlib import import_module
from typing import TYPE_CHECKING, Any, Optional, cast

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

from ...common import dtutcfromtimestamp
from ...config import DTOR
from ...config.observation_categories import ObservationCategories
from ...config.visualization import VisualizationConfig

if TYPE_CHECKING:
    from ...ditl import DITL, QueueDITL


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


def _gaussian_smooth(
    data: npt.NDArray[np.float64], sigma: float, wrap_ra: bool = True
) -> npt.NDArray[np.float64]:
    """Apply Gaussian smoothing to a 2D array for smooth constraint edges.

    Uses a separable 1D convolution approach for efficiency. This is a pure
    numpy implementation to avoid scipy dependency.

    Parameters
    ----------
    data : ndarray
        2D array to smooth (shape: dec x ra).
    sigma : float
        Standard deviation of the Gaussian kernel.
    wrap_ra : bool
        Whether to wrap RA axis (for 360-degree continuity).

    Returns
    -------
    ndarray
        Smoothed 2D array.
    """
    if sigma <= 0:
        return data

    # Create 1D Gaussian kernel
    kernel_size = int(6 * sigma + 1)  # Cover 3 sigma on each side
    if kernel_size % 2 == 0:
        kernel_size += 1
    x = np.arange(kernel_size) - kernel_size // 2
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel = kernel / kernel.sum()

    # Convolve along RA axis (axis=1) with wrapping
    if wrap_ra:
        # Pad with wrapped values for continuity at 0/360
        pad_width = kernel_size // 2
        padded = np.concatenate(
            [data[:, -pad_width:], data, data[:, :pad_width]], axis=1
        )
        smoothed = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="valid"), axis=1, arr=padded
        )
    else:
        smoothed = np.apply_along_axis(
            lambda m: np.convolve(m, kernel, mode="same"), axis=1, arr=data
        )

    # Convolve along Dec axis (axis=0) - no wrapping needed
    smoothed = np.apply_along_axis(
        lambda m: np.convolve(m, kernel, mode="same"), axis=0, arr=smoothed
    )

    return smoothed


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
        self._timer: Any | None = None

        # Plot elements (will be created in update_plot)
        self.constraint_patches: dict[str, Any] = {}
        self.current_pointing_marker: Any | None = None
        self.scheduled_obs_scatter: Any | None = None
        self.title_text: Any | None = None

        # Track categories present in current plot
        self.current_plot_categories: dict[str, str] = {}  # category_name -> color

        # Cache per-point optimal roll maps; Sun geometry changes slowly so nearby
        # frames can share the same solution.
        self._optimal_roll_cache: dict[tuple[Any, ...], npt.NDArray[np.float64]] = {}
        self._optimal_roll_cache_max_entries: int = (
            int(self.config.optimal_roll_cache_max_entries)
            if self.config is not None
            else 64
        )
        self._optimal_roll_cache_step_deg: float = (
            float(self.config.optimal_roll_cache_sun_bucket_deg)
            if self.config is not None
            else 0.02
        )

        # Control widgets
        self.slider: Slider
        self.play_button: Button
        self.prev_button: Button
        self.next_button: Button

    def _solar_panel_signature(self) -> tuple[Any, ...]:
        """Build a hashable signature for panel geometry/efficiency settings."""
        solar_panel = getattr(self.ditl.config, "solar_panel", None)
        if solar_panel is None:
            return ("no_solar_panel",)

        panels = getattr(solar_panel, "panels", None)
        if not isinstance(panels, list) or len(panels) == 0:
            return ("empty_panels",)

        panel_sig: list[tuple[Any, ...]] = []
        for panel in panels:
            normal = getattr(panel, "normal", None)
            max_power = getattr(panel, "max_power", None)
            panel_eff = getattr(panel, "conversion_efficiency", None)
            if normal is None or max_power is None:
                continue
            n_arr = np.asarray(normal, dtype=np.float64)
            panel_sig.append(
                (
                    round(float(n_arr[0]), 6),
                    round(float(n_arr[1]), 6),
                    round(float(n_arr[2]), 6),
                    round(float(max_power), 6),
                    None if panel_eff is None else round(float(panel_eff), 6),
                )
            )

        return (
            "panel",
            round(float(getattr(solar_panel, "conversion_efficiency", 1.0)), 6),
            tuple(panel_sig),
        )

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
        self._animation_step()

    def _animation_step(self) -> None:
        """Execute one animation step and schedule the next."""
        if not self.playing:
            return

        if self.current_time_idx < len(self.ditl.utime) - 1:
            self.current_time_idx += 1
            self.slider.set_val(self.current_time_idx)
            self.update_plot(self.ditl.utime[self.current_time_idx])

            # Schedule next step using the figure's timer
            if self.playing:
                self._timer = self.fig.canvas.new_timer(interval=100)
                self._timer.single_shot = True
                self._timer.add_callback(self._animation_step)
                self._timer.start()
        else:
            # Reached the end
            self.stop_animation()

    def stop_animation(self) -> None:
        """Stop playing animation."""
        self.playing = False
        if self.play_button is not None:
            self.play_button.label.set_text("Play")
        if hasattr(self, "_timer") and self._timer is not None:
            try:
                self._timer.stop()
            except (AttributeError, RuntimeError):
                pass
            self._timer = None
        self.fig.canvas.draw_idle()

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

        # Plot star tracker boresight directions
        self._plot_star_tracker_boresights(utime)

        # Plot hatched ST exclusion-zone circles around constrained bodies
        self._plot_st_exclusion_circles(utime)

        # Set up the plot
        self._setup_plot_appearance(utime)

        # Redraw
        self.fig.canvas.draw_idle()

    def _plot_star_tracker_boresights(self, utime: float) -> None:
        """Plot each star tracker's boresight direction as a hexagon marker.

        Parameters
        ----------
        utime : float
            Unix timestamp to display.
        """
        if not hasattr(self.ditl, "config") or not hasattr(
            self.ditl.config, "spacecraft_bus"
        ):
            return
        star_trackers = getattr(self.ditl.config.spacecraft_bus, "star_trackers", None)
        if star_trackers is None or not hasattr(star_trackers, "star_trackers"):
            return
        raw_trackers: object = getattr(star_trackers, "star_trackers", [])
        if not isinstance(raw_trackers, (list, tuple)) or not raw_trackers:
            return
        trackers: list[Any] = list(raw_trackers)

        # Current pointing and roll
        idx = self._find_time_index(utime)
        ra_deg = self.ditl.ra[idx]
        dec_deg = self.ditl.dec[idx]

        roll_list = getattr(self.ditl, "roll", [])
        roll_idx = min(idx, len(roll_list) - 1) if roll_list else -1
        roll_deg = float(roll_list[roll_idx]) if roll_idx >= 0 else 0.0

        # Spacecraft body-frame basis in ICRS at current (ra, dec, roll)
        ra_rad = np.deg2rad(ra_deg)
        dec_rad = np.deg2rad(dec_deg)
        x_hat = np.array(
            [
                np.cos(dec_rad) * np.cos(ra_rad),
                np.cos(dec_rad) * np.sin(ra_rad),
                np.sin(dec_rad),
            ],
            dtype=np.float64,
        )

        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y0 = np.cross(ref, x_hat)
        y0_norm = np.linalg.norm(y0)
        if y0_norm < 1e-12:
            ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            y0 = np.cross(ref, x_hat)
            y0_norm = np.linalg.norm(y0)
        y0 = y0 / y0_norm
        z0 = np.cross(x_hat, y0)
        z0 = z0 / np.linalg.norm(z0)

        roll_rad = np.deg2rad(roll_deg)
        c_r, s_r = np.cos(roll_rad), np.sin(roll_rad)
        y_hat = y0 * c_r - z0 * s_r
        z_hat = y0 * s_r + z0 * c_r

        # Resolve per-tracker functional status from housekeeping telemetry.
        # Green = functional (not in soft constraint), Red = degraded.
        hk_records = (
            list(self.ditl.telemetry.housekeeping)
            if hasattr(self.ditl, "telemetry")
            and hasattr(self.ditl.telemetry, "housekeeping")
            else []
        )
        hk_status: list[bool] | None = None
        if idx < len(hk_records):
            hk_status = hk_records[idx].star_tracker_status

        st_fallback_colors = ["cyan", "lime", "orange", "hotpink", "white", "yellow"]

        for i, st in enumerate(trackers):
            orientation = getattr(st, "orientation", None)
            if orientation is None:
                continue
            boresight = getattr(orientation, "boresight", None)
            if boresight is None:
                continue

            b = np.asarray(boresight, dtype=np.float64)
            v_st = b[0] * x_hat + b[1] * y_hat + b[2] * z_hat
            v_norm = np.linalg.norm(v_st)
            if v_norm < 1e-12:
                continue
            v_st = v_st / v_norm

            st_ra_deg = (np.degrees(np.arctan2(v_st[1], v_st[0])) + 360.0) % 360.0
            st_dec_deg = np.degrees(np.arcsin(np.clip(v_st[2], -1.0, 1.0)))

            if hk_status is not None and i < len(hk_status):
                color = "limegreen" if hk_status[i] else "red"
            else:
                color = st_fallback_colors[i % len(st_fallback_colors)]
            st_name = getattr(st, "name", f"ST-{i + 1}")
            ra_plot = self._convert_ra_for_plotting(np.array([st_ra_deg]))[0]

            self.ax.plot(
                np.deg2rad(ra_plot),
                np.deg2rad(st_dec_deg),
                marker="h",
                markersize=14,
                markerfacecolor=color,
                markeredgecolor="black",
                markeredgewidth=1.5,
                linestyle="none",
                label=st_name,
                zorder=6,
            )

    def _plot_st_exclusion_circles(self, utime: float) -> None:
        """Draw hatched exclusion-zone circles around constrained bodies for each ST.

        For each (tracker, tier, body) constraint spec, draws an analytical circle
        of angular radius ``min_angle`` (plus Earth disk radius for Earth constraints)
        centred on the corresponding celestial body.  This matches what
        ``in_soft_constraint`` / ``in_hard_constraint`` test against the ST boresight
        sky position.  Hard-constraint circles are shown with red diagonal hatching;
        soft-constraint circles with magenta cross-hatching.  Both use a transparent
        fill so they do not obscure the underlying constraint shading.
        """
        if not hasattr(self.ditl, "config") or not hasattr(
            self.ditl.config, "spacecraft_bus"
        ):
            return
        star_trackers = getattr(self.ditl.config.spacecraft_bus, "star_trackers", None)
        if star_trackers is None or not hasattr(star_trackers, "star_trackers"):
            return
        raw_trackers = getattr(star_trackers, "star_trackers", [])
        if not isinstance(raw_trackers, (list, tuple)) or not raw_trackers:
            return
        trackers: list[Any] = list(raw_trackers)

        # Collect (tracker_idx, tier, body, min_angle) specs.
        specs: list[tuple[int, str, str, float]] = []
        for ci, st in enumerate(trackers):
            for tier in ("hard", "soft"):
                cobj = getattr(st, f"{tier}_constraint", None)
                if cobj is None:
                    continue
                for body in ("sun", "earth", "moon"):
                    bcfg = getattr(cobj, f"{body}_constraint", None)
                    if bcfg is None:
                        continue
                    mangle = getattr(bcfg, "min_angle", None)
                    if mangle is not None and float(mangle) > 0:
                        specs.append((ci, tier, body, float(mangle)))
        if not specs:
            return

        # Ephemeris body positions at this time step.
        dt = dtutcfromtimestamp(utime)
        ephem = self.ditl.constraint.ephem
        if ephem is None:
            return
        ei = ephem.index(dt)
        body_pos = {
            "sun": (float(ephem.sun_ra_deg[ei]), float(ephem.sun_dec_deg[ei]), 0.0),
            "moon": (float(ephem.moon_ra_deg[ei]), float(ephem.moon_dec_deg[ei]), 0.0),
            "earth": (
                float(ephem.earth_ra_deg[ei]),
                float(ephem.earth_dec_deg[ei]),
                float(ephem.earth_radius_deg[ei]),  # add physical disk
            ),
        }

        # Sky grid (reuses the same resolution as other constraints).
        n_ra = self.n_grid_points * 2
        n_dec = self.n_grid_points
        ra_grid_rad, dec_grid_rad, ra_flat, dec_flat = self._create_regular_sky_grid(
            n_ra, n_dec
        )
        ra_flat_rad = np.deg2rad(ra_flat)
        dec_flat_rad = np.deg2rad(dec_flat)

        def _dist_mask(
            body_ra_d: float, body_dec_d: float, radius_d: float
        ) -> npt.NDArray[np.bool_]:
            br = np.deg2rad(body_ra_d)
            bd = np.deg2rad(body_dec_d)
            cos_d = np.sin(dec_flat_rad) * np.sin(bd) + np.cos(dec_flat_rad) * np.cos(
                bd
            ) * np.cos(ra_flat_rad - br)
            dist = np.degrees(np.arccos(np.clip(cos_d, -1.0, 1.0)))
            return cast(npt.NDArray[np.bool_], dist <= radius_d)

        # Accumulate union masks per tier.
        # The hatched zones are boresight exclusion zones: a circle of radius
        # min_angle (+ Earth disk) around each body, matching exactly what
        # in_soft_constraint / in_hard_constraint tests against the boresight
        # sky position.  The δ offset is NOT added here because the hexagon
        # markers are already plotted at the boresight sky position — not at
        # the spacecraft pointing direction.
        hard_mask: npt.NDArray[np.bool_] | None = None
        soft_mask: npt.NDArray[np.bool_] | None = None
        for tr_idx, tier, body, min_angle in specs:
            if body not in body_pos:
                continue
            body_ra, body_dec, extra_r = body_pos[body]
            eff_r = min_angle + extra_r
            if eff_r <= 0:
                continue
            m = _dist_mask(body_ra, body_dec, eff_r)
            if tier == "hard":
                hard_mask = m if hard_mask is None else (hard_mask | m)
            else:
                soft_mask = m if soft_mask is None else (soft_mask | m)

        alpha = max(0.03, self.constraint_alpha * 0.2)

        if hard_mask is not None and hard_mask.any():
            mask_2d = hard_mask.reshape((n_dec, n_ra)).astype(float)
            self.ax.contourf(
                ra_grid_rad,
                dec_grid_rad,
                mask_2d,
                levels=[0.5, 1.0],
                hatches=["/////"],
                colors=[mcolors.to_rgba("red", alpha)],
                zorder=2,
                alpha=alpha,
            )
            self.ax.contour(
                ra_grid_rad,
                dec_grid_rad,
                mask_2d,
                levels=[0.5],
                colors=["red"],
                linewidths=1.2,
                linestyles=["--"],
                zorder=2.1,
            )
            self.ax.plot(
                [],
                [],
                linestyle="--",
                color="red",
                linewidth=1.5,
                label="ST Hard Excl.",
            )

        if soft_mask is not None and soft_mask.any():
            mask_2d = soft_mask.reshape((n_dec, n_ra)).astype(float)
            self.ax.contourf(
                ra_grid_rad,
                dec_grid_rad,
                mask_2d,
                levels=[0.5, 1.0],
                hatches=["xxxxx"],
                colors=[mcolors.to_rgba("magenta", alpha)],
                zorder=2,
                alpha=alpha,
            )
            self.ax.contour(
                ra_grid_rad,
                dec_grid_rad,
                mask_2d,
                levels=[0.5],
                colors=["magenta"],
                linewidths=1.0,
                linestyles=[":"],
                zorder=2.1,
            )
            self.ax.plot(
                [],
                [],
                linestyle=":",
                color="magenta",
                linewidth=1.5,
                label="ST Soft Excl.",
            )

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

        # Grid dimensions for pcolormesh (use 2x n_grid_points for RA due to 360 degree range)
        n_ra = self.n_grid_points * 2
        n_dec = self.n_grid_points

        # Get regular sky grid points for pcolormesh
        ra_grid_rad, dec_grid_rad, ra_flat, dec_flat = self._create_regular_sky_grid(
            n_ra, n_dec
        )
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
            ("orbit", self.ditl.config.constraint.orbit_constraint),
            ("panel", self.ditl.config.constraint.panel_constraint),
        ]

        # Add aggregated star-tracker constraints (if configured)
        star_trackers = None
        if hasattr(self.ditl, "config") and hasattr(self.ditl.config, "spacecraft_bus"):
            star_trackers = getattr(
                self.ditl.config.spacecraft_bus, "star_trackers", None
            )

        if star_trackers is not None and hasattr(star_trackers, "num_trackers"):
            try:
                has_trackers = star_trackers.num_trackers() > 0
            except Exception:
                has_trackers = False

            if has_trackers:
                # Soft constraints are already exposed as a computed combined constraint
                st_soft_constraint = getattr(
                    star_trackers, "startracker_constraint", None
                )
                if st_soft_constraint is not None:
                    constraint_types.append(("star_tracker_soft", st_soft_constraint))

                st_hard_combined = star_trackers.startracker_hard_constraint
                if st_hard_combined is not None:
                    constraint_types.append(("star_tracker_hard", st_hard_combined))

        assert self.ditl.ephem is not None, (
            "Ephemeris must be set for constraint calculations."
        )

        for name, constraint_func in constraint_types:
            if constraint_func is None:
                continue
            # Batch evaluation with datetime array
            result = constraint_func.in_constraint_batch(
                ephemeris=self.ditl.ephem,
                target_ras=list(ra_flat),
                target_decs=list(dec_flat),
                times=times,  # Pass entire array of datetime objects
            )
            # Result shape is (n_points, n_times) from rust_ephem
            constraint_cache[name] = result

        # Store cache with grid info for contourf
        self._constraint_cache = {
            "ra_grid": ra_flat,
            "dec_grid": dec_flat,
            "ra_grid_rad": ra_grid_rad,
            "dec_grid_rad": dec_grid_rad,
            "grid_shape": (n_ra, n_dec),
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

        # Get celestial body positions from pre-computed arrays
        sun_ra = ephem.sun_ra_deg[idx]
        sun_dec = ephem.sun_dec_deg[idx]
        moon_ra = ephem.moon_ra_deg[idx]
        moon_dec = ephem.moon_dec_deg[idx]
        earth_ra = ephem.earth_ra_deg[idx]
        earth_dec = ephem.earth_dec_deg[idx]

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
                "Orbit",
                self.ditl.config.constraint.orbit_constraint,
                "purple",
                None,
                None,
            ),
            (
                "Panel",
                self.ditl.config.constraint.panel_constraint,
                "green",
                None,
                None,
            ),
            (
                "Radiator Hard",
                self.ditl.config.constraint.radiator_hard_constraint,
                "coral",
                None,
                None,
            ),
        ]

        for name, constraint_func, color, body_ra, body_dec in constraint_types:
            if constraint_func is None:
                continue
            self._plot_single_constraint(
                name,
                constraint_func,
                color,
                utime,
                body_ra,
                body_dec,
            )

        # Star tracker regions are evaluated with the current spacecraft roll.
        self._plot_roll_aware_star_tracker_constraints(utime)

        # Radiator KOZ: project each radiator normal into sky coordinates, then
        # evaluate the base sun/earth/moon constraints at those directions.
        self._plot_radiator_koz_regions(utime)

    def _plot_roll_aware_star_tracker_constraints(self, utime: float) -> None:
        """Plot roll-aware star tracker hard/soft constraint regions."""
        hard_mask = self._compute_roll_aware_star_tracker_mask(
            utime=utime,
            kind="hard",
        )
        if hard_mask is not None:
            self._plot_constraint_mask(
                name="Star Tracker Hard",
                color="red",
                constrained_flat=hard_mask,
            )

        soft_mask = self._compute_roll_aware_star_tracker_mask(
            utime=utime,
            kind="soft",
        )
        if soft_mask is not None:
            self._plot_constraint_mask(
                name="Star Tracker Soft",
                color="magenta",
                constrained_flat=soft_mask,
            )

    def _plot_radiator_koz_regions(self, utime: float) -> None:
        """Plot radiator hard keep-out zones on the sky map.

        For each configured radiator, projects its body-frame normal into sky
        coordinates for each grid point and sampled roll, then evaluates hard
        sun/earth/moon constraints at those directions.

        A sky point is marked as blocked only if no sampled roll satisfies the
        radiator hard constraints (field-of-regard semantics).
        """
        if not hasattr(self.ditl, "config") or not hasattr(
            self.ditl.config, "spacecraft_bus"
        ):
            return
        radiators_cfg = getattr(self.ditl.config.spacecraft_bus, "radiators", None)
        if radiators_cfg is None or not hasattr(radiators_cfg, "radiators"):
            return
        raw_radiators: object = getattr(radiators_cfg, "radiators", [])
        if not isinstance(raw_radiators, (list, tuple)) or not raw_radiators:
            return

        ephem = self.ditl.ephem
        if ephem is None:
            return
        time_dt = dtutcfromtimestamp(utime)

        n_ra = self.n_grid_points * 2
        n_dec = self.n_grid_points
        _, _, ra_flat, dec_flat = self._create_regular_sky_grid(n_ra, n_dec)

        # Build body-frame basis at roll=0 for each sky grid point (same convention
        # as _compute_roll_aware_star_tracker_mask).
        ra_rad_g = np.deg2rad(ra_flat)
        dec_rad_g = np.deg2rad(dec_flat)
        cos_dec = np.cos(dec_rad_g)
        x_hat = np.column_stack(
            [cos_dec * np.cos(ra_rad_g), cos_dec * np.sin(ra_rad_g), np.sin(dec_rad_g)]
        )
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y0 = np.cross(ref, x_hat)
        y0_norm = np.linalg.norm(y0, axis=1)
        pole_mask = y0_norm < 1e-12
        if np.any(pole_mask):
            alt_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            y0[pole_mask] = np.cross(alt_ref, x_hat[pole_mask])
            y0_norm = np.linalg.norm(y0, axis=1)
        y0 = y0 / y0_norm[:, None]
        z0 = np.cross(x_hat, y0)
        z0 = z0 / np.linalg.norm(z0, axis=1)[:, None]

        def _eval_radiators(
            y_hat: npt.NDArray[np.float64],
            z_hat: npt.NDArray[np.float64],
        ) -> npt.NDArray[np.bool_] | None:
            """Return per-point hard-constraint violations for one roll basis."""
            mask_roll = np.zeros(len(ra_flat), dtype=bool)
            has_any_constraint = False

            for rad in list(raw_radiators):
                hc = getattr(rad, "hard_constraint", None)
                if hc is None:
                    continue
                orientation = getattr(rad, "orientation", None)
                if orientation is None:
                    continue
                normal = getattr(orientation, "normal", None)
                if normal is None:
                    continue

                # Project radiator body-frame normal into sky coordinates.
                b = np.asarray(normal, dtype=np.float64)
                v_rad = b[0] * x_hat + b[1] * y_hat + b[2] * z_hat
                ra_dir = (
                    np.degrees(np.arctan2(v_rad[:, 1], v_rad[:, 0])) + 360.0
                ) % 360.0
                dec_dir = np.degrees(np.arcsin(np.clip(v_rad[:, 2], -1.0, 1.0)))

                # Any radiator hard sub-constraint violation blocks this roll.
                for attr in ("sun_constraint", "earth_constraint", "moon_constraint"):
                    c = getattr(hc, attr, None)
                    if c is None:
                        continue
                    result = c.in_constraint_batch(
                        ephemeris=ephem,
                        target_ras=ra_dir.tolist(),
                        target_decs=dec_dir.tolist(),
                        times=[time_dt],
                    )[:, 0]
                    mask_roll |= cast(
                        npt.NDArray[np.bool_], np.asarray(result, dtype=bool)
                    )
                    has_any_constraint = True

            if not has_any_constraint:
                return None
            return mask_roll

        # Field-of-regard semantics: a point is blocked only if EVERY sampled
        # roll violates at least one radiator hard constraint.
        per_point_clear = np.zeros(len(ra_flat), dtype=bool)
        has_any_roll_constraint = False
        for roll_deg_s in np.linspace(0.0, 360.0, 36, endpoint=False):
            rr = np.deg2rad(float(roll_deg_s))
            c_s, s_s = float(np.cos(rr)), float(np.sin(rr))
            y_hat_s = y0 * c_s - z0 * s_s
            z_hat_s = y0 * s_s + z0 * c_s
            mask = _eval_radiators(y_hat_s, z_hat_s)
            if mask is None:
                continue
            has_any_roll_constraint = True
            per_point_clear |= ~mask

        if not has_any_roll_constraint:
            return

        blocked_mask = cast(npt.NDArray[np.bool_], ~per_point_clear)
        if blocked_mask.any():
            self._plot_constraint_mask("Radiator Hard", "coral", blocked_mask)

    def _compute_roll_aware_star_tracker_mask(
        self,
        utime: float,
        kind: str,
    ) -> npt.NDArray[np.bool_] | None:
        """Compute a roll-aware star tracker constraint mask on the sky grid."""
        if not hasattr(self.ditl, "config") or not hasattr(
            self.ditl.config, "spacecraft_bus"
        ):
            return None

        star_trackers = getattr(self.ditl.config.spacecraft_bus, "star_trackers", None)
        if star_trackers is None or not hasattr(star_trackers, "star_trackers"):
            return None

        raw_trackers: object = getattr(star_trackers, "star_trackers", [])
        if not isinstance(raw_trackers, (list, tuple)) or not raw_trackers:
            return None
        trackers: list[Any] = list(raw_trackers)

        # Grid dimensions for contourf/pcolormesh
        n_ra = self.n_grid_points * 2
        n_dec = self.n_grid_points

        # Use the same regular grid as the other constraints
        _, _, ra_flat, dec_flat = self._create_regular_sky_grid(n_ra, n_dec)

        # Spacecraft boresight (+X) unit vectors for all grid points.
        ra_rad = np.deg2rad(ra_flat)
        dec_rad = np.deg2rad(dec_flat)
        cos_dec = np.cos(dec_rad)
        x_hat = np.column_stack(
            [
                cos_dec * np.cos(ra_rad),
                cos_dec * np.sin(ra_rad),
                np.sin(dec_rad),
            ]
        )

        # Construct body Y/Z basis around boresight using sky north as reference.
        ref = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        y0 = np.cross(ref, x_hat)
        y0_norm = np.linalg.norm(y0, axis=1)

        # Near poles, use alternate reference to avoid degeneracy.
        pole_mask = y0_norm < 1e-12
        if np.any(pole_mask):
            alt_ref = np.array([0.0, 1.0, 0.0], dtype=np.float64)
            y0[pole_mask] = np.cross(alt_ref, x_hat[pole_mask])
            y0_norm = np.linalg.norm(y0, axis=1)

        y0 = y0 / y0_norm[:, None]
        z0 = np.cross(x_hat, y0)
        z0 = z0 / np.linalg.norm(z0, axis=1)[:, None]

        time_dt = dtutcfromtimestamp(utime)
        ephem = self.ditl.ephem
        if ephem is None:
            return None

        # When ignore_roll is True, roll is a free parameter and constraint checks
        # use FOR (field-of-regard) semantics.  Visualise the hard exclusion zones:
        # sky points where no roll angle can satisfy the star tracker constraint.
        ignore_roll: bool = bool(
            getattr(
                getattr(getattr(self.ditl, "config", None), "constraint", None),
                "ignore_roll",
                False,
            )
        )

        # ------------------------------------------------------------------ #
        # Local helper: evaluate all trackers at a given body-frame y/z basis #
        # ------------------------------------------------------------------ #
        def _eval_trackers(
            y_hat: npt.NDArray[np.float64], z_hat: npt.NDArray[np.float64]
        ) -> npt.NDArray[np.bool_] | None:
            masks: list[npt.NDArray[np.bool_]] = []
            for st in trackers:
                orientation = getattr(st, "orientation", None)
                if orientation is None:
                    continue
                constraint_obj = (
                    getattr(st, "hard_constraint", None)
                    if kind == "hard"
                    else getattr(st, "soft_constraint", None)
                )
                if constraint_obj is None or constraint_obj.constraint is None:
                    continue
                boresight = getattr(orientation, "boresight", None)
                if boresight is None:
                    continue
                b = np.asarray(boresight, dtype=np.float64)
                v_st = b[0] * x_hat + b[1] * y_hat + b[2] * z_hat
                ra_st = (np.degrees(np.arctan2(v_st[:, 1], v_st[:, 0])) + 360.0) % 360.0
                dec_st = np.degrees(np.arcsin(np.clip(v_st[:, 2], -1.0, 1.0)))
                result = constraint_obj.constraint.in_constraint_batch(
                    ephemeris=ephem,
                    target_ras=ra_st.tolist(),
                    target_decs=dec_st.tolist(),
                    times=[time_dt],
                )[:, 0]
                masks.append(
                    cast(npt.NDArray[np.bool_], np.asarray(result, dtype=bool))
                )
            if not masks:
                return None
            stacked = cast(npt.NDArray[np.bool_], np.vstack(masks))
            if kind == "hard":
                return cast(npt.NDArray[np.bool_], np.any(stacked, axis=0))
            # Soft: AtLeast violation semantics
            min_func = int(getattr(star_trackers, "min_functional_trackers", 1))
            required = max(0, len(trackers) - min_func + 1)
            if required > len(masks):
                return None
            counts = cast(npt.NDArray[np.int_], np.sum(stacked, axis=0))
            return counts >= required

        # ------------------------------------------------------------------ #
        # ignore_roll=True: FOR semantics — blocked at ALL rolls = exclusion  #
        # ------------------------------------------------------------------ #
        if ignore_roll:
            # Sample 36 roll angles (10° steps) and mark a sky point as part of the
            # FOR exclusion zone only if the constraint is violated at every sample.
            per_point_clear = np.zeros(len(ra_flat), dtype=bool)
            has_any_constraint = False
            for roll_deg_s in np.linspace(0.0, 360.0, 36, endpoint=False):
                rr = np.deg2rad(float(roll_deg_s))
                c_s, s_s = float(np.cos(rr)), float(np.sin(rr))
                y_hat_s = y0 * c_s - z0 * s_s
                z_hat_s = y0 * s_s + z0 * c_s
                mask = _eval_trackers(y_hat_s, z_hat_s)
                if mask is None:
                    continue
                has_any_constraint = True
                per_point_clear |= ~mask  # clear at this roll → not in exclusion zone
            if not has_any_constraint:
                return None
            return cast(npt.NDArray[np.bool_], ~per_point_clear)

        # ------------------------------------------------------------------ #
        # Default: evaluate at optimal-power roll per sky point               #
        # ------------------------------------------------------------------ #
        # Compute/cache an optimal roll per sky point so star tracker masks represent
        # where constraints would be encountered when panel pointing is optimized.
        ephem_idx = ephem.index(time_dt)
        sun_ra = float(ephem.sun_ra_deg[ephem_idx])
        sun_dec = float(ephem.sun_dec_deg[ephem_idx])
        bucket_step = max(1e-6, self._optimal_roll_cache_step_deg)
        roll_cache_key = (
            n_ra,
            n_dec,
            int(np.round(sun_ra / bucket_step)),
            int(np.round(sun_dec / bucket_step)),
            self._solar_panel_signature(),
        )

        roll_deg_per_point = self._optimal_roll_cache.get(roll_cache_key)
        if roll_deg_per_point is None:
            sun_vec = np.asarray(
                ephem.sun_pv.position[ephem_idx], dtype=np.float64
            ) - np.asarray(ephem.gcrs_pv.position[ephem_idx], dtype=np.float64)
            sun_norm = np.linalg.norm(sun_vec)
            if sun_norm <= 0.0:
                return None
            sun_unit = sun_vec / sun_norm

            sx0 = x_hat @ sun_unit
            sy0 = y0 @ sun_unit
            sz0 = z0 @ sun_unit

            solar_panel = getattr(self.ditl.config, "solar_panel", None)
            if (
                solar_panel is not None
                and hasattr(solar_panel, "panels")
                and isinstance(solar_panel.panels, list)
                and len(solar_panel.panels) > 0
            ):
                panel_normals: list[npt.NDArray[np.float64]] = []
                panel_weights: list[float] = []
                default_eff = float(getattr(solar_panel, "conversion_efficiency", 1.0))

                for panel in solar_panel.panels:
                    normal = getattr(panel, "normal", None)
                    max_power = getattr(panel, "max_power", None)
                    if normal is None or max_power is None:
                        continue

                    panel_normals.append(np.asarray(normal, dtype=np.float64))
                    panel_eff = getattr(panel, "conversion_efficiency", None)
                    eff_val = float(panel_eff) if panel_eff is not None else default_eff
                    panel_weights.append(float(max_power) * eff_val)

                if len(panel_normals) == 0 or len(panel_weights) == 0:
                    roll_rad_per_point = np.arctan2(-sy0, sz0)
                    roll_deg_per_point = (roll_rad_per_point / DTOR) % 360.0
                else:
                    n_mat = np.asarray(panel_normals, dtype=np.float64)  # (P, 3)
                    w_vec = np.asarray(panel_weights, dtype=np.float64)  # (P,)

                    deg_grid = np.arange(360.0, dtype=np.float64)
                    ang_grid = deg_grid * DTOR
                    cos_grid = np.cos(ang_grid)[None, :]  # (1, 360)
                    sin_grid = np.sin(ang_grid)[None, :]  # (1, 360)

                    totals = np.zeros((len(ra_flat), 360), dtype=np.float64)
                    for p_idx in range(n_mat.shape[0]):
                        nx, ny, nz = n_mat[p_idx]
                        a_coef = nx * sx0
                        b_coef = ny * sy0 + nz * sz0
                        c_coef = ny * sz0 - nz * sy0
                        illum = (
                            a_coef[:, None]
                            + b_coef[:, None] * cos_grid
                            + c_coef[:, None] * sin_grid
                        )
                        np.maximum(illum, 0.0, out=illum)
                        totals += illum * w_vec[p_idx]

                    best_idx = np.argmax(totals, axis=1)
                    roll_deg_per_point = deg_grid[best_idx]
            else:
                roll_rad_per_point = np.arctan2(-sy0, sz0)
                roll_deg_per_point = (roll_rad_per_point / DTOR) % 360.0

            self._optimal_roll_cache[roll_cache_key] = roll_deg_per_point
            while len(self._optimal_roll_cache) > self._optimal_roll_cache_max_entries:
                self._optimal_roll_cache.pop(next(iter(self._optimal_roll_cache)))

        # Apply per-point roll as position-angle rotations around boresight (+X).
        roll_rad_per_point = np.deg2rad(roll_deg_per_point)
        c = np.cos(roll_rad_per_point)[:, None]
        s = np.sin(roll_rad_per_point)[:, None]
        y_hat = y0 * c - z0 * s
        z_hat = y0 * s + z0 * c

        return _eval_trackers(y_hat, z_hat)

    def _plot_constraint_mask(
        self,
        name: str,
        color: str,
        constrained_flat: npt.NDArray[np.bool_],
    ) -> None:
        """Render a precomputed 1D boolean constraint mask on the sky grid."""
        n_ra = self.n_grid_points * 2
        n_dec = self.n_grid_points
        ra_grid_rad, dec_grid_rad, _, _ = self._create_regular_sky_grid(n_ra, n_dec)

        constrained_2d = constrained_flat.reshape((n_dec, n_ra)).astype(float)
        if not constrained_2d.any():
            return

        sigma = max(1.0, self.n_grid_points / 50)
        smoothed = _gaussian_smooth(constrained_2d, sigma=sigma)

        self.ax.contourf(
            ra_grid_rad,
            dec_grid_rad,
            smoothed,
            levels=[0.5, 1.0],
            colors=[mcolors.to_rgba(color, self.constraint_alpha)],
            zorder=1,
            antialiased=True,
        )

        edge_smoothed = _gaussian_smooth(constrained_2d, sigma=0.8)
        self.ax.contour(
            ra_grid_rad,
            dec_grid_rad,
            edge_smoothed,
            levels=[0.5],
            colors=[color],
            linewidths=1.0,
            zorder=1.1,
        )

        self.ax.plot(
            [],
            [],
            marker="s",
            markersize=10,
            markerfacecolor=mcolors.to_rgba(color, self.constraint_alpha),
            markeredgecolor="none",
            linestyle="none",
            label=f"{name} Cons.",
        )

    def _plot_earth_disk(self, utime: float) -> None:
        """Plot the physical extent of Earth as seen from the spacecraft using a filled contour (``contourf``).

        Parameters
        ----------
        utime : float
            Unix timestamp for Earth position calculation.
        """

        dt = dtutcfromtimestamp(utime)
        ephem = self.ditl.ephem
        idx = ephem.index(dt)

        # Get Earth position and angular radius from pre-computed arrays
        earth_ra = ephem.earth_ra_deg[idx]
        earth_dec = ephem.earth_dec_deg[idx]
        earth_angular_radius = ephem.earth_radius_deg[idx]

        # Grid dimensions for pcolormesh
        n_ra = self.n_grid_points * 2
        n_dec = self.n_grid_points

        # Get regular grid for contourf
        if hasattr(self, "_constraint_cache") and self._constraint_cache.get(
            "grid_shape"
        ) == (n_ra, n_dec):
            ra_flat = self._constraint_cache["ra_grid"]
            dec_flat: npt.NDArray[np.float64] = self._constraint_cache["dec_grid"]
            ra_grid_rad = self._constraint_cache["ra_grid_rad"]
            dec_grid_rad = self._constraint_cache["dec_grid_rad"]
        else:
            ra_grid_rad, dec_grid_rad, ra_flat, dec_flat = (
                self._create_regular_sky_grid(n_ra, n_dec)
            )

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

        # Reshape to 2D grid for contourf
        inside_earth_2d = inside_earth.reshape((n_dec, n_ra)).astype(float)

        # Plot Earth disk using contourf for smooth, solid appearance
        if inside_earth_2d.any():
            # Apply Gaussian smoothing for smooth filled region appearance
            sigma = max(1.0, self.n_grid_points / 50)
            smoothed = _gaussian_smooth(inside_earth_2d, sigma=sigma)

            self.ax.contourf(
                ra_grid_rad,
                dec_grid_rad,
                smoothed,
                levels=[0.5, 1.0],  # Threshold at 0.5 for smooth boundary
                colors=[mcolors.to_rgba("darkblue", 0.8)],
                zorder=2.5,
                antialiased=True,
            )

            # Draw hard contour line at the TRUE edge using original binary mask
            # Apply minimal smoothing just for anti-aliasing the line
            edge_smoothed = _gaussian_smooth(inside_earth_2d, sigma=0.8)
            self.ax.contour(
                ra_grid_rad,
                dec_grid_rad,
                edge_smoothed,
                levels=[0.5],
                colors=["darkblue"],
                linewidths=1.5,
                zorder=2.6,
            )

            # Add a proxy artist for the legend (use plot with marker for compatibility with Mollweide)
            self.ax.plot(
                [],
                [],
                marker="s",
                markersize=10,
                markerfacecolor=mcolors.to_rgba("darkblue", 0.8),
                markeredgecolor="none",
                linestyle="none",
                label="Earth Disk",
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

    def _create_regular_sky_grid(
        self, n_ra: int, n_dec: int
    ) -> tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
    ]:
        """Create a regular rectangular grid for contourf plotting.

        Parameters
        ----------
        n_ra : int
            Number of RA points.
        n_dec : int
            Number of Dec points.

        Returns
        -------
        ra_grid_rad : array
            2D RA grid in radians for plotting (n_dec, n_ra).
        dec_grid_rad : array
            2D Dec grid in radians for plotting (n_dec, n_ra).
        ra_flat : array
            Flattened RA coordinates at grid points (0-360 degrees).
        dec_flat : array
            Flattened Dec coordinates at grid points (-90-90 degrees).
        """
        # Create grid coordinates
        # Use slightly smaller range for Dec to avoid numerical issues at poles with Mollweide projection
        ra_coords = np.linspace(0, 360, n_ra, endpoint=False)
        dec_coords = np.linspace(-89.5, 89.5, n_dec)

        # Create 2D meshgrid
        ra_grid, dec_grid = np.meshgrid(ra_coords, dec_coords)

        # Convert RA for plotting (-180 to 180)
        ra_grid_plot = ra_grid - 180

        # Convert to radians for mollweide projection
        ra_grid_rad = np.deg2rad(ra_grid_plot)
        dec_grid_rad = np.deg2rad(dec_grid)

        # Create flattened coordinates for constraint evaluation
        ra_flat = ra_grid.flatten()
        dec_flat = dec_grid.flatten()

        return ra_grid_rad, dec_grid_rad, ra_flat, dec_flat

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
        """Plot a single constraint region using pcolormesh for smooth appearance.

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
        body_ra : float or None
            RA of celestial body (for marker).
        body_dec : float or None
            Dec of celestial body (for marker).
        """
        # Grid dimensions for pcolormesh (use 2x n_grid_points for RA due to 360 degree range)
        n_ra = self.n_grid_points * 2
        n_dec = self.n_grid_points

        # Check if we have pre-computed constraints with matching grid
        use_cache = False
        if (
            hasattr(self, "_constraint_cache")
            and name.lower() in self._constraint_cache["constraints"]
            and self._constraint_cache.get("grid_shape") == (n_ra, n_dec)
        ):
            cache = self._constraint_cache
            # Find time index in cache
            time_idx = self._find_time_index(utime)
            cache_time_idx = np.where(cache["time_indices"] == time_idx)[0]
            if (
                len(cache_time_idx) > 0
                and cache_time_idx[0] < cache["constraints"][name.lower()].shape[1]
            ):
                use_cache = True
                cache_time_idx = cache_time_idx[0]
                # Cache shape is (n_points, n_times), so index with [:, time_idx]
                constrained_flat = cache["constraints"][name.lower()][:, cache_time_idx]
                constrained_flat = np.asarray(constrained_flat, dtype=bool)
                ra_grid_rad = cache["ra_grid_rad"]
                dec_grid_rad = cache["dec_grid_rad"]

        if not use_cache:
            # Fall back to real-time evaluation
            ra_grid_rad, dec_grid_rad, ra_flat, dec_flat = (
                self._create_regular_sky_grid(n_ra, n_dec)
            )

            constrained_flat = constraint_func.in_constraint_batch(
                ephemeris=self.ditl.ephem,
                target_ras=ra_flat.tolist(),
                target_decs=dec_flat.tolist(),
                times=[dtutcfromtimestamp(utime)],
            )[:, 0]

        # Reshape constraint mask to 2D grid for contourf
        constrained_2d = constrained_flat.reshape((n_dec, n_ra)).astype(float)

        # Only plot if there are constrained regions
        if constrained_2d.any():
            # Apply Gaussian smoothing for smooth filled region appearance
            # sigma controls smoothness: higher = smoother edges
            sigma = max(1.0, self.n_grid_points / 50)
            smoothed = _gaussian_smooth(constrained_2d, sigma=sigma)

            # Plot using contourf with smooth edges
            self.ax.contourf(
                ra_grid_rad,
                dec_grid_rad,
                smoothed,
                levels=[0.5, 1.0],  # Threshold at 0.5 for smooth boundary
                colors=[mcolors.to_rgba(color, self.constraint_alpha)],
                zorder=1,
                antialiased=True,
            )

            # Draw hard contour line at the TRUE edge using original binary mask
            # Apply minimal smoothing just for anti-aliasing the line
            edge_smoothed = _gaussian_smooth(constrained_2d, sigma=0.8)
            self.ax.contour(
                ra_grid_rad,
                dec_grid_rad,
                edge_smoothed,
                levels=[0.5],
                colors=[color],
                linewidths=1.0,
                zorder=1.1,
            )

            # Add a proxy artist for the legend (use plot with marker for compatibility with Mollweide)
            self.ax.plot(
                [],
                [],
                marker="s",
                markersize=10,
                markerfacecolor=mcolors.to_rgba(color, self.constraint_alpha),
                markeredgecolor="none",
                linestyle="none",
                label=f"{name} Cons.",
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
    progress_wrapper: Any = None
    if show_progress:
        try:
            module_name = "tqdm"
            tqdm_module = import_module(module_name)
            progress_wrapper = getattr(tqdm_module, "tqdm")
        except (ImportError, AttributeError):
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
            if progress_wrapper is not None:
                iterator = progress_wrapper(
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
