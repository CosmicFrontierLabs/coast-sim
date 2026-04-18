"""Sky Pointing Visualization

Interactive visualization showing spacecraft pointing on a mollweide projection
of the sky with scheduled observations and constraint regions.
"""

# pyright: reportMissingTypeStubs=false

from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import numpy as np
import numpy.typing as npt
import plotly.graph_objects as go
import rust_ephem

from ...common import dtutcfromtimestamp
from ...config.observation_categories import ObservationCategories
from ...config.visualization import VisualizationConfig
from ._constraint_helpers import (
    ConstraintPlotConfig,
    build_body_polygon_traces,
    build_optional_figure_traces,
    build_tail_constraint_traces,
    resolve_constraint_plot_config,
)
from ._helpers import (
    _marker_trace,
    _poly_trace,
    _ra_to_lon,
    _sky_circle_polygon,
    _to_rgba,
)
from ._helpers import (
    lighten_color as _lighten_color,
)
from .globe_pointing import (
    _build_st_boresights,
    _get_vis_config,
)

if TYPE_CHECKING:
    from ...ditl import DITL, QueueDITL


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


# ---------------------------------------------------------------------------
# Flat sky-map (Plotly/natural-earth) — Mollweide-style 2-D view
# ---------------------------------------------------------------------------


def plot_sky_pointing_plotly(
    ditl: "DITL | QueueDITL",
    n_frames: int | None = None,
    constraint_alpha: float = 0.35,
    config: VisualizationConfig | None = None,
    observation_categories: ObservationCategories | None = None,
) -> go.Figure:
    """Plot spacecraft pointing on an animated 2-D sky map using Plotly.

    Uses a ``natural earth`` flat projection (RA → longitude, Dec → latitude)
    so the result looks similar to the Matplotlib Mollweide view while
    remaining a fully interactive Plotly figure.

    Coordinate convention: Dec → latitude, RA → longitude (RA – 180°);
    RA = 180° appears at the centre of the map and RA = 0°/360° at the edges.

    What is shown
    -------------
    - All scheduled observations (static, colour-coded by category).
    - Sun / Moon / Earth exclusion-zone circles (animated).
    - Earth physical disk (animated).
    - Anti-Sun direction marker and exclusion zone (animated, orange).
    - Panel constraint exclusion zones (animated, green; hidden during eclipse).
    - Orbit (RAM direction) exclusion zone (animated, when configured).
    - Current spacecraft pointing — star marker, colour = ACS mode (animated).
    - Star-tracker boresights — hexagon markers (animated).
    - Star-tracker body-exclusion-zone circles (animated, hard=red, soft=magenta).

    Parameters
    ----------
    ditl : DITL or QueueDITL
        Completed DITL simulation object (``calc()`` must have been run first).
    n_frames : int, optional
        Maximum number of animation frames.  Defaults to
        ``min(len(ditl.utime), 300)``.  Reduce for lighter notebooks.
    constraint_alpha : float, optional
        Fill opacity of exclusion polygons (default: 0.35).
    config : VisualizationConfig, optional
        Visualisation config; falls back to ``ditl.config.visualization``.
    observation_categories : ObservationCategories, optional
        Observation-category colour map; falls back to
        ``ditl.config.observation_categories``.

    Returns
    -------
    go.Figure
        Interactive Plotly figure.  Display with ``fig.show()`` or evaluate the
        cell in Jupyter to render it inline.

    Examples
    --------
    >>> fig = plot_sky_pointing_plotly(ditl)
    >>> fig.show()
    """
    # ------------------------------------------------------------------
    # Validate
    # ------------------------------------------------------------------
    if not hasattr(ditl, "plan") or len(ditl.plan) == 0:
        raise ValueError("DITL has no pointings — run calc() first.")
    if not hasattr(ditl, "utime") or len(ditl.utime) == 0:
        raise ValueError("DITL has no time data — run calc() first.")
    if ditl.constraint.ephem is None:
        raise ValueError("DITL constraint has no ephemeris set.")

    vis_config = _get_vis_config(ditl, config)

    if observation_categories is None:
        if hasattr(ditl, "config") and hasattr(ditl.config, "observation_categories"):
            observation_categories = ditl.config.observation_categories
    if observation_categories is None:
        observation_categories = ObservationCategories.default_categories()

    ephem = ditl.constraint.ephem
    utimes = ditl.utime
    n_total = len(utimes)

    # ------------------------------------------------------------------
    # Frame subsampling
    # ------------------------------------------------------------------
    max_frames = n_frames if n_frames is not None else min(n_total, 300)
    step = max(1, n_total // max_frames)
    frame_indices: list[int] = list(range(0, n_total, step))

    # ------------------------------------------------------------------
    # ACS mode colours & star-tracker config
    # ------------------------------------------------------------------
    mode_colors: dict[str, str] = (
        vis_config.mode_colors
        if vis_config and hasattr(vis_config, "mode_colors")
        else {
            "SCIENCE": "green",
            "SLEWING": "orange",
            "SAA": "purple",
            "PASS": "cyan",
            "CHARGING": "yellow",
            "SAFE": "red",
        }
    )

    raw_trackers: list[Any] = []
    if hasattr(ditl, "config") and hasattr(ditl.config, "spacecraft_bus"):
        _st = getattr(ditl.config.spacecraft_bus, "star_trackers", None)
        if _st is not None and hasattr(_st, "star_trackers"):
            raw_trackers = list(getattr(_st, "star_trackers", []) or [])
    n_trackers = len(raw_trackers)
    _st_colors = ["cyan", "lime", "orange", "hotpink", "white", "yellow"]

    # ------------------------------------------------------------------
    # Resolve constraint plotting configuration (once, not per frame)
    # ------------------------------------------------------------------
    cc: ConstraintPlotConfig = resolve_constraint_plot_config(ditl, raw_trackers, ephem)
    _eclipse_con = rust_ephem.EclipseConstraint()

    # ------------------------------------------------------------------
    # Pre-compute radiator hard-constraint exclusion specs
    # ------------------------------------------------------------------
    radiator_constraint_specs: list[tuple[str, float]] = []  # (body, min_angle_deg)
    _rad_bus = getattr(
        getattr(getattr(ditl, "config", None), "spacecraft_bus", None),
        "radiators",
        None,
    )
    if _rad_bus is not None:
        for _rad in getattr(_rad_bus, "radiators", []):
            _hc = getattr(_rad, "hard_constraint", None)
            if _hc is None:
                continue
            for _body in ("sun", "earth", "moon"):
                _bcfg = getattr(_hc, f"{_body}_constraint", None)
                if _bcfg is None:
                    continue
                _mangle = getattr(_bcfg, "min_angle", None)
                if _mangle is not None and float(_mangle) > 0:
                    radiator_constraint_specs.append((_body, float(_mangle)))

    def _mode_color(idx: int) -> str:
        m = ditl.mode[idx]
        return mode_colors.get(m.name if hasattr(m, "name") else str(m), "red")

    # ------------------------------------------------------------------
    # Per-frame data builder
    # Let _B = 10+n_trackers+n_st_specs, _O = _B+(3 if orbit else 0),
    #         _A = _O+(3 if anti_sun else 0)
    # trace ordering:
    #   0: Sun min excl polygon   1: Sun max excl polygon   2: Sun body marker
    #   3: Moon min excl polygon  4: Moon max excl polygon  5: Moon body marker
    #   6: Earth min excl polygon 7: Earth max excl polygon 8: Earth disk polygon
    #   9: Pointing marker
    #   10 … 9+n_trackers: ST boresight markers
    #   10+n_trackers … 9+n_trackers+n_st_specs: ST exclusion-zone circles
    #   _B+0: Orbit RAM min exclusion     [if orbit configured]
    #   _B+1: Orbit RAM max exclusion     [if orbit configured]
    #   _B+2: RAM direction marker        [if orbit configured]
    #   _O+0: Anti-Sun min excl polygon   [if anti_sun configured]
    #   _O+1: Anti-Sun max excl polygon   [if anti_sun configured]
    #   _O+2: Anti-Sun direction marker   [if anti_sun configured]
    #   _A+0: Panel min excl polygon      [if panel configured]
    #   _A+1: Panel max excl polygon      [if panel configured]
    #   _A+2 … : Radiator hard exclusion-zone circles
    # ------------------------------------------------------------------
    def _frame_traces(idx: int) -> list[dict[str, Any]]:
        dt = dtutcfromtimestamp(utimes[idx])
        ei = ephem.index(dt)

        sun_ra = float(ephem.sun_ra_deg[ei])
        sun_dec = float(ephem.sun_dec_deg[ei])
        moon_ra = float(ephem.moon_ra_deg[ei])
        moon_dec = float(ephem.moon_dec_deg[ei])
        earth_ra = float(ephem.earth_ra_deg[ei])
        earth_dec = float(ephem.earth_dec_deg[ei])
        earth_disk_r = float(ephem.earth_radius_deg[ei])

        ra = float(ditl.ra[idx])
        dec = float(ditl.dec[idx])
        mc = _mode_color(idx)

        hk_records = (
            list(ditl.telemetry.housekeeping)
            if hasattr(ditl, "telemetry") and hasattr(ditl.telemetry, "housekeeping")
            else []
        )
        hk_status: list[bool] | None = None
        if idx < len(hk_records):
            hk_status = hk_records[idx].star_tracker_status

        st_positions = _build_st_boresights(ditl, idx)

        traces = build_body_polygon_traces(
            sun_ra,
            sun_dec,
            moon_ra,
            moon_dec,
            earth_ra,
            earth_dec,
            earth_disk_r,
            cc.body_angle_cfg,
        )
        traces.append(
            {
                "lon": [_ra_to_lon(ra)],
                "lat": [dec],
                "marker": {"color": mc, "size": 20, "symbol": "star"},
            }
        )
        for i in range(n_trackers):
            if i < len(st_positions):
                st_ra, st_dec, _color, _name = st_positions[i]
                st_marker_color = (
                    ("limegreen" if hk_status[i] else "red")
                    if hk_status is not None and i < len(hk_status)
                    else _color
                )
                traces.append(
                    {
                        "lon": [_ra_to_lon(st_ra)],
                        "lat": [st_dec],
                        "marker": {"color": st_marker_color},
                    }
                )
            else:
                traces.append({"lon": [], "lat": []})

        traces.extend(
            build_tail_constraint_traces(
                cc,
                sun_ra,
                sun_dec,
                moon_ra,
                moon_dec,
                earth_ra,
                earth_dec,
                earth_disk_r,
                ei,
                dt,
                _eclipse_con,
                ephem,
            )
        )

        for body, min_angle in radiator_constraint_specs:
            if body == "sun":
                body_ra_r, body_dec_r = sun_ra, sun_dec
                extra_r = 0.0
            elif body == "earth":
                body_ra_r, body_dec_r = earth_ra, earth_dec
                extra_r = earth_disk_r
            elif body == "moon":
                body_ra_r, body_dec_r = moon_ra, moon_dec
                extra_r = 0.0
            else:
                traces.append({"lon": [], "lat": []})
                continue
            eff_r = min_angle + extra_r
            if eff_r > 0:
                c_lons, c_lats = _sky_circle_polygon(body_ra_r, body_dec_r, eff_r)
            else:
                c_lons, c_lats = [], []
            traces.append({"lon": c_lons, "lat": c_lats})

        return traces

    # ------------------------------------------------------------------
    # Static observations trace
    # ------------------------------------------------------------------
    obs_lons: list[float] = []
    obs_lats: list[float] = []
    obs_colors: list[str] = []
    obs_texts: list[str] = []
    for ppt in ditl.plan:
        obs_lons.append(_ra_to_lon(float(ppt.ra)))
        obs_lats.append(float(ppt.dec))
        base_color = "steelblue"
        try:
            if observation_categories is not None and hasattr(ppt, "obsid"):
                cat = observation_categories.get_category(ppt.obsid)
                if hasattr(cat, "color") and isinstance(cat.color, str):
                    base_color = cat.color
        except Exception:
            pass
        obs_colors.append(mcolors.to_hex(base_color))
        obs_texts.append(f"Obs {getattr(ppt, 'obsid', '')}")

    # ------------------------------------------------------------------
    # Build initial trace list
    # ------------------------------------------------------------------
    init_idx = frame_indices[0]
    init_data = _frame_traces(init_idx)
    mc0 = _mode_color(init_idx)

    traces_fig: list[Any] = [
        # Trace 0: static observations
        go.Scattergeo(
            lon=obs_lons,
            lat=obs_lats,
            mode="markers",
            marker=dict(
                color=obs_colors,
                size=5,
                opacity=0.75,
                line=dict(color="black", width=0.5),
            ),
            text=obs_texts,
            hoverinfo="text",
            name="Observations",
            showlegend=True,
        ),
        # Traces 1–2: Sun exclusion polygons
        _poly_trace(
            init_data[0],
            "Sun min exclusion",
            _to_rgba("gold", constraint_alpha),
            "gold",
        ),
        _poly_trace(
            init_data[1],
            "Sun max exclusion",
            _to_rgba(_lighten_color("gold", 0.3), constraint_alpha * 0.8),
            _lighten_color("gold", 0.2),
            showlegend=cc.body_angle_cfg["sun"][1] is not None,
        ),
        # Trace 3: Sun body marker
        _marker_trace(init_data[2], "Sun", "circle", "yellow", 16),
        # Traces 4–5: Moon exclusion polygons
        _poly_trace(
            init_data[3],
            "Moon exclusion",
            _to_rgba("gray", constraint_alpha),
            "lightgray",
        ),
        _poly_trace(
            init_data[4],
            "Moon max exclusion",
            _to_rgba(_lighten_color("gray", 0.25), constraint_alpha * 0.75),
            _lighten_color("gray", 0.15),
            showlegend=cc.body_angle_cfg["moon"][1] is not None,
        ),
        # Trace 6: Moon body marker
        _marker_trace(init_data[5], "Moon", "circle", "lightgray", 12),
        # Traces 7–8: Earth exclusion polygons
        _poly_trace(
            init_data[6],
            "Earth exclusion",
            _to_rgba("royalblue", constraint_alpha),
            "dodgerblue",
        ),
        _poly_trace(
            init_data[7],
            "Earth max exclusion",
            _to_rgba(_lighten_color("royalblue", 0.25), constraint_alpha * 0.75),
            _lighten_color("royalblue", 0.15),
            showlegend=cc.body_angle_cfg["earth"][1] is not None,
        ),
        # Trace 9: Earth physical disk
        _poly_trace(
            init_data[8], "Earth disk", _to_rgba("darkblue", 0.75), "cornflowerblue"
        ),
        # Trace 10: Pointing marker
        go.Scattergeo(
            lon=init_data[9]["lon"],
            lat=init_data[9]["lat"],
            mode="markers",
            marker=dict(
                symbol="star",
                color=mc0,
                size=20,
                line=dict(color="white", width=2),
            ),
            name="Pointing",
            showlegend=True,
        ),
    ]

    # ST boresight traces
    for i in range(n_trackers):
        st_name = getattr(raw_trackers[i], "name", f"ST-{i + 1}")
        td = init_data[10 + i] if 10 + i < len(init_data) else {"lon": [], "lat": []}
        st_color = td.get("marker", {}).get("color", _st_colors[i % len(_st_colors)])
        traces_fig.append(
            go.Scattergeo(
                lon=td["lon"],
                lat=td["lat"],
                mode="markers",
                marker=dict(
                    symbol="hexagon",
                    color=st_color,
                    size=14,
                    line=dict(color="black", width=1.5),
                ),
                name=st_name,
                showlegend=True,
            )
        )

    # ST constraint circles, orbit, anti-sun, and panel traces
    traces_fig.extend(
        build_optional_figure_traces(cc, init_data, n_trackers, constraint_alpha)
    )

    # Radiator hard constraint exclusion-zone circle traces
    n_st_specs = len(cc.st_constraint_specs)
    _rad_base = (
        10 + n_trackers + n_st_specs
        + (3 if cc.orbit_constraint is not None else 0)
        + (3 if cc.anti_sun_constraint_cfg is not None else 0)
        + (2 if cc.panel_constraint_cfg is not None else 0)
    )
    _rad_legend_shown = False
    for k, (body, min_angle) in enumerate(radiator_constraint_specs):
        ci = _rad_base + k
        td_r = init_data[ci] if ci < len(init_data) else {"lon": [], "lat": []}
        traces_fig.append(
            go.Scattergeo(
                lon=td_r["lon"],
                lat=td_r["lat"],
                mode="lines",
                fill="toself",
                fillcolor=_to_rgba("coral", constraint_alpha),
                line=dict(color="coral", width=1, dash="dash"),
                name="Radiator KOZ",
                showlegend=not _rad_legend_shown,
                hoverinfo="skip",
            )
        )
        _rad_legend_shown = True

    animated_indices = list(range(1, len(traces_fig)))

    # ------------------------------------------------------------------
    # Build animation frames and slider steps
    # ------------------------------------------------------------------
    frames: list[go.Frame] = []
    slider_steps: list[dict[str, Any]] = []

    for fi, idx in enumerate(frame_indices):
        dt = dtutcfromtimestamp(utimes[idx])
        time_str = dt.strftime("%Y-%m-%d %H:%M UTC")
        label_str = dt.strftime("%H:%M")
        fdata = _frame_traces(idx)

        frames.append(
            go.Frame(
                data=[go.Scattergeo(**td) for td in fdata],
                traces=animated_indices,
                name=str(fi),
                layout=go.Layout(title_text=f"Sky Pointing — {time_str}"),
            )
        )
        slider_steps.append(
            {
                "args": [
                    [str(fi)],
                    {
                        "frame": {"duration": 0, "redraw": True},
                        "mode": "immediate",
                        "transition": {"duration": 0},
                    },
                ],
                "label": label_str,
                "method": "animate",
            }
        )

    # ------------------------------------------------------------------
    # Layout — flat natural-earth projection, dark background
    # ------------------------------------------------------------------
    init_time_str = dtutcfromtimestamp(utimes[init_idx]).strftime("%Y-%m-%d %H:%M UTC")
    layout = go.Layout(
        title=dict(
            text=f"Sky Pointing — {init_time_str}",
            font=dict(color="white", size=14),
        ),
        geo=dict(
            projection_type="natural earth",
            showland=False,
            showocean=False,
            showframe=True,
            framecolor="rgba(180,180,180,0.6)",
            bgcolor="rgb(10, 10, 25)",
            lataxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.12)",
                dtick=30,
            ),
            lonaxis=dict(
                showgrid=True,
                gridcolor="rgba(255,255,255,0.12)",
                dtick=30,
            ),
            showcoastlines=False,
            showcountries=False,
            showlakes=False,
            showrivers=False,
            showsubunits=False,
        ),
        paper_bgcolor="rgb(18, 18, 36)",
        plot_bgcolor="rgb(18, 18, 36)",
        font=dict(color="white"),
        legend=dict(
            bgcolor="rgba(25,25,45,0.85)",
            bordercolor="rgba(130,130,180,0.5)",
            borderwidth=1,
            font=dict(color="white", size=11),
        ),
        height=550,
        width=1000,
        margin=dict(l=0, r=0, t=50, b=80),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=0.0,
                x=0.0,
                xanchor="left",
                yanchor="top",
                pad=dict(t=0, r=10),
                bgcolor="rgba(40,40,70,0.9)",
                bordercolor="rgba(130,130,180,0.5)",
                font=dict(color="white"),
                buttons=[
                    dict(
                        label="▶ / ⏸",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 60, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                        args2=[
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                ],
            )
        ],
        sliders=[
            dict(
                active=0,
                currentvalue=dict(
                    prefix="",
                    visible=True,
                    xanchor="center",
                    font=dict(color="white", size=11),
                ),
                pad=dict(b=10, t=30),
                len=0.82,
                x=0.12,
                y=0.0,
                steps=slider_steps,
                bgcolor="rgba(40,40,70,0.8)",
                activebgcolor="rgba(100,100,160,0.9)",
                bordercolor="rgba(130,130,180,0.4)",
                font=dict(color="white"),
                tickcolor="rgba(255,255,255,0.4)",
            )
        ],
    )

    return go.Figure(data=traces_fig, frames=frames, layout=layout)
