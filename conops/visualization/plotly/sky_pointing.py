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

from ...common import dtutcfromtimestamp
from ...config.constants import EARTH_OCCULT, MOON_OCCULT, SUN_OCCULT
from ...config.observation_categories import ObservationCategories
from ...config.visualization import VisualizationConfig
from .globe_pointing import (
    _build_st_boresights,
    _get_vis_config,
    _ra_to_lon,
    _sky_circle_polygon,
    _to_rgba,
)

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
    # Resolve body-constraint plotting angles (min and max)
    # ------------------------------------------------------------------
    constraint_cfg = None
    if hasattr(ditl, "config") and hasattr(ditl.config, "constraint"):
        constraint_cfg = ditl.config.constraint

    body_angle_cfg: dict[str, tuple[float, float | None]] = {
        "sun": (float(SUN_OCCULT), None),
        "moon": (float(MOON_OCCULT), None),
        "earth": (float(EARTH_OCCULT), None),
    }
    for _body, _default in body_angle_cfg.items():
        _cfg = (
            getattr(constraint_cfg, f"{_body}_constraint", None)
            if constraint_cfg is not None
            else None
        )
        _min_angle = getattr(_cfg, "min_angle", _default[0])
        _max_angle = getattr(_cfg, "max_angle", _default[1])
        body_angle_cfg[_body] = (
            float(_min_angle) if _min_angle is not None else _default[0],
            float(_max_angle) if _max_angle is not None else None,
        )

    # ------------------------------------------------------------------
    # Pre-compute ST constraint specs
    # ------------------------------------------------------------------
    st_constraint_specs: list[tuple[int, str, str, str, float]] = []
    for _ci, _st_raw in enumerate(raw_trackers):
        for _tier in ("hard", "soft"):
            _cobj = getattr(_st_raw, f"{_tier}_constraint", None)
            if _cobj is None:
                continue
            for _body in ("sun", "earth", "moon"):
                _bcfg = getattr(_cobj, f"{_body}_constraint", None)
                if _bcfg is None:
                    continue
                _mangle = getattr(_bcfg, "min_angle", None)
                if _mangle is not None and float(_mangle) > 0:
                    st_constraint_specs.append(
                        (_ci, _tier, _body, "min", float(_mangle))
                    )
                _xangle = getattr(_bcfg, "max_angle", None)
                if _xangle is not None:
                    _max_circle_r = 180.0 - float(_xangle)
                    if _max_circle_r > 0:
                        st_constraint_specs.append(
                            (_ci, _tier, _body, "max", float(_xangle))
                        )

    def _mode_color(idx: int) -> str:
        m = ditl.mode[idx]
        return mode_colors.get(m.name if hasattr(m, "name") else str(m), "red")

    # ------------------------------------------------------------------
    # Per-frame data builder
    # trace ordering:
    #   0: Sun min excl polygon   1: Sun max excl polygon   2: Sun body marker
    #   3: Moon min excl polygon  4: Moon max excl polygon  5: Moon body marker
    #   6: Earth min excl polygon 7: Earth max excl polygon 8: Earth disk polygon
    #   9: Pointing marker
    #   10 … 9+n_trackers: ST boresight markers
    #   10+n_trackers … : ST exclusion-zone circles
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

        def _constraint_polygons(
            body_ra: float,
            body_dec: float,
            min_angle: float,
            max_angle: float | None,
            extra_r: float = 0.0,
        ) -> tuple[list[float], list[float], list[float], list[float]]:
            min_r = max(0.0, min_angle + extra_r)
            if min_r > 0:
                min_lons, min_lats = _sky_circle_polygon(body_ra, body_dec, min_r)
            else:
                min_lons, min_lats = [], []

            if max_angle is not None:
                # A max-angle violation from a body is equivalent to a cap around anti-body.
                anti_body_ra = (body_ra + 180.0) % 360.0
                anti_body_dec = -body_dec
                max_r = max(0.0, 180.0 - (max_angle + extra_r))
                if max_r > 0:
                    max_lons, max_lats = _sky_circle_polygon(
                        anti_body_ra,
                        anti_body_dec,
                        max_r,
                    )
                else:
                    max_lons, max_lats = [], []
            else:
                max_lons, max_lats = [], []
            return min_lons, min_lats, max_lons, max_lats

        sun_min_angle, sun_max_angle = body_angle_cfg["sun"]
        moon_min_angle, moon_max_angle = body_angle_cfg["moon"]
        earth_min_angle, earth_max_angle = body_angle_cfg["earth"]

        sun_lons, sun_lats, sun_max_lons, sun_max_lats = _constraint_polygons(
            sun_ra,
            sun_dec,
            sun_min_angle,
            sun_max_angle,
        )
        moon_lons, moon_lats, moon_max_lons, moon_max_lats = _constraint_polygons(
            moon_ra,
            moon_dec,
            moon_min_angle,
            moon_max_angle,
        )
        earth_excl_lons, earth_excl_lats, earth_max_lons, earth_max_lats = (
            _constraint_polygons(
                earth_ra,
                earth_dec,
                earth_min_angle,
                earth_max_angle,
                extra_r=earth_disk_r,
            )
        )
        earth_disk_lons, earth_disk_lats = _sky_circle_polygon(
            earth_ra, earth_dec, earth_disk_r
        )

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

        traces: list[dict[str, Any]] = [
            {"lon": sun_lons, "lat": sun_lats},
            {"lon": sun_max_lons, "lat": sun_max_lats},
            {
                "lon": [_ra_to_lon(sun_ra)],
                "lat": [sun_dec],
                "marker": {"color": "yellow", "size": 16, "symbol": "circle"},
            },
            {"lon": moon_lons, "lat": moon_lats},
            {"lon": moon_max_lons, "lat": moon_max_lats},
            {
                "lon": [_ra_to_lon(moon_ra)],
                "lat": [moon_dec],
                "marker": {"color": "lightgray", "size": 12, "symbol": "circle"},
            },
            {"lon": earth_excl_lons, "lat": earth_excl_lats},
            {"lon": earth_max_lons, "lat": earth_max_lats},
            {"lon": earth_disk_lons, "lat": earth_disk_lats},
            {
                "lon": [_ra_to_lon(ra)],
                "lat": [dec],
                "marker": {"color": mc, "size": 20, "symbol": "star"},
            },
        ]

        for i in range(n_trackers):
            if i < len(st_positions):
                st_ra, st_dec, _color, _name = st_positions[i]
                if hk_status is not None and i < len(hk_status):
                    st_marker_color = "limegreen" if hk_status[i] else "red"
                else:
                    st_marker_color = _color
                traces.append(
                    {
                        "lon": [_ra_to_lon(st_ra)],
                        "lat": [st_dec],
                        "marker": {"color": st_marker_color},
                    }
                )
            else:
                traces.append({"lon": [], "lat": []})

        for tr_idx, tier, body, bound_kind, angle in st_constraint_specs:
            if body == "sun":
                body_ra, body_dec = sun_ra, sun_dec
                extra_r = 0.0
            elif body == "earth":
                body_ra, body_dec = earth_ra, earth_dec
                extra_r = earth_disk_r
            elif body == "moon":
                body_ra, body_dec = moon_ra, moon_dec
                extra_r = 0.0
            else:
                traces.append({"lon": [], "lat": []})
                continue

            if bound_kind == "min":
                eff_r = angle + extra_r
                center_ra, center_dec = body_ra, body_dec
            else:
                eff_r = 180.0 - (angle + extra_r)
                center_ra, center_dec = (body_ra + 180.0) % 360.0, -body_dec

            if eff_r > 0:
                c_lons, c_lats = _sky_circle_polygon(center_ra, center_dec, eff_r)
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

    def _poly_trace(
        data: dict[str, Any],
        name: str,
        fill_color: str,
        line_color: str,
        showlegend: bool = True,
    ) -> go.Scattergeo:
        return go.Scattergeo(
            lon=data["lon"],
            lat=data["lat"],
            mode="lines",
            fill="toself",
            fillcolor=fill_color,
            line=dict(color=line_color, width=1),
            name=name,
            showlegend=showlegend,
            hoverinfo="skip",
        )

    def _marker_trace(
        data: dict[str, Any],
        name: str,
        symbol: str,
        color: str,
        size: int,
        edge_color: str = "white",
        showlegend: bool = True,
    ) -> go.Scattergeo:
        return go.Scattergeo(
            lon=data["lon"],
            lat=data["lat"],
            mode="markers",
            marker=dict(
                symbol=symbol,
                color=color,
                size=size,
                line=dict(color=edge_color, width=1),
            ),
            name=name,
            showlegend=showlegend,
        )

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
        # Trace 1: Sun exclusion polygon
        _poly_trace(
            init_data[0],
            "Sun min exclusion",
            _to_rgba("gold", constraint_alpha),
            "gold",
        ),
        # Trace 2: Sun max exclusion polygon (if configured)
        _poly_trace(
            init_data[1],
            "Sun max exclusion",
            _to_rgba(_lighten_color("gold", 0.3), constraint_alpha * 0.8),
            _lighten_color("gold", 0.2),
            showlegend=body_angle_cfg["sun"][1] is not None,
        ),
        # Trace 3: Sun body marker
        _marker_trace(init_data[2], "Sun", "circle", "yellow", 16),
        # Trace 4: Moon exclusion polygon
        _poly_trace(
            init_data[3],
            "Moon exclusion",
            _to_rgba("gray", constraint_alpha),
            "lightgray",
        ),
        # Trace 5: Moon max exclusion polygon (if configured)
        _poly_trace(
            init_data[4],
            "Moon max exclusion",
            _to_rgba(_lighten_color("gray", 0.25), constraint_alpha * 0.75),
            _lighten_color("gray", 0.15),
            showlegend=body_angle_cfg["moon"][1] is not None,
        ),
        # Trace 6: Moon body marker
        _marker_trace(init_data[5], "Moon", "circle", "lightgray", 12),
        # Trace 7: Earth exclusion polygon
        _poly_trace(
            init_data[6],
            "Earth exclusion",
            _to_rgba("royalblue", constraint_alpha),
            "dodgerblue",
        ),
        # Trace 8: Earth max exclusion polygon (if configured)
        _poly_trace(
            init_data[7],
            "Earth max exclusion",
            _to_rgba(_lighten_color("royalblue", 0.25), constraint_alpha * 0.75),
            _lighten_color("royalblue", 0.15),
            showlegend=body_angle_cfg["earth"][1] is not None,
        ),
        # Trace 9: Earth physical disk
        _poly_trace(
            init_data[8],
            "Earth disk",
            _to_rgba("darkblue", 0.75),
            "cornflowerblue",
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

    # ST constraint exclusion-zone circle traces
    _st_legend_shown: set[str] = set()
    for j, (tr_idx, tier, body, bound_kind, angle) in enumerate(st_constraint_specs):
        _legend_key = f"{tier}_{bound_kind}"
        _show_legend = _legend_key not in _st_legend_shown
        if _show_legend:
            _st_legend_shown.add(_legend_key)

        if tier == "hard":
            fill_color = _to_rgba("red", constraint_alpha)
            line_color = "red"
            line_dash = "solid" if bound_kind == "min" else "dash"
            label = "ST Hard Min Cons." if bound_kind == "min" else "ST Hard Max Cons."
        else:
            fill_color = _to_rgba("magenta", constraint_alpha * 0.7)
            line_color = "magenta"
            line_dash = "dot" if bound_kind == "min" else "dashdot"
            label = "ST Soft Min Cons." if bound_kind == "min" else "ST Soft Max Cons."

        ci = 10 + n_trackers + j
        td_c = init_data[ci] if ci < len(init_data) else {"lon": [], "lat": []}
        traces_fig.append(
            go.Scattergeo(
                lon=td_c["lon"],
                lat=td_c["lat"],
                mode="lines",
                fill="toself",
                fillcolor=fill_color,
                line=dict(color=line_color, width=1, dash=line_dash),
                name=label,
                showlegend=_show_legend,
                hoverinfo="skip",
            )
        )

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
