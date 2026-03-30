"""Globe Sky Pointing Visualization

Rotatable 3D celestial-sphere globe using Plotly's orthographic scattergeo
projection.  All scheduled observations, constraint exclusion zones, the current
spacecraft pointing, and star-tracker boresights are shown.  A Plotly slider lets
you step through the simulation timeline, and the globe can be rotated freely by
click-dragging.

Coordinate mapping
------------------
  lat  = Declination   (degrees, –90 … +90)
  lon  = RA – 180°     (degrees, –180 … +180)

This puts RA = 0° at lon = –180° (the "back" of the default globe view) and
RA = 180° at lon = 0° (the "front").  The user can spin the globe to any
orientation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go

from ..common import dtutcfromtimestamp
from ..config.constants import EARTH_OCCULT, MOON_OCCULT, SUN_OCCULT
from ..config.observation_categories import ObservationCategories
from ..config.visualization import VisualizationConfig

if TYPE_CHECKING:
    from ..ditl import DITL, QueueDITL


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _get_vis_config(
    ditl: DITL | QueueDITL, config: VisualizationConfig | None
) -> VisualizationConfig:
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


def _to_rgba(color: str, alpha: float) -> str:
    """Convert a named / hex colour + alpha value to a Plotly ``rgba()`` string."""
    rgb = mcolors.to_rgb(color)
    r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _ra_to_lon(ra_deg: float) -> float:
    """Map RA (0–360 °) to Plotly scattergeo longitude (–180 … +180 °)."""
    return float(ra_deg) - 180.0


def _sky_circle_polygon(
    ra0_deg: float,
    dec0_deg: float,
    r_deg: float,
    n: int = 120,
) -> tuple[list[float], list[float]]:
    """Return (lons, lats) in degrees for a small-circle polygon on the sky.

    Uses the standard spherical-offset formula.  The polygon is automatically
    closed (last point == first point).

    Parameters
    ----------
    ra0_deg, dec0_deg : float
        Centre of the circle in RA/Dec (degrees).
    r_deg : float
        Angular radius of the circle (degrees).
    n : int
        Number of polygon vertices (more → smoother circle).

    Returns
    -------
    lons : list[float]
        Longitudes in [–180, 180] degrees (RA – 180°).
    lats : list[float]
        Latitudes in [–90, 90] degrees (= Dec).
    """
    ra0 = np.radians(ra0_deg)
    dec0 = np.radians(dec0_deg)
    r = np.radians(r_deg)

    phi = np.linspace(0.0, 2.0 * np.pi, n + 1)  # +1 closes the polygon

    sin_dec = np.sin(dec0) * np.cos(r) + np.cos(dec0) * np.sin(r) * np.cos(phi)
    sin_dec = np.clip(sin_dec, -1.0, 1.0)
    dec = np.arcsin(sin_dec)

    y = np.sin(phi) * np.sin(r) * np.cos(dec0)
    x = np.cos(r) - np.sin(dec0) * sin_dec
    ra = ra0 + np.arctan2(y, x)

    ra_deg_arr = np.degrees(ra) % 360.0
    lons = (ra_deg_arr - 180.0).tolist()
    lats = np.degrees(dec).tolist()
    return lons, lats


def _build_st_boresights(
    ditl: DITL | QueueDITL,
    idx: int,
) -> list[tuple[float, float, str, str]]:
    """Return ``(ra_deg, dec_deg, color, name)`` for each ST boresight at ``idx``.

    Uses the same body-frame rotation as the Mollweide visualisation.
    """
    if not hasattr(ditl, "config") or not hasattr(ditl.config, "spacecraft_bus"):
        return []
    st_cfg = getattr(ditl.config.spacecraft_bus, "star_trackers", None)
    if st_cfg is None or not hasattr(st_cfg, "star_trackers"):
        return []
    raw = getattr(st_cfg, "star_trackers", [])
    if not isinstance(raw, (list, tuple)) or not raw:
        return []
    trackers: list[Any] = list(raw)

    ra_deg = float(ditl.ra[idx])
    dec_deg = float(ditl.dec[idx])
    roll_list = getattr(ditl, "roll", [])
    roll_idx = min(idx, len(roll_list) - 1) if roll_list else -1
    roll_deg = float(roll_list[roll_idx]) if roll_idx >= 0 else 0.0

    # Body-frame basis (boresight = +X, solar-roll around X)
    ra_rad = np.radians(ra_deg)
    dec_rad = np.radians(dec_deg)
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

    cr, sr = np.cos(np.radians(roll_deg)), np.sin(np.radians(roll_deg))
    y_hat = y0 * cr - z0 * sr
    z_hat = y0 * sr + z0 * cr

    st_colors = ["cyan", "lime", "orange", "hotpink", "white", "yellow"]
    result: list[tuple[float, float, str, str]] = []
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
        st_ra = (np.degrees(np.arctan2(v_st[1], v_st[0])) + 360.0) % 360.0
        st_dec = float(np.degrees(np.arcsin(np.clip(v_st[2], -1.0, 1.0))))
        result.append(
            (
                float(st_ra),
                st_dec,
                st_colors[i % len(st_colors)],
                getattr(st, "name", f"ST-{i + 1}"),
            )
        )
    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def plot_sky_pointing_globe(
    ditl: DITL | QueueDITL,
    n_frames: int | None = None,
    constraint_alpha: float = 0.35,
    config: VisualizationConfig | None = None,
    observation_categories: ObservationCategories | None = None,
) -> go.Figure:
    """Plot spacecraft pointing on a rotatable 3D celestial-sphere globe.

    Uses Plotly's ``orthographic`` scattergeo projection.  The globe can be
    rotated by click-dragging and a time slider (or Play button) steps through
    the simulation.

    Coordinate convention: Dec → latitude, RA → longitude (RA – 180°).

    What is shown
    -------------
    - All scheduled observations (static scatter, colour-coded by category).
    - Sun / Moon / Earth exclusion-zone circles (animated).
    - Earth physical disk (animated).
    - Current spacecraft pointing — star marker, colour = ACS mode (animated).
    - Star-tracker boresights — hexagon markers (animated, one per tracker).

    Parameters
    ----------
    ditl : DITL or QueueDITL
        Completed DITL simulation object (``calc()`` must have been run first).
    n_frames : int, optional
        Maximum number of animation frames.  Defaults to
        ``min(len(ditl.utime), 300)``.  Reduce for lighter notebooks.
    constraint_alpha : float, optional
        Fill opacity of the constraint exclusion polygons (default: 0.35).
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
    >>> fig = plot_sky_pointing_globe(ditl)
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
    # Pre-compute star-tracker boresight offsets and constraint specs
    # ------------------------------------------------------------------
    # δ = angular offset of each ST boresight from spacecraft +X (degrees).
    # For ignore_roll=True the field-of-regard exclusion radius around a
    # constrained body is exactly (δ + min_angle).
    st_boresight_offsets: list[float] = []
    for _st_raw in raw_trackers:
        _orient = getattr(_st_raw, "orientation", None)
        _b = getattr(_orient, "boresight", None) if _orient else None
        if _b is not None:
            _bv = np.asarray(_b, dtype=np.float64)
            _bn = np.linalg.norm(_bv)
            if _bn > 1e-12:
                _bv = _bv / _bn
            st_boresight_offsets.append(
                float(np.degrees(np.arccos(np.clip(_bv[0], -1.0, 1.0))))
            )
        else:
            st_boresight_offsets.append(0.0)

    # (tracker_idx, tier, body, min_angle_deg)
    st_constraint_specs: list[tuple[int, str, str, float]] = []
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
                    st_constraint_specs.append((_ci, _tier, _body, float(_mangle)))

    # n_st_circles = len(st_constraint_specs)
    ignore_roll: bool = bool(
        getattr(
            getattr(getattr(ditl, "config", None), "constraint", None),
            "ignore_roll",
            False,
        )
    )

    def _mode_color(idx: int) -> str:
        m = ditl.mode[idx]
        return mode_colors.get(m.name if hasattr(m, "name") else str(m), "red")

    # ------------------------------------------------------------------
    # Per-frame data builder
    # ------------------------------------------------------------------
    def _frame_traces(idx: int) -> list[dict[str, Any]]:
        """Return a list of partial trace-data dicts for all animated traces."""
        dt = dtutcfromtimestamp(utimes[idx])
        ei = ephem.index(dt)

        sun_ra = float(ephem.sun_ra_deg[ei])
        sun_dec = float(ephem.sun_dec_deg[ei])
        moon_ra = float(ephem.moon_ra_deg[ei])
        moon_dec = float(ephem.moon_dec_deg[ei])
        earth_ra = float(ephem.earth_ra_deg[ei])
        earth_dec = float(ephem.earth_dec_deg[ei])
        earth_disk_r = float(ephem.earth_radius_deg[ei])

        # Constraint exclusion radii (approximate from defaults; Earth adds disk)
        sun_r = float(SUN_OCCULT)
        moon_r = float(MOON_OCCULT)
        earth_excl_r = earth_disk_r + float(EARTH_OCCULT)

        sun_lons, sun_lats = _sky_circle_polygon(sun_ra, sun_dec, sun_r)
        moon_lons, moon_lats = _sky_circle_polygon(moon_ra, moon_dec, moon_r)
        earth_excl_lons, earth_excl_lats = _sky_circle_polygon(
            earth_ra, earth_dec, earth_excl_r
        )
        earth_disk_lons, earth_disk_lats = _sky_circle_polygon(
            earth_ra, earth_dec, earth_disk_r
        )

        ra = float(ditl.ra[idx])
        dec = float(ditl.dec[idx])
        mc = _mode_color(idx)

        st_positions = _build_st_boresights(ditl, idx)

        # Trace ordering (must match the initial trace list below):
        #   1: Sun exclusion polygon
        #   2: Sun body marker
        #   3: Moon exclusion polygon
        #   4: Moon body marker
        #   5: Earth exclusion polygon
        #   6: Earth disk polygon
        #   7: Pointing marker
        #   8 … 7+n_trackers: ST boresight markers
        traces: list[dict[str, Any]] = [
            {"lon": sun_lons, "lat": sun_lats},
            {
                "lon": [_ra_to_lon(sun_ra)],
                "lat": [sun_dec],
                "marker": {"color": "yellow", "size": 16, "symbol": "circle"},
            },
            {"lon": moon_lons, "lat": moon_lats},
            {
                "lon": [_ra_to_lon(moon_ra)],
                "lat": [moon_dec],
                "marker": {"color": "lightgray", "size": 12, "symbol": "circle"},
            },
            {"lon": earth_excl_lons, "lat": earth_excl_lats},
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
                traces.append({"lon": [_ra_to_lon(st_ra)], "lat": [st_dec]})
            else:
                traces.append({"lon": [], "lat": []})

        # --- ST constraint exclusion-zone circles ---
        # For ignore_roll=True the field-of-regard exclusion zone around each
        # constrained body is a circle of radius (δ + min_angle), where δ is the
        # ST boresight's angular offset from the spacecraft +X axis.
        # For ignore_roll=False we use the simpler min_angle circle around the body
        # (approximate — visually shows where the boresight must not point).
        for tr_idx, tier, body, min_angle in st_constraint_specs:
            if body == "sun":
                body_ra, body_dec = sun_ra, sun_dec
                extra_r = 0.0
            elif body == "earth":
                body_ra, body_dec = earth_ra, earth_dec
                extra_r = earth_disk_r  # add physical disk radius for Earth
            elif body == "moon":
                body_ra, body_dec = moon_ra, moon_dec
                extra_r = 0.0
            else:
                traces.append({"lon": [], "lat": []})
                continue

            delta = st_boresight_offsets[tr_idx] if ignore_roll else 0.0
            eff_r = delta + min_angle + extra_r
            if eff_r > 0:
                c_lons, c_lats = _sky_circle_polygon(body_ra, body_dec, eff_r)
            else:
                c_lons, c_lats = [], []
            traces.append({"lon": c_lons, "lat": c_lats})

        return traces

    # ------------------------------------------------------------------
    # Static observations trace (never changes between frames)
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
        # Plotly scattergeo rejects matplotlib `tab:*` names — normalise to hex.
        obs_colors.append(mcolors.to_hex(base_color))
        obs_texts.append(f"Obs {getattr(ppt, 'obsid', '')}")

    # ------------------------------------------------------------------
    # Build initial trace list (state at frame_indices[0])
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
        # --- Trace 0: static observations ---
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
        # --- Trace 1: Sun exclusion polygon ---
        _poly_trace(
            init_data[0],
            "Sun exclusion",
            _to_rgba("gold", constraint_alpha),
            "gold",
        ),
        # --- Trace 2: Sun body marker ---
        _marker_trace(init_data[1], "Sun", "circle", "yellow", 16),
        # --- Trace 3: Moon exclusion polygon ---
        _poly_trace(
            init_data[2],
            "Moon exclusion",
            _to_rgba("gray", constraint_alpha),
            "lightgray",
        ),
        # --- Trace 4: Moon body marker ---
        _marker_trace(init_data[3], "Moon", "circle", "lightgray", 12),
        # --- Trace 5: Earth exclusion polygon (disk + limb avoidance) ---
        _poly_trace(
            init_data[4],
            "Earth exclusion",
            _to_rgba("royalblue", constraint_alpha),
            "dodgerblue",
        ),
        # --- Trace 6: Earth physical disk polygon ---
        _poly_trace(
            init_data[5],
            "Earth disk",
            _to_rgba("darkblue", 0.75),
            "cornflowerblue",
            showlegend=True,
        ),
        # --- Trace 7: Current pointing ---
        go.Scattergeo(
            lon=init_data[6]["lon"],
            lat=init_data[6]["lat"],
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

    # Star-tracker boresight traces (one per tracker)
    for i in range(n_trackers):
        st_name = getattr(raw_trackers[i], "name", f"ST-{i + 1}")
        st_color = _st_colors[i % len(_st_colors)]
        td = init_data[7 + i] if 7 + i < len(init_data) else {"lon": [], "lat": []}
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

    # Star-tracker constraint exclusion-zone circles (one per spec)
    _st_legend_shown: set[str] = set()
    for j, (tr_idx, tier, body, min_angle) in enumerate(st_constraint_specs):
        _legend_key = tier
        _show_legend = _legend_key not in _st_legend_shown
        if _show_legend:
            _st_legend_shown.add(_legend_key)

        if tier == "hard":
            fill_color = _to_rgba("red", constraint_alpha)
            line_color = "red"
            line_dash = "solid"
            label = "ST Hard Cons."
        else:
            fill_color = _to_rgba("magenta", constraint_alpha * 0.7)
            line_color = "magenta"
            line_dash = "dot"
            label = "ST Soft Cons."

        ci = 7 + n_trackers + j
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

    # Animated trace indices: everything except trace 0 (observations)
    animated_indices = list(range(1, len(traces_fig)))

    # ------------------------------------------------------------------
    # Build Plotly animation frames & slider steps
    # ------------------------------------------------------------------
    frames: list[go.Frame] = []
    slider_steps: list[dict[str, Any]] = []

    for fi, idx in enumerate(frame_indices):
        dt = dtutcfromtimestamp(utimes[idx])
        time_str = dt.strftime("%Y-%m-%d %H:%M UTC")
        fdata = _frame_traces(idx)

        frames.append(
            go.Frame(
                data=[go.Scattergeo(**td) for td in fdata],
                traces=animated_indices,
                name=str(fi),
                layout=go.Layout(title_text=f"Spacecraft Pointing — {time_str}"),
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
                "label": time_str,
                "method": "animate",
            }
        )

    # ------------------------------------------------------------------
    # Layout
    # ------------------------------------------------------------------
    init_time_str = dtutcfromtimestamp(utimes[init_idx]).strftime("%Y-%m-%d %H:%M UTC")
    layout = go.Layout(
        title=dict(
            text=f"Spacecraft Pointing — {init_time_str}",
            font=dict(color="white", size=14),
        ),
        geo=dict(
            projection_type="orthographic",
            projection_rotation=dict(lon=0, lat=20, roll=0),
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
        margin=dict(l=0, r=0, t=50, b=80),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                y=-0.07,
                x=0.5,
                xanchor="center",
                yanchor="top",
                pad=dict(t=0, r=10),
                bgcolor="rgba(40,40,70,0.9)",
                bordercolor="rgba(130,130,180,0.5)",
                font=dict(color="white"),
                buttons=[
                    dict(
                        label="▶ Play",
                        method="animate",
                        args=[
                            None,
                            {
                                "frame": {"duration": 60, "redraw": True},
                                "fromcurrent": True,
                                "transition": {"duration": 0},
                            },
                        ],
                    ),
                    dict(
                        label="⏸ Pause",
                        method="animate",
                        args=[
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
                len=0.88,
                x=0.06,
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
