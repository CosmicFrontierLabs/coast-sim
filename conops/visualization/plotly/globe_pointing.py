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
    _to_rgba,
)
from ._helpers import (
    lighten_color as _lighten_color,
)

if TYPE_CHECKING:
    from ...ditl import DITL, QueueDITL


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
    - Anti-Sun direction marker and exclusion zone (animated, orange).
    - Panel constraint exclusion zones (animated, green; hidden during eclipse).
    - Orbit (RAM direction) exclusion zone — animated circle when configured.
    - Current spacecraft pointing — star marker, colour = ACS mode (animated).
    - Star-tracker boresights — hexagon markers (animated, one per tracker).
    - Star-tracker body-exclusion-zone circles (animated, hard=red, soft=magenta).

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
    # Resolve constraint plotting configuration (once, not per frame)
    # ------------------------------------------------------------------
    cc: ConstraintPlotConfig = resolve_constraint_plot_config(ditl, raw_trackers, ephem)
    _eclipse_con = rust_ephem.EclipseConstraint()

    def _mode_color(idx: int) -> str:
        m = ditl.mode[idx]
        return mode_colors.get(m.name if hasattr(m, "name") else str(m), "red")

    # ------------------------------------------------------------------
    # Per-frame data builder
    # ------------------------------------------------------------------
    # Trace ordering (0-indexed data items, one per animated trace):
    # Let _B = 10+n_trackers+n_st_specs, _O = _B+(3 if orbit else 0),
    #         _A = _O+(3 if anti_sun else 0)
    #   0: Sun min excl   1: Sun max excl   2: Sun marker
    #   3: Moon min excl  4: Moon max excl  5: Moon marker
    #   6: Earth min excl 7: Earth max excl 8: Earth disk
    #   9: Pointing marker
    #   10 … 9+n_trackers: ST boresight markers
    #   10+n_trackers … 9+n_trackers+n_st_specs: ST constraint circles
    #   _B+0: Orbit RAM min excl    [if orbit configured]
    #   _B+1: Orbit RAM max excl    [if orbit configured]
    #   _B+2: RAM direction marker  [if orbit configured]
    #   _O+0: Anti-Sun min excl     [if anti_sun configured]
    #   _O+1: Anti-Sun max excl     [if anti_sun configured]
    #   _O+2: Anti-Sun marker       [if anti_sun configured]
    #   _A+0: Panel min excl        [if panel configured]
    #   _A+1: Panel max excl        [if panel configured]
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
        obs_colors.append(mcolors.to_hex(base_color))
        obs_texts.append(f"Obs {getattr(ppt, 'obsid', '')}")

    # ------------------------------------------------------------------
    # Build initial trace list (state at frame_indices[0])
    # ------------------------------------------------------------------
    init_idx = frame_indices[0]
    init_data = _frame_traces(init_idx)
    mc0 = _mode_color(init_idx)

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
        # --- Traces 1–2: Sun exclusion polygons ---
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
        # --- Trace 3: Sun body marker ---
        _marker_trace(init_data[2], "Sun", "circle", "yellow", 16),
        # --- Traces 4–5: Moon exclusion polygons ---
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
        # --- Trace 6: Moon body marker ---
        _marker_trace(init_data[5], "Moon", "circle", "lightgray", 12),
        # --- Traces 7–8: Earth exclusion polygons ---
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
        # --- Trace 9: Earth physical disk polygon ---
        _poly_trace(
            init_data[8],
            "Earth disk",
            _to_rgba("darkblue", 0.75),
            "cornflowerblue",
            showlegend=True,
        ),
        # --- Trace 10: Current pointing ---
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

    # Star-tracker boresight traces (one per tracker)
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
        label_str = dt.strftime("%H:%M")
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
                "label": label_str,
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
        height=600,
        width=800,
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
