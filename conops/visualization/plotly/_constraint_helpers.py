"""Shared constraint-plotting helpers for sky and globe pointing visualizations."""

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any

import numpy as np
import plotly.graph_objects as go
import rust_ephem

from ...config.constants import EARTH_OCCULT, MOON_OCCULT, PANEL_CONSTRAINT, SUN_OCCULT
from ._helpers import (
    _marker_trace,
    _poly_trace,
    _ra_to_lon,
    _sky_circle_polygon,
    _to_rgba,
    lighten_color,
)

# (tracker_idx, tier, body, bound_kind, angle_deg)
ConstraintSpec = tuple[int, str, str, str, float]


@dataclass
class ConstraintPlotConfig:
    """All constraint-plotting configuration resolved from a DITL object."""

    body_angle_cfg: dict[str, tuple[float, float | None]]
    st_boresight_offsets: list[float]
    st_constraint_specs: list[ConstraintSpec]
    ignore_roll: bool
    orbit_constraint: Any | None
    orbit_min_angle: float
    orbit_max_angle: float | None
    gcrs_vel: np.ndarray | None
    anti_sun_constraint_cfg: Any | None
    anti_sun_min_angle: float
    anti_sun_max_angle: float | None
    panel_constraint_cfg: Any | None
    panel_min_angle: float
    panel_max_angle: float


def resolve_constraint_plot_config(
    ditl: Any,
    raw_trackers: list[Any],
    ephem: Any,
) -> ConstraintPlotConfig:
    """Resolve all constraint-plotting configuration from a DITL object."""
    constraint_cfg = None
    if hasattr(ditl, "config") and hasattr(ditl.config, "constraint"):
        constraint_cfg = ditl.config.constraint

    # Body angle config (min/max exclusion angles for Sun, Moon, Earth)
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

    # ST boresight angular offsets from spacecraft +X axis (degrees)
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

    # ST constraint specs
    st_constraint_specs: list[ConstraintSpec] = []
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
                if _xangle is not None and 180.0 - float(_xangle) > 0:
                    st_constraint_specs.append(
                        (_ci, _tier, _body, "max", float(_xangle))
                    )

    ignore_roll: bool = bool(
        getattr(constraint_cfg, "ignore_roll", False)
        if constraint_cfg is not None
        else False
    )

    # Orbit constraint
    orbit_constraint = (
        getattr(constraint_cfg, "orbit_constraint", None)
        if constraint_cfg is not None
        else None
    )
    orbit_min_angle: float = 0.0
    orbit_max_angle: float | None = None
    gcrs_vel: np.ndarray | None = None
    if orbit_constraint is not None:
        orbit_min_angle = float(getattr(orbit_constraint, "min_angle", 0.0))
        _orbit_max_raw = getattr(orbit_constraint, "max_angle", None)
        orbit_max_angle = float(_orbit_max_raw) if _orbit_max_raw is not None else None
        gcrs_vel = np.asarray(ephem.gcrs_pv.velocity, dtype=np.float64)

    # Anti-sun constraint
    anti_sun_constraint_cfg = (
        getattr(constraint_cfg, "anti_sun_constraint", None)
        if constraint_cfg is not None
        else None
    )
    anti_sun_min_angle: float = 0.0
    anti_sun_max_angle: float | None = None
    if anti_sun_constraint_cfg is not None:
        _as_min_raw = getattr(anti_sun_constraint_cfg, "min_angle", 0.0)
        anti_sun_min_angle = float(_as_min_raw) if _as_min_raw is not None else 0.0
        _as_max_raw = getattr(anti_sun_constraint_cfg, "max_angle", None)
        anti_sun_max_angle = float(_as_max_raw) if _as_max_raw is not None else None

    # Panel constraint
    panel_constraint_cfg = (
        getattr(constraint_cfg, "panel_constraint", None)
        if constraint_cfg is not None
        else None
    )
    panel_min_angle: float = float(PANEL_CONSTRAINT)
    panel_max_angle: float = float(180 - PANEL_CONSTRAINT)
    if panel_constraint_cfg is not None:
        _pc_min_raw = getattr(panel_constraint_cfg, "min_angle", None)
        _pc_max_raw = getattr(panel_constraint_cfg, "max_angle", None)
        if _pc_min_raw is not None:
            panel_min_angle = float(_pc_min_raw)
        if _pc_max_raw is not None:
            panel_max_angle = float(_pc_max_raw)

    return ConstraintPlotConfig(
        body_angle_cfg=body_angle_cfg,
        st_boresight_offsets=st_boresight_offsets,
        st_constraint_specs=st_constraint_specs,
        ignore_roll=ignore_roll,
        orbit_constraint=orbit_constraint,
        orbit_min_angle=orbit_min_angle,
        orbit_max_angle=orbit_max_angle,
        gcrs_vel=gcrs_vel,
        anti_sun_constraint_cfg=anti_sun_constraint_cfg,
        anti_sun_min_angle=anti_sun_min_angle,
        anti_sun_max_angle=anti_sun_max_angle,
        panel_constraint_cfg=panel_constraint_cfg,
        panel_min_angle=panel_min_angle,
        panel_max_angle=panel_max_angle,
    )


def constraint_polygons(
    body_ra: float,
    body_dec: float,
    min_angle: float,
    max_angle: float | None,
    extra_r: float = 0.0,
) -> tuple[list[float], list[float], list[float], list[float]]:
    """Return (min_lons, min_lats, max_lons, max_lats) exclusion polygons for a body.

    The min circle is centred on the body; the max circle is centred on the
    anti-body direction (because a max-angle violation means the pointing is
    too far from the body — equivalent to being inside a cap around anti-body).
    """
    min_r = max(0.0, min_angle + extra_r)
    if min_r > 0:
        min_lons, min_lats = _sky_circle_polygon(body_ra, body_dec, min_r)
    else:
        min_lons, min_lats = [], []

    if max_angle is not None:
        anti_body_ra = (body_ra + 180.0) % 360.0
        anti_body_dec = -body_dec
        max_r = max(0.0, 180.0 - (max_angle + extra_r))
        if max_r > 0:
            max_lons, max_lats = _sky_circle_polygon(anti_body_ra, anti_body_dec, max_r)
        else:
            max_lons, max_lats = [], []
    else:
        max_lons, max_lats = [], []

    return min_lons, min_lats, max_lons, max_lats


def build_body_polygon_traces(
    sun_ra: float,
    sun_dec: float,
    moon_ra: float,
    moon_dec: float,
    earth_ra: float,
    earth_dec: float,
    earth_disk_r: float,
    body_angle_cfg: dict[str, tuple[float, float | None]],
) -> list[dict[str, Any]]:
    """Build the 9 per-frame body-constraint trace dicts (indices 0–8).

    Trace ordering:
      0: Sun min excl polygon
      1: Sun max excl polygon
      2: Sun body marker
      3: Moon min excl polygon
      4: Moon max excl polygon
      5: Moon body marker
      6: Earth min excl polygon (disk + limb avoidance)
      7: Earth max excl polygon
      8: Earth physical disk polygon
    """
    sun_min_angle, sun_max_angle = body_angle_cfg["sun"]
    moon_min_angle, moon_max_angle = body_angle_cfg["moon"]
    earth_min_angle, earth_max_angle = body_angle_cfg["earth"]

    sun_lons, sun_lats, sun_max_lons, sun_max_lats = constraint_polygons(
        sun_ra, sun_dec, sun_min_angle, sun_max_angle
    )
    moon_lons, moon_lats, moon_max_lons, moon_max_lats = constraint_polygons(
        moon_ra, moon_dec, moon_min_angle, moon_max_angle
    )
    earth_excl_lons, earth_excl_lats, earth_max_lons, earth_max_lats = (
        constraint_polygons(
            earth_ra, earth_dec, earth_min_angle, earth_max_angle, extra_r=earth_disk_r
        )
    )
    earth_disk_lons, earth_disk_lats = _sky_circle_polygon(
        earth_ra, earth_dec, earth_disk_r
    )

    return [
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
    ]


def build_tail_constraint_traces(
    cc: ConstraintPlotConfig,
    sun_ra: float,
    sun_dec: float,
    moon_ra: float,
    moon_dec: float,
    earth_ra: float,
    earth_dec: float,
    earth_disk_r: float,
    ei: int,
    dt: datetime.datetime,
    eclipse_con: rust_ephem.EclipseConstraint,
    ephem: Any,
) -> list[dict[str, Any]]:
    """Build per-frame trace dicts for ST constraints, orbit, anti-sun, and panel.

    These are appended after the pointing marker and ST boresight markers
    (indices 10+n_trackers onward in the full trace list).
    """
    traces: list[dict[str, Any]] = []

    # ST constraint exclusion-zone circles
    for tr_idx, tier, body, bound_kind, angle in cc.st_constraint_specs:
        if body == "sun":
            body_ra, body_dec, extra_r = sun_ra, sun_dec, 0.0
        elif body == "earth":
            body_ra, body_dec, extra_r = earth_ra, earth_dec, earth_disk_r
        elif body == "moon":
            body_ra, body_dec, extra_r = moon_ra, moon_dec, 0.0
        else:
            traces.append({"lon": [], "lat": []})
            continue

        if bound_kind == "min":
            if cc.ignore_roll:
                # With free roll, tracker boresight sweeps a cone of half-angle delta
                # around +X, so the always-invalid cap shrinks by delta.
                eff_r = max(0.0, angle - cc.st_boresight_offsets[tr_idx]) + extra_r
            else:
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

    # Orbit (RAM direction) constraint
    if cc.orbit_constraint is not None and cc.gcrs_vel is not None:
        v = cc.gcrs_vel[ei].copy()
        v_n = np.linalg.norm(v)
        if v_n > 1e-12:
            v = v / v_n
        ram_ra_deg = (np.degrees(np.arctan2(v[1], v[0])) + 360.0) % 360.0
        ram_dec_deg = float(np.degrees(np.arcsin(np.clip(v[2], -1.0, 1.0))))

        oc_lons, oc_lats = (
            _sky_circle_polygon(ram_ra_deg, ram_dec_deg, cc.orbit_min_angle)
            if cc.orbit_min_angle > 0
            else ([], [])
        )

        if cc.orbit_max_angle is not None:
            anti_ram_ra = (ram_ra_deg + 180.0) % 360.0
            anti_ram_dec = -ram_dec_deg
            oc_max_r = max(0.0, 180.0 - cc.orbit_max_angle)
            oc_max_lons, oc_max_lats = (
                _sky_circle_polygon(anti_ram_ra, anti_ram_dec, oc_max_r)
                if oc_max_r > 0
                else ([], [])
            )
        else:
            oc_max_lons, oc_max_lats = [], []

        traces.append({"lon": oc_lons, "lat": oc_lats})
        traces.append({"lon": oc_max_lons, "lat": oc_max_lats})
        traces.append(
            {
                "lon": [_ra_to_lon(ram_ra_deg)],
                "lat": [ram_dec_deg],
                "marker": {"color": "mediumpurple", "size": 12, "symbol": "circle"},
            }
        )

    # Anti-sun constraint
    if cc.anti_sun_constraint_cfg is not None:
        anti_sun_ra_v = (sun_ra + 180.0) % 360.0
        anti_sun_dec_v = -sun_dec
        as_min_lons, as_min_lats, as_max_lons, as_max_lats = constraint_polygons(
            sun_ra, sun_dec, cc.anti_sun_min_angle, cc.anti_sun_max_angle
        )
        traces.append({"lon": as_min_lons, "lat": as_min_lats})
        traces.append({"lon": as_max_lons, "lat": as_max_lats})
        traces.append(
            {
                "lon": [_ra_to_lon(anti_sun_ra_v)],
                "lat": [anti_sun_dec_v],
                "marker": {"color": "orange", "size": 12, "symbol": "diamond"},
            }
        )

    # Panel constraint (only visible in sunlit orbit)
    if cc.panel_constraint_cfg is not None:
        in_eclipse_now = bool(
            eclipse_con.in_constraint(
                ephemeris=ephem, target_ra=0.0, target_dec=0.0, time=dt
            )
        )
        if not in_eclipse_now:
            pc_min_lons, pc_min_lats, pc_max_lons, pc_max_lats = constraint_polygons(
                sun_ra, sun_dec, cc.panel_min_angle, cc.panel_max_angle
            )
        else:
            pc_min_lons, pc_min_lats, pc_max_lons, pc_max_lats = [], [], [], []
        traces.append({"lon": pc_min_lons, "lat": pc_min_lats})
        traces.append({"lon": pc_max_lons, "lat": pc_max_lats})

    return traces


def build_optional_figure_traces(
    cc: ConstraintPlotConfig,
    init_data: list[dict[str, Any]],
    n_trackers: int,
    constraint_alpha: float,
) -> list[go.Scattergeo]:
    """Build static Scattergeo traces for ST constraints, orbit, anti-sun, and panel.

    These are the non-per-frame (initial-state) traces appended after the
    pointing marker and ST boresight markers in the figure trace list.
    """
    traces: list[go.Scattergeo] = []
    n_st_specs = len(cc.st_constraint_specs)

    # ST constraint exclusion-zone circles
    _st_legend_shown: set[str] = set()
    for j, (_tr_idx, tier, _body, bound_kind, _angle) in enumerate(
        cc.st_constraint_specs
    ):
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
        traces.append(
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

    # Orbit constraint exclusion polygons + RAM direction marker
    if cc.orbit_constraint is not None:
        _oc_base = 10 + n_trackers + n_st_specs
        _oc_poly = (
            init_data[_oc_base] if _oc_base < len(init_data) else {"lon": [], "lat": []}
        )
        _oc_max = (
            init_data[_oc_base + 1]
            if _oc_base + 1 < len(init_data)
            else {"lon": [], "lat": []}
        )
        _oc_ram = (
            init_data[_oc_base + 2]
            if _oc_base + 2 < len(init_data)
            else {"lon": [], "lat": []}
        )
        traces.append(
            _poly_trace(
                _oc_poly,
                "Orbit (RAM) min excl.",
                _to_rgba("mediumpurple", constraint_alpha),
                "mediumpurple",
            )
        )
        traces.append(
            _poly_trace(
                _oc_max,
                "Orbit (RAM) max excl.",
                _to_rgba(lighten_color("mediumpurple", 0.25), constraint_alpha * 0.75),
                lighten_color("mediumpurple", 0.15),
                showlegend=cc.orbit_max_angle is not None,
            )
        )
        traces.append(
            _marker_trace(_oc_ram, "RAM direction", "circle", "mediumpurple", 12)
        )

    # Anti-sun constraint traces
    if cc.anti_sun_constraint_cfg is not None:
        _as_base = (
            10 + n_trackers + n_st_specs + (3 if cc.orbit_constraint is not None else 0)
        )
        _as_min = (
            init_data[_as_base] if _as_base < len(init_data) else {"lon": [], "lat": []}
        )
        _as_max = (
            init_data[_as_base + 1]
            if _as_base + 1 < len(init_data)
            else {"lon": [], "lat": []}
        )
        _as_marker = (
            init_data[_as_base + 2]
            if _as_base + 2 < len(init_data)
            else {"lon": [], "lat": []}
        )
        traces.append(
            _poly_trace(
                _as_min,
                "Anti-Sun min exclusion",
                _to_rgba("darkorange", constraint_alpha),
                "darkorange",
                showlegend=cc.anti_sun_min_angle > 0,
            )
        )
        traces.append(
            _poly_trace(
                _as_max,
                "Anti-Sun exclusion",
                _to_rgba("orange", constraint_alpha),
                "orange",
                showlegend=(
                    cc.anti_sun_max_angle is not None
                    and (180.0 - cc.anti_sun_max_angle) > 0
                ),
            )
        )
        traces.append(_marker_trace(_as_marker, "Anti-Sun", "diamond", "orange", 12))

    # Panel constraint traces
    if cc.panel_constraint_cfg is not None:
        _pc_base = (
            10
            + n_trackers
            + n_st_specs
            + (3 if cc.orbit_constraint is not None else 0)
            + (3 if cc.anti_sun_constraint_cfg is not None else 0)
        )
        _pc_min = (
            init_data[_pc_base] if _pc_base < len(init_data) else {"lon": [], "lat": []}
        )
        _pc_max = (
            init_data[_pc_base + 1]
            if _pc_base + 1 < len(init_data)
            else {"lon": [], "lat": []}
        )
        traces.append(
            _poly_trace(
                _pc_min,
                "Panel min exclusion",
                _to_rgba("limegreen", constraint_alpha),
                "limegreen",
                showlegend=cc.panel_min_angle > 0,
            )
        )
        traces.append(
            _poly_trace(
                _pc_max,
                "Panel max exclusion",
                _to_rgba("green", constraint_alpha * 0.8),
                "green",
                showlegend=(180.0 - cc.panel_max_angle) > 0,
            )
        )

    return traces
