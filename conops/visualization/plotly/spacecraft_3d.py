"""Interactive 3-D spacecraft configuration visualizer.

Renders the spacecraft body, solar panels, radiators, star trackers and
telescope as a Plotly 3-D scene that is freely rotatable in the browser or
Jupyter.

Coordinate system (spacecraft body frame)
------------------------------------------
  +X  boresight / telescope pointing direction
  +Y  spacecraft "up"
  +Z  completes the right-handed system

Component placement rules
--------------------------
  Telescope   – cylinder extending in the boresight direction from the +X bus face.
  Solar panels – use ``PanelGeometry.center_m`` when present; otherwise placed
                 on the bus face whose outward normal best matches the panel normal.
  Radiators   – placed on the bus face whose outward normal best matches the
                radiator normal; sized by ``width_m`` × ``height_m``.
  Star trackers – small prism on the bus face matching the boresight direction.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np
import plotly.graph_objects as go

if TYPE_CHECKING:
    from ...config import MissionConfig

# ---------------------------------------------------------------------------
# Visual palette (dark-themed to match the rest of the codebase)
# ---------------------------------------------------------------------------
_BG = "rgb(12, 12, 28)"
_C_BUS = "rgb(155, 160, 175)"
_C_SCOPE = "rgb(45, 50, 62)"
_C_SCOPE_RING = "rgb(200, 160, 50)"
_C_SOLAR = "rgb(15, 55, 145)"
_C_SOLAR_FRAME = "rgb(195, 200, 215)"
_C_RADIATOR = "rgb(228, 232, 240)"
_C_ST = "rgb(65, 70, 88)"
_C_X = "red"
_C_Y = "limegreen"
_C_Z = "dodgerblue"
_C_NORMAL = "rgba(255,255,180,0.9)"
_C_BORESIGHT = "cyan"

# Axis / arrow lengths
_AXIS_LEN: float = 1.5
_NORMAL_LEN: float = 0.45
_CONE_SIZE: float = 0.07


# ---------------------------------------------------------------------------
# Low-level geometry helpers
# ---------------------------------------------------------------------------


def _unit(v: Any) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = np.linalg.norm(a)
    return a / n if n > 1e-12 else a


def _perp_pair(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors perpendicular to *v*."""
    v = _unit(v)
    ref = np.array([0.0, 0.0, 1.0]) if abs(v[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    u = np.cross(ref, v)
    u = _unit(u)
    w = _unit(np.cross(v, u))
    return u, w


def _box_mesh(
    hx: float,
    hy: float,
    hz: float,
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    color: str = _C_BUS,
    name: str = "Bus",
    opacity: float = 0.88,
    show_legend: bool = True,
) -> go.Mesh3d:
    cx, cy, cz = center
    # 8 vertices (sx, sy, sz) ∈ {-1,+1}³
    verts = np.array(
        [
            [-hx, -hy, -hz],  # 0
            [+hx, -hy, -hz],  # 1
            [+hx, +hy, -hz],  # 2
            [-hx, +hy, -hz],  # 3
            [-hx, -hy, +hz],  # 4
            [+hx, -hy, +hz],  # 5
            [+hx, +hy, +hz],  # 6
            [-hx, +hy, +hz],  # 7
        ]
    ) + [cx, cy, cz]
    # 12 triangles covering 6 faces
    ii = [0, 0, 1, 1, 0, 0, 3, 3, 0, 0, 4, 4]
    jj = [3, 2, 2, 6, 4, 5, 7, 6, 3, 7, 5, 6]
    kk = [2, 1, 6, 5, 5, 6, 4, 2, 7, 4, 6, 7]
    return go.Mesh3d(
        x=verts[:, 0].tolist(),
        y=verts[:, 1].tolist(),
        z=verts[:, 2].tolist(),
        i=ii,
        j=jj,
        k=kk,
        color=color,
        opacity=opacity,
        name=name,
        showlegend=show_legend,
        flatshading=True,
        lighting=dict(ambient=0.65, diffuse=0.75, specular=0.25, roughness=0.7),
        hoverinfo="name",
    )


def _flat_rect_mesh(
    center: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    width: float,
    height: float,
    color: str,
    name: str,
    opacity: float = 0.88,
    show_legend: bool = True,
    legendgroup: str | None = None,
) -> go.Mesh3d:
    """Flat rectangular panel as a two-triangle Mesh3d."""
    c0 = center - u * width / 2 - v * height / 2
    c1 = center + u * width / 2 - v * height / 2
    c2 = center + u * width / 2 + v * height / 2
    c3 = center - u * width / 2 + v * height / 2
    pts = np.stack([c0, c1, c2, c3])
    return go.Mesh3d(
        x=pts[:, 0].tolist(),
        y=pts[:, 1].tolist(),
        z=pts[:, 2].tolist(),
        i=[0, 0],
        j=[1, 3],
        k=[2, 2],
        color=color,
        opacity=opacity,
        name=name,
        showlegend=show_legend,
        legendgroup=legendgroup,
        flatshading=True,
        lighting=dict(ambient=0.55, diffuse=0.80, specular=0.35, roughness=0.6),
        hoverinfo="name",
    )


def _cylinder_mesh(
    start: np.ndarray,
    axis: np.ndarray,
    radius: float,
    length: float,
    color: str,
    name: str,
    n_theta: int = 32,
    opacity: float = 0.90,
    show_legend: bool = True,
    legendgroup: str | None = None,
) -> go.Mesh3d:
    """Side surface of a cylinder as a Mesh3d."""
    u, v = _perp_pair(axis)
    theta = np.linspace(0, 2 * math.pi, n_theta, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    ring0 = start[:, None] + radius * (np.outer(u, cos_t) + np.outer(v, sin_t))
    ring1 = (start + axis * length)[:, None] + radius * (
        np.outer(u, cos_t) + np.outer(v, sin_t)
    )
    xs = np.concatenate([ring0[0], ring1[0]]).tolist()
    ys = np.concatenate([ring0[1], ring1[1]]).tolist()
    zs = np.concatenate([ring0[2], ring1[2]]).tolist()
    ii, jj, kk = [], [], []
    for idx in range(n_theta):
        nxt = (idx + 1) % n_theta
        ii.extend([idx, nxt])
        jj.extend([nxt, nxt + n_theta])
        kk.extend([idx + n_theta, idx + n_theta])
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=ii,
        j=jj,
        k=kk,
        color=color,
        opacity=opacity,
        name=name,
        showlegend=show_legend,
        legendgroup=legendgroup,
        flatshading=True,
        lighting=dict(ambient=0.55, diffuse=0.80, specular=0.45, roughness=0.5),
        hoverinfo="name",
    )


def _disk_mesh(
    center: np.ndarray,
    normal: np.ndarray,
    outer_r: float,
    inner_r: float,
    color: str,
    name: str,
    n_theta: int = 32,
    opacity: float = 0.90,
    show_legend: bool = False,
    legendgroup: str | None = None,
) -> go.Mesh3d:
    """Annulus (ring) or filled disk (inner_r=0) as a Mesh3d."""
    u, v = _perp_pair(normal)
    theta = np.linspace(0, 2 * math.pi, n_theta, endpoint=False)
    outer = center[:, None] + outer_r * (
        np.outer(u, np.cos(theta)) + np.outer(v, np.sin(theta))
    )
    if inner_r > 0:
        inner = center[:, None] + inner_r * (
            np.outer(u, np.cos(theta)) + np.outer(v, np.sin(theta))
        )
        xs = np.concatenate([outer[0], inner[0]]).tolist()
        ys = np.concatenate([outer[1], inner[1]]).tolist()
        zs = np.concatenate([outer[2], inner[2]]).tolist()
        ii, jj, kk = [], [], []
        for idx in range(n_theta):
            nxt = (idx + 1) % n_theta
            ii.extend([idx, nxt])
            jj.extend([nxt, nxt + n_theta])
            kk.extend([idx + n_theta, idx + n_theta])
    else:
        xs = [center[0]] + outer[0].tolist()
        ys = [center[1]] + outer[1].tolist()
        zs = [center[2]] + outer[2].tolist()
        ii, jj, kk = [], [], []
        for idx in range(n_theta):
            nxt = (idx % n_theta) + 1
            nxt2 = ((idx + 1) % n_theta) + 1
            ii.append(0)
            jj.append(nxt)
            kk.append(nxt2)
    return go.Mesh3d(
        x=xs,
        y=ys,
        z=zs,
        i=ii,
        j=jj,
        k=kk,
        color=color,
        opacity=opacity,
        name=name,
        showlegend=show_legend,
        legendgroup=legendgroup,
        flatshading=True,
        hoverinfo="name",
    )


def _arrow_traces(
    origin: np.ndarray,
    direction: np.ndarray,
    length: float,
    color: str,
    name: str,
    cone_sz: float = _CONE_SIZE,
    show_legend: bool = True,
    legendgroup: str | None = None,
    line_width: int = 4,
) -> list[Any]:
    """[Scatter3d shaft, Cone arrowhead] for a labelled vector arrow."""
    d = _unit(direction)
    if np.linalg.norm(d) < 0.5:
        return []
    tip = origin + d * length
    shaft_end = origin + d * (length - cone_sz * 1.2)
    line = go.Scatter3d(
        x=[float(origin[0]), float(shaft_end[0])],
        y=[float(origin[1]), float(shaft_end[1])],
        z=[float(origin[2]), float(shaft_end[2])],
        mode="lines",
        line=dict(color=color, width=line_width),
        name=name,
        showlegend=show_legend,
        legendgroup=legendgroup,
        hoverinfo="name",
    )
    cone = go.Cone(
        x=[float(tip[0])],
        y=[float(tip[1])],
        z=[float(tip[2])],
        u=[float(d[0]) * cone_sz],
        v=[float(d[1]) * cone_sz],
        w=[float(d[2]) * cone_sz],
        colorscale=[[0, color], [1, color]],
        showscale=False,
        name=name,
        showlegend=False,
        legendgroup=legendgroup,
        sizemode="absolute",
        sizeref=cone_sz,
        anchor="tail",
        hoverinfo="name",
    )
    return [line, cone]


def _axis_label(pos: np.ndarray, text: str, color: str) -> go.Scatter3d:
    return go.Scatter3d(
        x=[float(pos[0])],
        y=[float(pos[1])],
        z=[float(pos[2])],
        mode="text",
        text=[text],
        textfont=dict(color=color, size=14),
        showlegend=False,
        hoverinfo="skip",
    )


# ---------------------------------------------------------------------------
# Bus-face placement helper
# ---------------------------------------------------------------------------


def _face_placement(
    normal: np.ndarray,
    bus_hd: tuple[float, float, float],
    gap: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Given a component normal/boresight, return (center, span_u, span_v)
    placing the component on the bus face whose outward normal best matches.
    """
    hx, hy, hz = bus_hd
    face_n = np.array(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ]
    )
    face_c = np.array(
        [
            [hx, 0.0, 0.0],
            [-hx, 0.0, 0.0],
            [0.0, hy, 0.0],
            [0.0, -hy, 0.0],
            [0.0, 0.0, hz],
            [0.0, 0.0, -hz],
        ]
    )
    best = int(np.argmax(face_n @ normal))
    fn = face_n[best]
    center = face_c[best] + fn * gap
    u, v = _perp_pair(fn)
    return center, u, v


# ---------------------------------------------------------------------------
# Main public function
# ---------------------------------------------------------------------------


def plot_spacecraft_3d(
    config: MissionConfig,
    show_normals: bool = True,
    show_axes: bool = True,
    bus_half_dims: tuple[float, float, float] | None = None,
    title: str | None = None,
) -> go.Figure:
    """Build an interactive rotatable 3-D model of the spacecraft.

    The figure uses Plotly's 3-D scene so the user can rotate, zoom and pan
    freely.  All geometric parameters are read directly from *config*.

    Parameters
    ----------
    config:
        A fully-constructed ``MissionConfig``.  Defaults are used for any
        subsystem that is not explicitly configured.
    show_normals:
        Draw outward-normal / boresight arrows for panels, radiators and
        star trackers (default ``True``).
    show_axes:
        Draw +X / +Y / +Z spacecraft-body-frame axes from the origin
        (default ``True``).
    bus_half_dims:
        ``(half_x, half_y, half_z)`` extents of the spacecraft bus box in
        metres.  Defaults to ``(1.0, 0.55, 0.55)``.
    title:
        Figure title.  Falls back to the spacecraft bus name from config.

    Returns
    -------
    go.Figure
        Interactive Plotly figure.  Call ``fig.show()`` or evaluate in
        Jupyter to render.

    Examples
    --------
    >>> config = MissionConfig.from_json_file("config.json")
    >>> fig = plot_spacecraft_3d(config)
    >>> fig.show()
    """
    traces: list[Any] = []

    bus_hd: tuple[float, float, float] = bus_half_dims or (1.0, 0.55, 0.55)
    hx, hy, hz = bus_hd

    # Telescope geometry captured during section 2; used for ST mounting in section 5.
    # (tube_start, bore, outer_r, tube_len)
    _scope_geom: tuple[np.ndarray, np.ndarray, float, float] | None = None

    # ------------------------------------------------------------------ #
    # 1. Spacecraft bus body                                               #
    # ------------------------------------------------------------------ #
    bus_name = getattr(
        getattr(config, "spacecraft_bus", None), "name", "Spacecraft Bus"
    )
    traces.append(_box_mesh(hx, hy, hz, name=bus_name, color=_C_BUS))

    # ------------------------------------------------------------------ #
    # 2. Telescope / payload instruments                                   #
    # ------------------------------------------------------------------ #
    payload = getattr(config, "payload", None)
    instruments = list(getattr(payload, "instruments", []) or [])

    scope_legend_shown = False
    for inst in instruments:
        inst_type = getattr(inst, "instrument_type", None)
        if inst_type != "Telescope":
            continue

        bore = np.asarray(getattr(inst, "boresight", (1.0, 0.0, 0.0)), dtype=float)
        bore = _unit(bore)
        optics = getattr(inst, "optics", None)
        aperture_r: float = (getattr(optics, "aperture_m", None) or 0.5) / 2.0
        tube_len: float = getattr(optics, "tube_length_m", None) or max(
            aperture_r * 5.0, 1.2
        )

        # Outer baffle tube starts at the +X face of the bus
        # (projected along boresight to the face)
        # Approximate: place the tube starting at the bus face in the boresight direction
        face_offset: float = max(
            hx * abs(bore[0]), hy * abs(bore[1]), hz * abs(bore[2])
        )
        tube_start = bore * face_offset

        outer_r = aperture_r + 0.06  # baffle slightly wider than aperture
        scope_name = getattr(inst, "name", "Telescope")

        # Save for star-tracker mounting below (last telescope wins if multiple).
        _scope_geom = (tube_start, bore, outer_r, tube_len)

        # Outer baffle cylinder
        traces.append(
            _cylinder_mesh(
                tube_start,
                bore,
                outer_r,
                tube_len,
                color=_C_SCOPE,
                name=scope_name,
                show_legend=not scope_legend_shown,
                legendgroup="telescope",
            )
        )
        scope_legend_shown = True

        # Aperture end ring (gold)
        aper_center = tube_start + bore * tube_len
        traces.append(
            _disk_mesh(
                aper_center,
                bore,
                outer_r,
                aperture_r,
                color=_C_SCOPE_RING,
                name="Aperture ring",
                legendgroup="telescope",
            )
        )

        # Aperture cover disk (dark, shows the opening)
        traces.append(
            _disk_mesh(
                aper_center,
                bore,
                aperture_r,
                0.0,
                color="rgb(8, 8, 15)",
                name="Aperture opening",
                opacity=0.95,
                legendgroup="telescope",
            )
        )

        # Mid-tube baffle ring for the "space telescope" look
        mid_center = tube_start + bore * (tube_len * 0.45)
        traces.append(
            _disk_mesh(
                mid_center,
                bore,
                outer_r,
                outer_r * 0.75,
                color=_C_SCOPE_RING,
                name="Baffle ring",
                legendgroup="telescope",
            )
        )

        # Secondary mirror housing (small cylinder near aperture)
        sec_r = aperture_r * 0.25
        sec_start = aper_center - bore * (tube_len * 0.15)
        traces.append(
            _cylinder_mesh(
                sec_start,
                bore,
                sec_r,
                tube_len * 0.12,
                color="rgb(80, 85, 100)",
                name="Secondary",
                show_legend=False,
                legendgroup="telescope",
            )
        )

        # Boresight arrow
        if show_normals:
            bsight_origin = aper_center + bore * 0.05
            traces.extend(
                _arrow_traces(
                    bsight_origin,
                    bore,
                    _NORMAL_LEN,
                    color=_C_BORESIGHT,
                    name=f"{scope_name} boresight",
                    legendgroup="tel_normals",
                    show_legend=True,
                )
            )

    # ------------------------------------------------------------------ #
    # 3. Solar panels                                                      #
    # ------------------------------------------------------------------ #
    solar_set = getattr(config, "solar_panel", None)
    panels = list(getattr(solar_set, "panels", []) or [])
    panel_legend_shown = False
    frame_legend_shown = False

    for pidx, panel in enumerate(panels):
        pname = getattr(panel, "name", f"Panel {pidx + 1}")
        normal_raw = getattr(panel, "normal", (0.0, 1.0, 0.0))
        n_vec = _unit(np.asarray(normal_raw, dtype=float))
        geom = getattr(panel, "geometry", None)

        if geom is not None:
            # Use explicit PanelGeometry
            c = np.asarray(geom.center_m, dtype=float)
            u = _unit(np.asarray(geom.u, dtype=float))
            v = _unit(np.asarray(geom.v, dtype=float))
            pw = float(geom.width_m)
            ph = float(geom.height_m)
        else:
            # Place panel on the matching bus face
            # Panel size estimated from max_power (assume ~150 W/m²)
            max_pwr = float(getattr(panel, "max_power", 800.0))
            eff = float(getattr(solar_set, "conversion_efficiency", 0.95))
            area_est = max_pwr / max(150.0 * eff, 1.0)
            pw = max(math.sqrt(area_est * 1.6), 0.8)
            ph = max(math.sqrt(area_est / 1.6), 0.5)
            c, u, v = _face_placement(n_vec, bus_hd, gap=0.02)

        show_sol = not panel_legend_shown
        traces.append(
            _flat_rect_mesh(
                c,
                u,
                v,
                pw,
                ph,
                color=_C_SOLAR,
                name=pname,
                show_legend=show_sol,
                legendgroup="solar_panels",
            )
        )
        panel_legend_shown = True

        # Thin frame border (slightly larger, behind the cell)
        frame_offset = n_vec * 0.003
        show_fr = not frame_legend_shown
        traces.append(
            _flat_rect_mesh(
                c - frame_offset,
                u,
                v,
                pw + 0.04,
                ph + 0.04,
                color=_C_SOLAR_FRAME,
                name="Solar panel frame",
                opacity=0.75,
                show_legend=show_fr,
                legendgroup="solar_frames",
            )
        )
        frame_legend_shown = True

        if show_normals:
            traces.extend(
                _arrow_traces(
                    c + n_vec * 0.05,
                    n_vec,
                    _NORMAL_LEN,
                    color=_C_NORMAL,
                    name=f"{pname} normal",
                    show_legend=(pidx == 0),
                    legendgroup="panel_normals",
                )
            )

    # ------------------------------------------------------------------ #
    # 4. Radiators                                                         #
    # ------------------------------------------------------------------ #
    rad_cfg = getattr(getattr(config, "spacecraft_bus", None), "radiators", None)
    radiators = list(getattr(rad_cfg, "radiators", []) or [])
    rad_legend_shown = False

    for ridx, rad in enumerate(radiators):
        rname = getattr(rad, "name", f"Radiator {ridx + 1}")
        orient = getattr(rad, "orientation", None)
        normal_raw = getattr(orient, "normal", (0.0, 1.0, 0.0))
        n_vec = _unit(np.asarray(normal_raw, dtype=float))
        rw = float(getattr(rad, "width_m", 1.0))
        rh = float(getattr(rad, "height_m", 1.0))

        geom = getattr(rad, "geometry", None)
        if geom is not None:
            c = np.asarray(geom.center_m, dtype=float)
            u = _unit(np.asarray(geom.u, dtype=float))
            v = _unit(np.asarray(geom.v, dtype=float))
            rw = float(geom.width_m)
            rh = float(geom.height_m)
        else:
            c, u, v = _face_placement(n_vec, bus_hd, gap=0.005)

        show_r = not rad_legend_shown
        traces.append(
            _flat_rect_mesh(
                c,
                u,
                v,
                rw,
                rh,
                color=_C_RADIATOR,
                name=rname,
                opacity=0.85,
                show_legend=show_r,
                legendgroup="radiators",
            )
        )
        rad_legend_shown = True

        if show_normals:
            traces.extend(
                _arrow_traces(
                    c + n_vec * 0.02,
                    n_vec,
                    _NORMAL_LEN * 0.8,
                    color=_C_NORMAL,
                    name=f"{rname} normal",
                    show_legend=(ridx == 0),
                    legendgroup="rad_normals",
                )
            )

    # ------------------------------------------------------------------ #
    # 5. Star trackers                                                     #
    # ------------------------------------------------------------------ #
    st_cfg = getattr(getattr(config, "spacecraft_bus", None), "star_trackers", None)
    trackers = list(getattr(st_cfg, "star_trackers", []) or [])
    st_colors = ["orange", "hotpink", "cyan", "limegreen", "gold", "violet"]
    st_legend_shown = False

    for sidx, st in enumerate(trackers):
        sname = getattr(st, "name", f"StarTracker {sidx + 1}")
        orient = getattr(st, "orientation", None)
        bore_raw = getattr(orient, "boresight", (1.0, 0.0, 0.0))
        b_vec = _unit(np.asarray(bore_raw, dtype=float))
        sc = st_colors[sidx % len(st_colors)]

        body_size = 0.12

        if _scope_geom is not None:
            # Mount on the telescope tube surface.
            s_start, s_bore, s_outer_r, s_tube_len = _scope_geom
            # Radial direction: component of boresight perpendicular to the tube axis.
            b_perp = b_vec - float(np.dot(b_vec, s_bore)) * s_bore
            radial = (
                _unit(b_perp) if np.linalg.norm(b_perp) > 0.1 else _perp_pair(s_bore)[0]
            )
            # Position at 65 % along the tube, just outside the baffle surface.
            c = s_start + s_bore * (s_tube_len * 0.65) + radial * (s_outer_r + 0.005)
            u = s_bore  # span along tube
            v = _unit(np.cross(s_bore, radial))  # span tangential to tube
            prism_start = c
            prism_end = c + radial * 0.08
        else:
            # No telescope — fall back to bus-face placement.
            c, u, v = _face_placement(b_vec, bus_hd, gap=0.0)
            prism_start = c
            prism_end = c + b_vec * 0.08
        traces.append(
            _flat_rect_mesh(
                (prism_start + prism_end) / 2,
                u,
                v,
                body_size,
                body_size,
                color=_C_ST,
                name=sname,
                opacity=0.92,
                show_legend=not st_legend_shown,
                legendgroup="star_trackers",
            )
        )
        st_legend_shown = True

        # Front face of the star tracker (in tracker color = lens aperture).
        # Offset slightly outward from the prism end in the mounting direction.
        mount_dir = _unit(prism_end - prism_start)
        traces.append(
            _flat_rect_mesh(
                prism_end + mount_dir * 0.001,
                u,
                v,
                body_size * 0.6,
                body_size * 0.6,
                color=sc,
                name=sname,
                opacity=0.80,
                show_legend=False,
                legendgroup="star_trackers",
            )
        )

        if show_normals:
            bore_origin = prism_end + b_vec * 0.05
            traces.extend(
                _arrow_traces(
                    bore_origin,
                    b_vec,
                    _NORMAL_LEN * 0.9,
                    color=sc,
                    name=f"{sname} boresight",
                    show_legend=(sidx == 0),
                    legendgroup="st_normals",
                    line_width=3,
                )
            )

    # ------------------------------------------------------------------ #
    # 6. Spacecraft body-frame coordinate axes                             #
    # ------------------------------------------------------------------ #
    if show_axes:
        origin = np.zeros(3)
        for direction, color, label in [
            (np.array([1.0, 0.0, 0.0]), _C_X, "+X (boresight)"),
            (np.array([0.0, 1.0, 0.0]), _C_Y, "+Y (up)"),
            (np.array([0.0, 0.0, 1.0]), _C_Z, "+Z"),
        ]:
            traces.extend(
                _arrow_traces(
                    origin,
                    direction,
                    _AXIS_LEN,
                    color=color,
                    name=label,
                    cone_sz=0.09,
                    legendgroup="axes",
                    line_width=5,
                )
            )
            label_pos = direction * (_AXIS_LEN + 0.12)
            traces.append(_axis_label(label_pos, label.split()[0], color))

    # ------------------------------------------------------------------ #
    # 7. Figure layout                                                     #
    # ------------------------------------------------------------------ #
    fig_title = title or f"{bus_name} — 3-D Configuration"
    fig = go.Figure(data=traces)

    # Compute a sensible axis range
    scope_reach = hx + 2.5  # telescope extends in +X
    panel_reach = max(hy + 2.5, hz + 1.5)
    rng = max(scope_reach, panel_reach, _AXIS_LEN + 0.5)

    fig.update_layout(
        title=dict(
            text=fig_title,
            font=dict(color="white", size=15),
        ),
        paper_bgcolor=_BG,
        plot_bgcolor=_BG,
        font=dict(color="white"),
        scene=dict(
            xaxis=dict(
                title="X (boresight)",
                range=[-rng, rng],
                backgroundcolor=_BG,
                gridcolor="rgba(80,80,120,0.4)",
                zerolinecolor="rgba(120,120,160,0.6)",
                color="rgba(200,200,220,0.8)",
            ),
            yaxis=dict(
                title="Y (up)",
                range=[-rng, rng],
                backgroundcolor=_BG,
                gridcolor="rgba(80,80,120,0.4)",
                zerolinecolor="rgba(120,120,160,0.6)",
                color="rgba(200,200,220,0.8)",
            ),
            zaxis=dict(
                title="Z",
                range=[-rng, rng],
                backgroundcolor=_BG,
                gridcolor="rgba(80,80,120,0.4)",
                zerolinecolor="rgba(120,120,160,0.6)",
                color="rgba(200,200,220,0.8)",
            ),
            bgcolor=_BG,
            aspectmode="cube",
            camera=dict(
                eye=dict(x=1.6, y=0.9, z=0.7),
                up=dict(x=0, y=1, z=0),
            ),
        ),
        legend=dict(
            bgcolor="rgba(20,20,40,0.85)",
            bordercolor="rgba(120,120,170,0.5)",
            borderwidth=1,
            font=dict(color="white", size=11),
            itemsizing="constant",
        ),
        margin=dict(l=0, r=0, t=50, b=0),
        height=700,
    )

    return fig
