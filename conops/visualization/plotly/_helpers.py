"""Shared helper utilities for Plotly visualizations."""

from __future__ import annotations

from typing import Any

import matplotlib.colors as mcolors
import numpy as np
import plotly.graph_objects as go


def lighten_color(color: str, factor: float = 0.5) -> str:
    """Lighten a color by blending it with white.

    Parameters
    ----------
    color : str
        Input color (named color, hex string, or RGB-compatible value).
    factor : float
        Blend factor where 0.0 keeps the original color and 1.0 returns white.

    Returns
    -------
    str
        Lightened color as a hex string.
    """
    rgb = mcolors.to_rgb(color)
    r, g, b = rgb
    lightened = (
        (1 - factor) * r + factor,
        (1 - factor) * g + factor,
        (1 - factor) * b + factor,
    )
    return mcolors.to_hex(lightened)


def _to_rgba(color: str, alpha: float) -> str:
    """Convert a named/hex colour + alpha to a Plotly ``rgba()`` string."""
    rgb = mcolors.to_rgb(color)
    r, g, b = int(rgb[0] * 255), int(rgb[1] * 255), int(rgb[2] * 255)
    return f"rgba({r},{g},{b},{alpha:.3f})"


def _ra_to_lon(ra_deg: float) -> float:
    """Map RA (0–360°) to Plotly scattergeo longitude (–180…+180°)."""
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

    phi = np.linspace(0.0, 2.0 * np.pi, n + 1)

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
