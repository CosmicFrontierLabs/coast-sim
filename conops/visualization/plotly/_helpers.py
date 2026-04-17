"""Shared helper utilities for Plotly visualizations."""

import matplotlib.colors as mcolors


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
