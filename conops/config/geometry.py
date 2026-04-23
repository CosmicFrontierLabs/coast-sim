"""Panel geometry and inter-component shadow computation.

Provides ``PanelGeometry``, a Pydantic model describing a rectangular panel in the
spacecraft body frame, and ``compute_shadow_fraction``, which calculates what fraction
of a receiver panel is shadowed by one or more occluder panels for a given sun direction.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from pydantic import Field, field_validator
from shapely.geometry import Polygon
from shapely.ops import unary_union

from ._base import ConfigModel

if TYPE_CHECKING:
    from shapely.geometry.base import BaseGeometry

__all__ = ["PanelGeometry", "compute_shadow_fraction"]


class PanelGeometry(ConfigModel):
    """3D rectangular panel geometry in spacecraft body frame.

    Describes the spatial extent of a flat rectangular panel (solar panel or radiator)
    for inter-component shadow computation.

    The panel occupies the region::

        center_m + a * u * width_m/2 + b * v * height_m/2   for a, b ∈ [-1, 1]

    The outward normal is implicitly ``u × v`` (right-hand rule).  When attached to a
    ``Radiator``, this should be consistent with ``orientation.normal``.

    Example — solar panel in the XZ plane, facing +Y, 2 m × 1 m::

        PanelGeometry(
            center_m=(0.0, 0.0, 0.0),
            u=(1.0, 0.0, 0.0),
            v=(0.0, 0.0, 1.0),
            width_m=2.0,
            height_m=1.0,
        )
    """

    center_m: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="Panel centre position in spacecraft body frame (m)",
    )
    u: tuple[float, float, float] = Field(
        description="First spanning direction (unit vector, width axis)",
    )
    v: tuple[float, float, float] = Field(
        description="Second spanning direction (unit vector, height axis). Should be orthogonal to u.",
    )
    width_m: float = Field(default=1.0, gt=0.0, description="Panel width along u (m)")
    height_m: float = Field(default=1.0, gt=0.0, description="Panel height along v (m)")

    @field_validator("u", "v")
    @classmethod
    def _validate_unit_vector(
        cls, vec: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        mag = float(np.sqrt(sum(x**2 for x in vec)))
        if mag < 0.99 or mag > 1.01:
            raise ValueError(
                f"PanelGeometry spanning vectors must be unit vectors, got magnitude {mag:.4f}"
            )
        return vec

    @property
    def normal(self) -> npt.NDArray[np.float64]:
        """Outward normal as u × v (unit vector when u ⊥ v)."""
        return np.cross(
            np.array(self.u, dtype=np.float64), np.array(self.v, dtype=np.float64)
        )

    @property
    def corners(self) -> npt.NDArray[np.float64]:
        """Four corners of the panel ordered CCW, shape (4, 3)."""
        c = np.array(self.center_m, dtype=np.float64)
        hu = np.array(self.u, dtype=np.float64) * (self.width_m / 2.0)
        hv = np.array(self.v, dtype=np.float64) * (self.height_m / 2.0)
        return np.array([c - hu - hv, c + hu - hv, c + hu + hv, c - hu + hv])


def _shadow_poly_2d(
    sun_unit: npt.NDArray[np.float64],
    occluder: PanelGeometry,
    receiver: PanelGeometry,
) -> BaseGeometry | None:
    """Project occluder corners along sun direction onto receiver plane.

    Returns a Shapely geometry in receiver (u, v) coordinates, or None if no
    valid shadow exists (e.g. sun is parallel to receiver, or occluder is on
    the wrong side of the receiver plane).
    """
    n_rec = receiver.normal
    s_dot_n = float(np.dot(sun_unit, n_rec))

    # Use abs: n_rec = u×v may be antiparallel to the component's orientation.normal.
    # The sign of s_dot_n only affects the sign of t, which we check explicitly below.
    if abs(s_dot_n) < 1e-9:
        # Sun nearly parallel to receiver face — no projection possible.
        return None

    c_rec = np.array(receiver.center_m, dtype=np.float64)
    rec_u = np.array(receiver.u, dtype=np.float64)
    rec_v = np.array(receiver.v, dtype=np.float64)

    pts_2d: list[tuple[float, float]] = []
    for q in occluder.corners:
        # t is the parameter such that (q - t*sun_unit) lies on the receiver plane.
        # t > 0 means the occluder corner is on the sun-side of the receiver plane.
        # Include t == 0 (corner exactly on receiver plane) as a valid boundary point.
        t = float(np.dot(q - c_rec, n_rec)) / s_dot_n
        if t < -1e-12:
            continue
        shadow_pt = q - t * sun_unit
        pts_2d.append(
            (
                float(np.dot(shadow_pt - c_rec, rec_u)),
                float(np.dot(shadow_pt - c_rec, rec_v)),
            )
        )

    if len(pts_2d) < 3:
        return None

    geom: BaseGeometry = Polygon(pts_2d)
    if not geom.is_valid or geom.is_empty:
        geom = geom.convex_hull
    if not geom.is_valid or geom.is_empty:
        return None
    return geom


def compute_shadow_fraction(
    sun_unit: npt.NDArray[np.float64],
    occluders: list[PanelGeometry],
    receiver: PanelGeometry,
) -> float:
    """Return the fraction [0, 1] of receiver area in shadow from all occluders.

    Shadows from multiple occluders are unioned before computing the overlap with the
    receiver rectangle, so the result is exact for non-overlapping shadows and correct
    for overlapping ones.

    Returns 0.0 when:
    - ``occluders`` is empty
    - The sun direction is nearly parallel to the receiver face
    - No occluder projects onto the receiver

    Args:
        sun_unit: Unit vector pointing toward the sun in the spacecraft body frame.
        occluders: List of occluder panel geometries (e.g. solar panels).
        receiver: The panel whose shadowed area is to be computed (e.g. radiator).

    Returns:
        Shadow fraction in [0, 1].
    """
    if not occluders:
        return 0.0

    shadow_polys: list[BaseGeometry] = []
    for occ in occluders:
        poly = _shadow_poly_2d(sun_unit, occ, receiver)
        if poly is not None:
            shadow_polys.append(poly)

    if not shadow_polys:
        return 0.0

    hw = receiver.width_m / 2.0
    hh = receiver.height_m / 2.0
    receiver_poly = Polygon([(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)])

    combined_shadow: BaseGeometry = (
        unary_union(shadow_polys) if len(shadow_polys) > 1 else shadow_polys[0]
    )

    try:
        intersection = receiver_poly.intersection(combined_shadow)
    except Exception:
        return 0.0

    if intersection.is_empty:
        return 0.0

    return float(intersection.area) / (receiver.width_m * receiver.height_m)
