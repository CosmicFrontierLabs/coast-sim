from __future__ import annotations

import numpy as np
import rust_ephem

from ..common import dtutcfromtimestamp, scbodyvector
from ..common.vector import boresight_axis_permutation
from ..config import DTOR, Constraint, SolarPanelSet

"""Roll computation helpers."""


def _roll_valid_mask(
    ra: float,
    dec: float,
    utime: float,
    ephem: rust_ephem.Ephemeris,
    constraint: Constraint | None,
) -> np.ndarray | None:
    """Return a (360,) bool mask of valid rolls, or None if unconstrained.

    Calls ``roll_range`` on the combined rust-ephem constraint object.  Returns
    ``None`` when no constraint is present, when the target is fully blocked at
    every roll (fall back to unconstrained), or when every roll is valid
    (shortcut: no restriction needed).
    """
    if constraint is None or constraint.roll_dependent_constraint is None:
        return None
    # Only apply constraint masking when ignore_roll=True.
    # With ignore_roll=False the scheduler already gated visibility on the
    # solar-optimal roll satisfying constraints, so re-sweeping roll_range()
    # at every ACS step (for every 60-second DITL tick) is unnecessary and
    # expensive for constraints that include roll-dependent components like
    # BoresightOffsetConstraint (star-tracker keep-outs).
    if not constraint.ignore_roll:
        return None
    # Snap to the nearest ephemeris timestamp — roll_range() requires an exact
    # match and utime may fall between grid points.
    idx = ephem.index(dtutcfromtimestamp(utime))
    snapped_dt = ephem.timestamp[idx]
    # Use only roll-dependent sub-constraints (star trackers, radiators, telescope
    # offsets). Roll-independent constraints (sun/earth/moon on the main boresight)
    # return [] from roll_range(), which OrConstraint misinterprets as "no valid
    # rolls" when combined via |.
    valid_ranges: list[tuple[float, float]] = (
        constraint.roll_dependent_constraint.roll_range(
            time=snapped_dt, ephemeris=ephem, target_ra=ra, target_dec=dec
        )
    )
    if not valid_ranges:
        # Fully blocked at all rolls — return None and let caller fall back
        return None
    mask = np.zeros(360, dtype=bool)
    for start, end in valid_ranges:
        lo = int(round(start)) % 360
        hi = int(round(end)) % 360
        if lo <= hi:
            mask[lo : hi + 1] = True
        else:
            # Interval wraps around 0°/360°
            mask[lo:] = True
            mask[: hi + 1] = True
    if mask.all():
        return None  # All rolls valid — no restriction
    return mask


def optimum_roll(
    ra: float,
    dec: float,
    utime: float,
    ephem: rust_ephem.Ephemeris,
    solar_panel: SolarPanelSet | None = None,
    constraint: Constraint | None = None,
    boresight_axis: str = "+X",
) -> float:
    """Calculate the optimum roll angle (degrees in [0,360)).

    - If `solar_panel` is None: return the closed-form optimum that **maximises
      the Sun's Y-component** in the spacecraft body frame (i.e. maximises
      illumination on a +Y-normal panel), obtained by differentiating
      ``s_y(θ) = s_y0·cos(θ) − s_z0·sin(θ)`` and solving.
    - If provided: maximise the total weighted power across all panels by
      scanning roll in 1° increments.
    - If `constraint` is provided: restrict candidate rolls to those allowed by
      the combined constraint (via ``roll_range``).  If the constraint blocks all
      rolls (fully blocked pointing) the function falls back to the unconstrained
      optimum.
    """
    # Fetch ephemeris index and Sun vector from pre-computed arrays
    index = ephem.index(dtutcfromtimestamp(utime))
    sunvec = ephem.sun_pv.position[index] - ephem.gcrs_pv.position[index]  # km

    # Sun vector in body coordinates for roll=0
    s_body_0 = scbodyvector(ra * DTOR, dec * DTOR, 0.0, sunvec)
    s = np.asarray(s_body_0, dtype=float)
    s_norm = s / np.linalg.norm(s)

    # Build valid-roll mask from constraint (None if unconstrained or all valid)
    valid_mask = _roll_valid_mask(ra, dec, utime, ephem, constraint)

    def _analytic_roll() -> float:
        roll_rad = np.arctan2(-s_norm[2], s_norm[1])
        return float((roll_rad / DTOR) % 360.0)

    if solar_panel is None or not solar_panel.panels:
        # Analytic optimum for side-mounted panel (0,1,0): max y_body = cos(θ)*y0 - sin(θ)*z0
        # d/dθ = 0 → θ = atan2(-z0, y0)
        if valid_mask is None:
            return _analytic_roll()
        # Constraint present: scan 360° with illumination model for a (0,1,0) panel
        deg = np.arange(360.0, dtype=float)
        ang = deg * DTOR
        illum = np.cos(ang) * s_norm[1] - np.sin(ang) * s_norm[2]
        totals = np.where(valid_mask, illum, -np.inf)
        if not valid_mask.any():
            return _analytic_roll()
        return float(deg[int(np.argmax(totals))])

    # Weighted optimization using actual panel geometry (vectorized).
    # solar_panel is non-None and has panels here.
    panels = solar_panel.panels
    default_eff = solar_panel.conversion_efficiency
    base_normals = []
    weights = []  # max_power * efficiency
    for p in panels:
        base_normals.append(np.array(p.normal, dtype=float))
        eff = (
            p.conversion_efficiency
            if p.conversion_efficiency is not None
            else default_eff
        )
        weights.append(p.max_power * eff)

    # Convert lists to arrays
    n_mat = np.asarray(base_normals, dtype=float)  # shape (P,3) in user's frame
    w_vec = np.asarray(weights, dtype=float)  # shape (P,)

    # Convert panel normals from user's body frame to the internal +X-boresight
    # frame.  scbodyvector() always returns vectors in the internal frame, so
    # the illumination dot-products below must use consistent coordinates.
    # n_internal[i] = P^T @ n_user[i]; in matrix form: N_int = N_user @ P.
    if boresight_axis != "+X":
        p_mat = boresight_axis_permutation(boresight_axis)
        n_mat = n_mat @ p_mat  # shape (P,3) now in internal frame

    # For a spacecraft roll of θ about the body +X (boresight) axis the Sun
    # vector expressed in the body frame evolves as (right-hand rule):
    #   s_body_y(θ) = s_y · cos(θ) − s_z · sin(θ)
    #   s_body_z(θ) = s_y · sin(θ) + s_z · cos(θ)
    # (s_x is unchanged; s_y, s_z are the roll=0 body-frame components)
    #
    # Panel illumination = n · s_body(θ)  (panel normal n is fixed in the body frame):
    #   illum(θ) = n_x·s_x + cos(θ)·(n_y·s_y + n_z·s_z) + sin(θ)·(n_z·s_y − n_y·s_z)

    # Precompute per-panel coefficients:
    #   illum(θ) = a + cos(θ)·b + sin(θ)·c
    # where a = nx·sx, b = ny·sy + nz·sz, c = nz·sy − ny·sz
    a_coef = n_mat[:, 0] * s_norm[0]
    b_coef = n_mat[:, 1] * s_norm[1] + n_mat[:, 2] * s_norm[2]
    c_coef = n_mat[:, 2] * s_norm[1] - n_mat[:, 1] * s_norm[2]

    # Angles 0..359 degrees
    deg = np.arange(360.0, dtype=float)
    ang = deg * DTOR
    cos_t = np.cos(ang)  # (360,)
    sin_t = np.sin(ang)  # (360,)

    # Illumination per angle and panel: (360,P)
    # Broadcasting: cos_t[:,None]*B[None,:] etc.
    illum = (
        a_coef[None, :]
        + cos_t[:, None] * b_coef[None, :]
        + sin_t[:, None] * c_coef[None, :]
    )
    illum = np.maximum(illum, 0.0)

    # Total weighted power per angle: (360,)
    totals = illum * w_vec[None, :]
    totals = totals.sum(axis=1)

    # Apply valid-roll mask if present
    if valid_mask is not None and valid_mask.any():
        totals = np.where(valid_mask, totals, -np.inf)

    # Argmax over angles
    best_idx = int(np.argmax(totals))
    return float(deg[best_idx])


def optimum_roll_sidemount(
    ra: float,
    dec: float,
    utime: float,
    ephem: rust_ephem.Ephemeris,
    boresight_axis: str = "+X",
) -> float:
    """Calculate the optimum Roll angle (in degrees) for a given Ra, Dec
    and Unix Time"""
    # Analytic optimum: choose roll that minimizes the Z-component of the
    # Sun vector in the spacecraft body frame (roll=free about X).
    # This maximizes illumination for side-mounted panels (and general canted
    # panels derived from -Z toward +X), independent of panel cant magnitude.

    # Fetch ephemeris index and Sun vector from pre-computed arrays
    index = ephem.index(dtutcfromtimestamp(utime))
    sunvec = ephem.sun_pv.position[index] - ephem.gcrs_pv.position[index]  # km

    # Sun vector in body coordinates for roll=0
    s_body_0 = scbodyvector(ra * DTOR, dec * DTOR, 0.0, sunvec)
    y0 = s_body_0[1]
    z0 = s_body_0[2]

    # Maximize y_body = cos(roll)*y0 - sin(roll)*z0
    # d/d(roll) = 0 -> -sin(roll)*y0 - cos(roll)*z0 = 0
    # => tan(roll) = -z0 / y0 -> roll = atan2(-z0, y0)
    roll_rad = np.arctan2(-z0, y0)

    # Return degrees in [0, 360)
    roll_deg = (roll_rad / DTOR) % 360.0
    return float(roll_deg)
