"""Roll computation helpers."""

from __future__ import annotations

import numpy as np
import rust_ephem

from ..common import ACSMode, dtutcfromtimestamp, scbodyvector
from ..config import DTOR, Constraint, SolarPanelSet


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
    reference_roll: float | None = None,
    max_roll_delta: float | None = None,
    acs_mode: ACSMode | None = None,
    in_eclipse: bool | None = None,
    drive_preview_seconds: float = 0.0,
) -> float:
    """Calculate the optimum roll angle (degrees in [0,360)).

    - If `solar_panel` is None: return the closed-form optimum that **maximises
      the Sun's Y-component** in the spacecraft body frame (i.e. maximises
      illumination on a +Y-normal panel), obtained by differentiating
      ``s_y(θ) = s_y0·cos(θ) + s_z0·sin(θ)`` and solving.
    - If provided: maximise the total weighted power across all panels by
      scanning roll in 1° increments.
    - If `constraint` is provided: restrict candidate rolls to those allowed by
      the combined constraint (via ``roll_range``).  If the constraint blocks all
      rolls (fully blocked pointing) the function falls back to the unconstrained
      optimum.
    - If `reference_roll` and `max_roll_delta` are provided: restrict the search
      to rolls within that shortest-path angular distance. If no integer-degree
      candidate is reachable, hold the reference roll.
    - Finite array drives hold their current physical angle during candidate
      evaluation unless ``drive_preview_seconds`` explicitly grants motion.
      ``in_eclipse`` controls whether each drive's eclipse tracking policy permits
      that motion and is required when the interval is positive.
    """
    if (reference_roll is None) != (max_roll_delta is None):
        raise ValueError("reference_roll and max_roll_delta must be provided together")
    if max_roll_delta is not None and max_roll_delta < 0:
        raise ValueError("max_roll_delta must be non-negative")
    if not np.isfinite(drive_preview_seconds) or drive_preview_seconds < 0.0:
        raise ValueError("drive_preview_seconds must be finite and non-negative")
    if drive_preview_seconds > 0.0 and in_eclipse is None:
        raise ValueError(
            "in_eclipse must be provided when candidate drive motion is enabled"
        )

    # Fetch ephemeris index and Sun vector from pre-computed arrays
    index = ephem.index(dtutcfromtimestamp(utime))
    sunvec = ephem.sun_pv.position[index] - ephem.gcrs_pv.position[index]  # km

    # Sun vector in body coordinates for roll=0
    s_body_0 = scbodyvector(ra * DTOR, dec * DTOR, 0.0, sunvec)
    s = np.asarray(s_body_0, dtype=float)
    s_norm = s / np.linalg.norm(s)

    # Build valid-roll mask from constraint (None if unconstrained or all valid)
    candidate_mask = _roll_valid_mask(ra, dec, utime, ephem, constraint)
    deg = np.arange(360.0, dtype=float)
    if reference_roll is not None and max_roll_delta is not None:
        reference_roll %= 360.0
        roll_delta = np.abs((deg - reference_roll + 180.0) % 360.0 - 180.0)
        reachable_mask = roll_delta <= max_roll_delta + 1e-9
        candidate_mask = (
            reachable_mask
            if candidate_mask is None
            else candidate_mask & reachable_mask
        )
        if not candidate_mask.any():
            return reference_roll

    def _analytic_roll() -> float:
        roll_rad = np.arctan2(s_norm[2], s_norm[1])
        return float((roll_rad / DTOR) % 360.0)

    if solar_panel is None or not solar_panel.panels:
        # Analytic optimum for side-mounted panel (0,1,0): max y_body = cos(θ)*y0 + sin(θ)*z0
        # d/dθ = 0 → θ = atan2(z0, y0)
        if candidate_mask is None:
            return _analytic_roll()
        # Constraint present: scan 360° with illumination model for a (0,1,0) panel
        ang = deg * DTOR
        illum = np.cos(ang) * s_norm[1] + np.sin(ang) * s_norm[2]
        totals = np.where(candidate_mask, illum, -np.inf)
        return float(deg[int(np.argmax(totals))])

    angles_rad = deg * DTOR
    sun_body_candidates = np.column_stack(
        (
            np.full_like(deg, s_norm[0]),
            np.cos(angles_rad) * s_norm[1] + np.sin(angles_rad) * s_norm[2],
            -np.sin(angles_rad) * s_norm[1] + np.cos(angles_rad) * s_norm[2],
        )
    )
    totals = solar_panel.preview_power_from_normalized_sun_body(
        sun_body_candidates,
        acs_mode=acs_mode,
        in_eclipse=bool(in_eclipse),
        elapsed_seconds=drive_preview_seconds,
    )
    if candidate_mask is not None:
        totals = np.where(candidate_mask, totals, -np.inf)
    return float(deg[int(np.argmax(totals))])


def optimum_roll_sidemount(
    ra: float, dec: float, utime: float, ephem: rust_ephem.Ephemeris
) -> float:
    """Calculate the optimum Roll angle (in degrees) for a given Ra, Dec
    and Unix Time"""
    return optimum_roll(ra, dec, utime, ephem)
