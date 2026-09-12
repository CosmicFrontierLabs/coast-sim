"""Roll computation helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import rust_ephem

from ..common import dtutcfromtimestamp, scbodyvector
from ..common.enums import ACSMode
from ..config import DTOR, Constraint, SolarPanelSet, Telescope
from ..config.constraint import (
    AttitudeConstraintScope,
    mounted_science_attitude_constraint_names,
)


def _panel_power_inputs(
    solar_panel: SolarPanelSet | None,
    telescope: Telescope | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return panel normals, weights, and gimbal flags in the pointing frame."""
    panels = solar_panel.panels if solar_panel is not None else []
    if not panels:
        normals = [(0.0, 1.0, 0.0)]
        weights = [1.0]
        gimbled = [False]
    else:
        assert solar_panel is not None
        default_efficiency = solar_panel.conversion_efficiency
        normals = [panel.normal for panel in panels]
        weights = [
            panel.max_power
            * (
                panel.conversion_efficiency
                if panel.conversion_efficiency is not None
                else default_efficiency
            )
            for panel in panels
        ]
        gimbled = [panel.gimbled for panel in panels]

    if telescope is not None:
        normals = [telescope.mounting.instrument_vector(normal) for normal in normals]
    return (
        np.asarray(normals, dtype=float),
        np.asarray(weights, dtype=float),
        np.asarray(gimbled, dtype=bool),
    )


def _panel_power_by_roll(
    sun_at_zero_roll: npt.NDArray[np.float64],
    normals: npt.NDArray[np.float64],
    weights: npt.NDArray[np.float64],
    gimbled: npt.NDArray[np.bool_],
) -> npt.NDArray[np.float64]:
    """Vectorized panel power score for integer rolls from 0 through 359 degrees."""
    sun = sun_at_zero_roll / np.linalg.norm(sun_at_zero_roll)
    angles = np.arange(360.0) * DTOR
    cosine = np.cos(angles)[:, None]
    sine = np.sin(angles)[:, None]
    illumination = (
        normals[None, :, 0] * sun[0]
        + cosine * (normals[None, :, 1] * sun[1] + normals[None, :, 2] * sun[2])
        + sine * (normals[None, :, 1] * sun[2] - normals[None, :, 2] * sun[1])
    )
    illumination = np.maximum(illumination, 0.0)
    illumination[:, gimbled] = 1.0
    result: npt.NDArray[np.float64] = np.asarray(
        illumination @ weights, dtype=np.float64
    )
    return result


def _mounted_optimum_roll(
    ra: float,
    dec: float,
    utime: float,
    ephem: rust_ephem.Ephemeris,
    telescope: Telescope,
    solar_panel: SolarPanelSet | None,
    constraint: Constraint | None,
    reference_roll: float | None,
    max_roll_delta: float | None,
) -> float:
    """Optimize instrument roll while evaluating the physical body attitude."""
    degrees = np.arange(360.0, dtype=float)
    candidate_mask = np.ones(360, dtype=bool)
    if reference_roll is not None and max_roll_delta is not None:
        reference_roll %= 360.0
        roll_delta = np.abs((degrees - reference_roll + 180.0) % 360.0 - 180.0)
        candidate_mask &= roll_delta <= max_roll_delta + 1e-9
        if not candidate_mask.any():
            return reference_roll

    index = ephem.index(dtutcfromtimestamp(utime))
    sun_eci = np.asarray(
        ephem.sun_pv.position[index] - ephem.gcrs_pv.position[index],
        dtype=float,
    )
    sun_instrument = scbodyvector(ra * DTOR, dec * DTOR, 0.0, sun_eci)
    scores = _panel_power_by_roll(
        sun_instrument, *_panel_power_inputs(solar_panel, telescope)
    )
    scores[~candidate_mask] = -np.inf

    # Test candidates in descending power order. Break equal-power ties by the
    # smallest roll change when a reference exists, then by increasing angle.
    # This keeps a flat power curve at 0 degrees instead of selecting 359 degrees.
    tie_distance = (
        np.abs((degrees - reference_roll + 180.0) % 360.0 - 180.0)
        if reference_roll is not None
        else degrees
    )
    candidate_order = np.lexsort((degrees, tie_distance, -scores))
    for candidate in candidate_order:
        if not np.isfinite(scores[candidate]):
            break
        attitude = telescope.target_body_attitude(ra, dec, float(candidate))
        violations = (
            mounted_science_attitude_constraint_names(
                constraint,
                list(AttitudeConstraintScope),
                (ra, dec, float(candidate)),
                attitude,
                utime,
                ACSMode.SCIENCE,
            )
            if constraint is not None
            else []
        )
        if not violations:
            return float(candidate)

    # Match the legacy fail-open return contract; the caller's locked-attitude
    # validation rejects the target when every candidate is constrained.
    best = int(np.argmax(scores))
    return float(best) if np.isfinite(scores[best]) else float(reference_roll or 0.0)


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
    telescope: Telescope | None = None,
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
    """
    if (reference_roll is None) != (max_roll_delta is None):
        raise ValueError("reference_roll and max_roll_delta must be provided together")
    if max_roll_delta is not None and max_roll_delta < 0:
        raise ValueError("max_roll_delta must be non-negative")

    if telescope is not None and not telescope.mounting.is_identity:
        return _mounted_optimum_roll(
            ra,
            dec,
            utime,
            ephem,
            telescope,
            solar_panel,
            constraint,
            reference_roll,
            max_roll_delta,
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

    totals = _panel_power_by_roll(s_norm, *_panel_power_inputs(solar_panel))

    # Apply valid-roll mask if present
    if candidate_mask is not None:
        totals = np.where(candidate_mask, totals, -np.inf)

    # Argmax over angles
    best_idx = int(np.argmax(totals))
    return float(deg[best_idx])


def optimum_roll_sidemount(
    ra: float, dec: float, utime: float, ephem: rust_ephem.Ephemeris
) -> float:
    """Calculate the optimum Roll angle (in degrees) for a given Ra, Dec
    and Unix Time"""
    return optimum_roll(ra, dec, utime, ephem)
