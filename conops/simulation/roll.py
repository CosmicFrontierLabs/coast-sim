"""Roll computation helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import rust_ephem

from ..common import dtutcfromtimestamp, scbodyvector
from ..common.enums import ACSMode
from ..config import (
    DTOR,
    Constraint,
    SolarArrayDriveState,
    SolarPanelSet,
    Telescope,
)
from ..config.constraint import (
    AttitudeConstraintScope,
    mounted_science_attitude_constraint_names,
)

_POWER_SCORE_RTOL = 1e-12
_POWER_SCORE_ATOL_W = 1e-12
_ROLL_DEGREES = np.arange(360.0, dtype=np.float64)


def _candidate_sun_vectors(
    sun_at_zero_roll: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """Return normalized pointing-frame Sun vectors for integer roll candidates."""
    sun = np.asarray(sun_at_zero_roll, dtype=np.float64)
    sun /= np.linalg.norm(sun)
    angles = _ROLL_DEGREES * DTOR
    cosine = np.cos(angles)
    sine = np.sin(angles)
    return np.column_stack(
        (
            np.full_like(_ROLL_DEGREES, sun[0]),
            cosine * sun[1] + sine * sun[2],
            -sine * sun[1] + cosine * sun[2],
        )
    )


def _power_scores(
    sun_body_candidates: npt.NDArray[np.float64],
    solar_panel: SolarPanelSet | None,
    drive_state: SolarArrayDriveState | None,
) -> npt.NDArray[np.float64]:
    """Score candidate body-frame Sun vectors at one immutable drive state."""
    if solar_panel is None or not solar_panel.panels:
        return np.maximum(sun_body_candidates[:, 1], 0.0)
    return solar_panel.power_from_normalized_sun_body(
        sun_body_candidates, drive_state=drive_state
    )


def _power_score_order(
    scores: npt.NDArray[np.float64],
    reference_roll: float | None,
) -> npt.NDArray[np.int64]:
    """Order finite candidates by power with deterministic tolerance-aware ties."""
    finite = np.flatnonzero(np.isfinite(scores))
    ranked = finite[np.argsort(-scores[finite], kind="stable")]
    tie_distance = (
        np.abs((_ROLL_DEGREES - reference_roll + 180.0) % 360.0 - 180.0)
        if reference_roll is not None
        else _ROLL_DEGREES
    )
    ordered: list[int] = []
    start = 0
    while start < ranked.size:
        stop = start + 1
        while stop < ranked.size and np.isclose(
            scores[ranked[stop]],
            scores[ranked[start]],
            rtol=_POWER_SCORE_RTOL,
            atol=_POWER_SCORE_ATOL_W,
        ):
            stop += 1
        tied = ranked[start:stop]
        tie_order = np.lexsort((_ROLL_DEGREES[tied], tie_distance[tied]))
        ordered.extend(int(candidate) for candidate in tied[tie_order])
        start = stop
    return np.asarray(ordered, dtype=np.int64)


def _validate_reachable_rolls(
    reference_roll: float | None,
    max_roll_delta: float | None,
) -> tuple[float | None, npt.NDArray[np.bool_] | None]:
    if (reference_roll is None) != (max_roll_delta is None):
        raise ValueError("reference_roll and max_roll_delta must be provided together")
    if max_roll_delta is not None and max_roll_delta < 0.0:
        raise ValueError("max_roll_delta must be non-negative")
    if reference_roll is None or max_roll_delta is None:
        return None, None
    reference = reference_roll % 360.0
    delta = np.abs((_ROLL_DEGREES - reference + 180.0) % 360.0 - 180.0)
    return reference, delta <= max_roll_delta + 1e-9


def _roll_valid_mask(
    ra: float,
    dec: float,
    utime: float,
    ephem: rust_ephem.Ephemeris,
    constraint: Constraint | None,
) -> npt.NDArray[np.bool_] | None:
    """Return valid body-roll candidates, or ``None`` for an unrestricted search."""
    if constraint is None or constraint.roll_dependent_constraint is None:
        return None
    if not constraint.ignore_roll:
        return None
    idx = ephem.index(dtutcfromtimestamp(utime))
    valid_ranges: list[tuple[float, float]] = (
        constraint.roll_dependent_constraint.roll_range(
            time=ephem.timestamp[idx],
            ephemeris=ephem,
            target_ra=ra,
            target_dec=dec,
        )
    )
    if not valid_ranges:
        return None
    mask = np.zeros(360, dtype=bool)
    for start, end in valid_ranges:
        lo = int(round(start)) % 360
        hi = int(round(end)) % 360
        if lo <= hi:
            mask[lo : hi + 1] = True
        else:
            mask[lo:] = True
            mask[: hi + 1] = True
    return None if mask.all() else mask


def _sun_at_zero_roll(
    ra: float, dec: float, utime: float, ephem: rust_ephem.Ephemeris
) -> npt.NDArray[np.float64]:
    index = ephem.index(dtutcfromtimestamp(utime))
    sun_eci = ephem.sun_pv.position[index] - ephem.gcrs_pv.position[index]
    return np.asarray(scbodyvector(ra * DTOR, dec * DTOR, 0.0, sun_eci), dtype=float)


def optimum_body_roll(
    ra: float,
    dec: float,
    utime: float,
    ephem: rust_ephem.Ephemeris,
    solar_panel: SolarPanelSet | None = None,
    constraint: Constraint | None = None,
    reference_roll: float | None = None,
    max_roll_delta: float | None = None,
    drive_state: SolarArrayDriveState | None = None,
) -> float:
    """Return the power-optimal physical body roll at the current array state."""
    reference, reachable = _validate_reachable_rolls(reference_roll, max_roll_delta)
    sun_at_zero = _sun_at_zero_roll(ra, dec, utime, ephem)
    candidate_mask = _roll_valid_mask(ra, dec, utime, ephem, constraint)
    if reachable is not None:
        candidate_mask = (
            reachable if candidate_mask is None else candidate_mask & reachable
        )
        if not candidate_mask.any():
            assert reference is not None
            return reference

    if (
        (solar_panel is None or not solar_panel.panels)
        and candidate_mask is None
        and reference is None
    ):
        sun = sun_at_zero / np.linalg.norm(sun_at_zero)
        return float((np.arctan2(sun[2], sun[1]) / DTOR) % 360.0)

    scores = _power_scores(
        _candidate_sun_vectors(sun_at_zero), solar_panel, drive_state
    )
    if candidate_mask is not None:
        scores = np.where(candidate_mask, scores, -np.inf)
    order = _power_score_order(scores, reference)
    return float(_ROLL_DEGREES[order[0]]) if order.size else float(reference or 0.0)


def optimum_instrument_roll(
    ra: float,
    dec: float,
    utime: float,
    ephem: rust_ephem.Ephemeris,
    telescope: Telescope,
    solar_panel: SolarPanelSet | None = None,
    constraint: Constraint | None = None,
    reference_roll: float | None = None,
    max_roll_delta: float | None = None,
    drive_state: SolarArrayDriveState | None = None,
) -> float:
    """Optimize science roll while scoring panels in the physical body frame."""
    if telescope.mounting.is_identity:
        return optimum_body_roll(
            ra,
            dec,
            utime,
            ephem,
            solar_panel,
            constraint,
            reference_roll,
            max_roll_delta,
            drive_state,
        )

    reference, reachable = _validate_reachable_rolls(reference_roll, max_roll_delta)
    sun_instrument = _candidate_sun_vectors(_sun_at_zero_roll(ra, dec, utime, ephem))
    mounting = telescope.mounting
    body_from_instrument = np.column_stack(
        (
            mounting.body_vector((1.0, 0.0, 0.0)),
            mounting.body_vector((0.0, 1.0, 0.0)),
            mounting.body_vector((0.0, 0.0, 1.0)),
        )
    )
    sun_body = sun_instrument @ body_from_instrument.T
    scores = _power_scores(sun_body, solar_panel, drive_state)
    if reachable is not None:
        scores = np.where(reachable, scores, -np.inf)
        if not reachable.any():
            assert reference is not None
            return reference

    order = _power_score_order(scores, reference)
    for candidate in order:
        instrument_roll = float(_ROLL_DEGREES[candidate])
        attitude = telescope.target_body_attitude(ra, dec, instrument_roll)
        violations = (
            mounted_science_attitude_constraint_names(
                constraint,
                list(AttitudeConstraintScope),
                (ra, dec, instrument_roll),
                attitude,
                utime,
                ACSMode.SCIENCE,
            )
            if constraint is not None
            else []
        )
        if not violations:
            return instrument_roll

    # Preserve the established fail-open contract. Locked-attitude validation
    # rejects a target when every science-roll candidate is constrained.
    return float(_ROLL_DEGREES[order[0]]) if order.size else float(reference or 0.0)


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
    drive_state: SolarArrayDriveState | None = None,
) -> float:
    """Compatibility wrapper selecting body- or instrument-frame optimization."""
    if telescope is not None:
        return optimum_instrument_roll(
            ra,
            dec,
            utime,
            ephem,
            telescope,
            solar_panel,
            constraint,
            reference_roll,
            max_roll_delta,
            drive_state,
        )
    return optimum_body_roll(
        ra,
        dec,
        utime,
        ephem,
        solar_panel,
        constraint,
        reference_roll,
        max_roll_delta,
        drive_state,
    )


def optimum_roll_sidemount(
    ra: float, dec: float, utime: float, ephem: rust_ephem.Ephemeris
) -> float:
    """Return the legacy +Y-panel optimum body roll."""
    return optimum_body_roll(ra, dec, utime, ephem)
