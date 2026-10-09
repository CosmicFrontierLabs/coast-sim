"""Lazy tie grouping must preserve the original NumPy ranking exactly."""

import numpy as np
import pytest

from conops.simulation.roll import _power_score_order


def candidate_order(scores, reference_roll):
    return _power_score_order(scores, reference_roll)


def numpy_reference(scores, reference_roll):
    finite = np.flatnonzero(np.isfinite(scores))
    ranked = finite[np.argsort(-scores[finite], kind="stable")]
    angles = np.arange(360.0)
    distance = (
        angles
        if reference_roll is None
        else np.abs((angles - reference_roll + 180.0) % 360.0 - 180.0)
    )
    result = []
    start = 0
    while start < len(ranked):
        stop = start + 1
        while stop < len(ranked) and np.isclose(
            scores[ranked[stop]], scores[ranked[start]], rtol=1e-12, atol=1e-12
        ):
            stop += 1
        tied = ranked[start:stop]
        result.extend(tied[np.lexsort((angles[tied], distance[tied]))])
        start = stop
    return np.asarray(result, dtype=np.int64)


@pytest.mark.parametrize("reference_roll", [None, 0.0, 359.5, 123.4])
def test_order_matches_numpy_for_exact_near_and_nonfinite_ties(reference_roll):
    random = np.random.default_rng(19)
    for scores in (
        np.zeros(360),
        np.full(360, -np.inf),
        np.linspace(-6000.0, 6000.0, 360),
        random.uniform(0, 6000, 360),
        6000.0 + random.uniform(-2e-8, 2e-8, 360),
        random.uniform(-2e-12, 2e-12, 360),
        np.resize([np.nan, np.inf, -np.inf, 0.0, 0.5e-12, 1.5e-12], 360),
    ):
        np.testing.assert_array_equal(
            list(candidate_order(scores, reference_roll)),
            numpy_reference(scores, reference_roll),
        )


def test_only_requested_tie_groups_are_sorted(monkeypatch):
    from unittest.mock import Mock

    lexsort = Mock(wraps=np.lexsort)
    monkeypatch.setattr(np, "lexsort", lexsort)
    # The best candidate is unique; the 359 lower-power ties are never needed.
    scores = np.zeros(360)
    scores[123] = 100.0
    order = candidate_order(scores, None)
    assert next(order) == 123
    lexsort.assert_not_called()
    assert list(order) == [i for i in range(360) if i != 123]
    lexsort.assert_called_once()


@pytest.mark.parametrize(
    "allowed,expected,checked", [(5, 5, [5]), (7, 7, [5, 7]), (None, 5, [5, 7])]
)
def test_mounted_roll_preserves_lazy_selection_and_rejected_fallback(
    monkeypatch, mock_ephem, allowed, expected, checked
):
    from unittest.mock import Mock

    from conops.config import Telescope
    from conops.simulation import roll

    scores = np.full(360, -np.inf)
    scores[5], scores[7] = 100.0, 90.0
    monkeypatch.setattr(roll, "_power_scores", lambda *args: scores)
    evaluated = []

    def constraints(constraint, scopes, instrument, *args):
        candidate = instrument[2]
        evaluated.append(candidate)
        return [] if candidate == allowed else ["Sun"]

    monkeypatch.setattr(roll, "mounted_science_attitude_constraint_names", constraints)
    assert (
        roll.optimum_instrument_roll(
            0, 0, 0, mock_ephem, Telescope(boresight=(0, 1, 0)), constraint=Mock()
        )
        == expected
    )
    assert evaluated == checked


@pytest.mark.parametrize("reference", [None, 42.5])
def test_mounted_roll_preserves_empty_ranking_fallback(
    monkeypatch, mock_ephem, reference
):
    from conops.config import Telescope
    from conops.simulation import roll

    scores = np.resize([np.nan, np.inf, -np.inf], 360)
    monkeypatch.setattr(roll, "_power_scores", lambda *args: scores)
    assert roll.optimum_instrument_roll(
        0,
        0,
        0,
        mock_ephem,
        Telescope(boresight=(0, 1, 0)),
        reference_roll=reference,
        max_roll_delta=None if reference is None else 180,
    ) == float(reference or 0.0)
