"""Lazy tie grouping must preserve the original NumPy ranking exactly."""

import numpy as np
import pytest

from conops.simulation.roll import _power_score_order


def candidate_order(scores, reference_roll):
    angles = np.arange(360.0)
    distance = (
        angles
        if reference_roll is None
        else np.abs((angles - reference_roll + 180.0) % 360.0 - 180.0)
    )
    return _power_score_order(scores, distance, angles)


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
