"""Winner-only power selection must match the full mounted-roll ranking."""

import numpy as np
import pytest

from conops.simulation.roll import _best_power_roll, _power_score_order


@pytest.mark.parametrize("reference", [None, 0.0, 17.5, 180.0, 359.5])
@pytest.mark.parametrize("scale", [0.0, 1e-13, 1.0, 1000.0])
def test_best_roll_matches_full_ranking(reference, scale):
    rng = np.random.default_rng(281)
    for _ in range(20):
        scores = rng.uniform(0.0, scale, 360)
        scores[rng.random(360) < 0.2] = -np.inf
        scores[rng.choice(360, 3, replace=False)] = [np.nan, np.inf, -np.inf]
        original = scores.copy()
        order = list(_power_score_order(scores, reference))

        assert _best_power_roll(scores, reference) == float(order[0])
        np.testing.assert_array_equal(scores, original)


@pytest.mark.parametrize(
    ("candidates", "reference", "expected"),
    [
        ({10: 100.0, 20: 101.0}, None, 20.0),
        ({10: 100.0, 20: 101.0}, 10.0, 20.0),
        ({0: 1.0, 359: 1.0}, 359.5, 0.0),
        ({1: 1.0, 180: 1.0}, 359.0, 1.0),
        ({5: 1.0, 10: 1.0}, None, 5.0),
        # 0 is close to 5, but not to the maximum at 10: do not chain ties.
        ({0: 1000.0 - 1.5e-9, 5: 1000.0 - 0.75e-9, 10: 1000.0}, None, 5.0),
        ({0: 1000.0 - 1.5e-9, 5: 1000.0 - 0.75e-9, 10: 1000.0}, 10.0, 10.0),
        ({0: -1.5e-12, 5: -0.75e-12, 10: 0.0}, None, 5.0),
    ],
)
def test_best_roll_preserves_power_and_tie_breaks(candidates, reference, expected):
    scores = np.full(360, -np.inf)
    for index, score in candidates.items():
        scores[index] = score

    assert _best_power_roll(scores, reference) == expected
    assert next(_power_score_order(scores, reference)) == expected


@pytest.mark.parametrize("reference", [None, 0.0, 17.5, 359.5])
def test_best_roll_preserves_empty_fallback(reference):
    scores = np.resize(np.array([np.nan, np.inf, -np.inf]), 360)

    assert not list(_power_score_order(scores, reference))
    assert _best_power_roll(scores, reference) == float(reference or 0.0)


def test_body_roll_does_not_rank_masked_candidates(monkeypatch, mock_ephem):
    from conops.simulation import roll

    scores = np.zeros(360)
    scores[180] = 1000.0
    scores[7] = 2.0
    mask = np.zeros(360, dtype=bool)
    mask[[7, 8]] = True
    monkeypatch.setattr(roll, "_power_scores", lambda *args: scores)
    monkeypatch.setattr(roll, "_roll_valid_mask", lambda *args: mask)

    def unexpected_ranking(*args):
        pytest.fail("Body-roll selection must not build a complete ranking")

    monkeypatch.setattr(roll, "_power_score_order", unexpected_ranking)
    assert roll.optimum_body_roll(0, 0, 0, mock_ephem) == 7.0
