"""Late acceptance counts evaluated changes without skipping history slots."""

from datetime import timedelta

from pydantic import BaseModel, Field

from conops.schedulers import LocalSearchScheduler

from .planning_scenario import BEGIN, make_config


class _ScoreState(BaseModel):
    score: float
    timeline: list[object] = Field(default_factory=list)


class _Candidate(BaseModel):
    final: _ScoreState


def test_noop_preserves_bridge_acceptance_without_extra_evaluations(monkeypatch):
    scheduler = LocalSearchScheduler(
        make_config(1),
        [],
        BEGIN,
        BEGIN + timedelta(hours=1),
        history_length=2,
        max_iterations=6,
        time_limit=60,
    )
    start = _Candidate(final=_ScoreState(score=10))
    moves = iter([(1,), (2,), None, (4,), (5,), (6,)])
    decoded_from = []

    def decode(current, attempt):
        decoded_from.append(current.final.score)
        scores = {1: 12, 2: 11, 4: 13, 5: 11.5}
        # The bridge must compare with score 11, rather than a stale score 12
        # left by a skipped history slot. Accepting that bridge
        # exposes an improvement reachable from it within the same six tries.
        score = (
            (14 if current.final.score == 11.5 else 13)
            if attempt == 6
            else scores[attempt]
        )
        return _Candidate(final=_ScoreState(score=score))

    monkeypatch.setattr(scheduler, "_random_move", lambda *args: next(moves))
    monkeypatch.setattr(scheduler, "_redecode", decode)
    result = scheduler._search(start)

    assert result.final.score == 14
    assert decoded_from == [10, 12, 11, 13, 11.5]
    assert scheduler.search_iterations == [6]
    assert scheduler.search_limits == ["iteration budget"]
    assert len(decoded_from) == 5


def test_noop_does_not_skip_an_evaluated_history_slot(monkeypatch):
    outcomes = []
    for proposals in [[(1,), (2,), (3,)], [(1,), None, (2,), (3,)]]:
        scheduler = LocalSearchScheduler(
            make_config(1),
            [],
            BEGIN,
            BEGIN + timedelta(hours=1),
            history_length=2,
            max_iterations=len(proposals),
            time_limit=60,
        )
        moves = iter(proposals)
        decoded_from = []

        def decode(current, step):
            decoded_from.append(current.final.score)
            if step == 3:
                score = 13 if current.final.score == 11 else 12
            else:
                score = {1: 12, 2: 11}[step]
            return _Candidate(final=_ScoreState(score=score))

        monkeypatch.setattr(scheduler, "_random_move", lambda *args: next(moves))
        monkeypatch.setattr(scheduler, "_redecode", decode)
        best = scheduler._search(_Candidate(final=_ScoreState(score=10)))
        outcomes.append((best.final.score, decoded_from))
        assert scheduler.search_iterations == [len(proposals)]
        assert scheduler.search_limits == ["iteration budget"]
        assert len(decoded_from) == 3

    # A rejected/no-op proposal uses an attempt, without changing the sequence
    # of evaluated neighbours or the acceptance history for those neighbours.
    assert outcomes == [(13, [10, 12, 11]), (13, [10, 12, 11])]
