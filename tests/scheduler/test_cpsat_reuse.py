"""A reused native response must solve the current exact model and mapping."""

import numpy as np
import pytest

pytest.importorskip("ortools")

from ortools.sat.python import cp_model  # noqa: E402

from conops.schedulers.cpsat_scheduler import (  # noqa: E402
    _ChunkModel,
    _ChunkRecorder,
    _ChunkSolveCache,
    _Span,
    _Task,
    _TaskHint,
)


def chunk(*, obsid=1, entering=None, fixed=0, hint_seconds=60, allocated=False):
    pose = (90.0, 0.0, 0.0)
    tasks = [
        _Task(None, (pose, pose), (0, 0)),
        _Task(
            (obsid, 0.0),
            (pose, pose),
            (0, 600),
            collection=(1, 60),
            window=(0, 600),
            rate=1.0,
            spans=[_Span((obsid, 0), (10, 540), 2.0)] if allocated else [],
        ),
    ]
    recorder = (
        _ChunkRecorder([], entering, [0], [fixed], [0], [0], 100, [(0, 0), (1, 0)])
        if entering is not None
        else None
    )
    result = _ChunkModel(
        tasks,
        np.zeros((2, 2), dtype=np.int64),
        1,
        600,
        {obsid: 60},
        recorder,
        allocated_left={(obsid, 0): 60} if allocated else {},
    )
    arrival = 10 if allocated else 0
    result.add_hint({1: _TaskHint(arrival, arrival, hint_seconds)})
    return result


def solve(model, cache, *, origin=100.0, workers=1, budget=0.05, seed=2029):
    model.solve_cache = cache
    return model.solve(2.0, workers, seed, origin, budget)


@pytest.fixture
def native_calls(monkeypatch):
    calls = []
    original = cp_model.CpSolver.solve

    def counted(self, *args, **kwargs):
        if self.parameters.fix_variables_to_their_hinted_value:
            # Completing the hint is not a search.
            return original(self, *args, **kwargs)
        calls.append(True)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(cp_model.CpSolver, "solve", counted)
    return calls


def test_equal_models_decode_response_with_current_target_and_origin(native_calls):
    cache = _ChunkSolveCache()
    before, after = chunk(obsid=11, allocated=True), chunk(obsid=22, allocated=True)
    assert str(before.model.proto) == str(after.model.proto)
    status, _, first = solve(before, cache)
    repeated_status, ended, second = solve(after, cache, origin=200.0)
    assert status == repeated_status == "OPTIMAL"
    assert ended == "reused"
    assert first is not None and second is not None
    assert first[0].obsid == 11 and second[0].obsid == 22
    assert first[0].seconds == second[0].seconds == 60
    assert first[0].release is not None and second[0].release == first[0].release + 100
    assert len(native_calls) == 1


@pytest.mark.parametrize("changed", ["recorder", "hint"])
def test_statistics_collision_still_checks_complete_model(
    monkeypatch, native_calls, changed
):
    cache = _ChunkSolveCache()
    monkeypatch.setattr(
        cp_model.CpModel, "model_stats", lambda self: "same fingerprint"
    )
    before = chunk(entering=40)
    after = (
        chunk(entering=41)
        if changed == "recorder"
        else chunk(entering=40, hint_seconds=30)
    )
    assert len(before.model.proto.variables) == len(after.model.proto.variables)
    assert len(before.model.proto.constraints) == len(after.model.proto.constraints)
    assert str(before.model.proto) != str(after.model.proto)
    solve(before, cache)
    result = solve(after, cache)
    assert result[1] != "reused"
    assert result[2] is not None
    assert result[2][0].seconds == (59 if changed == "recorder" else 60)
    assert len(native_calls) == 2


def test_stored_model_is_an_owned_snapshot(native_calls):
    cache = _ChunkSolveCache()
    original = chunk()
    solve(original, cache)
    original.model.add(original.collection[1] <= 30)
    changed = solve(original, cache)
    restored = solve(chunk(), cache)
    assert changed[2] is not None and changed[2][0].seconds == 30
    assert restored[1] == "reused"
    assert restored[2] is not None and restored[2][0].seconds == 60
    assert len(native_calls) == 2


@pytest.mark.parametrize("options", [{"workers": 2}, {"budget": None}, {"seed": 2030}])
def test_different_or_nondeterministic_settings_do_not_reuse(native_calls, options):
    cache = _ChunkSolveCache()
    solve(chunk(), cache)
    assert solve(chunk(), cache, **options)[1] != "reused"
    assert len(native_calls) == 2


@pytest.mark.parametrize(
    "status,wall,first,second,calls",
    [
        # A wall-limited result is never cached.
        (cp_model.FEASIBLE, 2.0, "wall clock", "wall clock", 2),
        # An unknown at the deterministic budget reuses the same fallback.
        (cp_model.UNKNOWN, 0.001, "deterministic budget", "reused", 1),
    ],
)
def test_limited_results_are_cached_only_at_the_deterministic_budget(
    monkeypatch, native_calls, status, wall, first, second, calls
):
    cache = _ChunkSolveCache()
    counted = cp_model.CpSolver.solve

    def limited(self, *args, **kwargs):
        counted(self, *args, **kwargs)
        return status

    monkeypatch.setattr(cp_model.CpSolver, "solve", limited)
    monkeypatch.setattr(cp_model.CpSolver, "wall_time", property(lambda self: wall))
    monkeypatch.setattr(
        cp_model.CpSolver, "deterministic_time", property(lambda self: 0.05)
    )
    assert solve(chunk(), cache)[1] == first
    assert solve(chunk(), cache)[1] == second
    assert len(native_calls) == calls
