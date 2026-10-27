"""Executed-plan files retain completion; program files omit execution state."""

import json
from pathlib import Path

import pytest

from conops.targets import MovingPointing, Plan, Pointing, track_from_table
from conops.targets.visits import VisitWindow


# Every pair of kind, windowed and complete values is covered.
@pytest.mark.parametrize(
    ("kind", "windowed", "complete"),
    [
        (Pointing, False, True),
        (Pointing, True, False),
        (MovingPointing, True, True),
        (MovingPointing, False, False),
    ],
)
def test_executed_plan_preserves_completion_on_reexport(
    tmp_path: Path, kind: type[Pointing], windowed: bool, complete: bool
) -> None:
    entry = kind(
        obsid=701,
        ra=10,
        dec=20,
        ss_min=600,
        ss_max=600,
        exptime=600,
        track=track_from_table([1000, 2000], [10, 10.1], [20, 20])
        if kind is MovingPointing
        else [],
    )
    if windowed:
        entry.visit_windows = [VisitWindow(start=1000, deadline=2000)]
        if complete:
            entry.record_collection(1600, 600, visit=0)
    else:
        # A completed snapshot can retain the parent's remaining exposure.
        entry.done = complete
    assert entry.done is complete
    original = Plan(entries=[entry])
    original.save(tmp_path / "original.json")
    loaded = Plan.load(tmp_path / "original.json")
    assert isinstance(loaded.entries[0], kind)
    assert loaded.entries[0].done is complete
    loaded.save(tmp_path / "reexported.json")
    assert json.loads((tmp_path / "reexported.json").read_text()) == json.loads(
        (tmp_path / "original.json").read_text()
    )


def test_legacy_plan_without_done_derives_exhausted_exposure(tmp_path: Path) -> None:
    path = tmp_path / "legacy.json"
    path.write_text(
        json.dumps({"entries": [{"kind": "pointing", "obsid": 701, "exptime": 0}]})
    )
    loaded = Plan.load(path)
    assert loaded.entries[0].done
