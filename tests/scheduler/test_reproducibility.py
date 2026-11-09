"""Cross-process dispatch comparisons retain decisions and failure evidence."""

import json
import os
import subprocess
import sys
from pathlib import Path

from conops.benchmark.reproducibility import compare_runs, first_difference


def test_comparison_does_not_round_away_a_boundary_difference() -> None:
    difference = first_difference(
        {"entries": [{"begin": 100.0}]},
        {"entries": [{"begin": 100.00000001}]},
    )
    assert difference is not None
    assert difference["path"] == "$.entries[0].begin"


def test_identical_failed_runs_do_not_pass() -> None:
    run = {
        "provenance": {},
        "execution": {
            "completed": False,
            "error": "recharge failure",
            "mismatches": None,
        },
    }
    result = compare_runs(run, run)
    assert result["equal"] and not result["passed"]


def test_changed_provenance_or_invalid_execution_does_not_pass() -> None:
    run = {
        "provenance": {"ephemeris": "first"},
        "execution": {"completed": True, "error": None, "mismatches": []},
    }
    assert compare_runs(run, run)["passed"]
    changed = {**run, "provenance": {"ephemeris": "second"}}
    assert not compare_runs(run, changed)["passed"]
    invalid = {**run, "execution": {**run["execution"], "mismatches": ["wrong mode"]}}
    assert not compare_runs(invalid, invalid)["passed"]


def test_first_changed_event_is_reported_separately() -> None:
    a = {
        "execution": {
            "completed": True,
            "error": None,
            "mismatches": [],
            "entries": [1],
            "events": [{"obsid": 10}],
        },
        "provenance": {},
    }
    b = {
        "execution": {**a["execution"], "entries": [2], "events": [{"obsid": 20}]},
        "provenance": {},
    }
    result = compare_runs(a, b)
    assert result["first_difference"]["path"] == "$.entries[0]"
    assert result["first_event_difference"]["path"] == "$.events[0].obsid"


def test_fresh_process_built_loaded_and_round_tripped_programs_agree(
    tmp_path: Path,
) -> None:
    # The shared program includes timed visits, an observation sequence, a moving
    # target with a saved track, a ToO, a sliding calibration and a group.
    code = """
import json, sys
from pathlib import Path
from conops import create_ditl
from conops.benchmark.reproducibility import execution_record
from conops.targets import Program
from tests.scheduler.test_program import _config, _program, HOURS, BEGIN, END
cfg = _config(HOURS)
program = _program(cfg)
program.requests = [r for r in program.requests if r.obsid != 104]
path = Path(sys.argv[1])
if sys.argv[2] != 'built':
    program.save(path.with_suffix('.program.json'))
    program = Program.load(path.with_suffix('.program.json'))
simulation = create_ditl(cfg, program=program, begin=BEGIN, end=END)
record = execution_record(simulation, completed=simulation.calc())
path.write_text(json.dumps(record, sort_keys=True, allow_nan=False))
"""
    records = []
    root = Path(__file__).resolve().parents[2]
    for index, (seed, mode) in enumerate(
        ((2029, "built"), (2029, "loaded"), (1, "loaded"))
    ):
        path = tmp_path / f"run-{index}.json"
        env = {
            **os.environ,
            "PYTHONHASHSEED": str(seed),
            "PYTHONPATH": str(root),
            "MPLCONFIGDIR": str(tmp_path / "mpl-cache"),
            "MPLBACKEND": "Agg",
        }
        subprocess.run(
            [sys.executable, "-c", code, str(path), mode],
            cwd=root,
            env=env,
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
        records.append(json.loads(path.read_text()))
    assert records[0]["completed"] and records[0]["mismatches"] == []
    for record in records[1:]:
        assert first_difference(records[0], record) is None
