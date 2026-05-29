"""Tests for the QueueDITL seed scan helper script."""

import json
from unittest.mock import Mock

import pytest

from conops import ACSMode, PlanExecutionMismatchError
from conops.common.enums import ObsType
from scripts import scan_queue_ditl_seeds


def _scan_args(tmp_path, seed: int = 7) -> Mock:
    return Mock(
        start_seed=seed,
        seed_count=1,
        target_count=10,
        shard_index=0,
        shard_count=1,
        max_failures=10,
        output_dir=str(tmp_path),
    )


def test_scan_writes_success_stats(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    entry = Mock()
    entry.obstype = ObsType.AT
    entry.begin = 0.0
    entry.end = 120.0
    entry.slewtime = 30.0
    entry.slewdist = 4.5

    event = Mock()
    event.event_type = "QUEUE"
    event.description = "Target skipped during scan"

    ditl = Mock()
    ditl.plan = [entry]
    ditl.utime = [0.0, 60.0]
    ditl.mode = [ACSMode.SCIENCE, ACSMode.IDLE]
    ditl.step_size = 60.0
    ditl.batterylevel = [0.8, 0.7]
    ditl.recorder_fill_fraction = [0.1, 0.2]
    ditl.log.events = [event]
    ditl.validate_plan_matches_execution.return_value = []

    monkeypatch.setattr(
        scan_queue_ditl_seeds,
        "build_seeded_example_ditl",
        Mock(return_value=ditl),
    )

    assert scan_queue_ditl_seeds.scan_seeds(_scan_args(tmp_path)) == 0

    stats = json.loads((tmp_path / "seed_7_stats.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert stats["executed_science_seconds"] == 60.0
    assert stats["max_slew_degrees"] == 4.5
    assert stats["validation_mismatch_count"] == 0
    assert summary["stats_rollup"]["executed_science_seconds"]["mean"] == 60.0


def test_scan_fails_on_plan_execution_constraint_violation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    event = Mock()
    event.event_type = "ERROR"
    event.description = "constraint violation"

    ditl = Mock()
    ditl.plan = []
    ditl.utime = [0.0]
    ditl.mode = [ACSMode.SLEWING]
    ditl.step_size = 60.0
    ditl.batterylevel = []
    ditl.recorder_fill_fraction = []
    ditl.log.events = [event]
    ditl.calc.side_effect = PlanExecutionMismatchError(
        "Plan execution validation failed: constraint_violation"
    )
    ditl.validate_plan_matches_execution.return_value = [
        "attitude constraint_violation: mode SLEWING violates ST Hard"
    ]

    monkeypatch.setattr(
        scan_queue_ditl_seeds,
        "build_seeded_example_ditl",
        Mock(return_value=ditl),
    )

    assert scan_queue_ditl_seeds.scan_seeds(_scan_args(tmp_path)) == 1

    failure = json.loads((tmp_path / "seed_7_failure.json").read_text())
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert failure["error_type"] == "PlanExecutionMismatchError"
    assert "constraint_violation" in failure["mismatches"][0]
    assert failure["stats"]["constraint_mismatch_count"] == 1
    assert summary["failure_count"] == 1
