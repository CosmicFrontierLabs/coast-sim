"""The CLI must reject a changed captured runtime, even if workers agree."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import rust_ephem

from conops.benchmark.reproducibility import compare_runs
from conops.config import MissionConfig
from conops.config.scheduler import SchedulerMode
from conops.targets import Program


@pytest.fixture
def cli():
    path = (
        Path(__file__).resolve().parents[2] / "scripts/check_program_repeatability.py"
    )
    spec = importlib.util.spec_from_file_location("repeatability_cli_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def worker(cli, tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    for name in ("mission.yaml", "program.json", "orbit.e", "latest_eop2.short"):
        (inputs / name).write_text("captured input")
    files = {}
    for name in ("library", "driver", "dependency"):
        path = tmp_path / name
        path.write_text("captured runtime")
        files[name] = path

    def runtime():
        return {
            "sources": {
                name: cli.digest(files[name]) for name in ("library", "driver")
            },
            "dependencies": {"example": files["dependency"].read_text()},
        }

    manifest = {
        "inputs": {p.name: {"sha256": cli.digest(p)} for p in inputs.iterdir()},
        "runtime": runtime(),
        "planetary_model": "analytic",
        "step_size": 120,
    }
    cli.write_json(tmp_path / "manifest.json", manifest)
    monkeypatch.setattr(cli, "runtime_provenance", runtime)
    monkeypatch.setattr(
        MissionConfig,
        "from_yaml_file",
        lambda *_: SimpleNamespace(
            scheduler=SimpleNamespace(mode=SchedulerMode.DISPATCH)
        ),
    )
    monkeypatch.setattr(
        Program,
        "load",
        lambda *_: SimpleNamespace(begin=0, end=60, requests=[], toos=[]),
    )
    monkeypatch.setattr(rust_ephem, "FileEphemeris", lambda *_, **__: object())
    monkeypatch.setattr(rust_ephem, "is_planetary_ephemeris_initialized", lambda: False)
    monkeypatch.setattr(
        rust_ephem,
        "get_eop_provenance",
        lambda: {
            "ut1": {
                "available": True,
                "sha256": cli.digest(inputs / "latest_eop2.short"),
            }
        },
    )
    calculate = Mock(return_value=True)
    monkeypatch.setattr(
        "conops.create_ditl", lambda *_, **__: SimpleNamespace(calc=calculate)
    )
    monkeypatch.setattr(
        "conops.benchmark.reproducibility.execution_record",
        lambda simulation, **state: {**state, "entries": [], "mismatches": []},
    )
    return SimpleNamespace(
        args=argparse.Namespace(output=tmp_path, worker=1, round_trip=False),
        inputs=inputs,
        runtime_files=files,
        calculate=calculate,
    )


@pytest.mark.parametrize("changed", ["library", "driver", "dependency"])
def test_changed_runtime_is_rejected_before_execution(cli, worker, changed):
    worker.runtime_files[changed].write_text("different runtime")
    with pytest.raises(ValueError, match="Captured runtime changed"):
        cli.run_worker(worker.args)
    worker.calculate.assert_not_called()


@pytest.mark.parametrize("changed", ["library", "driver", "input"])
def test_a_change_during_execution_cannot_be_accepted(cli, worker, changed):
    path = (
        worker.inputs / "program.json"
        if changed == "input"
        else worker.runtime_files[changed]
    )

    def calculate():
        path.write_text("changed during execution")
        return True

    worker.calculate.side_effect = calculate
    assert cli.run_worker(worker.args) == 1
    payload = json.loads((worker.args.output / "run-1.json").read_text())
    assert not payload["execution"]["completed"]
    assert "changed" in payload["execution"]["error"]
    assert not compare_runs(payload, payload)["passed"]


def test_unchanged_snapshot_can_complete(cli, worker):
    assert cli.run_worker(worker.args) == 0
    payload = json.loads((worker.args.output / "run-1.json").read_text())
    assert payload["execution"]["completed"]
    assert payload["execution"]["error"] is None
    assert compare_runs(payload, payload)["passed"]


def test_runtime_snapshot_includes_the_cli_driver(cli):
    provenance = cli.runtime_provenance()
    expected = hashlib.sha256(Path(cli.__file__).read_bytes()).hexdigest()
    assert provenance["sources"]["scripts/check_program_repeatability.py"] == expected
