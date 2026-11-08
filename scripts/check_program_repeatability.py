#!/usr/bin/env python3
"""Freeze inputs and compare dispatch executions in fresh Python processes.

Run from the repository root, for example::

    PYTHONPATH=. .venv/bin/python scripts/check_program_repeatability.py \
        --config mission.yaml --program program.json --ephemeris orbit.e \
        --output out/repeatability --hash-seeds 2029 2029 1

The first two workers load the frozen program. A third saves and reloads it
before execution, checking serialization as well as process/hash-seed effects.
No moving-body lookup is allowed: every MovingPointing must carry its track.
This checks saved requests and their round trip; notebook-built programs can be
checked with the same execution_record/compare_runs helpers independently.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def runtime_provenance() -> dict:
    """Fingerprint code and dependencies, not only editable-install metadata."""
    import rust_ephem

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO, text=True
        ).strip(),
        "sources": {
            str(p.relative_to(REPO)): digest(p)
            for p in sorted((REPO / "conops").rglob("*.py"))
        },
        "dependencies": dict(
            sorted(
                (d.metadata["Name"], d.version)
                for d in importlib.metadata.distributions()
                if d.metadata["Name"]
            )
        ),
        "rust_ephem_binary": {
            p.name: digest(p)
            for p in sorted(Path(rust_ephem.__file__).parent.glob("*.so"))
        },
    }


def check_tracks(program) -> None:
    from conops.targets import MovingPointing

    requests = [*program.requests, *(t.request for t in program.toos)]
    for request in requests:
        if isinstance(request, MovingPointing) and not request.track:
            raise ValueError(
                f"Moving request {request.obsid} has no saved track; fetch and save "
                "it before capturing a repeatability baseline"
            )


def run_worker(args) -> int:
    import rust_ephem

    from conops import create_ditl
    from conops.benchmark.reproducibility import execution_record
    from conops.config import MissionConfig
    from conops.config.scheduler import SchedulerMode
    from conops.targets import Program

    manifest = json.loads((args.output / "manifest.json").read_text())
    inputs = args.output / "inputs"
    # Detect input edits between capture and execution rather than comparing
    # results from files whose manifest no longer describes them.
    for name, expected in manifest["inputs"].items():
        if digest(inputs / name) != expected["sha256"]:
            raise ValueError(f"Frozen input changed: {name}")
    config = MissionConfig.from_yaml_file(str(inputs / "mission.yaml"))
    if config.scheduler.mode is not SchedulerMode.DISPATCH:
        raise ValueError("This baseline checks dispatch; supply a dispatch config")
    program = Program.load(inputs / "program.json")
    check_tracks(program)
    if args.round_trip:
        path = args.output / f"round-trip-{args.worker}.json"
        program.save(path)
        program = Program.load(path)
    begin = datetime.fromtimestamp(program.begin, timezone.utc)
    end = datetime.fromtimestamp(program.end, timezone.utc)
    # A fresh notebook uses the analytic Sun/Moon models unless it explicitly
    # initializes a kernel. Do not silently change that physical input here.
    # Older snapshots from this tool explicitly initialized DE440s.
    planetary_model = manifest.get("planetary_model", "spice")
    if planetary_model == "spice":
        rust_ephem.init_planetary_ephemeris(
            str(
                inputs / "planetary.bsp"
                if (inputs / "planetary.bsp").exists()
                else inputs / "de440s.bsp"
            )
        )
    elif rust_ephem.is_planetary_ephemeris_initialized():
        raise ValueError(
            "Analytic baseline requires a fresh process without a planetary kernel"
        )
    ephem = rust_ephem.FileEphemeris(
        str(inputs / "orbit.e"),
        begin=begin,
        end=end,
        step_size=manifest["step_size"],
        position_unit="m",
        velocity_unit="m/s",
    )
    simulation = create_ditl(config, program=program, begin=begin, end=end, ephem=ephem)
    started = time.perf_counter()
    error = None
    completed = False
    try:
        completed = simulation.calc()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
    record = execution_record(simulation, completed=completed, error=error)
    provenance = runtime_provenance()
    provenance["inputs"] = {k: v["sha256"] for k, v in manifest["inputs"].items()}
    provenance["step_size"] = manifest["step_size"]
    provenance["planetary_model"] = planetary_model
    eop = rust_ephem.get_eop_provenance()
    provenance["eop"] = {
        provider: {
            key: value
            for key, value in data.items()
            if key in ("available", "source_url", "sha256")
        }
        for provider, data in eop.items()
    }
    expected_eop = manifest["inputs"]["latest_eop2.short"]["sha256"]
    for provider, data in provenance["eop"].items():
        if data.get("available") and data["sha256"] != expected_eop:
            record["error"] = f"{provider} loaded EOP different from the captured input"
            record["completed"] = False
    payload = {
        "provenance": provenance,
        "execution": record,
        "run": {
            "hash_seed": os.environ.get("PYTHONHASHSEED"),
            "round_trip": args.round_trip,
            "seconds": time.perf_counter() - started,
            "eop_loading": eop,
        },
    }
    write_json(args.output / f"run-{args.worker}.json", payload)
    print(
        json.dumps(
            {
                "completed": record["completed"],
                "error": record["error"],
                "entries": len(record["entries"]),
                "run": payload["run"],
            }
        )
    )
    return (
        0
        if record["completed"]
        and record["error"] is None
        and record["mismatches"] == []
        else 1
    )


def capture_and_compare(args) -> int:
    import rust_ephem

    from conops.benchmark.reproducibility import compare_runs
    from conops.targets import Program

    if len(args.hash_seeds) < 2:
        raise ValueError("At least two fresh-process runs are required")
    check_tracks(Program.load(args.program))
    # A new destination keeps failed and earlier baselines intact.
    args.output.mkdir(parents=True, exist_ok=False)
    inputs = args.output / "inputs"
    inputs.mkdir()
    cache = Path(rust_ephem.get_cache_dir())
    sources = {
        "mission.yaml": args.config,
        "program.json": args.program,
        "orbit.e": args.ephemeris,
        "latest_eop2.short": cache / "latest_eop2.short",
    }
    if args.planetary_kernel is not None:
        sources["planetary.bsp"] = args.planetary_kernel
    for name, source in sources.items():
        shutil.copy2(source, inputs / name)
    manifest = {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "step_size": args.step_size,
        "planetary_model": "spice" if args.planetary_kernel is not None else "analytic",
        "hash_seeds": args.hash_seeds,
        "inputs": {
            name: {"source": str(source.resolve()), "sha256": digest(inputs / name)}
            for name, source in sources.items()
        },
        "runtime": runtime_provenance(),
        "git_status": subprocess.check_output(
            ["git", "status", "--short"], cwd=REPO, text=True
        ),
        "eop_note": "Provider data is fingerprinted per worker; changed EOP fails comparison.",
    }
    write_json(args.output / "manifest.json", manifest)
    runs = []
    for index, seed in enumerate(args.hash_seeds, 1):
        env = {
            **os.environ,
            "PYTHONHASHSEED": str(seed),
            "PYTHONPATH": str(REPO),
            "MPLCONFIGDIR": str(args.output / "mpl-cache"),
            "MPLBACKEND": "Agg",
        }
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--output",
            str(args.output),
            "--worker",
            str(index),
        ]
        if index > 2:
            command.append("--round-trip")
        print(
            f"Starting run {index}: hash seed {seed}, round trip {index > 2}",
            flush=True,
        )
        with (args.output / f"run-{index}.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        path = args.output / f"run-{index}.json"
        runs.append(json.loads(path.read_text()) if path.exists() else None)
        print(f"Run {index} exited {result.returncode}", flush=True)
    comparisons = [
        {"run": index, **compare_runs(runs[0], run)}
        if runs[0] is not None and run is not None
        else {
            "run": index,
            "passed": False,
            "error": "Worker produced no record; see log",
        }
        for index, run in enumerate(runs[1:], 2)
    ]
    report = {
        "passed": all(c["passed"] for c in comparisons),
        "comparisons": comparisons,
    }
    write_json(args.output / "comparison.json", report)
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--ephemeris", type=Path)
    parser.add_argument(
        "--planetary-kernel",
        type=Path,
        help="Explicit SPICE kernel; default is the fresh notebook's analytic Sun/Moon models",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--step-size", type=int, default=120)
    parser.add_argument("--hash-seeds", type=int, nargs="+", default=[2029, 2029, 1])
    parser.add_argument("--worker", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--round-trip", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.step_size <= 0:
        parser.error("--step-size must be positive")
    if args.worker is not None:
        return run_worker(args)
    if any(value is None for value in (args.config, args.program, args.ephemeris)):
        parser.error("--config, --program and --ephemeris are required")
    return capture_and_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
