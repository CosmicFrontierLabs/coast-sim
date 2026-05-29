"""Smoke-test seeded QueueDITL plan generation runs."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from rust_ephem import TLEEphemeris

from conops.config.config import MissionConfig
from conops.ditl.queue_ditl import PlanExecutionMismatchError, QueueDITL

REPO_ROOT = Path(__file__).resolve().parents[1]


def build_seeded_example_ditl(seed: int, target_count: int) -> QueueDITL:
    """Build the generic seeded QueueDITL scenario used by CI."""
    config = MissionConfig(random_seed=seed)
    config.constraint.ignore_roll = True
    ephemeris = TLEEphemeris(
        begin=datetime(2025, 12, 1, 0, 0, 0),
        end=datetime(2025, 12, 2, 0, 0, 0),
        step_size=60,
        tle=str(REPO_ROOT / "examples/example.tle"),
    )

    ditl = QueueDITL(config=config, ephem=ephemeris)
    rng = np.random.default_rng(seed)
    for index in range(target_count):
        ditl.queue.add(
            ra=rng.uniform(0, 360),
            dec=rng.uniform(-90, 90),
            exptime=1000,
            obsid=10000 + index,
        )
    return ditl


def run_seeded_example_plan(seed: int, target_count: int) -> QueueDITL:
    """Run one seeded generic QueueDITL scenario through validation."""
    ditl = build_seeded_example_ditl(seed=seed, target_count=target_count)
    ditl.calc()
    return ditl


def shard_seeds(
    start_seed: int, seed_count: int, shard_index: int, shard_count: int
) -> list[int]:
    """Return the seeds assigned to one shard."""
    return [
        start_seed + offset
        for offset in range(seed_count)
        if offset % shard_count == shard_index
    ]


def failure_record(seed: int, exc: Exception, ditl: QueueDITL | None) -> dict[str, Any]:
    """Build a JSON-serializable failure record for one seed."""
    record: dict[str, Any] = {
        "seed": seed,
        "error_type": type(exc).__name__,
        "message": str(exc),
    }
    if ditl is None:
        return record

    record["plan_entries"] = len(ditl.plan)
    record["telemetry_samples"] = len(ditl.utime)
    record["last_events"] = [str(event) for event in ditl.log.events[-20:]]
    record["stats"] = collect_ditl_stats(seed, ditl)
    if isinstance(exc, PlanExecutionMismatchError):
        record["mismatches"] = [
            str(mismatch) for mismatch in ditl.validate_plan_matches_execution()[:20]
        ]
    return record


def _name(value: Any) -> str:
    name = getattr(value, "name", None)
    if isinstance(name, str):
        return name
    return str(value)


def _duration_seconds(begin: Any, end: Any) -> float:
    try:
        return max(0.0, float(end) - float(begin))
    except (TypeError, ValueError):
        return 0.0


def _mode_sample_durations(ditl: QueueDITL) -> list[float]:
    utime = [float(t) for t in getattr(ditl, "utime", [])]
    if not utime:
        return []

    fallback = float(getattr(ditl, "step_size", 0.0) or 0.0)
    durations: list[float] = []
    for index, start in enumerate(utime):
        if index + 1 < len(utime):
            durations.append(max(0.0, float(utime[index + 1]) - start))
        else:
            durations.append(fallback)
    return durations


def _event_counts(ditl: QueueDITL) -> dict[str, int]:
    return dict(
        Counter(_name(getattr(event, "event_type", "")) for event in ditl.log.events)
    )


def _event_description_count(ditl: QueueDITL, needles: tuple[str, ...]) -> int:
    count = 0
    for event in ditl.log.events:
        description = str(getattr(event, "description", "")).lower()
        if any(needle in description for needle in needles):
            count += 1
    return count


def _numeric(values: list[Any]) -> list[float]:
    result: list[float] = []
    for value in values:
        if value is None:
            continue
        try:
            result.append(float(value))
        except (TypeError, ValueError):
            continue
    return result


def collect_ditl_stats(seed: int, ditl: QueueDITL) -> dict[str, Any]:
    """Collect non-failing summary statistics for one completed DITL run."""
    entry_counts: Counter[str] = Counter()
    planned_seconds_by_type: Counter[str] = Counter()
    slew_times: list[float] = []
    slew_distances: list[float] = []

    for entry in ditl.plan:
        entry_type = _name(getattr(entry, "obstype", "unknown"))
        entry_counts[entry_type] += 1
        planned_seconds_by_type[entry_type] += _duration_seconds(
            getattr(entry, "begin", None),
            getattr(entry, "end", None),
        )
        slew_times.extend(_numeric([getattr(entry, "slewtime", None)]))
        slew_distances.extend(_numeric([getattr(entry, "slewdist", None)]))

    mode_seconds: Counter[str] = Counter()
    for mode, duration in zip(getattr(ditl, "mode", []), _mode_sample_durations(ditl)):
        mode_seconds[_name(mode)] += duration

    validation_mismatches = ditl.validate_plan_matches_execution()
    constraint_mismatches = [
        mismatch
        for mismatch in validation_mismatches
        if "constraint_violation" in str(mismatch)
    ]

    battery_levels = _numeric(getattr(ditl, "batterylevel", []))
    recorder_fill = _numeric(getattr(ditl, "recorder_fill_fraction", []))

    return {
        "seed": seed,
        "plan_entries": len(ditl.plan),
        "telemetry_samples": len(getattr(ditl, "utime", [])),
        "entries_by_type": dict(entry_counts),
        "planned_seconds_by_type": dict(planned_seconds_by_type),
        "executed_seconds_by_mode": dict(mode_seconds),
        "executed_science_seconds": float(mode_seconds.get("SCIENCE", 0.0)),
        "executed_contact_seconds": float(mode_seconds.get("PASS", 0.0)),
        "executed_slew_seconds": float(mode_seconds.get("SLEWING", 0.0)),
        "executed_idle_seconds": float(mode_seconds.get("IDLE", 0.0)),
        "max_slew_seconds": max(slew_times, default=0.0),
        "max_slew_degrees": max(slew_distances, default=0.0),
        "battery_min": min(battery_levels, default=None),
        "battery_final": battery_levels[-1] if battery_levels else None,
        "recorder_fill_final": recorder_fill[-1] if recorder_fill else None,
        "event_counts": _event_counts(ditl),
        "queue_rejection_events": _event_description_count(
            ditl, ("rejected", "skipped")
        ),
        "constraint_log_events": _event_description_count(ditl, ("constraint",)),
        "validation_mismatch_count": len(validation_mismatches),
        "constraint_mismatch_count": len(constraint_mismatches),
    }


def _rollup_values(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "min": min(values),
        "max": max(values),
        "mean": sum(values) / len(values),
    }


def rollup_stats(seed_stats: list[dict[str, Any]]) -> dict[str, Any]:
    """Return aggregate stats for all successful seeds in one shard."""
    numeric_keys = (
        "plan_entries",
        "executed_science_seconds",
        "executed_contact_seconds",
        "executed_slew_seconds",
        "executed_idle_seconds",
        "max_slew_seconds",
        "max_slew_degrees",
        "queue_rejection_events",
        "constraint_log_events",
        "validation_mismatch_count",
        "constraint_mismatch_count",
    )
    return {
        key: rollup
        for key in numeric_keys
        if (
            rollup := _rollup_values(
                [
                    float(stats[key])
                    for stats in seed_stats
                    if stats.get(key) is not None
                ]
            )
        )
        is not None
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def scan_seeds(args: argparse.Namespace) -> int:
    seeds = shard_seeds(
        start_seed=args.start_seed,
        seed_count=args.seed_count,
        shard_index=args.shard_index,
        shard_count=args.shard_count,
    )
    output_dir = Path(args.output_dir)
    failures: list[dict[str, Any]] = []
    seed_stats: list[dict[str, Any]] = []
    scanned_seeds: list[int] = []

    print(
        f"Scanning {len(seeds)} seed(s): start={args.start_seed}, "
        f"count={args.seed_count}, shard={args.shard_index}/{args.shard_count}, "
        f"target_count={args.target_count}"
    )
    for seed in seeds:
        scanned_seeds.append(seed)
        ditl: QueueDITL | None = None
        try:
            ditl = build_seeded_example_ditl(
                seed=seed,
                target_count=args.target_count,
            )
            ditl.calc()
        except Exception as exc:
            failure = failure_record(seed, exc, ditl)
            failures.append(failure)
            write_json(output_dir / f"seed_{seed}_failure.json", failure)
            print(f"FAIL seed={seed}: {type(exc).__name__}: {exc}")
            if len(failures) >= args.max_failures:
                break
        else:
            stats = collect_ditl_stats(seed, ditl)
            seed_stats.append(stats)
            write_json(output_dir / f"seed_{seed}_stats.json", stats)
            print(
                f"PASS seed={seed}: entries={stats['plan_entries']} "
                f"science_s={stats['executed_science_seconds']:.0f} "
                f"slew_s={stats['executed_slew_seconds']:.0f}"
            )

    summary = {
        "start_seed": args.start_seed,
        "seed_count": args.seed_count,
        "shard_index": args.shard_index,
        "shard_count": args.shard_count,
        "target_count": args.target_count,
        "scanned_seeds": scanned_seeds,
        "failure_count": len(failures),
        "failures": failures,
        "seed_stats": seed_stats,
        "stats_rollup": rollup_stats(seed_stats),
    }
    write_json(output_dir / "summary.json", summary)
    return 1 if failures else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test seeded QueueDITL plan generation runs."
    )
    parser.add_argument("--start-seed", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=32)
    parser.add_argument("--target-count", type=int, default=1000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--shard-count", type=int, default=1)
    parser.add_argument("--max-failures", type=int, default=10)
    parser.add_argument("--output-dir", default="seed-scan-results")
    args = parser.parse_args()

    if args.seed_count < 1:
        parser.error("--seed-count must be positive")
    if args.target_count < 1:
        parser.error("--target-count must be positive")
    if args.shard_count < 1:
        parser.error("--shard-count must be positive")
    if not 0 <= args.shard_index < args.shard_count:
        parser.error("--shard-index must satisfy 0 <= index < shard-count")
    if args.max_failures < 1:
        parser.error("--max-failures must be positive")
    return args


def main() -> int:
    return scan_seeds(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
