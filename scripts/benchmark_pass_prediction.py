"""Time rolling-horizon runs with ground stations, and the pass prediction in them.

For each run length, builds the real-orbit test scenario with the default
ground stations, runs ``RollingHorizonDITL`` with twelve-hour plans and
six-hour replans, and reports the wall time, how many times ``PassTimes.get``
was called and how long those calls took, the contacts planned, and a
fingerprint of the plan so that runs on two revisions can be compared.

Usage::

    .venv/bin/python scripts/benchmark_pass_prediction.py [--hours 24 48 72] [--out results.json]

Run it on two revisions and compare the ``plan`` lists in the JSON: they are
expected to be identical.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

from scheduler.planning_scenario import BEGIN, PATCH, make_config, make_target  # noqa: E402

from conops.ditl import RollingHorizonDITL  # noqa: E402
from conops.simulation.passes import PassTimes  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hours", type=int, nargs="+", default=[24, 48, 72])
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    calls = {"count": 0, "seconds": 0.0}
    original = PassTimes.get

    def timed(self: PassTimes, *a: object, **kw: object) -> None:
        started = time.perf_counter()
        try:
            original(self, *a, **kw)  # type: ignore[arg-type]
        finally:
            calls["count"] += 1
            calls["seconds"] += time.perf_counter() - started

    PassTimes.get = timed  # type: ignore[method-assign]

    results = []
    for hours in args.hours:
        calls["count"], calls["seconds"] = 0, 0.0
        config = make_config(hours, stations=True)
        targets = [
            make_target(
                config, 100 + i, ra, dec, merit=90 - i, minutes=120, snapshot=20
            )
            for i, (ra, dec) in enumerate(PATCH)
        ]
        ditl = RollingHorizonDITL(
            config,
            targets,
            begin=BEGIN,
            end=BEGIN + timedelta(hours=hours),
            horizon=timedelta(hours=12),
            replan_interval=timedelta(hours=6),
        )
        ditl.step_size = 60
        started = time.perf_counter()
        if not ditl.calc():
            raise SystemExit(f"the {hours} h run failed")
        wall = time.perf_counter() - started
        result = {
            "hours": hours,
            "plans": len(ditl.replans),
            "contacts": sum(1 for e in ditl.plan if e.obstype.name == "GSP"),
            "predictions": calls["count"],
            "predicting_seconds": round(calls["seconds"], 1),
            "wall_seconds": round(wall, 1),
            "mismatches": len(ditl.validate_plan_matches_execution()),
            "plan": [
                (int(e.obsid), round(float(e.begin)), round(float(e.end)))
                for e in ditl.plan
            ],
        }
        results.append(result)
        shown = {k: v for k, v in result.items() if k != "plan"}
        print(f"{hours} h: {shown}", flush=True)
    if args.out is not None:
        args.out.write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
