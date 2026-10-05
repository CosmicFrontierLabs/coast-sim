#!/usr/bin/env python
"""Compare COASTSim's scheduling modes on the standard benchmark scenarios.

Each scenario (see conops.benchmark.scenarios) runs every contender on fresh
copies of it:

* dispatch: QueueDITL picks each next target;
* planned:priority / planned:local_search: one plan built up front, executed
  by DITL (a plan built in advance cannot react to ToOs);
* rolling:priority / rolling:local_search: rolling-horizon replanning with
  rapid replans for ToOs;
* planned:cp_sat / rolling:cp_sat: the same with the CP-SAT planner, when
  OR-Tools is installed (``pip install coast-sim[cpsat]``);
* dispatch+alloc / rolling:*+alloc: for runs of two days or more, the same
  steered by a long-range allocator with one-day bins.

Example:
    uv run python scripts/benchmark_schedulers.py --scenario all
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from datetime import timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from conops.benchmark import (  # noqa: E402
    BenchmarkScenario,
    Contender,
    dispatch,
    format_results,
    planned,
    rolling,
    run_benchmark,
)
from conops.benchmark.scenarios import SCENARIOS  # noqa: E402
from conops.schedulers import (  # noqa: E402
    CpSatPlanner,
    LocalSearchPlanner,
    PriorityPlanner,
)

TLE = REPO_ROOT / "examples" / "example.tle"


def build_contenders(
    scenario: BenchmarkScenario, time_limit: float, lead_minutes: float
) -> list[Contender]:
    """Return the scheduling modes to compare, with replanning scaled to the run.

    Rolling contenders replan every six hours, or at least four times in a
    shorter run, and plan twice as far ahead as they replan. Runs of two days
    or more also compare dispatch and rolling steered by a long-range
    allocator with one-day bins.
    """
    hours = (scenario.end - scenario.begin).total_seconds() / 3600
    interval = min(6.0, hours / 4)
    replanning = {
        "horizon": timedelta(hours=min(hours, 2 * interval)),
        "replan_interval": timedelta(hours=interval),
        "commit_lead_time": timedelta(minutes=lead_minutes),
    }
    search = {"time_limit": time_limit}
    contenders = [
        dispatch(),
        planned(PriorityPlanner),
        planned(LocalSearchPlanner, **search),
        rolling(PriorityPlanner, **replanning),  # type: ignore[arg-type]
        rolling(LocalSearchPlanner, planner_options=search, **replanning),  # type: ignore[arg-type]
    ]
    cp_sat = importlib.util.find_spec("ortools") is not None
    solver = {"solver_time_limit": time_limit}
    if cp_sat:
        contenders += [
            planned(CpSatPlanner, **solver),
            rolling(CpSatPlanner, planner_options=solver, **replanning),  # type: ignore[arg-type]
        ]
    if hours >= 48:
        day = timedelta(days=1)
        contenders += [
            dispatch(allocation=day),
            rolling(PriorityPlanner, allocation=day, **replanning),  # type: ignore[arg-type]
            rolling(
                LocalSearchPlanner,
                planner_options=search,
                allocation=day,
                **replanning,  # type: ignore[arg-type]
            ),
        ]
        if cp_sat:
            contenders.append(
                rolling(
                    CpSatPlanner,
                    planner_options=solver,
                    allocation=day,
                    **replanning,  # type: ignore[arg-type]
                )
            )
    return contenders


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--scenario",
        choices=[*SCENARIOS, "all"],
        default="baseline",
        help="scenario to run, or all of them",
    )
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        help="seconds of local search or CP-SAT solving per plan",
    )
    parser.add_argument(
        "--lead-minutes", type=float, default=30.0, help="commit lead time"
    )
    args = parser.parse_args(argv)

    names = list(SCENARIOS) if args.scenario == "all" else [args.scenario]
    for name in names:
        scenario = SCENARIOS[name](TLE, seed=args.seed)
        contenders = build_contenders(scenario, args.time_limit, args.lead_minutes)
        print(f"\nScenario: {scenario.name}")
        print(format_results(run_benchmark(scenario, contenders)), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
