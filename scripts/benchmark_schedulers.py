#!/usr/bin/env python
"""Compare COASTSim's scheduling modes on one realistic scenario.

The scenario uses the example TLE, Sun and Earth-limb avoidance, the default
ground-station network, a pool of random targets and one Target of
Opportunity halfway through the run with a one-hour deadline. Each contender
runs on fresh copies of it:

* dispatch: QueueDITL picks each next target;
* planned:priority / planned:local_search: one plan built up front, executed
  by DITL (a plan built in advance cannot react to the ToO);
* rolling:priority / rolling:local_search: rolling-horizon replanning with
  rapid replans for the ToO.

Example:
    uv run python scripts/benchmark_schedulers.py --hours 24 --targets 200
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from rust_ephem import EarthLimbConstraint, SunConstraint, TLEEphemeris

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from conops.benchmark import (  # noqa: E402
    BenchmarkScenario,
    Contender,
    TOOSpec,
    dispatch,
    format_results,
    planned,
    rolling,
    run_benchmark,
)
from conops.config import (  # noqa: E402
    AttitudeControlSystem,
    Battery,
    Constraint,
    GroundStationRegistry,
    MissionConfig,
    RadiatorConfiguration,
    SolarPanelSet,
    SpacecraftBus,
    StarTrackerConfiguration,
)
from conops.schedulers import LocalSearchPlanner, PriorityPlanner  # noqa: E402
from conops.targets import Pointing  # noqa: E402

TLE = REPO_ROOT / "examples" / "example.tle"
BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)


def build_scenario(
    hours: int, targets: int, seed: int, stations: bool
) -> BenchmarkScenario:
    """Return the default benchmark scenario."""
    end = BEGIN + timedelta(hours=hours)

    def make_config() -> MissionConfig:
        config = MissionConfig(
            constraint=Constraint(
                sun_constraint=SunConstraint(min_angle=45),
                earth_constraint=EarthLimbConstraint(min_angle=10),
            ),
            ground_stations=GroundStationRegistry.default()
            if stations
            else GroundStationRegistry(stations=[]),
            solar_panel=SolarPanelSet(panels=[]),
            battery=Battery(watthour=100_000.0),
            spacecraft_bus=SpacecraftBus(
                attitude_control=AttitudeControlSystem(
                    max_slew_rate=1.0, slew_acceleration=0.5, settle_time=30.0
                ),
                star_trackers=StarTrackerConfiguration(
                    star_trackers=[], min_functional_trackers=0, modes_require_lock=[]
                ),
                radiators=RadiatorConfiguration(radiators=[]),
            ),
        )
        config.random_seed = seed
        config.constraint.ephem = TLEEphemeris(
            tle=str(TLE), begin=BEGIN, end=end, step_size=60
        )
        return config

    def make_targets(config: MissionConfig) -> list[Pointing]:
        rng = np.random.default_rng(seed)
        pool = []
        for k in range(targets):
            merit = float(rng.integers(10, 100))
            target = Pointing(
                config=config,
                ra=float(rng.uniform(0, 360)),
                dec=float(np.degrees(np.arcsin(rng.uniform(-1, 1)))),
                obsid=10000 + k,
                name=f"t{k}",
                merit=merit,
                fom=merit,
                ss_min=300,
                ss_max=1200,
            )
            target.exptime = int(rng.choice([600, 1200, 2400]))
            pool.append(target)
        return pool

    middle = BEGIN.timestamp() + hours * 3600 / 2
    return BenchmarkScenario(
        name=f"{hours}h, {targets} targets" + (", ground stations" if stations else ""),
        begin=BEGIN,
        end=end,
        make_config=make_config,
        make_targets=make_targets,
        toos=[
            TOOSpec(
                obsid=1_000_001,
                ra=105.0,
                dec=10.0,
                merit=500.0,
                exptime=900,
                name="ToO",
                submit_time=middle,
                deadline=middle + 3600,
            )
        ],
    )


def build_contenders(
    hours: int, time_limit: float, replan_hours: float, lead_minutes: float
) -> list[Contender]:
    """Return the scheduling modes to compare."""
    replanning = {
        "horizon": timedelta(hours=min(hours, 2 * replan_hours)),
        "replan_interval": timedelta(hours=replan_hours),
        "commit_lead_time": timedelta(minutes=lead_minutes),
    }
    search = {"time_limit": time_limit}
    return [
        dispatch(),
        planned(PriorityPlanner),
        planned(LocalSearchPlanner, **search),
        rolling(PriorityPlanner, **replanning),  # type: ignore[arg-type]
        rolling(LocalSearchPlanner, planner_options=search, **replanning),  # type: ignore[arg-type]
    ]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--hours", type=int, default=24, help="run length")
    parser.add_argument("--targets", type=int, default=200, help="target pool size")
    parser.add_argument("--seed", type=int, default=1234, help="random seed")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=10.0,
        help="seconds of local search per plan",
    )
    parser.add_argument(
        "--replan-hours", type=float, default=6.0, help="hours between replans"
    )
    parser.add_argument(
        "--lead-minutes", type=float, default=30.0, help="commit lead time"
    )
    parser.add_argument(
        "--no-stations", action="store_true", help="plan without ground passes"
    )
    args = parser.parse_args(argv)

    scenario = build_scenario(
        args.hours, args.targets, args.seed, stations=not args.no_stations
    )
    contenders = build_contenders(
        args.hours, args.time_limit, args.replan_hours, args.lead_minutes
    )
    print(f"Scenario: {scenario.name}")
    print(format_results(run_benchmark(scenario, contenders)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
