#!/usr/bin/env python
"""Check COAST's default deterministic plan output against a golden baseline.

The scenario in this file is intentionally owned by COAST itself.  It does not
depend on downstream mission configuration repositories, live EOP downloads, or
ambient user cache state.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

_MPLCONFIGDIR = Path(tempfile.gettempdir()) / "coast-sim-matplotlib"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BASELINE = REPO_ROOT / "tests" / "baselines" / "default_plan_output.golden"
# Only absorb platform-level float roundoff; planner or timing changes should fail.
DEFAULT_NUMERIC_ABS_TOL = 1e-9
sys.path.insert(0, str(REPO_ROOT))

from conops import QueueDITL  # noqa: E402
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
from conops.targets import PlanSchema  # noqa: E402


class DeterministicConstraint(Constraint):
    """Constraint set for the golden scenario.

    The individual constraint fields are all ``None`` via ``Constraint``.  The
    eclipse check is overridden so the scenario never calls rust-ephem's live
    eclipse/EOP path.
    """

    def in_eclipse(self, ra: float, dec: float, time: float) -> bool:
        return False


class DeterministicEphemeris:
    """Small deterministic ephemeris adapter for QueueDITL tests."""

    def __init__(
        self, begin: datetime, end: datetime, step_size_seconds: int = 300
    ) -> None:
        self.begin = begin
        self.end = end
        self.step_size = float(step_size_seconds)
        sample_count = int((end - begin).total_seconds() / step_size_seconds) + 1
        self.timestamp = [
            begin + timedelta(seconds=i * step_size_seconds)
            for i in range(sample_count)
        ]

        sample = np.arange(sample_count, dtype=float)
        theta = np.linspace(0.0, 2.0 * np.pi, sample_count)

        self.earth_ra_deg = (180.0 + np.rad2deg(theta)) % 360.0
        self.earth_dec_deg = 15.0 * np.sin(theta)
        self.sun_ra_deg = (45.0 + 0.985647 * sample / 1440.0) % 360.0
        self.sun_dec_deg = np.full(sample_count, 10.0)
        self.moon_ra_deg = (120.0 + 13.0 * sample / 1440.0) % 360.0
        self.moon_dec_deg = np.full(sample_count, -5.0)

        orbit_radius_km = 7000.0
        self.gcrs_pv = SimpleNamespace(
            position=np.column_stack(
                (
                    orbit_radius_km * np.cos(theta),
                    orbit_radius_km * np.sin(theta),
                    500.0 * np.sin(2.0 * theta),
                )
            )
        )
        self.sun_pv = SimpleNamespace(
            position=np.tile(np.array([1.5e8, 1.0e7, 2.0e7]), (sample_count, 1))
        )

    def index(self, time: datetime | float) -> int:
        if isinstance(time, (int, float)):
            seconds = float(time) - self.begin.timestamp()
        else:
            if time.tzinfo is None:
                time = time.replace(tzinfo=timezone.utc)
            seconds = time.timestamp() - self.begin.timestamp()
        idx = int(round(seconds / self.step_size))
        return max(0, min(len(self.timestamp) - 1, idx))


@dataclass(frozen=True)
class ScenarioTarget:
    obsid: int
    name: str
    ra: float
    dec: float
    merit: float
    exptime: int
    ss_min: int = 180
    ss_max: int = 900


SCENARIO_BEGIN = datetime(2026, 1, 1, tzinfo=timezone.utc)
SCENARIO_END = SCENARIO_BEGIN + timedelta(days=1)
SCENARIO_TARGETS = (
    ScenarioTarget(10000, "coast-default-0", 20.0, -10.0, 90.0, 900),
    ScenarioTarget(10001, "coast-default-1", 75.0, 5.0, 92.0, 900),
    ScenarioTarget(10002, "coast-default-2", 140.0, 20.0, 88.0, 1200),
    ScenarioTarget(10003, "coast-default-3", 215.0, -25.0, 95.0, 600),
    ScenarioTarget(10004, "coast-default-4", 300.0, 35.0, 85.0, 900),
)


def build_default_plan_payload() -> dict[str, Any]:
    ephem = DeterministicEphemeris(SCENARIO_BEGIN, SCENARIO_END)
    config = MissionConfig(
        name="COAST default plan regression",
        constraint=DeterministicConstraint(),
        ground_stations=GroundStationRegistry(stations=[]),
        solar_panel=SolarPanelSet(panels=[]),
        battery=Battery(watthour=100_000.0),
        spacecraft_bus=SpacecraftBus(
            attitude_control=AttitudeControlSystem(
                max_slew_rate=1.0,
                slew_acceleration=0.5,
                settle_time=10.0,
            ),
            star_trackers=StarTrackerConfiguration(
                star_trackers=[],
                min_functional_trackers=0,
                modes_require_lock=[],
            ),
            radiators=RadiatorConfiguration(radiators=[]),
        ),
    )
    config.constraint.ephem = ephem
    config.random_seed = 8675309
    config.targets.slew_distance_weight = 0.01
    config.targets.slew_time_weight = 0.05
    config.targets.collection_time_weight = 0.02

    ditl = QueueDITL(config=config, ephem=ephem, calculate_field_of_regard=False)
    for target in SCENARIO_TARGETS:
        ditl.queue.add(
            ra=target.ra,
            dec=target.dec,
            exptime=target.exptime,
            ss_min=target.ss_min,
            ss_max=target.ss_max,
            obsid=target.obsid,
            merit=target.merit,
            name=target.name,
        )

    if not ditl.calc():
        raise RuntimeError("Default plan scenario failed to calculate")

    schema = PlanSchema.from_plan(ditl.plan)
    plan_payload = schema.model_dump(mode="json", exclude_none=True)
    plan_payload.pop("created_at", None)
    plan_payload.pop("coast_sim_version", None)

    attitude_payload = None
    if schema.attitude_timeseries is not None:
        attitude_payload = schema.attitude_timeseries.model_dump(
            mode="json", exclude_none=True
        )
        attitude_payload.pop("created_at", None)
        attitude_payload.pop("coast_sim_version", None)

    return {
        "scenario": {
            "name": config.name,
            "begin": SCENARIO_BEGIN.isoformat(),
            "end": SCENARIO_END.isoformat(),
            "step_size_seconds": int(ephem.step_size),
            "targets": [target.__dict__ for target in SCENARIO_TARGETS],
        },
        "summary": {
            "plan_entries": len(plan_payload["entries"]),
            "attitude_samples": len(
                (attitude_payload or {"samples": []}).get("samples", [])
            ),
            "obsids": [entry["obsid"] for entry in plan_payload["entries"]],
            "obstypes": [entry["obstype"] for entry in plan_payload["entries"]],
        },
        "plan": plan_payload,
        "attitude_timeseries": attitude_payload,
    }


def _format_json(payload: dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _compare_payloads(
    expected: Any,
    actual: Any,
    path: str = "$",
    *,
    abs_tol: float,
    diffs: list[str],
) -> None:
    if len(diffs) >= 100:
        return
    if isinstance(expected, dict) and isinstance(actual, dict):
        expected_keys = set(expected)
        actual_keys = set(actual)
        for key in sorted(expected_keys - actual_keys):
            diffs.append(f"{path}.{key}: missing from actual payload")
        for key in sorted(actual_keys - expected_keys):
            diffs.append(f"{path}.{key}: unexpected in actual payload")
        for key in sorted(expected_keys & actual_keys):
            _compare_payloads(
                expected[key],
                actual[key],
                f"{path}.{key}",
                abs_tol=abs_tol,
                diffs=diffs,
            )
        return
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            diffs.append(
                f"{path}: list length changed from {len(expected)} to {len(actual)}"
            )
            return
        for index, (expected_item, actual_item) in enumerate(zip(expected, actual)):
            _compare_payloads(
                expected_item,
                actual_item,
                f"{path}[{index}]",
                abs_tol=abs_tol,
                diffs=diffs,
            )
        return
    if _is_number(expected) and _is_number(actual):
        if not math.isclose(float(expected), float(actual), abs_tol=abs_tol, rel_tol=0):
            diffs.append(f"{path}: expected {expected!r}, got {actual!r}")
        return
    if expected != actual:
        diffs.append(f"{path}: expected {expected!r}, got {actual!r}")


def compare_to_baseline(
    actual: dict[str, Any], baseline_path: Path, *, abs_tol: float
) -> list[str]:
    if not baseline_path.exists():
        return [
            f"{baseline_path} does not exist; run "
            "`python scripts/check_default_plan_output.py --update` to create it."
        ]
    expected = json.loads(baseline_path.read_text(encoding="utf-8"))
    diffs: list[str] = []
    _compare_payloads(expected, actual, abs_tol=abs_tol, diffs=diffs)
    return diffs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help=f"baseline JSON path (default: {DEFAULT_BASELINE})",
    )
    parser.add_argument(
        "--update",
        action="store_true",
        help="overwrite the baseline with the current default plan output",
    )
    parser.add_argument(
        "--abs-tol",
        type=float,
        default=DEFAULT_NUMERIC_ABS_TOL,
        help="absolute tolerance for numeric comparisons",
    )
    args = parser.parse_args(argv)

    actual = build_default_plan_payload()
    if args.update:
        args.baseline.parent.mkdir(parents=True, exist_ok=True)
        args.baseline.write_text(_format_json(actual), encoding="utf-8")
        print(f"Updated default plan output baseline: {args.baseline}")
        print(_format_json(actual["summary"]).rstrip())
        return 0

    diffs = compare_to_baseline(actual, args.baseline, abs_tol=args.abs_tol)
    if diffs:
        print("Default COAST plan output changed from baseline.")
        print("First differences:")
        for diff in diffs[:25]:
            print(f"  - {diff}")
        if len(diffs) > 25:
            print(f"  - ... {len(diffs) - 25} more differences")
        print(
            "If this planner/output change is intentional, run "
            "`python scripts/check_default_plan_output.py --update` and commit the "
            "baseline diff."
        )
        return 1

    print("Default COAST plan output matches baseline.")
    print(_format_json(actual["summary"]).rstrip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
