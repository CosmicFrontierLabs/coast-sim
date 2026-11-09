"""Exact, semantic comparisons of program executions across fresh processes.

Keep simulated timestamps and floating-point values unchanged. Run provenance and
wall-clock timings belong outside the compared record; do not round away a
different attitude or dispatch boundary to make a run appear repeatable.
"""

from typing import Any

from ..ditl.queue_ditl import QueueDITL


def first_difference(left: Any, right: Any, path: str = "$") -> dict[str, Any] | None:
    """Return the first unequal JSON value, including its path and both values."""
    if type(left) is not type(right):
        return {"path": path, "left": left, "right": right}
    if isinstance(left, dict):
        for key in sorted(left.keys() | right.keys()):
            if key not in left or key not in right:
                return {
                    "path": f"{path}.{key}",
                    "left_present": key in left,
                    "right_present": key in right,
                    "left": left.get(key),
                    "right": right.get(key),
                }
            difference = first_difference(left[key], right[key], f"{path}.{key}")
            if difference is not None:
                return difference
    elif isinstance(left, list):
        for index, (a, b) in enumerate(zip(left, right)):
            difference = first_difference(a, b, f"{path}[{index}]")
            if difference is not None:
                return difference
        if len(left) != len(right):
            return {
                "path": f"{path}.length",
                "left": len(left),
                "right": len(right),
            }
    elif left != right:
        return {"path": path, "left": left, "right": right}
    return None


def execution_record(
    simulation: QueueDITL, *, completed: bool, error: str | None = None
) -> dict[str, Any]:
    """Capture execution, decisions and telemetry, including a failed run's tail.

    A partial run's open entries are diagnostic evidence, not a validated plan.
    In particular, never call plan validation after an exception and replace the
    original failure with an error about an unfinished entry.
    """
    mismatches = (
        [str(m) for m in simulation.validate_plan_matches_execution()]
        if completed and error is None
        else None
    )
    return {
        "completed": completed,
        "error": error,
        "entries": [e.model_dump(mode="json") for e in simulation.plan],
        "events": [e.model_dump(mode="json") for e in simulation.log],
        "toos": [t.model_dump(mode="json") for t in simulation.too_register],
        "categories": [s.model_dump() for s in simulation.summary_by_category()],
        "groups": [s.model_dump() for s in simulation.summary_by_group()],
        "visits": [s.model_dump() for s in simulation.visit_table(per_window=True)],
        "instrument_mode_hours": simulation.instrument_mode_hours(),
        "housekeeping": [
            h.model_dump(mode="json") for h in simulation.telemetry.housekeeping
        ],
        "mismatches": mismatches,
        "failure_state": (
            {
                "next_tick": simulation.ustart
                + len(simulation.telemetry.housekeeping) * simulation.step_size,
                "attitude": [
                    simulation.acs.ra,
                    simulation.acs.dec,
                    simulation.acs.roll,
                ],
                "acs_mode": int(simulation.acs.acsmode),
                "battery_level": simulation.battery.battery_level,
                "active_observation": (
                    simulation.ppt.model_dump(mode="json")
                    if simulation.ppt is not None
                    else None
                ),
            }
            if error is not None
            else None
        ),
    }


def compare_runs(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    """Distinguish equality from successful, valid repeatability.

    Two identical failures are useful evidence but must not pass the baseline.
    Compare the event stream separately so the first changed dispatch decision
    is easy to locate even when the entry difference appears earlier in JSON.
    """
    a, b = left["execution"], right["execution"]
    difference = first_difference(a, b)
    provenance_difference = first_difference(
        left["provenance"], right["provenance"], "$.provenance"
    )
    valid = all(
        r["completed"] and r["error"] is None and r["mismatches"] == [] for r in (a, b)
    )
    return {
        "equal": difference is None,
        "same_provenance": provenance_difference is None,
        "passed": difference is None and provenance_difference is None and valid,
        "first_difference": difference,
        "first_event_difference": first_difference(
            a.get("events", []), b.get("events", []), "$.events"
        ),
        "provenance_difference": provenance_difference,
    }
