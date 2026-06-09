from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .plan import Plan


def _format_utc_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def tle_plan_metadata(
    tle_record: Any,
    tle_file: str | Path | None = None,
    *,
    source: str = "TLE",
) -> dict[str, Any]:
    """Build generic plan metadata for a TLE-backed ephemeris.

    COAST owns the plan metadata shape, while rust-ephem owns TLE parsing and
    element derivation through ``TLERecord.classical_elements()``.
    """
    classical_elements = getattr(tle_record, "classical_elements", None)
    if not callable(classical_elements):
        raise TypeError(
            "tle_record must provide classical_elements(); install rust-ephem >= 0.11"
        )

    return {
        "ephemeris": {
            "source": source,
            "tle_file": str(tle_file) if tle_file is not None else None,
            "tle_name": tle_record.name,
            "tle_epoch_utc": _format_utc_datetime(tle_record.epoch),
            "norad_id": tle_record.norad_id,
            "line1": tle_record.line1,
            "line2": tle_record.line2,
            "classical_elements": classical_elements(),
            "classical_elements_note": (
                "TLE mean elements at the TLE epoch; RightAscension_deg is RAAN. "
                "SemimajorAxis_m is derived from TLE mean motion, and "
                "TrueAnomaly_deg is derived from TLE mean anomaly."
            ),
        }
    }


def attach_tle_plan_metadata(
    plan: Plan,
    tle_record: Any,
    tle_file: str | Path | None = None,
    *,
    source: str = "TLE",
) -> None:
    """Attach TLE ephemeris metadata to a plan before schema export."""
    plan.metadata = {
        **getattr(plan, "metadata", {}),
        **tle_plan_metadata(tle_record=tle_record, tle_file=tle_file, source=source),
    }
