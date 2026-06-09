from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .plan import Plan


def _format_utc_datetime(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


class TLEEphemerisMetadata(BaseModel):
    source: str = "TLE"
    tle_file: str | None = None
    tle_name: str
    tle_epoch_utc: str
    norad_id: int
    line1: str
    line2: str
    classical_elements: dict[str, float]
    classical_elements_note: str = Field(
        default=(
            "TLE mean elements at the TLE epoch; RightAscension_deg is RAAN. "
            "SemimajorAxis_m is derived from TLE mean motion, and "
            "TrueAnomaly_deg is derived from TLE mean anomaly."
        )
    )


class PlanMetadata(BaseModel):
    """Top-level metadata envelope persisted under ``PlanSchema.metadata``.

    ``ephemeris`` is typed when present, while producer-specific fields remain
    supported via ``extra=allow``.
    """

    model_config = ConfigDict(extra="allow")

    ephemeris: TLEEphemerisMetadata | None = None

    @classmethod
    def from_tle_record(
        cls,
        tle_record: Any,
        tle_file: str | Path | None = None,
        *,
        source: str = "TLE",
    ) -> PlanMetadata:
        classical_elements = getattr(tle_record, "classical_elements", None)
        if not callable(classical_elements):
            raise TypeError(
                "tle_record must provide classical_elements(); install rust-ephem >= 0.11"
            )

        return cls(
            ephemeris=TLEEphemerisMetadata(
                source=source,
                tle_file=str(tle_file) if tle_file is not None else None,
                tle_name=tle_record.name,
                tle_epoch_utc=_format_utc_datetime(tle_record.epoch),
                norad_id=tle_record.norad_id,
                line1=tle_record.line1,
                line2=tle_record.line2,
                classical_elements=classical_elements(),
            )
        )


def attach_tle_plan_metadata(
    plan: Plan,
    tle_record: Any,
    tle_file: str | Path | None = None,
    *,
    source: str = "TLE",
) -> None:
    """Attach TLE metadata to ``plan.metadata`` while preserving existing keys."""
    existing = PlanMetadata.model_validate(getattr(plan, "metadata", None) or {})
    ephemeris_metadata = PlanMetadata.from_tle_record(
        tle_record=tle_record,
        tle_file=tle_file,
        source=source,
    )

    plan.metadata = PlanMetadata.model_validate(
        {
            **existing.model_dump(mode="json", exclude_none=True),
            **ephemeris_metadata.model_dump(mode="json", exclude_none=True),
        }
    ).model_dump(mode="json", exclude_none=True)
