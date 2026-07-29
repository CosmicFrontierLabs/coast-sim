from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

import rust_ephem
from pydantic import BaseModel, ConfigDict

from .plan import Plan


def _as_utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _format_utc_datetime(value: datetime) -> str:
    return _as_utc_datetime(value).isoformat().replace("+00:00", "Z")


TLE_MEAN_ELEMENTS_NOTE = (
    "TLE mean elements for SGP4, not propagated osculating elements. "
    "RightAscension_deg is RAAN. SemimajorAxis_m is derived from TLE mean "
    "motion, and TrueAnomaly_deg is derived from TLE mean anomaly."
)

OSCULATING_ELEMENTS_NOTE = (
    "Instantaneous two-body osculating elements derived from the GCRS "
    "position and velocity at epoch_utc. RightAscension_deg is RAAN. "
    "Angles are normalized to [0, 360) degrees."
)


class TLEMeanElementsMetadata(BaseModel):
    epoch_utc: str
    elements: dict[str, float]
    note: str


class OsculatingElementsMetadata(BaseModel):
    epoch_utc: str
    frame: Literal["GCRS"] = "GCRS"
    origin: Literal["Earth center"] = "Earth center"
    elements: dict[str, float]
    note: str = OSCULATING_ELEMENTS_NOTE


class EphemerisMetadata(BaseModel):
    model_config = ConfigDict(extra="allow")

    source: str | None = None
    tle_file: str | None = None
    tle_name: str | None = None
    tle_epoch_utc: str | None = None
    norad_id: int | None = None
    line1: str | None = None
    line2: str | None = None
    tle_mean_elements: TLEMeanElementsMetadata | None = None
    osculating_elements: OsculatingElementsMetadata | None = None


class PlanMetadata(BaseModel):
    """Top-level metadata envelope persisted under ``Plan.metadata``.

    ``ephemeris`` is typed when present, while producer-specific fields remain
    supported via ``extra=allow``.
    """

    model_config = ConfigDict(extra="allow")

    ephemeris: EphemerisMetadata | None = None

    @classmethod
    def from_tle_record(
        cls,
        tle_record: rust_ephem.TLERecord,
        tle_file: str | Path | None = None,
        *,
        source: str = "TLE",
    ) -> PlanMetadata:
        classical_elements = getattr(tle_record, "classical_elements", None)
        if not callable(classical_elements):
            raise TypeError(
                "tle_record must provide classical_elements(); install rust-ephem >= 0.11"
            )

        tle_epoch_utc = _format_utc_datetime(tle_record.epoch)
        return cls(
            ephemeris=EphemerisMetadata(
                source=source,
                tle_file=str(tle_file) if tle_file is not None else None,
                tle_name=tle_record.name,
                tle_epoch_utc=tle_epoch_utc,
                norad_id=tle_record.norad_id,
                line1=tle_record.line1,
                line2=tle_record.line2,
                tle_mean_elements=TLEMeanElementsMetadata(
                    epoch_utc=tle_epoch_utc,
                    elements=classical_elements(),
                    note=TLE_MEAN_ELEMENTS_NOTE,
                ),
            )
        )


def _attach_ephemeris_metadata(
    plan: Plan,
    ephemeris_update: EphemerisMetadata,
) -> None:
    existing = PlanMetadata.model_validate(getattr(plan, "metadata", None) or {})
    existing_data = existing.model_dump(mode="json", exclude_none=True)
    existing_ephemeris = existing_data.get("ephemeris", {})
    update_ephemeris = ephemeris_update.model_dump(
        mode="json",
        exclude_none=True,
    )
    plan.metadata = PlanMetadata.model_validate(
        {
            **existing_data,
            "ephemeris": {
                **existing_ephemeris,
                **update_ephemeris,
            },
        }
    ).model_dump(mode="json", exclude_none=True)


def attach_tle_plan_metadata(
    plan: Plan,
    tle_record: rust_ephem.TLERecord,
    tle_file: str | Path | None = None,
    *,
    source: str = "TLE",
) -> None:
    """Attach TLE metadata to ``plan.metadata`` while preserving existing keys."""
    ephemeris_metadata = PlanMetadata.from_tle_record(
        tle_record=tle_record,
        tle_file=tle_file,
        source=source,
    ).ephemeris
    if ephemeris_metadata is None:
        raise ValueError("TLE plan metadata must include ephemeris metadata")
    _attach_ephemeris_metadata(plan, ephemeris_metadata)


def attach_osculating_elements_metadata(
    plan: Plan,
    ephemeris: rust_ephem.Ephemeris,
    epoch: datetime,
) -> None:
    """Attach GCRS osculating elements at an exact ephemeris timestamp.

    rust-ephem owns the Cartesian-state conversion, including selection of its
    central-body gravitational parameter. The exact value returned by
    rust-ephem is included in the serialized elements.
    """
    timestamps = ephemeris.timestamp
    positions = ephemeris.gcrs_pv.position
    velocities = ephemeris.gcrs_pv.velocity
    if not (len(timestamps) == len(positions) == len(velocities)):
        raise ValueError(
            "Ephemeris timestamps, GCRS positions, and GCRS velocities must "
            "have matching lengths"
        )

    epoch_utc = _as_utc_datetime(epoch)
    matching_indices = [
        index
        for index, timestamp in enumerate(timestamps)
        if _as_utc_datetime(timestamp) == epoch_utc
    ]
    if len(matching_indices) != 1:
        raise ValueError(
            "Osculating-element epoch must match exactly one ephemeris "
            f"timestamp; found {len(matching_indices)} matches for "
            f"{_format_utc_datetime(epoch_utc)}"
        )

    index = matching_indices[0]
    elements = rust_ephem.osculating_elements_from_state(
        position_km=positions[index],
        velocity_km_s=velocities[index],
    )
    serialized_elements = {
        "SemimajorAxis_m": elements["semimajor_axis_km"] * 1_000.0,
        "Eccentricity": elements["eccentricity"],
        "Inclination_deg": elements["inclination_deg"],
        "RightAscension_deg": elements["right_ascension_of_ascending_node_deg"],
        "ArgPeriapsis_deg": elements["argument_of_periapsis_deg"],
        "TrueAnomaly_deg": elements["true_anomaly_deg"],
        "GravitationalParameter_m3_s2": (
            elements["gravitational_parameter_km3_s2"] * 1.0e9
        ),
    }

    _attach_ephemeris_metadata(
        plan,
        EphemerisMetadata(
            osculating_elements=OsculatingElementsMetadata(
                epoch_utc=_format_utc_datetime(epoch_utc),
                elements=serialized_elements,
            )
        ),
    )
