from __future__ import annotations

from datetime import datetime, timezone

import pytest
from rust_ephem.tle import TLERecord

from conops.targets import Plan, PlanMetadata, attach_tle_plan_metadata

_TLE1 = "1 43613U 18070A   26060.00000000  .00000000  00000-0  00000-0 0  9991"
_TLE2 = "2 43613  97.7898  39.6457 0016466  83.3495 116.0254 15.13083683    09"


@pytest.fixture
def tle_record() -> TLERecord:
    return TLERecord(
        name="Aperture-1",
        line1=_TLE1,
        line2=_TLE2,
        epoch=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )


def test_plan_metadata_from_tle_record_uses_rust_ephem_elements(
    tle_record: TLERecord,
) -> None:
    metadata = PlanMetadata.from_tle_record(
        tle_record=tle_record,
        tle_file="TLEs/Aperture-1_TLE_20260301.tle",
    ).model_dump(mode="json")

    ephemeris = metadata["ephemeris"]
    assert ephemeris["source"] == "TLE"
    assert ephemeris["tle_file"] == "TLEs/Aperture-1_TLE_20260301.tle"
    assert ephemeris["tle_name"] == "Aperture-1"
    assert ephemeris["tle_epoch_utc"] == "2026-03-01T00:00:00Z"
    assert ephemeris["norad_id"] == 43613
    assert ephemeris["line1"] == _TLE1
    assert ephemeris["line2"] == _TLE2

    elements = ephemeris["classical_elements"]
    assert elements == tle_record.classical_elements()
    assert elements["SemimajorAxis_m"] == pytest.approx(6904941.542146514)
    assert elements["Inclination_deg"] == pytest.approx(97.7898)
    assert elements["RightAscension_deg"] == pytest.approx(39.6457)


def test_attach_tle_plan_metadata_preserves_existing_metadata(
    tle_record: TLERecord,
) -> None:
    plan = Plan()
    plan.metadata = {"producer": {"name": "mission-generator"}}

    attach_tle_plan_metadata(plan, tle_record=tle_record)

    assert plan.metadata["producer"] == {"name": "mission-generator"}
    assert plan.metadata["ephemeris"]["classical_elements"] == (
        tle_record.classical_elements()
    )


def test_plan_metadata_requires_rust_ephem_element_api() -> None:
    class OldRecord:
        name = "legacy"
        epoch = datetime(2026, 3, 1, tzinfo=timezone.utc)
        norad_id = 43613
        line1 = _TLE1
        line2 = _TLE2

    with pytest.raises(TypeError, match="rust-ephem >= 0.11"):
        PlanMetadata.from_tle_record(OldRecord())


def test_plan_metadata_preserves_non_tle_ephemeris_payload() -> None:
    metadata = PlanMetadata.model_validate(
        {"ephemeris": {"source": "SPICE", "kernel": "example.bsp"}}
    ).model_dump(mode="json", exclude_none=True)

    assert metadata["ephemeris"]["source"] == "SPICE"
    assert metadata["ephemeris"]["kernel"] == "example.bsp"
    assert "classical_elements_note" not in metadata["ephemeris"]
