from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import numpy as np
import pytest
import rust_ephem
from rust_ephem.tle import TLERecord

from conops.targets import (
    Plan,
    PlanMetadata,
    attach_osculating_elements_metadata,
    attach_tle_plan_metadata,
)

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

    tle_mean_elements = ephemeris["tle_mean_elements"]
    assert tle_mean_elements["epoch_utc"] == ephemeris["tle_epoch_utc"]
    assert "not propagated osculating elements" in tle_mean_elements["note"]

    elements = tle_mean_elements["elements"]
    assert elements == tle_record.classical_elements()
    assert elements["SemimajorAxis_m"] == pytest.approx(6904941.542146514)
    assert elements["Inclination_deg"] == pytest.approx(97.7898)
    assert elements["RightAscension_deg"] == pytest.approx(39.6457)
    assert "classical_elements" not in ephemeris


def test_attach_tle_plan_metadata_preserves_existing_metadata(
    tle_record: TLERecord,
) -> None:
    plan = Plan()
    plan.metadata = {"producer": {"name": "mission-generator"}}

    attach_tle_plan_metadata(plan, tle_record=tle_record)

    assert plan.metadata["producer"] == {"name": "mission-generator"}
    assert plan.metadata["ephemeris"]["tle_mean_elements"]["elements"] == (
        tle_record.classical_elements()
    )


def test_attach_osculating_elements_uses_exact_gcrs_state_and_preserves_tle(
    tle_record: TLERecord,
) -> None:
    epoch = datetime(2026, 3, 1, 18, 0, tzinfo=timezone.utc)
    ephemeris = Mock(spec=rust_ephem.Ephemeris)
    ephemeris.timestamp = np.array(
        [epoch - timedelta(minutes=1), epoch],
        dtype=object,
    )
    ephemeris.gcrs_pv = Mock(
        position=np.array(
            [
                [7000.0, 0.0, 0.0],
                [-6045.0, -3490.0, 2500.0],
            ]
        ),
        velocity=np.array(
            [
                [0.0, 7.5, 0.0],
                [-3.457, 6.618, 2.533],
            ]
        ),
    )
    plan = Plan()
    plan.metadata = {"producer": {"name": "mission-generator"}}
    attach_tle_plan_metadata(plan, tle_record=tle_record)

    attach_osculating_elements_metadata(plan, ephemeris, epoch)

    assert plan.metadata["producer"] == {"name": "mission-generator"}
    ephemeris_metadata = plan.metadata["ephemeris"]
    assert "tle_mean_elements" in ephemeris_metadata

    osculating = ephemeris_metadata["osculating_elements"]
    assert osculating["epoch_utc"] == "2026-03-01T18:00:00Z"
    assert osculating["frame"] == "GCRS"
    assert osculating["origin"] == "Earth center"
    assert "Instantaneous two-body osculating elements" in osculating["note"]

    elements = osculating["elements"]
    assert elements["SemimajorAxis_m"] == pytest.approx(
        8_788_070.943,
        abs=0.001,
    )
    assert elements["Eccentricity"] == pytest.approx(0.171210, abs=1.0e-6)
    assert elements["Inclination_deg"] == pytest.approx(153.249, abs=0.001)
    assert elements["RightAscension_deg"] == pytest.approx(255.279, abs=0.001)
    assert elements["ArgPeriapsis_deg"] == pytest.approx(20.068, abs=0.001)
    assert elements["TrueAnomaly_deg"] == pytest.approx(28.446, abs=0.001)
    assert elements["GravitationalParameter_m3_s2"] == pytest.approx(
        rust_ephem.WGS72_EARTH_MU_KM3_S2 * 1.0e9
    )

    attach_tle_plan_metadata(plan, tle_record=tle_record)
    assert plan.metadata["ephemeris"]["osculating_elements"]["elements"] == elements


def test_attach_osculating_elements_requires_exact_epoch() -> None:
    epoch = datetime(2026, 3, 1, 18, 0, tzinfo=timezone.utc)
    ephemeris = Mock(spec=rust_ephem.Ephemeris)
    ephemeris.timestamp = np.array(
        [epoch - timedelta(minutes=1)],
        dtype=object,
    )
    ephemeris.gcrs_pv = Mock(
        position=np.array([[7000.0, 0.0, 0.0]]),
        velocity=np.array([[0.0, 7.5, 0.0]]),
    )

    with pytest.raises(ValueError, match="found 0 matches"):
        attach_osculating_elements_metadata(Plan(), ephemeris, epoch)


def test_attach_osculating_elements_rejects_duplicate_epoch() -> None:
    epoch = datetime(2026, 3, 1, 18, 0, tzinfo=timezone.utc)
    ephemeris = Mock(spec=rust_ephem.Ephemeris)
    ephemeris.timestamp = np.array([epoch, epoch], dtype=object)
    ephemeris.gcrs_pv = Mock(
        position=np.array([[7000.0, 0.0, 0.0]] * 2),
        velocity=np.array([[0.0, 7.5, 0.0]] * 2),
    )

    with pytest.raises(ValueError, match="found 2 matches"):
        attach_osculating_elements_metadata(Plan(), ephemeris, epoch)


def test_attach_osculating_elements_requires_aligned_state_arrays() -> None:
    epoch = datetime(2026, 3, 1, 18, 0, tzinfo=timezone.utc)
    ephemeris = Mock(spec=rust_ephem.Ephemeris)
    ephemeris.timestamp = np.array([epoch], dtype=object)
    ephemeris.gcrs_pv = Mock(
        position=np.array([[7000.0, 0.0, 0.0]]),
        velocity=np.empty((0, 3)),
    )

    with pytest.raises(ValueError, match="matching lengths"):
        attach_osculating_elements_metadata(Plan(), ephemeris, epoch)


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
    assert "tle_mean_elements" not in metadata["ephemeris"]


def test_plan_metadata_preserves_missing_source_ephemeris_payload() -> None:
    metadata = PlanMetadata.model_validate(
        {"ephemeris": {"kernel": "example.bsp"}}
    ).model_dump(mode="json", exclude_none=True)

    assert metadata["ephemeris"]["kernel"] == "example.bsp"
    assert "source" not in metadata["ephemeris"]
    assert "tle_mean_elements" not in metadata["ephemeris"]


def test_plan_metadata_preserves_empty_ephemeris_payload() -> None:
    metadata = PlanMetadata.model_validate({"ephemeris": {}}).model_dump(
        mode="json", exclude_none=True
    )

    assert metadata["ephemeris"] == {}
