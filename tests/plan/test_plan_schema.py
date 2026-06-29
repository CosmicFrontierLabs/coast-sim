"""Tests for conops.targets.plan_schema (PlanSchema / PlanEntrySchema)."""

import json
import pathlib
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
from pydantic import ValidationError
from rust_ephem.tle import TLERecord

from conops.common.enums import ObsType
from conops.common.vector import attitude_to_quat
from conops.targets import (
    AttitudePointingSchema,
    AttitudeRotationSchema,
    AttitudeSampleSchema,
    AttitudeTimeseriesSchema,
    PlanEntrySchema,
    PlanSchema,
    TargetAttitudeSchema,
)
from conops.targets.plan_metadata import PlanMetadata

# ── Helpers ───────────────────────────────────────────────────────────────────

_ENTRY_DICT = {
    "name": "TEST",
    "ra": 123.45,
    "dec": -45.0,
    "roll": 0.0,
    "begin": 1_000_000.0,
    "end": 1_001_000.0,
    "merit": 50.0,
    "slewtime": 120,
    "insaa": 0,
    "obsid": 99,
    "obstype": "AT",
    "slewdist": 10.0,
    "ss_min": 300.0,
    "ss_max": 1_000_000.0,
    "exptime": 880,
    "exporig": 1000,
    "isat": False,
    "done": True,
    "exposure": 880,
}


def _make_schema(n: int = 2) -> PlanSchema:
    """Build a PlanSchema from raw dicts (no Plan object required)."""
    entries = [
        dict(
            _ENTRY_DICT,
            obsid=i,
            begin=float(1_000_000 + i * 2000),
            end=float(1_001_000 + i * 2000),
        )
        for i in range(n)
    ]
    return PlanSchema(
        version="test-1.0",
        created_at="2025-01-01T00:00:00+00:00",
        start=entries[0]["begin"],
        end=entries[-1]["end"],
        num_entries=n,
        entries=[PlanEntrySchema(**e) for e in entries],
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


# ── PlanEntrySchema ────────────────────────────────────────────────────────────


class TestPlanEntrySchema:
    def test_from_dict_roundtrip(self):
        entry = PlanEntrySchema(**_ENTRY_DICT)
        assert entry.name == "TEST"
        assert entry.ra == pytest.approx(123.45)
        assert entry.dec == pytest.approx(-45.0)
        assert entry.obsid == 99
        assert entry.done is True
        assert entry.isat is False
        assert entry.exptime == 880
        assert entry.exporig == 1000

    def test_defaults_for_missing_optional_fields(self):
        entry = PlanEntrySchema(
            **{k: v for k, v in _ENTRY_DICT.items() if k not in ("isat", "done")}
        )
        assert entry.isat is False
        assert entry.done is False

    def test_model_dump_contains_all_keys(self):
        entry = PlanEntrySchema(**_ENTRY_DICT)
        dumped = entry.model_dump()
        for key in (
            "name",
            "ra",
            "dec",
            "roll",
            "begin",
            "end",
            "merit",
            "slewtime",
            "insaa",
            "obsid",
            "obstype",
            "slewdist",
            "ss_min",
            "ss_max",
            "exptime",
            "exporig",
            "isat",
            "done",
            "exposure",
        ):
            assert key in dumped, f"Missing key: {key}"

    def test_static_entries_export_generic_target_attitude(self):
        entry = PlanEntrySchema(**dict(_ENTRY_DICT, ra=15.0, dec=45.0, roll=10.0))

        dumped = entry.model_dump(mode="json")

        attitude = dumped["target_attitude"]
        assert attitude["frame"] == "GCRS"
        assert attitude["body_frame"] == "COAST_BODY"
        assert attitude["rotation"]["representation"] == "quaternion"
        assert attitude["rotation"]["direction"] == "inertial_to_body"
        assert attitude["rotation"]["order"] == "wxyz"
        assert attitude["rotation"]["values"] == pytest.approx(
            attitude_to_quat(15.0, 45.0, 10.0).tolist()
        )
        assert attitude["pointing"] == {
            "ra_deg": 15.0,
            "dec_deg": 45.0,
            "roll_deg": 10.0,
            "boresight_axis": "+X",
            "roll_axis": "+X",
            "roll_source": "planned",
        }

    def test_target_attitude_metadata_fields_are_typed_literals(self):
        rotation = AttitudeRotationSchema(values=(1.0, 0.0, 0.0, 0.0))
        pointing = AttitudePointingSchema(
            ra_deg=15.0,
            dec_deg=45.0,
            roll_deg=10.0,
            boresight_axis="+Z",
        )

        attitude = TargetAttitudeSchema(rotation=rotation, pointing=pointing)

        assert attitude.rotation.representation == "quaternion"
        assert attitude.pointing.boresight_axis == "+Z"

        with pytest.raises(ValidationError):
            AttitudeRotationSchema(
                representation="matrix",
                values=(1.0, 0.0, 0.0, 0.0),
            )
        with pytest.raises(ValidationError):
            AttitudePointingSchema(
                ra_deg=15.0,
                dec_deg=45.0,
                roll_deg=10.0,
                boresight_axis="Z",
            )
        with pytest.raises(ValidationError):
            TargetAttitudeSchema(
                frame="ICRF",
                rotation=rotation,
                pointing=pointing,
            )

    def test_unconstrained_roll_sentinel_exports_zero_roll_attitude(self):
        entry = PlanEntrySchema(**dict(_ENTRY_DICT, roll=-1.0))

        attitude = entry.model_dump(mode="json")["target_attitude"]

        assert attitude["rotation"]["values"] == pytest.approx(
            attitude_to_quat(_ENTRY_DICT["ra"], _ENTRY_DICT["dec"], 0.0).tolist()
        )
        assert attitude["pointing"]["roll_deg"] == 0.0
        assert (
            attitude["pointing"]["roll_source"]
            == "defaulted_from_unconstrained_sentinel"
        )

    def test_gsp_contact_metadata_roundtrips(self):
        entry = PlanEntrySchema(
            **dict(
                _ENTRY_DICT,
                obstype="GSP",
                station="TRO",
                station_lat_deg=12.34,
                station_lon_deg=-56.78,
                station_alt_m=910.0,
                contact_begin=1_000_120.0,
                contact_end=1_000_720.0,
                track_start_ra=12.5,
                track_start_dec=-4.25,
                track_start_roll=11.0,
                track_end_ra=48.75,
                track_end_dec=9.5,
                track_end_roll=13.0,
            )
        )

        dumped = entry.model_dump(mode="json")
        assert dumped["station"] == "TRO"
        assert dumped["station_lat_deg"] == pytest.approx(12.34)
        assert dumped["station_lon_deg"] == pytest.approx(-56.78)
        assert dumped["station_alt_m"] == pytest.approx(910.0)
        assert dumped["contact_begin"] == "1970-01-12T13:48:40+00:00"
        assert dumped["contact_end"] == "1970-01-12T13:58:40+00:00"
        assert dumped["track_start_ra"] == pytest.approx(12.5)
        assert dumped["track_start_dec"] == pytest.approx(-4.25)
        assert dumped["track_start_roll"] == pytest.approx(11.0)
        assert dumped["track_end_ra"] == pytest.approx(48.75)
        assert dumped["track_end_dec"] == pytest.approx(9.5)
        assert dumped["track_end_roll"] == pytest.approx(13.0)

        reloaded = PlanEntrySchema(**dumped)
        assert reloaded.station == "TRO"
        assert reloaded.station_lat_deg == pytest.approx(12.34)
        assert reloaded.station_lon_deg == pytest.approx(-56.78)
        assert reloaded.station_alt_m == pytest.approx(910.0)
        assert reloaded.contact_begin == pytest.approx(1_000_120.0)
        assert reloaded.contact_end == pytest.approx(1_000_720.0)
        assert reloaded.track_start_ra == pytest.approx(12.5)
        assert reloaded.track_start_dec == pytest.approx(-4.25)
        assert reloaded.track_start_roll == pytest.approx(11.0)
        assert reloaded.track_end_ra == pytest.approx(48.75)
        assert reloaded.track_end_dec == pytest.approx(9.5)
        assert reloaded.track_end_roll == pytest.approx(13.0)

    def test_dynamic_pass_entries_do_not_export_fixed_target_attitude(self):
        entry = PlanEntrySchema(**dict(_ENTRY_DICT, obstype="GSP"))

        dumped = entry.model_dump(mode="json", exclude_none=True)

        assert "target_attitude" not in dumped


# ── PlanSchema ─────────────────────────────────────────────────────────────────


class TestPlanSchema:
    def test_construction_and_basic_fields(self):
        schema = _make_schema(3)
        assert schema.num_entries == 3
        assert len(schema.entries) == 3
        assert schema.version == 0  # coerced from "test-1.0"
        assert schema.start == pytest.approx(1_000_000.0)

    def test_save_to_directory_autogenerates_filename(self, tmp_path):
        schema = _make_schema(1)
        returned_path = schema.save(str(tmp_path) + "/")
        assert returned_path.parent == tmp_path.resolve()
        assert returned_path.suffix == ".json"
        assert returned_path.name.startswith("plan_")
        assert returned_path.exists()

    def test_save_to_existing_directory_autogenerates_filename(self, tmp_path):
        schema = _make_schema(1)
        returned_path = schema.save(tmp_path)  # tmp_path already exists as dir
        assert returned_path.parent == tmp_path.resolve()
        assert returned_path.name.startswith("plan_")

    def test_default_filename_format(self):
        schema = _make_schema(1)
        name = schema._default_filename()
        # e.g. plan_19700101T277H46M40Z_..._vtest-1.0.json
        assert name.startswith("plan_")
        assert name.endswith(".json")
        parts = name[len("plan_") : name.rfind("_v")]
        assert "T" in parts  # ISO date component present

    def test_save_and_load_roundtrip(self, tmp_path):
        original = _make_schema(2)
        dest = tmp_path / "plan.json"
        returned_path = original.save(dest)
        assert returned_path == dest.resolve()

        loaded = PlanSchema.load(dest)
        assert loaded.version == original.version
        assert loaded.created_at == original.created_at
        assert loaded.num_entries == original.num_entries
        assert len(loaded.entries) == len(original.entries)
        assert loaded.entries[0].obsid == original.entries[0].obsid
        assert loaded.entries[0].ra == pytest.approx(original.entries[0].ra)
        assert loaded.entries[0].begin == pytest.approx(original.entries[0].begin)
        assert loaded.entries[0].end == pytest.approx(original.entries[0].end)
        assert loaded.start == pytest.approx(original.start)
        assert loaded.end == pytest.approx(original.end)

    def test_save_and_load_roundtrip_metadata(self, tmp_path):
        original = _make_schema(1)
        original.metadata = PlanMetadata.model_validate(
            {
                "ephemeris": {
                    "source": "TLE",
                    "tle_name": "Aperture-1",
                    "tle_epoch_utc": "2026-03-01T00:00:00Z",
                    "norad_id": 43613,
                    "line1": "1 43613U 18070A   26060.00000000  .00000000  00000-0  00000-0 0  9991",
                    "line2": "2 43613  97.7898  39.6457 0016466  83.3495 116.0254 15.13083683    09",
                    "classical_elements": {"SemimajorAxis_m": 6_900_000.0},
                }
            }
        )
        dest = tmp_path / "plan.json"

        original.save(dest)

        raw = json.loads(dest.read_text())
        assert raw["metadata"] == original.metadata.model_dump(
            mode="json", exclude_none=True
        )
        loaded = PlanSchema.load(dest)
        assert loaded.metadata is not None
        assert loaded.metadata.model_dump(
            mode="json", exclude_none=True
        ) == original.metadata.model_dump(mode="json", exclude_none=True)

    def test_save_omits_empty_metadata(self, tmp_path):
        schema = _make_schema(1)
        dest = tmp_path / "plan.json"

        schema.save(dest)

        raw = json.loads(dest.read_text())
        assert "metadata" not in raw

    def test_save_writes_target_attitude_for_static_entries(self, tmp_path):
        schema = _make_schema(1)
        dest = tmp_path / "plan.json"

        schema.save(dest)

        raw = json.loads(dest.read_text())
        attitude = raw["entries"][0]["target_attitude"]
        assert attitude["rotation"]["direction"] == "inertial_to_body"
        assert attitude["rotation"]["order"] == "wxyz"
        assert attitude["pointing"]["boresight_axis"] == "+X"

    def test_save_produces_valid_json(self, tmp_path):
        schema = _make_schema(1)
        dest = tmp_path / "plan.json"
        schema.save(dest)
        raw = json.loads(dest.read_text())
        assert "version" in raw
        assert "created_at" in raw
        assert "start" in raw
        assert "end" in raw
        assert "num_entries" in raw
        assert isinstance(raw["start"], str), "start should be an ISO-8601 string"
        assert isinstance(raw["end"], str), "end should be an ISO-8601 string"
        assert "T" in raw["start"]  # sanity-check ISO format
        assert isinstance(raw["entries"], list)
        assert len(raw["entries"]) == 1

    def test_save_writes_linked_attitude_timeseries(self, tmp_path):
        schema = _make_schema(1)
        schema.attitude_timeseries = AttitudeTimeseriesSchema(
            samples=[
                AttitudeSampleSchema(
                    utime=1_000_000.0,
                    timestamp="1970-01-12T13:46:40+00:00",
                    ra=12.0,
                    dec=-4.0,
                    roll=30.0,
                    mode="SCIENCE",
                    obsid=99,
                    quat_w=1.0,
                    quat_x=0.0,
                    quat_y=0.0,
                    quat_z=0.0,
                )
            ]
        )
        dest = tmp_path / "plan.json"

        schema.save(dest)

        raw = json.loads(dest.read_text())
        assert raw["attitude_timeseries_file"] == "plan_attitude_timeseries.json"
        assert "attitude_timeseries" not in raw

        attitude_path = tmp_path / raw["attitude_timeseries_file"]
        attitude_raw = json.loads(attitude_path.read_text())
        assert attitude_raw["plan_file"] == "plan.json"
        assert attitude_raw["plan_version"] == raw["version"]
        assert attitude_raw["plan_start"] == raw["start"]
        assert attitude_raw["plan_end"] == raw["end"]
        assert attitude_raw["num_samples"] == 1
        assert attitude_raw["samples"][0]["mode"] == "SCIENCE"

    def test_entry_fields_in_json(self, tmp_path):
        schema = _make_schema(1)
        dest = tmp_path / "plan.json"
        schema.save(dest)
        raw = json.loads(dest.read_text())
        entry = raw["entries"][0]
        for key in (
            "name",
            "ra",
            "dec",
            "roll",
            "begin",
            "end",
            "merit",
            "slewtime",
            "insaa",
            "obsid",
            "obstype",
            "slewdist",
            "ss_min",
            "ss_max",
            "exptime",
            "exporig",
            "isat",
            "done",
            "exposure",
        ):
            assert key in entry, f"JSON entry missing key: {key}"
        assert isinstance(entry["begin"], str), "begin should be an ISO-8601 string"
        assert isinstance(entry["end"], str), "end should be an ISO-8601 string"
        assert "T" in entry["begin"]
        assert "station" not in entry
        assert "station_lat_deg" not in entry
        assert "station_lon_deg" not in entry
        assert "station_alt_m" not in entry
        assert "contact_begin" not in entry
        assert "contact_end" not in entry
        assert "track_start_ra" not in entry
        assert "track_start_dec" not in entry
        assert "track_start_roll" not in entry
        assert "track_end_ra" not in entry
        assert "track_end_dec" not in entry
        assert "track_end_roll" not in entry

    def test_load_existing_example_json(self):
        """Load a plan JSON file produced by a previous version (backward compat)."""
        example = (
            pathlib.Path(__file__).parent.parent.parent
            / "examples"
            / "plan_20251201T000000Z_20251201T235900Z_v0.json"
        )
        if not example.exists():
            pytest.skip("Example JSON file not found")
        schema = PlanSchema.load(example)
        assert schema.num_entries >= 0
        assert isinstance(schema.entries, list)
        assert isinstance(schema.version, int)

    def test_load_legacy_json_without_metadata(self, tmp_path):
        """Files produced before PlanSchema existed may lack created_at / num_entries."""
        legacy = {
            "version": "0.1.0",
            "start": 1_000_000,
            "end": 1_002_000,
            "entries": [_ENTRY_DICT],
        }
        dest = tmp_path / "legacy.json"
        dest.write_text(json.dumps(legacy))
        schema = PlanSchema.load(dest)
        assert schema.version == 0  # legacy semver coerced to 0
        assert len(schema.entries) == 1
        assert schema.num_entries == len(schema.entries)
        # num_entries and created_at will have schema defaults
        assert isinstance(schema.created_at, str)

    def test_reconcile_num_entries_when_missing(self, tmp_path):
        """num_entries is recomputed from entries when absent in JSON."""
        raw = {
            "version": 0,
            "start": 1_000_000,
            "end": 1_002_000,
            "entries": [_ENTRY_DICT, _ENTRY_DICT],
            # no num_entries key
        }
        dest = tmp_path / "plan.json"
        dest.write_text(json.dumps(raw))
        schema = PlanSchema.load(dest)
        assert schema.num_entries == 2

    def test_reconcile_num_entries_when_stale(self, tmp_path):
        """num_entries is corrected when it disagrees with the actual entries list."""
        raw = {
            "version": 0,
            "start": 1_000_000,
            "end": 1_002_000,
            "num_entries": 99,  # intentionally wrong
            "entries": [_ENTRY_DICT],
        }
        dest = tmp_path / "plan.json"
        dest.write_text(json.dumps(raw))
        schema = PlanSchema.load(dest)
        assert schema.num_entries == 1

    def test_reconcile_start_when_missing(self, tmp_path):
        """start is inferred from the first entry's begin time when absent."""
        raw = {
            "version": 0,
            "end": 1_001_000,
            "entries": [_ENTRY_DICT],
            # no start key
        }
        dest = tmp_path / "plan.json"
        dest.write_text(json.dumps(raw))
        schema = PlanSchema.load(dest)
        assert schema.start == pytest.approx(_ENTRY_DICT["begin"])

    def test_reconcile_end_when_missing(self, tmp_path):
        """end is inferred from the last entry's end time when absent."""
        raw = {
            "version": 0,
            "start": 1_000_000,
            "entries": [_ENTRY_DICT],
            # no end key
        }
        dest = tmp_path / "plan.json"
        dest.write_text(json.dumps(raw))
        schema = PlanSchema.load(dest)
        assert schema.end == pytest.approx(_ENTRY_DICT["end"])

    def test_from_plan_classmethod_empty_plan(self):
        """from_plan on an empty Plan should produce zero entries and start/end=0."""
        from conops.targets.plan import Plan

        plan = Plan()
        schema = PlanSchema.from_plan(plan)
        assert schema.num_entries == 0
        assert schema.entries == []
        assert schema.start == 0.0
        assert schema.end == 0.0

    def test_from_plan_manual_tle_metadata_merge(self, tle_record: TLERecord):
        from conops.targets.plan import Plan

        plan = Plan()
        schema = PlanSchema.from_plan(plan)
        schema.metadata = PlanMetadata.model_validate(
            {"generator": {"name": "unit-test"}}
        )

        schema.metadata = PlanMetadata.model_validate(
            {
                **(schema.metadata.model_dump(mode="json") if schema.metadata else {}),
                **PlanMetadata.from_tle_record(
                    tle_record=tle_record,
                    tle_file="tle/example.tle",
                ).model_dump(mode="json"),
            }
        )

        metadata = schema.metadata.model_dump(mode="json") if schema.metadata else {}
        assert metadata["generator"] == {"name": "unit-test"}
        assert metadata["ephemeris"]["source"] == "TLE"
        assert metadata["ephemeris"]["norad_id"] == 43613
        assert metadata["ephemeris"]["tle_file"] == "tle/example.tle"

    def test_from_plan_preserves_metadata(self):
        from conops.targets.plan import Plan

        plan = Plan()
        plan.metadata = {"generator": {"name": "test"}}

        schema = PlanSchema.from_plan(plan)

        assert schema.metadata is not None
        assert (
            schema.metadata.model_dump(mode="json", exclude_none=True) == plan.metadata
        )

    def test_from_plan_clamps_negative_exposure(self, tmp_path):
        """Generated JSON should not contain negative exposure for truncated entries."""
        from conops.targets.plan import Plan
        from conops.targets.plan_entry import PlanEntry

        config = Mock()
        config.constraint = Mock()
        config.constraint.ephem = Mock()
        config.spacecraft_bus = Mock()
        config.spacecraft_bus.attitude_control = Mock()

        entry = PlanEntry(config=config)
        entry.obstype = "AT"
        entry.begin = 1000.0
        entry.end = 1060.0
        entry.slewtime = 224
        entry.insaa = 0

        plan = Plan()
        plan.append(entry)

        schema = PlanSchema.from_plan(plan)
        assert schema.entries[0].exposure == 0

        dest = tmp_path / "plan.json"
        schema.save(dest)
        raw = json.loads(dest.read_text())
        assert raw["entries"][0]["exposure"] == 0

    def test_from_plan_computes_exposure_without_mutating_entry(self) -> None:
        """Serialising a plan should not modify the source PlanEntry."""
        from conops.targets.plan import Plan
        from conops.targets.plan_entry import PlanEntry

        config = Mock()
        config.constraint = Mock()
        config.constraint.ephem = Mock()
        config.spacecraft_bus = Mock()
        config.spacecraft_bus.attitude_control = Mock()

        entry = PlanEntry(config=config)
        entry.obstype = ObsType.AT
        entry.begin = 1000.0
        entry.end = 1200.0
        entry.slewtime = 50
        entry.insaa = 30

        plan = Plan()
        plan.append(entry)

        schema = PlanSchema.from_plan(plan)

        assert schema.entries[0].exposure == 120
        assert entry.insaa == 30
