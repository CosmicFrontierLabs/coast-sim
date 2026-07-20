"""Tests for the JSON-export/import behavior of PlanEntry / Pointing / Plan.

These tests used to target the standalone ``PlanEntrySchema`` / ``PlanSchema``
mirror classes (removed from ``conops.targets.plan_schema``). That behavior
now lives directly on ``PlanEntry`` (``conops.targets.plan_entry``),
``Pointing`` (``conops.targets.pointing``), and ``Plan``
(``conops.targets.plan``), so these tests exercise those classes directly.

Coverage that is already exercised by ``tests/plan/test_plan.py`` (basic
save/load roundtrips, directory auto-naming/versioning, attitude-timeseries
companion files, plan-level metadata roundtrips) is intentionally not
duplicated here; this file focuses on behavior unique to the old schema
tests: target-attitude generation, GSP field roundtripping, version
coercion, legacy-JSON compatibility, and TLE metadata merging.
"""

import json
import pathlib
from datetime import datetime, timezone
from unittest.mock import Mock

import pytest
import rust_ephem
from pydantic import ValidationError
from rust_ephem.tle import TLERecord

from conops import (
    AttitudeControlSystem,
    Constraint,
    MissionConfig,
    Plan,
    PlanEntry,
    Pointing,
)
from conops.common.enums import ObsType
from conops.common.vector import attitude_to_quat
from conops.targets import (
    AttitudePointingSchema,
    AttitudeRotationSchema,
    TargetAttitudeSchema,
)
from conops.targets.plan_metadata import PlanMetadata, attach_tle_plan_metadata

# ── Helpers ───────────────────────────────────────────────────────────────────

_ENTRY_KWARGS = dict(
    name="TEST",
    ra=123.45,
    dec=-45.0,
    roll=0.0,
    begin=1_000_000.0,
    end=1_001_000.0,
    merit=50.0,
    slewtime=120,
    insaa=0,
    obsid=99,
    obstype=ObsType.AT,
    slewdist=10.0,
    ss_min=300.0,
    ss_max=1_000_000.0,
)


def _make_entry(
    exptime: int = 880, exporig: int = 1000, **overrides: object
) -> PlanEntry:
    """Build a bare PlanEntry with private exptime/exporig attrs initialized."""
    kwargs = dict(_ENTRY_KWARGS)
    kwargs.update(overrides)
    entry = PlanEntry(**kwargs)
    entry._exptime = exptime
    entry._exporig = exporig
    return entry


def _make_pointing(
    exptime: int = 880, exporig: int = 1000, **overrides: object
) -> Pointing:
    """Build a bare Pointing with private exptime/exporig attrs initialized."""
    kwargs = dict(_ENTRY_KWARGS)
    kwargs.update(overrides)
    entry = Pointing(**kwargs)
    entry._exptime = exptime
    entry._exporig = exporig
    return entry


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


# ── PlanEntry / Pointing export ───────────────────────────────────────────────


class TestPlanEntryExport:
    def test_model_dump_contains_expected_keys_and_excludes_config(self):
        entry = _make_entry()
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
            "exposure",
        ):
            assert key in dumped, f"Missing key: {key}"
        for key in ("config", "constraint", "acs_config", "ephem", "saa", "windows"):
            assert key not in dumped, f"Field should be excluded: {key}"

    def test_pointing_model_dump_contains_isat_and_done_but_not_fom(self):
        entry = _make_pointing(exptime=880, exporig=1000)
        dumped = entry.model_dump()
        assert dumped["isat"] is False
        assert dumped["done"] is False
        assert "fom" not in dumped

        done_entry = _make_pointing(exptime=0, exporig=1000)
        assert done_entry.model_dump()["done"] is True

    def test_pointing_defaults_isat_false_done_false(self):
        entry = _make_pointing(exptime=880, exporig=1000)
        assert entry.isat is False
        assert entry.done is False

    def test_static_entries_export_generic_target_attitude(self):
        entry = _make_entry(ra=15.0, dec=45.0, roll=10.0, obstype=ObsType.AT)

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
        entry = _make_entry(roll=-1.0, obstype=ObsType.AT)

        attitude = entry.model_dump(mode="json")["target_attitude"]

        assert attitude["rotation"]["values"] == pytest.approx(
            attitude_to_quat(_ENTRY_KWARGS["ra"], _ENTRY_KWARGS["dec"], 0.0).tolist()
        )
        assert attitude["pointing"]["roll_deg"] == 0.0
        assert (
            attitude["pointing"]["roll_source"]
            == "defaulted_from_unconstrained_sentinel"
        )

    def test_gsp_contact_metadata_roundtrips(self):
        entry = _make_entry(
            obstype=ObsType.GSP,
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

        reloaded = PlanEntry(**dumped)
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

    def test_gsp_entries_do_not_export_fixed_target_attitude(self):
        entry = _make_entry(obstype=ObsType.GSP)

        dumped = entry.model_dump(mode="json", exclude_none=True)

        assert "target_attitude" not in dumped

    def test_entry_fields_absent_when_none(self):
        entry = _make_entry(obstype=ObsType.AT)
        dumped = entry.model_dump(mode="json", exclude_none=True)
        for key in (
            "station",
            "station_lat_deg",
            "station_lon_deg",
            "station_alt_m",
            "contact_begin",
            "contact_end",
            "track_start_ra",
            "track_start_dec",
            "track_start_roll",
            "track_end_ra",
            "track_end_dec",
            "track_end_roll",
        ):
            assert key not in dumped, f"Field should be absent when None: {key}"


# ── Plan version coercion ─────────────────────────────────────────────────────


class TestPlanVersionCoercion:
    def test_legacy_semver_string_coerces_to_zero(self):
        plan = Plan(version="test-1.0")
        assert plan.version == 0

    def test_integer_version_passes_through(self):
        plan = Plan(version=5)
        assert plan.version == 5

    def test_numeric_string_version_coerces_to_int(self):
        plan = Plan(version="3")
        assert plan.version == 3


# ── Plan I/O: filenames, legacy compatibility, TLE metadata ──────────────────


class TestPlanIO:
    def test_default_filename_format(self):
        plan = Plan()
        plan.append(_make_entry(begin=1_000_000.0, end=1_001_000.0))
        name = plan._default_filename()
        assert name.startswith("plan_")
        assert name.endswith("_v0.json")
        parts = name[len("plan_") : name.rfind("_v")]
        assert "T" in parts  # ISO date component present

    def test_save_writes_target_attitude_for_static_entries(self, tmp_path):
        plan = Plan()
        plan.append(_make_entry(obstype=ObsType.AT))
        dest = tmp_path / "plan.json"

        plan.save(dest)

        raw = json.loads(dest.read_text())
        attitude = raw["entries"][0]["target_attitude"]
        assert attitude["rotation"]["direction"] == "inertial_to_body"
        assert attitude["rotation"]["order"] == "wxyz"
        assert attitude["pointing"]["boresight_axis"] == "+X"

    def test_load_existing_example_json(self):
        """Load a plan JSON file produced by a previous version (backward compat)."""
        example = (
            pathlib.Path(__file__).parent.parent.parent
            / "examples"
            / "plan_20251201T000000Z_20251201T235900Z_v0.json"
        )
        if not example.exists():
            pytest.skip("Example JSON file not found")
        plan = Plan.load(example)
        assert plan.num_entries >= 0
        assert isinstance(plan.entries, list)
        assert isinstance(plan.version, int)

    def test_load_legacy_json_without_metadata(self, tmp_path):
        """Files produced before created_at/num_entries existed should still load."""
        legacy = {
            "version": "0.1.0",
            "start": 1_000_000,
            "end": 1_002_000,
            "entries": [{"begin": 1_000_000.0, "end": 1_001_000.0}],
        }
        dest = tmp_path / "legacy.json"
        dest.write_text(json.dumps(legacy))
        plan = Plan.load(dest)
        assert plan.version == 0  # legacy semver coerced to 0
        assert len(plan.entries) == 1
        assert plan.num_entries == len(plan.entries)
        assert isinstance(plan.created_at, str)

    def test_reconcile_num_entries_when_missing(self, tmp_path):
        """num_entries is recomputed from entries when absent in JSON."""
        raw = {
            "version": 0,
            "start": 1_000_000,
            "end": 1_002_000,
            "entries": [
                {"begin": 1_000_000.0, "end": 1_001_000.0},
                {"begin": 1_001_000.0, "end": 1_002_000.0},
            ],
            # no num_entries key
        }
        dest = tmp_path / "plan.json"
        dest.write_text(json.dumps(raw))
        plan = Plan.load(dest)
        assert plan.num_entries == 2

    def test_reconcile_num_entries_when_stale(self, tmp_path):
        """num_entries (even if present, since it's computed) reflects real entries."""
        raw = {
            "version": 0,
            "start": 1_000_000,
            "end": 1_002_000,
            "num_entries": 99,  # intentionally wrong / ignored since computed
            "entries": [{"begin": 1_000_000.0, "end": 1_001_000.0}],
        }
        dest = tmp_path / "plan.json"
        dest.write_text(json.dumps(raw))
        plan = Plan.load(dest)
        assert plan.num_entries == 1

    def test_reconcile_start_when_missing(self, tmp_path):
        """start is always derived live from the first entry's begin time,
        regardless of whether a (now-ignored) legacy 'start' key is present."""
        raw = {
            "version": 0,
            "entries": [{"begin": 1_000_000.0, "end": 1_001_000.0}],
            # no start key
        }
        dest = tmp_path / "plan.json"
        dest.write_text(json.dumps(raw))
        plan = Plan.load(dest)
        assert plan._start_ts == pytest.approx(1_000_000.0)

    def test_reconcile_end_when_missing(self, tmp_path):
        """end is always derived live from the last entry's end time,
        regardless of whether a (now-ignored) legacy 'end' key is present."""
        raw = {
            "version": 0,
            "entries": [{"begin": 1_000_000.0, "end": 1_001_000.0}],
            # no end key
        }
        dest = tmp_path / "plan.json"
        dest.write_text(json.dumps(raw))
        plan = Plan.load(dest)
        assert plan._end_ts == pytest.approx(1_001_000.0)

    def test_tle_metadata_merge_survives_save_and_load(self, tmp_path, tle_record):
        """attach_tle_plan_metadata-populated metadata roundtrips through save/load."""
        plan = Plan()
        plan.metadata = {"generator": {"name": "unit-test"}}

        attach_tle_plan_metadata(
            plan, tle_record=tle_record, tle_file="tle/example.tle"
        )

        assert plan.metadata["generator"] == {"name": "unit-test"}
        assert plan.metadata["ephemeris"]["source"] == "TLE"
        assert plan.metadata["ephemeris"]["norad_id"] == 43613
        assert plan.metadata["ephemeris"]["tle_file"] == "tle/example.tle"

        dest = tmp_path / "plan.json"
        plan.save(dest)

        raw = json.loads(dest.read_text())
        assert raw["metadata"] == plan.metadata

        loaded = Plan.load(dest)
        assert loaded.metadata == plan.metadata

    def test_manual_tle_metadata_dict_merge(self, tle_record):
        """Manually merging PlanMetadata dumps into plan.metadata (fact #6 pattern)."""
        plan = Plan()
        plan.metadata = PlanMetadata.model_validate(
            {"generator": {"name": "unit-test"}}
        ).model_dump(mode="json", exclude_none=True)

        plan.metadata = PlanMetadata.model_validate(
            {
                **plan.metadata,
                **PlanMetadata.from_tle_record(
                    tle_record=tle_record,
                    tle_file="tle/example.tle",
                ).model_dump(mode="json"),
            }
        ).model_dump(mode="json", exclude_none=True)

        assert plan.metadata["generator"] == {"name": "unit-test"}
        assert plan.metadata["ephemeris"]["source"] == "TLE"
        assert plan.metadata["ephemeris"]["norad_id"] == 43613
        assert plan.metadata["ephemeris"]["tle_file"] == "tle/example.tle"


# ── Exposure computation (non-mutating, clamped) ─────────────────────────────


def _make_from_config_entry() -> PlanEntry:
    config = Mock()
    config.__class__ = MissionConfig
    config.fault_management = None
    config.constraint = Mock()
    config.constraint.__class__ = Constraint
    config.constraint.ephem = Mock()
    config.constraint.ephem.__class__ = rust_ephem.Ephemeris
    config.spacecraft_bus = Mock()
    config.spacecraft_bus.attitude_control = Mock()
    config.spacecraft_bus.attitude_control.__class__ = AttitudeControlSystem

    return PlanEntry(config=config)


class TestExposureViaPlan:
    def test_clamps_negative_exposure(self, tmp_path):
        """Generated JSON should not contain negative exposure for truncated entries."""
        entry = _make_from_config_entry()
        entry.obstype = ObsType.AT
        entry.begin = 1000.0
        entry.end = 1060.0
        entry.slewtime = 224
        entry.insaa = 0

        plan = Plan()
        plan.append(entry)

        assert plan.entries[0].exposure == 0

        dest = tmp_path / "plan.json"
        plan.save(dest)
        raw = json.loads(dest.read_text())
        assert raw["entries"][0]["exposure"] == 0

    def test_computes_exposure_without_mutating_entry(self):
        """Reading exposure should not modify the source PlanEntry."""
        entry = _make_from_config_entry()
        entry.obstype = ObsType.AT
        entry.begin = 1000.0
        entry.end = 1200.0
        entry.slewtime = 50
        entry.insaa = 30

        plan = Plan()
        plan.append(entry)

        assert plan.entries[0].exposure == 120
        assert entry.insaa == 30
