"""Tests for conops.targets.plan_schema (PlanSchema / PlanEntrySchema)."""

import json
import pathlib

import pytest

from conops.targets import PlanEntrySchema, PlanSchema

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


# ── PlanSchema ─────────────────────────────────────────────────────────────────


class TestPlanSchema:
    def test_construction_and_basic_fields(self):
        schema = _make_schema(3)
        assert schema.num_entries == 3
        assert len(schema.entries) == 3
        assert schema.version == "test-1.0"
        assert schema.start == pytest.approx(1_000_000.0)

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
        assert isinstance(raw["entries"], list)
        assert len(raw["entries"]) == 1

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
        # version string should be non-empty
        assert schema.version

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
        assert schema.version == "0.1.0"
        assert len(schema.entries) == 1
        assert schema.num_entries == len(schema.entries)
        # num_entries and created_at will have schema defaults
        assert isinstance(schema.created_at, str)

    def test_from_plan_classmethod_empty_plan(self):
        """from_plan on an empty Plan should produce zero entries and start/end=0."""
        from conops.targets.plan import Plan

        plan = Plan()
        schema = PlanSchema.from_plan(plan)
        assert schema.num_entries == 0
        assert schema.entries == []
        assert schema.start == 0.0
        assert schema.end == 0.0
