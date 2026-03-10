"""Tests for conops.plan module."""

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from conops import Plan, PlanEntry
from conops.common.enums import ObsType


def _make_plan_entry(obsid: int, begin: float, end: float) -> PlanEntry:
    """Return a bare PlanEntry instance (bypasses __init__) with the minimal
    attributes that PlanEntrySchema._coerce_from_plan_entry reads."""
    entry = object.__new__(PlanEntry)
    entry.name = f"TARGET_{obsid}"
    entry.ra = 10.0 + obsid
    entry.dec = -20.0 + obsid
    entry.roll = -1.0
    entry.begin = begin
    entry.end = end
    entry.merit = 100.0
    entry.slewtime = 60
    entry.insaa = 0
    entry.obsid = obsid
    entry.obstype = ObsType.AT
    entry.slewdist = 5.0
    entry.ss_min = 300.0
    entry.ss_max = 1_000_000.0
    entry._exptime = 1000
    entry._exporig = 1000
    entry.isat = False
    entry.done = False
    entry.exposure = 1000
    return entry


def _make_plan(n: int = 2) -> Plan:
    """Build a Plan with *n* real PlanEntry objects."""
    plan = Plan()
    for i in range(n):
        plan.append(
            _make_plan_entry(
                obsid=i,
                begin=float(1_000_000 + i * 2000),
                end=float(1_001_000 + i * 2000),
            )
        )
    return plan


class TestTargetList:
    """Test TargetList class."""

    def test_target_list_initialization_length(self, empty_target_list):
        """Test that TargetList initializes with empty list (length)."""
        assert len(empty_target_list) == 0

    def test_target_list_initialization_targets(self, empty_target_list):
        """Test that TargetList initializes with empty list (targets)."""
        assert empty_target_list.targets == []

    def test_target_list_add_target_first(self, empty_target_list):
        """Test adding first target to TargetList."""
        target1 = Mock(spec=PlanEntry)
        empty_target_list.add_target(target1)
        assert len(empty_target_list) == 1

    def test_target_list_add_target_first_item(self, target_list_with_one):
        """Test adding first target to TargetList (item check)."""
        tl, target1 = target_list_with_one
        assert tl[0] == target1

    def test_target_list_add_target_second(self, target_list_with_two):
        """Test adding second target to TargetList."""
        tl, _, _ = target_list_with_two
        assert len(tl) == 2

    def test_target_list_add_target_second_item(self, target_list_with_two):
        """Test adding second target to TargetList (item check)."""
        tl, _, target2 = target_list_with_two
        assert tl[1] == target2

    def test_target_list_getitem_first(self, target_list_with_two):
        """Test getting first item from TargetList by index."""
        tl, target1, _ = target_list_with_two
        assert tl[0] == target1

    def test_target_list_getitem_second(self, target_list_with_two):
        """Test getting second item from TargetList by index."""
        tl, _, target2 = target_list_with_two
        assert tl[1] == target2


class TestPlan:
    """Test Plan class."""

    def test_plan_initialization_length(self, empty_plan):
        """Test that Plan initializes with empty list (length)."""
        assert len(empty_plan) == 0

    def test_plan_initialization_entries(self, empty_plan):
        """Test that Plan initializes with empty list (entries)."""
        assert empty_plan.entries == []

    def test_plan_getitem_first(self, plan_with_two_entries):
        """Test getting first item from Plan by index."""
        plan, ppt1, _ = plan_with_two_entries
        assert plan[0] == ppt1

    def test_plan_getitem_second(self, plan_with_two_entries):
        """Test getting second item from Plan by index."""
        plan, _, ppt2 = plan_with_two_entries
        assert plan[1] == ppt2

    def test_plan_which_ppt_finds_current_pointing_ppt1(self, plan_with_two_entries):
        """Test which_ppt finds ppt1 at time 50.0."""
        plan, ppt1, _ = plan_with_two_entries
        assert plan.which_ppt(50.0) == ppt1

    def test_plan_which_ppt_finds_current_pointing_ppt2(self, plan_with_two_entries):
        """Test which_ppt finds ppt2 at time 150.0."""
        plan, _, ppt2 = plan_with_two_entries
        assert plan.which_ppt(150.0) == ppt2

    def test_plan_which_ppt_finds_current_pointing_boundary(
        self, plan_with_two_entries
    ):
        """Test which_ppt at boundary (should match ppt2)."""
        plan, _, ppt2 = plan_with_two_entries
        assert plan.which_ppt(100.0) == ppt2

    def test_plan_which_ppt_finds_current_pointing_outside(self, plan_with_two_entries):
        """Test which_ppt outside any range."""
        plan, _, _ = plan_with_two_entries
        assert plan.which_ppt(300.0) is None

    def test_plan_extend_length(self, empty_plan):
        """Test extending Plan with list of PPTs (length)."""
        ppt1 = Mock(spec=PlanEntry)
        ppt2 = Mock(spec=PlanEntry)
        empty_plan.extend([ppt1, ppt2])
        assert len(empty_plan) == 2

    def test_plan_extend_first_item(self, empty_plan):
        """Test extending Plan with list of PPTs (first item)."""
        ppt1 = Mock(spec=PlanEntry)
        ppt2 = Mock(spec=PlanEntry)
        empty_plan.extend([ppt1, ppt2])
        assert empty_plan[0] == ppt1

    def test_plan_extend_second_item(self, empty_plan):
        """Test extending Plan with list of PPTs (second item)."""
        ppt1 = Mock(spec=PlanEntry)
        ppt2 = Mock(spec=PlanEntry)
        empty_plan.extend([ppt1, ppt2])
        assert empty_plan[1] == ppt2

    def test_plan_append_first_length(self, empty_plan):
        """Test appending first PPT to Plan (length)."""
        ppt1 = Mock(spec=PlanEntry)
        empty_plan.append(ppt1)
        assert len(empty_plan) == 1

    def test_plan_append_first_item(self, empty_plan):
        """Test appending first PPT to Plan (item)."""
        ppt1 = Mock(spec=PlanEntry)
        empty_plan.append(ppt1)
        assert empty_plan[0] == ppt1

    def test_plan_append_second_length(self, empty_plan):
        """Test appending second PPT to Plan (length)."""
        ppt1 = Mock(spec=PlanEntry)
        ppt2 = Mock(spec=PlanEntry)
        empty_plan.append(ppt1)
        empty_plan.append(ppt2)
        assert len(empty_plan) == 2

    def test_plan_append_second_item(self, empty_plan):
        """Test appending second PPT to Plan (item)."""
        ppt1 = Mock(spec=PlanEntry)
        ppt2 = Mock(spec=PlanEntry)
        empty_plan.append(ppt1)
        empty_plan.append(ppt2)
        assert empty_plan[1] == ppt2


class TestPlanSaveLoad:
    """Tests for Plan.save() and Plan.load()."""

    def test_save_to_explicit_path_returns_resolved_path(self, tmp_path):
        """Plan.save(file) returns the resolved Path of the written file."""
        plan = _make_plan(2)
        dest = tmp_path / "plan.json"
        result = plan.save(dest)
        assert isinstance(result, Path)
        assert result == dest.resolve()

    def test_save_creates_json_file(self, tmp_path):
        """Plan.save() writes a file that exists and is valid JSON."""
        plan = _make_plan(2)
        dest = tmp_path / "plan.json"
        plan.save(dest)
        assert dest.exists()
        raw = json.loads(dest.read_text())
        assert isinstance(raw, dict)

    def test_save_json_contains_expected_metadata(self, tmp_path):
        """Saved JSON contains version, start, end, num_entries, and entries keys."""
        plan = _make_plan(2)
        dest = tmp_path / "plan.json"
        plan.save(dest)
        raw = json.loads(dest.read_text())
        assert "version" in raw
        assert "coast_sim_version" in raw
        assert "created_at" in raw
        assert "start" in raw
        assert "end" in raw
        assert raw["num_entries"] == 2
        assert len(raw["entries"]) == 2

    def test_save_times_are_iso_strings(self, tmp_path):
        """start, end, and entry begin/end are serialised as ISO-8601 strings."""
        plan = _make_plan(1)
        dest = tmp_path / "plan.json"
        plan.save(dest)
        raw = json.loads(dest.read_text())
        assert isinstance(raw["start"], str) and "T" in raw["start"]
        assert isinstance(raw["end"], str) and "T" in raw["end"]
        entry = raw["entries"][0]
        assert isinstance(entry["begin"], str) and "T" in entry["begin"]
        assert isinstance(entry["end"], str) and "T" in entry["end"]

    def test_save_to_directory_autogenerates_filename(self, tmp_path):
        """Plan.save(directory) auto-generates a filename."""
        plan = _make_plan(1)
        result = plan.save(str(tmp_path) + "/")
        assert result.parent == tmp_path.resolve()
        assert result.suffix == ".json"
        assert result.name.startswith("plan_")
        assert result.exists()

    def test_save_to_directory_increments_version(self, tmp_path):
        """Saving to the same directory twice increments the version number."""
        plan = _make_plan(1)
        first = plan.save(tmp_path)
        second = plan.save(tmp_path)
        assert first != second
        assert "_v0." in first.name
        assert "_v1." in second.name

    def test_load_returns_plan_schema(self, tmp_path):
        """Plan.load() returns a PlanSchema, not a plain Plan."""
        from conops.targets.plan_schema import PlanSchema

        plan = _make_plan(2)
        dest = tmp_path / "plan.json"
        plan.save(dest)
        loaded = Plan.load(dest)
        assert isinstance(loaded, PlanSchema)

    def test_load_entry_count_matches(self, tmp_path):
        """Loaded PlanSchema has the same number of entries as saved."""
        plan = _make_plan(3)
        dest = tmp_path / "plan.json"
        plan.save(dest)
        loaded = Plan.load(dest)
        assert loaded.num_entries == 3
        assert len(loaded.entries) == 3

    def test_load_entry_fields_roundtrip(self, tmp_path):
        """Key scalar fields survive a save/load roundtrip."""

        plan = _make_plan(1)
        dest = tmp_path / "plan.json"
        plan.save(dest)
        loaded = Plan.load(dest)
        original_entry = plan.entries[0]
        loaded_entry = loaded.entries[0]
        assert loaded_entry.obsid == original_entry.obsid
        assert loaded_entry.ra == pytest.approx(original_entry.ra)
        assert loaded_entry.dec == pytest.approx(original_entry.dec)
        assert loaded_entry.begin == pytest.approx(original_entry.begin)
        assert loaded_entry.end == pytest.approx(original_entry.end)
        assert loaded_entry.obstype == original_entry.obstype

    def test_load_metadata_roundtrip(self, tmp_path):
        """start, end, and version survive a save/load roundtrip."""

        plan = _make_plan(2)
        dest = tmp_path / "plan.json"
        plan.save(dest)
        loaded = Plan.load(dest)
        assert loaded.start == pytest.approx(plan.entries[0].begin)
        assert loaded.end == pytest.approx(plan.entries[-1].end)
        assert isinstance(loaded.version, int)
