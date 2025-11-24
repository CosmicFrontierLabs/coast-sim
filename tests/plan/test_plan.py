"""Tests for conops.plan module."""

from unittest.mock import Mock

from conops import PlanEntry


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
