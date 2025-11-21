"""Tests for conops.ppst module."""

from unittest.mock import Mock

from conops.plan_entry import PlanEntry
from conops.ppst import Plan, TargetList


class TestTargetList:
    """Test TargetList class."""

    def test_target_list_initialization(self):
        """Test that TargetList initializes with empty list."""
        tl = TargetList()
        assert len(tl) == 0
        assert tl.targets == []

    def test_target_list_add_target(self):
        """Test adding targets to TargetList."""
        tl = TargetList()
        target1 = Mock(spec=PlanEntry)
        target2 = Mock(spec=PlanEntry)

        tl.add_target(target1)
        assert len(tl) == 1
        assert tl[0] == target1

        tl.add_target(target2)
        assert len(tl) == 2
        assert tl[1] == target2

    def test_target_list_getitem(self):
        """Test getting items from TargetList by index."""
        tl = TargetList()
        target1 = Mock(spec=PlanEntry)
        target2 = Mock(spec=PlanEntry)
        tl.targets = [target1, target2]

        assert tl[0] == target1
        assert tl[1] == target2


class TestPlan:
    """Test Plan class."""

    def test_plan_initialization(self):
        """Test that Plan initializes with empty list."""
        plan = Plan()
        assert len(plan) == 0
        assert plan.entries == []

    def test_plan_getitem(self):
        """Test getting items from Plan by index."""
        plan = Plan()
        ppt1 = Mock(spec=PlanEntry)
        ppt2 = Mock(spec=PlanEntry)
        plan.entries = [ppt1, ppt2]

        assert plan[0] == ppt1
        assert plan[1] == ppt2

    def test_plan_which_ppt_finds_current_pointing(self):
        """Test which_ppt finds the current pointing at given time."""
        plan = Plan()

        ppt1 = Mock(spec=PlanEntry)
        ppt1.begin = 0.0
        ppt1.end = 100.0

        ppt2 = Mock(spec=PlanEntry)
        ppt2.begin = 100.0
        ppt2.end = 200.0

        plan.entries = [ppt1, ppt2]

        # Test finding ppt1
        assert plan.which_ppt(50.0) == ppt1
        # Test finding ppt2
        assert plan.which_ppt(150.0) == ppt2
        # Test at boundary (should not match end boundary)
        assert plan.which_ppt(100.0) == ppt2
        # Test outside any range
        assert plan.which_ppt(300.0) is None

    def test_plan_extend(self):
        """Test extending Plan with list of PPTs."""
        plan = Plan()
        ppt1 = Mock(spec=PlanEntry)
        ppt2 = Mock(spec=PlanEntry)

        plan.extend([ppt1, ppt2])
        assert len(plan) == 2
        assert plan[0] == ppt1
        assert plan[1] == ppt2

    def test_plan_append(self):
        """Test appending single PPT to Plan."""
        plan = Plan()
        ppt1 = Mock(spec=PlanEntry)
        ppt2 = Mock(spec=PlanEntry)

        plan.append(ppt1)
        assert len(plan) == 1
        assert plan[0] == ppt1

        plan.append(ppt2)
        assert len(plan) == 2
        assert plan[1] == ppt2
