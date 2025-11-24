"""Test fixtures for plan subsystem tests."""

from unittest.mock import Mock

import pytest

from conops import Plan, PlanEntry, TargetList


@pytest.fixture
def empty_target_list():
    """Fixture for an empty TargetList."""
    return TargetList()


@pytest.fixture
def target_list_with_one():
    """Fixture for TargetList with one target."""
    tl = TargetList()
    target1 = Mock(spec=PlanEntry)
    tl.add_target(target1)
    return tl, target1


@pytest.fixture
def target_list_with_two():
    """Fixture for TargetList with two targets."""
    tl = TargetList()
    target1 = Mock(spec=PlanEntry)
    target2 = Mock(spec=PlanEntry)
    tl.add_target(target1)
    tl.add_target(target2)
    return tl, target1, target2


@pytest.fixture
def empty_plan():
    """Fixture for an empty Plan."""
    return Plan()


@pytest.fixture
def plan_with_two_entries():
    """Fixture for Plan with two entries."""
    plan = Plan()
    ppt1 = Mock(spec=PlanEntry)
    ppt1.begin = 0.0
    ppt1.end = 100.0
    ppt2 = Mock(spec=PlanEntry)
    ppt2.begin = 100.0
    ppt2.end = 200.0
    plan.entries = [ppt1, ppt2]
    return plan, ppt1, ppt2
