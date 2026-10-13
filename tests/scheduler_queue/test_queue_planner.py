"""QueuePlanner: plan building, reruns, and its DumbQueueScheduler shim."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from conops import DumbQueueScheduler, QueuePlanner

BEGIN = datetime(2021, 1, 4, tzinfo=timezone.utc)


@pytest.fixture
def planner(mock_queue: Mock) -> QueuePlanner:
    return QueuePlanner(mock_queue, BEGIN, BEGIN + timedelta(hours=1))


def test_plans_queue_selections_in_order(
    planner, make_targets, queue_get_from_list
) -> None:
    targets = make_targets(count=3)
    for k, target in enumerate(targets):
        target.begin = planner.ustart + 600 * k
        target.end = target.begin + 600
    queue_get_from_list(planner, targets)

    plan = planner.schedule()

    assert plan is planner.plan
    assert [e.obsid for e in plan] == [t.obsid for t in targets]
    assert all(t.done for t in targets)
    assert plan.metadata == {"planner": "queue"}


def test_rerun_starts_fresh(planner, make_target, queue_get_from_list) -> None:
    target = make_target(targetid=1)
    target.begin, target.end = planner.ustart, planner.ustart + 600
    queue_get_from_list(planner, [target])
    first = planner.schedule()

    planner.queue.get = Mock(return_value=None)
    second = planner.schedule()

    assert second is not first
    assert len(second) == 0


def test_next_selection_starts_from_previous_attitude(
    planner, make_targets, queue_get_from_list
) -> None:
    targets = make_targets(count=2)
    for k, target in enumerate(targets):
        target.begin = planner.ustart + 600 * k
        target.end = target.begin + 600
        target.spacecraft_attitude = None
    positions = queue_get_from_list(planner, targets, track_positions=True)

    planner.schedule()

    assert positions[:2] == [(0.0, 0.0), (targets[0].ra, targets[0].dec)]


class TestDumbQueueSchedulerShim:
    def test_is_a_queue_planner(self) -> None:
        assert issubclass(DumbQueueScheduler, QueuePlanner)

    def test_horizon_may_be_set_later(self, mock_queue: Mock) -> None:
        legacy = DumbQueueScheduler(queue=mock_queue)
        legacy.queue.get = Mock(return_value=None)
        with pytest.raises(AssertionError, match="Begin and end"):
            legacy.schedule()

        legacy.begin, legacy.end = BEGIN, BEGIN + timedelta(hours=1)
        assert len(legacy.schedule()) == 0
