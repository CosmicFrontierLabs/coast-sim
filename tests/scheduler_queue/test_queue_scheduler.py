"""QueueScheduler: plan building, reruns, and its DumbQueueScheduler shim."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest

from conops import DumbQueueScheduler, QueueScheduler

BEGIN = datetime(2021, 1, 4, tzinfo=timezone.utc)


@pytest.fixture
def scheduler(mock_queue: Mock) -> QueueScheduler:
    return QueueScheduler(mock_queue, BEGIN, BEGIN + timedelta(hours=1))


def test_plans_queue_selections_in_order(
    scheduler, make_targets, queue_get_from_list
) -> None:
    targets = make_targets(count=3)
    for k, target in enumerate(targets):
        target.begin = scheduler.ustart + 600 * k
        target.end = target.begin + 600
    queue_get_from_list(scheduler, targets)

    plan = scheduler.schedule()

    assert plan is scheduler.plan
    assert [e.obsid for e in plan] == [t.obsid for t in targets]
    assert all(t.done for t in targets)
    assert plan.metadata == {"scheduler": "queue"}


def test_rerun_starts_fresh(scheduler, make_target, queue_get_from_list) -> None:
    target = make_target(targetid=1)
    target.begin, target.end = scheduler.ustart, scheduler.ustart + 600
    queue_get_from_list(scheduler, [target])
    first = scheduler.schedule()

    scheduler.queue.get = Mock(return_value=None)
    second = scheduler.schedule()

    assert second is not first
    assert len(second) == 0


def test_next_selection_starts_from_previous_attitude(
    scheduler, make_targets, queue_get_from_list
) -> None:
    targets = make_targets(count=2)
    for k, target in enumerate(targets):
        target.begin = scheduler.ustart + 600 * k
        target.end = target.begin + 600
        target.spacecraft_attitude = None
    positions = queue_get_from_list(scheduler, targets, track_positions=True)

    scheduler.schedule()

    assert positions[:2] == [(0.0, 0.0), (targets[0].ra, targets[0].dec)]


class TestDumbQueueSchedulerShim:
    def test_is_a_queue_scheduler(self) -> None:
        assert issubclass(DumbQueueScheduler, QueueScheduler)

    def test_horizon_may_be_set_later(self, mock_queue: Mock) -> None:
        legacy = DumbQueueScheduler(queue=mock_queue)
        legacy.queue.get = Mock(return_value=None)
        with pytest.raises(AssertionError, match="Begin and end"):
            legacy.schedule()

        legacy.begin, legacy.end = BEGIN, BEGIN + timedelta(hours=1)
        assert len(legacy.schedule()) == 0
