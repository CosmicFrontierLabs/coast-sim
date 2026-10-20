"""Unit tests for the DumbQueueScheduler class."""

from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import pytest  # type: ignore[import-untyped]
from astropy.time import Time  # type: ignore[import-untyped]

from conops import DAY_SECONDS, DumbQueueScheduler, Plan


class TestDumbQueueSchedulerInit:
    """Test DumbQueueScheduler initialization."""

    def test_init_default_queue_not_none(self, mock_queue):
        scheduler = DumbQueueScheduler(queue=mock_queue)
        assert scheduler.queue is not None

    def test_init_default_plan_not_none(self, mock_queue):
        scheduler = DumbQueueScheduler(queue=mock_queue)
        assert scheduler.plan is not None

    def test_init_default_begin_value(self, mock_queue):
        scheduler = DumbQueueScheduler(queue=mock_queue)
        assert scheduler.begin is None

    def test_init_default_end_value(self, mock_queue):
        scheduler = DumbQueueScheduler(queue=mock_queue)
        assert scheduler.end is None

    def test_init_with_custom_parameter_queue(
        self, mock_queue, scheduler_2022_100_len2
    ):
        assert scheduler_2022_100_len2.queue is mock_queue

    def test_init_with_custom_parameter_plan(self, mock_queue):
        plan = Plan()
        scheduler = DumbQueueScheduler(
            queue=mock_queue,
            plan=plan,
            begin=datetime(2022, 4, 10),
            end=datetime(2022, 4, 11),
        )
        assert scheduler.plan is plan

    def test_init_with_custom_parameter_begin(self, mock_queue):
        plan = Plan()
        begin = datetime(2022, 4, 10, tzinfo=timezone.utc)
        end = begin + timedelta(days=2)
        scheduler = DumbQueueScheduler(
            queue=mock_queue, plan=plan, begin=begin, end=end
        )
        assert scheduler.begin == begin

    def test_init_with_custom_parameter_end(self, mock_queue):
        plan = Plan()
        begin = datetime(2022, 4, 10, tzinfo=timezone.utc)
        end = begin + timedelta(days=2)
        scheduler = DumbQueueScheduler(
            queue=mock_queue, plan=plan, begin=begin, end=end
        )
        assert scheduler.end == end

    def test_init_creates_empty_plan(self):
        scheduler = DumbQueueScheduler(queue=Mock())
        assert len(scheduler.plan) == 0

    def test_init_creates_empty_queue_not_none(self):
        scheduler = DumbQueueScheduler(queue=Mock())
        assert scheduler.queue is not None


class TestDumbQueueSchedulerSchedule:
    """Test the schedule method."""

    def test_schedule_returns_plan(self, scheduler):
        result = scheduler.schedule()
        assert isinstance(result, Plan)

    def test_schedule_precondition_plan_nonempty(self, scheduler):
        scheduler.plan.extend([Mock()])  # Add dummy entry
        assert len(scheduler.plan) > 0

    def test_schedule_returns_plan_when_prepopulated(self, scheduler):
        scheduler.plan.extend([Mock()])  # Add dummy entry
        result = scheduler.schedule()
        assert isinstance(result, Plan)

    def test_schedule_with_single_target_returns_at_least_one_entry(
        self, scheduler, make_target, queue_get_from_list
    ):
        target = make_target(targetid=1, ra=45.0, dec=30.0, merit=100, ss_min=300)
        target.begin = int(scheduler.ustart)
        target.end = int(scheduler.ustart + 600)  # 10 minutes

        queue_get_from_list(scheduler, [target])

        result = scheduler.schedule()
        assert len(result) >= 1

    def test_schedule_with_multiple_targets_returns_at_least_one_entry(
        self, scheduler, make_targets, queue_get_from_list
    ):
        targets = make_targets(count=3)
        targets[0].begin = scheduler.ustart
        targets[0].end = scheduler.ustart + 600

        targets[1].begin = scheduler.ustart + 600
        targets[1].end = scheduler.ustart + 1200

        targets[2].begin = scheduler.ustart + 1200
        targets[2].end = scheduler.ustart + 1800

        queue_get_from_list(scheduler, targets)

        result = scheduler.schedule()
        assert len(result) >= 1

    def test_schedule_stops_when_queue_empty(self, scheduler):
        scheduler.queue.get = Mock(return_value=None)

        scheduler.schedule()
        scheduler.queue.get.assert_called()

    def test_schedule_respects_time_window(
        self, scheduler, make_target, queue_get_from_list
    ):
        target = make_target(targetid=1, ra=45.0, dec=30.0, merit=100, ss_min=300)

        queue_get_from_list(scheduler, [target])
        target.begin = scheduler.ustart
        target.end = scheduler.ustart + DAY_SECONDS * 2  # Beyond window

        result = scheduler.schedule()
        assert isinstance(result, Plan)


class TestDumbQueueSchedulerStartTime:
    """Test start time calculation."""

    def test_ustart_calculation(self, scheduler):
        scheduler.schedule()
        assert scheduler.ustart > 0

    @pytest.mark.parametrize(
        "begin_date",
        [
            datetime(2020, 1, 1, tzinfo=timezone.utc),
            datetime(2021, 4, 10, tzinfo=timezone.utc),
            datetime(2022, 12, 31, tzinfo=timezone.utc),
        ],
    )
    def test_different_begin_dates_set_ustart(self, begin_date):
        end = begin_date + timedelta(days=1)
        scheduler = DumbQueueScheduler(queue=Mock(), begin=begin_date, end=end)
        scheduler.queue.get = Mock(return_value=None)
        scheduler.schedule()
        assert scheduler.ustart > 0

    def test_multi_day_scheduling_end_preserved(self):
        begin = datetime(2021, 1, 4, tzinfo=timezone.utc)
        end = begin + timedelta(days=3)
        scheduler = DumbQueueScheduler(queue=Mock(), begin=begin, end=end)
        scheduler.queue.get = Mock(return_value=None)
        scheduler.schedule()
        assert scheduler.end == end


class TestDumbQueueSchedulerTargetProcessing:
    """Test target processing during scheduling."""

    def test_target_marked_done_after_scheduling(
        self, scheduler, make_target, queue_get_from_list
    ):
        target = make_target(targetid=1, ra=45.0, dec=30.0, merit=100)
        target.begin = scheduler.ustart
        target.end = scheduler.ustart + 600
        target.done = False

        queue_get_from_list(scheduler, [target])

        _ = scheduler.schedule()
        assert target.done is True

    def test_target_position_updated_during_scheduling_adds_entries(
        self, scheduler, make_targets, queue_get_from_list
    ):
        targets = make_targets(count=2)

        targets[0].begin = scheduler.ustart
        targets[0].end = scheduler.ustart + 600

        targets[1].begin = scheduler.ustart + 600
        targets[1].end = scheduler.ustart + 1200

        queue_get_from_list(scheduler, targets)
        result = scheduler.schedule()

        assert len(result) >= 1


class TestDumbQueueSchedulerEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_duration_target_returns_plan(
        self, scheduler, make_target, queue_get_from_list
    ):
        target = make_target(targetid=1, ra=45.0, dec=30.0, merit=100)
        target.begin = scheduler.ustart
        target.end = scheduler.ustart  # Zero duration

        queue_get_from_list(scheduler, [target])

        result = scheduler.schedule()
        assert isinstance(result, Plan)

    def test_negative_duration_target_returns_plan(
        self, scheduler, make_target, queue_get_from_list
    ):
        target = make_target(targetid=1, ra=45.0, dec=30.0, merit=100)
        target.begin = scheduler.ustart
        target.end = scheduler.ustart - 100  # Negative duration

        queue_get_from_list(scheduler, [target])

        result = scheduler.schedule()
        assert isinstance(result, Plan)

    def test_very_long_scheduling_window(
        self, scheduler, make_target, queue_get_from_list
    ):
        # Extend the scheduling window to a full year
        scheduler.end = scheduler.begin + timedelta(days=365)

        target = make_target(targetid=1, ra=45.0, dec=30.0, merit=100)
        target.begin = scheduler.ustart
        target.end = scheduler.ustart + 600

        queue_get_from_list(scheduler, [target])
        result = scheduler.schedule()
        assert isinstance(result, Plan)

    def test_many_targets_in_queue_returns_plan(
        self, scheduler, make_targets, queue_get_from_list
    ):
        targets = make_targets(count=50)

        for i, target in enumerate(targets):
            target.begin = scheduler.ustart + i * 1000
            target.end = scheduler.ustart + (i + 1) * 1000

        queue_get_from_list(scheduler, targets)
        result = scheduler.schedule()
        assert isinstance(result, Plan)


class TestDumbQueueSchedulerIntegration:
    """Integration tests."""

    def test_full_scheduling_workflow_returns_plan(
        self, scheduler, make_targets, queue_get_from_list
    ):
        scheduler.ustart = Time("2021-01-04 00:00:00", scale="utc").unix

        targets = make_targets(count=2)

        for i, target in enumerate(targets):
            target.begin = scheduler.ustart + i * 1000
            target.end = scheduler.ustart + (i + 1) * 1000

        queue_get_from_list(scheduler, targets)
        result = scheduler.schedule()
        assert isinstance(result, Plan)

    def test_full_scheduling_workflow_all_entries_begin_after_ustart(
        self, scheduler, make_targets, queue_get_from_list
    ):
        scheduler.ustart = Time("2021-01-04 00:00:00", scale="utc").unix

        targets = make_targets(count=2)

        for i, target in enumerate(targets):
            target.begin = scheduler.ustart + i * 1000
            target.end = scheduler.ustart + (i + 1) * 1000

        queue_get_from_list(scheduler, targets)
        result = scheduler.schedule()
        assert all(entry.begin >= scheduler.ustart for entry in result.entries)

    def test_full_scheduling_workflow_all_entries_end_within_window(
        self, scheduler, make_targets, queue_get_from_list
    ):
        scheduler.ustart = Time("2021-01-04 00:00:00", scale="utc").unix

        targets = make_targets(count=2)

        for i, target in enumerate(targets):
            target.begin = scheduler.ustart + i * 1000
            target.end = scheduler.ustart + (i + 1) * 1000

        queue_get_from_list(scheduler, targets)
        result = scheduler.schedule()
        window_end = scheduler.end.timestamp()
        assert all(entry.end <= window_end for entry in result.entries)

    def test_scheduling_with_position_tracking_records_positions(
        self, scheduler, make_targets, queue_get_from_list
    ):
        targets = make_targets(count=3)

        for i, target in enumerate(targets):
            target.begin = scheduler.ustart + i * 1000
            target.end = scheduler.ustart + (i + 1) * 1000

        positions = queue_get_from_list(scheduler, targets, track_positions=True)
        _ = scheduler.schedule()
        assert len(positions) >= 1

    def test_scheduling_with_position_tracking_initial_position(
        self, scheduler, make_targets, queue_get_from_list
    ):
        targets = make_targets(count=3)

        for i, target in enumerate(targets):
            target.begin = scheduler.ustart + i * 1000
            target.end = scheduler.ustart + (i + 1) * 1000

        positions = queue_get_from_list(scheduler, targets, track_positions=True)
        _ = scheduler.schedule()
        assert positions[0] == (0.0, 0.0)

    def test_plan_entries_in_sequence_type(
        self, scheduler, make_targets, queue_get_from_list
    ):
        targets = make_targets(count=3)

        base_time = scheduler.ustart
        for i, target in enumerate(targets):
            target.begin = base_time + i * 1000
            target.end = base_time + (i + 1) * 1000

        queue_get_from_list(scheduler, targets)
        plan = scheduler.schedule()

        assert isinstance(plan, Plan)

    def test_plan_entries_in_time_sequence(
        self, scheduler, make_targets, queue_get_from_list
    ):
        targets = make_targets(count=3)

        base_time = scheduler.ustart
        for i, target in enumerate(targets):
            target.begin = base_time + i * 1000
            target.end = base_time + (i + 1) * 1000

        queue_get_from_list(scheduler, targets)
        plan = scheduler.schedule()

        assert all(
            plan.entries[i].end <= plan.entries[i + 1].begin
            for i in range(max(0, len(plan.entries) - 1))
        )


class TestDumbQueueSchedulerStateManagement:
    """Test state management and plan reuse."""

    def test_plan_reset_between_runs_independent_plans(
        self, scheduler, make_target, queue_get_from_list
    ):
        target = make_target(targetid=1, ra=45.0, dec=30.0, merit=100)
        target.begin = scheduler.ustart
        target.end = scheduler.ustart + 600

        queue_get_from_list(scheduler, [target])
        plan1 = scheduler.schedule()

        # Ensure a fresh plan will be returned on a new run
        # Reuse the queue behavior by re-attaching the same target
        plan2_targets = [target]
        queue_get_from_list(scheduler, plan2_targets)
        plan2 = scheduler.schedule()

        assert plan1 is not plan2

    def test_scheduler_parameters_preserved_begin(self, scheduler):
        original_begin = scheduler.begin
        scheduler.queue.get = Mock(return_value=None)
        scheduler.schedule()
        assert scheduler.begin == original_begin

    def test_scheduler_parameters_preserved_end(self, scheduler):
        original_end = scheduler.end
        scheduler.queue.get = Mock(return_value=None)
        scheduler.schedule()
        assert scheduler.end == original_end


class TestDumbQueueSchedulerConfiguration:
    """Test configuration options."""

    @pytest.mark.parametrize("begin_year", [2000, 2021, 2050])
    def test_begin_parameter(self, begin_year):
        begin = datetime(begin_year, 1, 1, tzinfo=timezone.utc)
        end = begin + timedelta(days=1)
        scheduler = DumbQueueScheduler(queue=Mock(), begin=begin, end=end)
        assert scheduler.begin == begin

    @pytest.mark.parametrize("end_year", [2000, 2021, 2050])
    def test_end_parameter(self, end_year):
        begin = datetime(2021, 1, 1, tzinfo=timezone.utc)
        end = datetime(end_year, 1, 2, tzinfo=timezone.utc)
        scheduler = DumbQueueScheduler(queue=Mock(), begin=begin, end=end)
        assert scheduler.end == end
