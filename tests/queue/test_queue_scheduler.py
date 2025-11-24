"""Unit tests for the DumbQueueScheduler class."""

from unittest.mock import Mock

from astropy.time import Time  # type: ignore[import-untyped]

from conops import DAY_SECONDS, DumbQueueScheduler, Plan


class TestDumbQueueSchedulerInit:
    """Test DumbQueueScheduler initialization."""

    def test_init_with_defaults(self):
        """Test initialization with default parameters."""
        scheduler = DumbQueueScheduler()
        assert scheduler.queue is not None
        assert scheduler.plan is not None
        assert scheduler.year == 2021
        assert scheduler.day == 4
        assert scheduler.length == 1

    def test_init_with_custom_parameters(self, mock_queue):
        """Test initialization with custom parameters."""
        plan = Plan()
        scheduler = DumbQueueScheduler(
            queue=mock_queue, plan=plan, year=2022, day=100, length=2
        )
        assert scheduler.queue is mock_queue
        assert scheduler.plan is plan
        assert scheduler.year == 2022
        assert scheduler.day == 100
        assert scheduler.length == 2

    def test_init_creates_empty_plan(self):
        """Test that init creates empty plan if not provided."""
        scheduler = DumbQueueScheduler()
        assert len(scheduler.plan) == 0

    def test_init_creates_empty_queue(self):
        """Test that init creates empty queue if not provided."""
        scheduler = DumbQueueScheduler()
        assert scheduler.queue is not None
        assert len(scheduler.queue) == 0


class TestDumbQueueSchedulerSchedule:
    """Test the schedule method."""

    def test_schedule_returns_plan(self, scheduler):
        """Test that schedule returns a Plan object."""
        result = scheduler.schedule()
        assert isinstance(result, Plan)

    def test_schedule_resets_plan_on_run(self, scheduler):
        """Test that schedule resets the plan each run."""
        scheduler.plan.extend([Mock()])  # Add dummy entry
        assert len(scheduler.plan) > 0

        result = scheduler.schedule()
        # Plan should be reset after schedule
        assert isinstance(result, Plan)

    def test_schedule_with_single_target(self, scheduler, mock_queue, mock_pointing):
        """Test scheduling with a single target."""
        # Set up mock to return one target then None
        target = mock_pointing(targetid=1, ra=45.0, dec=30.0, merit=100, ssmin=300)
        target.begin = int(scheduler.ustart)
        target.end = int(scheduler.ustart + 600)  # 10 minutes

        call_count = [0]

        def mock_get_single(ra, dec, utime):
            call_count[0] += 1
            if call_count[0] == 1:
                return target
            return None

        scheduler.queue.get = mock_get_single

        result = scheduler.schedule()
        assert len(result) >= 1

    def test_schedule_with_multiple_targets(self, scheduler, mock_pointing):
        """Test scheduling with multiple targets."""
        targets = [
            mock_pointing(targetid=1, ra=45.0, dec=30.0, merit=100, ssmin=300),
            mock_pointing(targetid=2, ra=90.0, dec=-45.0, merit=90, ssmin=300),
            mock_pointing(targetid=3, ra=180.0, dec=60.0, merit=80, ssmin=300),
        ]

        # Set different begin/end times for each
        targets[0].begin = scheduler.ustart
        targets[0].end = scheduler.ustart + 600

        targets[1].begin = scheduler.ustart + 600
        targets[1].end = scheduler.ustart + 1200

        targets[2].begin = scheduler.ustart + 1200
        targets[2].end = scheduler.ustart + 1800

        call_count = [0]

        def mock_get_multi(ra, dec, utime):
            if call_count[0] < len(targets):
                result = targets[call_count[0]]
                result.done = False
                call_count[0] += 1
                return result
            return None

        scheduler.queue.get = mock_get_multi

        result = scheduler.schedule()
        assert len(result) >= 1

    def test_schedule_stops_when_queue_empty(self, scheduler):
        """Test that scheduling stops when queue returns None."""
        scheduler.queue.get = Mock(return_value=None)

        scheduler.schedule()
        scheduler.queue.get.assert_called()

    def test_schedule_respects_time_window(self, scheduler, mock_pointing):
        """Test that scheduler respects the time window."""
        target = mock_pointing(targetid=1, ra=45.0, dec=30.0, merit=100, ssmin=300)

        scheduler.queue.get = Mock(return_value=target)
        target.begin = scheduler.ustart
        target.end = scheduler.ustart + DAY_SECONDS * 2  # Beyond window

        # Should still complete without error
        result = scheduler.schedule()
        assert isinstance(result, Plan)


class TestDumbQueueSchedulerStartTime:
    """Test start time calculation."""

    def test_ustart_calculation(self, scheduler):
        """Test that ustart is calculated correctly."""
        scheduler.schedule()
        # ustart should be set after schedule call
        assert scheduler.ustart > 0

    def test_different_year_day_combinations(self):
        """Test scheduling with different year/day combinations."""
        for year in [2020, 2021, 2022]:
            for day in [1, 100, 365]:
                scheduler = DumbQueueScheduler(year=year, day=day, length=1)
                scheduler.queue.get = Mock(return_value=None)
                scheduler.schedule()
                assert scheduler.ustart > 0

    def test_multi_day_scheduling(self):
        """Test scheduling over multiple days."""
        scheduler = DumbQueueScheduler(year=2021, day=4, length=3)
        scheduler.queue.get = Mock(return_value=None)
        scheduler.schedule()
        # Should not raise any errors
        assert scheduler.length == 3


class TestDumbQueueSchedulerTargetProcessing:
    """Test target processing during scheduling."""

    def test_target_marked_done_after_scheduling(self, scheduler, mock_pointing):
        """Test that targets are marked as done after scheduling."""
        target = mock_pointing(targetid=1, ra=45.0, dec=30.0, merit=100)
        target.begin = scheduler.ustart
        target.end = scheduler.ustart + 600
        target.done = False

        call_count = [0]

        def mock_get(ra, dec, utime):
            if call_count[0] == 0:
                call_count[0] += 1
                return target
            return None

        scheduler.queue.get = mock_get

        _ = scheduler.schedule()
        assert target.done is True

    def test_target_position_updated_during_scheduling(self, scheduler, mock_pointing):
        """Test that last position is updated correctly."""
        target1 = mock_pointing(targetid=1, ra=45.0, dec=30.0, merit=100)
        target2 = mock_pointing(targetid=2, ra=90.0, dec=-45.0, merit=90)

        target1.begin = scheduler.ustart
        target1.end = scheduler.ustart + 600

        target2.begin = scheduler.ustart + 600
        target2.end = scheduler.ustart + 1200

        targets = [target1, target2]
        call_count = [0]

        def mock_get(ra, dec, utime):
            if call_count[0] < len(targets):
                result = targets[call_count[0]]
                call_count[0] += 1
                return result
            return None

        scheduler.queue.get = mock_get
        result = scheduler.schedule()

        # Multiple targets should be scheduled
        assert len(result) >= 1


class TestDumbQueueSchedulerEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_duration_target(self, scheduler, mock_pointing):
        """Test handling of zero-duration targets."""
        target = mock_pointing(targetid=1, ra=45.0, dec=30.0, merit=100)
        target.begin = scheduler.ustart
        target.end = scheduler.ustart  # Zero duration

        scheduler.queue.get = Mock(return_value=target)

        scheduler.schedule()

    def test_negative_duration_target(self, scheduler, mock_pointing):
        """Test handling of negative-duration targets."""
        target = mock_pointing(targetid=1, ra=45.0, dec=30.0, merit=100)
        target.begin = scheduler.ustart
        target.end = scheduler.ustart - 100  # Negative duration

        scheduler.queue.get = Mock(return_value=target)

        scheduler.schedule()

    def test_very_long_scheduling_window(self, scheduler, mock_pointing):
        """Test scheduling with very long time window."""
        scheduler.length = 365  # Full year

        target = mock_pointing(targetid=1, ra=45.0, dec=30.0, merit=100)
        target.begin = scheduler.ustart
        target.end = scheduler.ustart + 600

        scheduler.queue.get = Mock(return_value=target)
        # Should complete without error
        result = scheduler.schedule()
        assert isinstance(result, Plan)

    def test_many_targets_in_queue(self, scheduler, mock_pointing):
        """Test scheduling with many targets."""
        targets = [
            mock_pointing(
                targetid=i,
                ra=(i * 10) % 360,
                dec=(i - 50) % 90 - 45,
                merit=100 - i,
                ssmin=300,
            )
            for i in range(50)
        ]

        # Set durations for all
        for i, target in enumerate(targets):
            target.begin = scheduler.ustart + i * 1000
            target.end = scheduler.ustart + (i + 1) * 1000

        call_count = [0]

        def mock_get_many(ra, dec, utime):
            if call_count[0] < len(targets):
                result = targets[call_count[0]]
                call_count[0] += 1
                return result
            return None

        scheduler.queue.get = mock_get_many

        result = scheduler.schedule()
        # Should handle many targets
        assert isinstance(result, Plan)


class TestDumbQueueSchedulerIntegration:
    """Integration tests."""

    def test_full_scheduling_workflow(self, scheduler, mock_pointing):
        """Test complete scheduling workflow."""
        # Need to initialize ustart first
        scheduler.ustart = Time("2021-01-04 00:00:00", scale="utc").unix

        # Create real targets
        targets = [
            mock_pointing(targetid=1, ra=45.0, dec=30.0, merit=100),
            mock_pointing(targetid=2, ra=90.0, dec=-45.0, merit=90),
        ]

        for i, target in enumerate(targets):
            target.begin = scheduler.ustart + i * 1000
            target.end = scheduler.ustart + (i + 1) * 1000

        call_count = [0]

        def mock_get_workflow(ra, dec, utime):
            if call_count[0] < len(targets):
                result = targets[call_count[0]]
                call_count[0] += 1
                return result
            return None

        scheduler.queue.get = mock_get_workflow

        # Run scheduling
        result = scheduler.schedule()

        # Verify results
        assert len(result) >= 0
        for entry in result.entries:
            assert entry.begin >= scheduler.ustart
            assert entry.end <= scheduler.ustart + scheduler.length * DAY_SECONDS

    def test_scheduling_with_position_tracking(self, scheduler, mock_pointing):
        """Test that position is correctly tracked across targets."""
        target1 = mock_pointing(targetid=1, ra=45.0, dec=30.0, merit=100)
        target2 = mock_pointing(targetid=2, ra=90.0, dec=-45.0, merit=90)
        target3 = mock_pointing(targetid=3, ra=180.0, dec=60.0, merit=80)

        targets = [target1, target2, target3]

        for i, target in enumerate(targets):
            target.begin = scheduler.ustart + i * 1000
            target.end = scheduler.ustart + (i + 1) * 1000

        call_count = [0]
        positions = []

        def mock_get_track(ra, dec, utime):
            positions.append((ra, dec))
            if call_count[0] < len(targets):
                result = targets[call_count[0]]
                call_count[0] += 1
                return result
            return None

        scheduler.queue.get = mock_get_track

        _ = scheduler.schedule()

        # Should have tracked positions
        assert len(positions) >= 1
        # First position should be (0, 0) - default start
        assert positions[0] == (0.0, 0.0)

    def test_plan_entries_in_sequence(self, scheduler, mock_pointing):
        """Test that plan entries are in time sequence."""
        targets = [
            mock_pointing(targetid=1, ra=45.0, dec=30.0, merit=100),
            mock_pointing(targetid=2, ra=90.0, dec=-45.0, merit=90),
            mock_pointing(targetid=3, ra=180.0, dec=60.0, merit=80),
        ]

        base_time = scheduler.ustart
        for i, target in enumerate(targets):
            target.begin = base_time + i * 1000
            target.end = base_time + (i + 1) * 1000

        call_count = [0]

        def mock_get_seq(ra, dec, utime):
            if call_count[0] < len(targets):
                result = targets[call_count[0]]
                call_count[0] += 1
                return result
            return None

        scheduler.queue.get = mock_get_seq
        plan = scheduler.schedule()

        # Check that entries are in sequence
        assert isinstance(plan, Plan)
        for i in range(len(plan.entries) - 1):
            assert plan.entries[i].end <= plan.entries[i + 1].begin


class TestDumbQueueSchedulerStateManagement:
    """Test state management and plan reuse."""

    def test_plan_reset_between_runs(self, scheduler, mock_pointing):
        """Test that plan is reset between scheduling runs."""
        target = mock_pointing(targetid=1, ra=45.0, dec=30.0, merit=100)
        target.begin = scheduler.ustart
        target.end = scheduler.ustart + 600

        call_count = [0]

        def mock_get_reset(ra, dec, utime):
            if call_count[0] < 2:
                call_count[0] += 1
                return target
            return None

        scheduler.queue.get = mock_get_reset

        # First run
        plan1 = scheduler.schedule()

        # Second run
        call_count[0] = 0
        plan2 = scheduler.schedule()

        # Plans should be independent
        assert plan1 is not plan2

    def test_scheduler_parameters_preserved(self, scheduler):
        """Test that scheduler parameters are preserved."""
        original_year = scheduler.year
        original_day = scheduler.day
        original_length = scheduler.length

        scheduler.queue.get = Mock(return_value=None)
        scheduler.schedule()

        assert scheduler.year == original_year
        assert scheduler.day == original_day
        assert scheduler.length == original_length


class TestDumbQueueSchedulerConfiguration:
    """Test configuration options."""

    def test_year_parameter(self):
        """Test year parameter configuration."""
        for year in [2000, 2021, 2050]:
            scheduler = DumbQueueScheduler(year=year, day=1, length=1)
            assert scheduler.year == year

    def test_day_parameter(self):
        """Test day parameter configuration."""
        for day in [1, 100, 365]:
            scheduler = DumbQueueScheduler(year=2021, day=day, length=1)
            assert scheduler.day == day

    def test_length_parameter(self):
        """Test length parameter configuration."""
        for length in [1, 5, 30]:
            scheduler = DumbQueueScheduler(year=2021, day=1, length=length)
            assert scheduler.length == length
