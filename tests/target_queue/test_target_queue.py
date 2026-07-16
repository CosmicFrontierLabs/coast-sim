from unittest.mock import Mock, patch

from conops import Queue, TargetSlewEstimate


class TestQueueInitAndAppend:
    def test_queue_init_targets_empty(self, mock_config) -> None:
        queue = Queue(config=mock_config)
        assert queue.targets == []

    def test_queue_init_ephem_none(self, mock_config) -> None:
        queue = Queue(config=mock_config)
        assert queue.ephem is None

    def test_queue_init_utime_none(self, mock_config) -> None:
        queue = Queue(config=mock_config)
        assert queue.utime == 0.0

    def test_queue_append_len(self, mock_target, mock_config) -> None:
        queue = Queue(config=mock_config)
        queue.append(mock_target)
        assert len(queue.targets) == 1

    def test_queue_append_target_equals(self, mock_target, mock_config) -> None:
        queue = Queue(config=mock_config)
        queue.append(mock_target)
        assert queue.targets[0] == mock_target


class TestQueueBasicOps:
    def test_queue_len(self, queue_instance) -> None:
        """Test the length of the queue."""
        assert len(queue_instance) == 5

    def test_queue_getitem(self, queue_instance, mock_targets) -> None:
        """Test getting an item from the queue."""
        assert queue_instance[2] == mock_targets[2]

    def test_queue_reset_reset_called(self, queue_instance) -> None:
        """Test resetting the queue calls reset() on each target."""
        for target in queue_instance.targets:
            target.done = True

        queue_instance.reset()

        for target in queue_instance.targets:
            target.reset.assert_called_once()

    def test_queue_reset_targets_done_false(self, queue_instance) -> None:
        """Test resetting the queue clears done flag on each target."""
        for target in queue_instance.targets:
            target.done = True

        queue_instance.reset()

        for target in queue_instance.targets:
            assert not target.done


class TestMeritsort:
    def test_meritsort_invisible_target_merit(self, queue_instance) -> None:
        """The invisible target should get the hidden-target merit."""
        invisible_target = queue_instance.targets[1]
        invisible_target.visible.return_value = False

        queue_instance.meritsort()

        assert invisible_target.merit == -900

    def test_meritsort_sorted_descending(self, queue_instance):
        """Check that the targets are sorted by merit descending."""
        queue_instance.meritsort()
        for i in range(len(queue_instance.targets) - 1):
            assert (
                queue_instance.targets[i].merit >= queue_instance.targets[i + 1].merit
            )

    def test_meritsort_invisible_target_last(self, queue_instance):
        """Invisible target should be last after sort (lowest merit)."""
        invisible_target = queue_instance.targets[1]
        invisible_target.visible.return_value = False

        queue_instance.meritsort()

        assert queue_instance.targets[-1] is invisible_target

    def test_meritsort_uses_deterministic_equal_merit_tie_break(self, queue_instance):
        """Equal-merit targets should not depend on original queue order."""
        for index, target in enumerate(queue_instance.targets):
            target.fom = 100
            target.obsid = 1000 + index

        queue_instance.meritsort()
        expected_obsids = [target.obsid for target in queue_instance.targets]

        queue_instance.targets.reverse()
        queue_instance.meritsort()

        assert [target.obsid for target in queue_instance.targets] == expected_obsids


class TestGetTarget:
    def test_get_target_calls_meritsort(self, queue_instance):
        """Test that meritsort is called with provided coordinates."""
        utime = 1762924800.0
        with patch.object(queue_instance, "meritsort") as mock_meritsort:
            _ = queue_instance.get(ra=0, dec=0, utime=utime)
            mock_meritsort.assert_called_once_with()

    def test_get_target_returns_not_none(self, queue_instance):
        """Test that get returns a target when available."""
        utime = 1762924800.0
        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(ra=0, dec=0, utime=utime)
        assert target is not None

    def test_get_target_returns_first_target(self, queue_instance):
        """Test that get returns the first target in the queue."""
        utime = 1762924800.0
        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(ra=0, dec=0, utime=utime)
        assert target == queue_instance.targets[0]

    def test_get_target_calc_slewtime_called(self, queue_instance):
        """Test that calc_slewtime is called on the returned target."""
        utime = 1762924800.0
        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(ra=0, dec=0, utime=utime)
        target.calc_slewtime.assert_called_once_with(0, 0)

    def test_get_target_begin_set(self, queue_instance):
        """Test that the begin time is set correctly."""
        utime = 1762924800.0
        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(ra=0, dec=0, utime=utime)
        expected_begin = int(utime)
        assert target.begin == expected_begin

    def test_get_target_end_set(self, queue_instance):
        """Test that the end time is set correctly."""
        utime = 1762924800.0
        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(ra=0, dec=0, utime=utime)
        expected_end = int(utime + target.slewtime + target.ss_max)
        assert target.end == expected_end

    def test_get_target_none_available(self, queue_instance):
        """Test getting a target when none are available."""
        utime = 1762924800.0

        # Make all targets not visible
        for target in queue_instance.targets:
            target.visible.return_value = False

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(ra=0, dec=0, utime=utime)

        assert target is None

    def test_get_target_endtime_exceeds_ephem(self, queue_instance):
        """Test when observation end time exceeds ephemeris."""
        utime = queue_instance.ephem.timestamp[-1].timestamp() - 50

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(ra=0, dec=0, utime=utime)

        endtime = utime + target.slewtime + target.ss_min
        expected_endtime_check = queue_instance.ephem.timestamp[-1].timestamp()
        assert endtime > expected_endtime_check

    def test_get_target_visible_called_with_constrained_end(self, queue_instance):
        """Test that visible() is called with the constrained ephemeris end."""
        utime = queue_instance.ephem.timestamp[-1].timestamp() - 50

        with patch.object(queue_instance, "meritsort"):
            _ = queue_instance.get(ra=0, dec=0, utime=utime)

        expected_endtime_check = queue_instance.ephem.timestamp[-1].timestamp()
        queue_instance.targets[0].visible.assert_called_with(
            utime, expected_endtime_check
        )

    def test_get_target_returns_target_still_visible(self, queue_instance):
        """Test that get() can still return a target when observation is constrained by ephem."""
        utime = queue_instance.ephem.timestamp[-1].timestamp() - 50

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(ra=0, dec=0, utime=utime)

        assert (
            target is not None
        )  # Assuming it is still visible in the shortened window


class TestSlewDistanceWeight:
    """Tests for slew distance weight feature in target selection."""

    def test_zero_weight_returns_first_visible_target(self, queue_instance):
        """With slew_distance_weight=0.0, should return first visible target (highest merit)."""
        utime = 1762924800.0
        queue_instance.slew_distance_weight = 0.0

        # Set different slewdist values - shouldn't matter with weight=0
        for i, target in enumerate(queue_instance.targets):
            target.slewdist = (i + 1) * 10.0  # 10, 20, 30, 40, 50

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(ra=0, dec=0, utime=utime)

        # Should return first target (highest merit) regardless of slewdist
        assert target == queue_instance.targets[0]

    def test_positive_weight_prefers_closer_targets(self, queue_instance):
        """With positive slew_distance_weight, should prefer targets with shorter slews."""
        utime = 1762924800.0
        queue_instance.slew_distance_weight = 5.0  # Significant weight

        # Target 0: merit=100, slewdist=100 -> score = 100 - 5*100 = -400
        # Target 1: merit=90, slewdist=10 -> score = 90 - 5*10 = 40
        # Target 2: merit=80, slewdist=5 -> score = 80 - 5*5 = 55 (best!)
        # Target 3: merit=70, slewdist=50 -> score = 70 - 5*50 = -180
        # Target 4: merit=60, slewdist=50 -> score = 60 - 5*50 = -190
        queue_instance.targets[0].merit = 100
        queue_instance.targets[0].slewdist = 100.0
        queue_instance.targets[1].merit = 90
        queue_instance.targets[1].slewdist = 10.0
        queue_instance.targets[2].merit = 80
        queue_instance.targets[2].slewdist = 5.0
        queue_instance.targets[3].merit = 70
        queue_instance.targets[3].slewdist = 50.0
        queue_instance.targets[4].merit = 60
        queue_instance.targets[4].slewdist = 50.0

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(ra=0, dec=0, utime=utime)

        # Should return target[2] with best score despite lower merit
        assert target == queue_instance.targets[2]

    def test_weight_breaks_merit_ties(self, queue_instance):
        """When merits are equal, slew distance weight should break ties."""
        utime = 1762924800.0
        queue_instance.slew_distance_weight = 1.0

        # All targets have same merit, different distances
        for i, target in enumerate(queue_instance.targets):
            target.merit = 100
            target.slewdist = (i + 1) * 10.0  # 10, 20, 30, 40, 50

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(ra=0, dec=0, utime=utime)

        # Should return target[0] with shortest slew distance
        assert target == queue_instance.targets[0]

    def test_missing_slewdist_defaults_to_zero(self, queue_instance):
        """Targets without slewdist attribute should default to 0.0."""
        utime = 1762924800.0
        queue_instance.slew_distance_weight = 10.0

        # Remove slewdist from all targets
        for target in queue_instance.targets:
            if hasattr(target, "slewdist"):
                delattr(target, "slewdist")

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(ra=0, dec=0, utime=utime)

        # Should still return first target (no penalty applied)
        assert target == queue_instance.targets[0]


class TestCollectionTimeWeight:
    """Tests for collection time reward in target selection."""

    def test_positive_weight_prefers_longer_collection_window(self, queue_instance):
        """Collection time can choose a longer useful observation over a shorter slew."""
        utime = 1762924800.0
        queue_instance.slew_distance_weight = 0.1
        queue_instance.collection_time_weight = 1.0

        close_target = queue_instance.targets[0]
        close_target.merit = 100
        close_target.slewdist = 0.0
        close_target.calc_slewtime.return_value = 10
        close_target.exptime = 1500
        close_target.ss_max = 1500
        close_target.visible.return_value = [utime, utime + 400]

        longer_target = queue_instance.targets[1]
        longer_target.merit = 100
        longer_target.slewdist = 30.0
        longer_target.calc_slewtime.return_value = 20
        longer_target.exptime = 1500
        longer_target.ss_max = 1500
        longer_target.visible.return_value = [utime, utime + 1700]

        for target in queue_instance.targets[2:]:
            target.visible.return_value = False

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(ra=0, dec=0, utime=utime)

        assert target == longer_target

    def test_collection_time_is_capped_by_remaining_exposure(self, queue_instance):
        """The collection reward should not exceed remaining target exposure."""
        utime = 1762924800.0
        target = queue_instance.targets[0]
        target.slewtime = 20
        target.exptime = 300
        target.ss_max = 1000

        collection_seconds = queue_instance._candidate_collection_seconds(
            target=target,
            visibility_window=[utime, utime + 2000],
            utime=utime,
        )

        assert collection_seconds == 300

    def test_collection_time_is_capped_by_deadline(self, queue_instance):
        """Collection reward should use the earliest downstream science deadline."""
        utime = 1762924800.0
        target = queue_instance.targets[0]
        target.slewtime = 20
        target.exptime = 1000
        target.ss_max = 1000

        collection_seconds = queue_instance._candidate_collection_seconds(
            target=target,
            visibility_window=[utime, utime + 2000],
            utime=utime,
            deadline=utime + 500,
        )

        assert collection_seconds == 480

    def test_deadline_skips_target_that_cannot_meet_minimum_snapshot(
        self, queue_instance
    ):
        """A downstream deadline should keep infeasible targets out of scoring."""
        utime = 1762924800.0
        queue_instance.slew_distance_weight = 0.1
        queue_instance.collection_time_weight = 1.0

        infeasible_target = queue_instance.targets[0]
        infeasible_target.merit = 100
        infeasible_target.slewdist = 0.0
        infeasible_target.calc_slewtime.return_value = 10
        infeasible_target.ss_min = 300
        infeasible_target.exptime = 1500
        infeasible_target.ss_max = 1500
        infeasible_target.visible.return_value = [utime, utime + 1700]

        feasible_target = queue_instance.targets[1]
        feasible_target.merit = 100
        feasible_target.slewdist = 40.0
        feasible_target.calc_slewtime.return_value = 20
        feasible_target.ss_min = 300
        feasible_target.exptime = 1500
        feasible_target.ss_max = 1500
        feasible_target.visible.return_value = [utime, utime + 1700]

        for target in queue_instance.targets[2:]:
            target.visible.return_value = False

        def deadline(target, slew_end):
            if target is infeasible_target:
                return slew_end + 120
            return slew_end + 900

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(
                ra=0,
                dec=0,
                utime=utime,
                collection_deadline=deadline,
            )

        assert target == feasible_target

    def test_zero_slew_prefilter_skips_impossible_visibility_before_slew_estimate(
        self, queue_instance
    ):
        """Do not estimate slew when even zero slew cannot fit the visibility window."""
        utime = 1762924800.0
        queue_instance.slew_time_weight = 1.0

        impossible_target = queue_instance.targets[0]
        impossible_target.merit = 100
        impossible_target.visible.return_value = False

        feasible_target = queue_instance.targets[1]
        feasible_target.merit = 90
        feasible_target.visible.return_value = [utime, utime + 1000]

        for target in queue_instance.targets[2:]:
            target.visible.return_value = False

        estimator = Mock(return_value=TargetSlewEstimate(slewtime=10.0, slewdist=1.0))

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(
                ra=0,
                dec=0,
                utime=utime,
                slew_estimator=estimator,
            )

        assert target == feasible_target
        estimator.assert_called_once_with(feasible_target)
        impossible_target.calc_slewtime.assert_not_called()

    def test_zero_slew_prefilter_skips_impossible_deadline_before_slew_estimate(
        self, queue_instance
    ):
        """Do not estimate slew when zero slew cannot satisfy the deadline."""
        utime = 1762924800.0
        queue_instance.slew_time_weight = 1.0

        impossible_target = queue_instance.targets[0]
        impossible_target.merit = 100
        impossible_target.ss_min = 300
        impossible_target.exptime = 1500
        impossible_target.ss_max = 1500
        impossible_target.visible.return_value = [utime, utime + 1000]

        feasible_target = queue_instance.targets[1]
        feasible_target.merit = 90
        feasible_target.ss_min = 300
        feasible_target.exptime = 1500
        feasible_target.ss_max = 1500
        feasible_target.visible.return_value = [utime, utime + 1000]

        for target in queue_instance.targets[2:]:
            target.visible.return_value = False

        def deadline(target, _slew_end):
            if target is impossible_target:
                return utime + 120
            return utime + 900

        estimator = Mock(return_value=TargetSlewEstimate(slewtime=10.0, slewdist=1.0))

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(
                ra=0,
                dec=0,
                utime=utime,
                collection_deadline=deadline,
                slew_estimator=estimator,
            )

        assert target == feasible_target
        estimator.assert_called_once_with(feasible_target)
        impossible_target.calc_slewtime.assert_not_called()

    def test_deadline_prefilter_is_skipped_when_scoring_disabled(self, queue_instance):
        """Preserve the unscored fast path, which ignores collection deadlines."""
        utime = 1762924800.0
        target = queue_instance.targets[0]
        target.merit = 100
        target.ss_min = 300
        target.exptime = 1500
        target.ss_max = 1500
        target.visible.return_value = [utime, utime + 1000]

        def deadline(_target, _slew_end):
            return utime + 120

        with patch.object(queue_instance, "meritsort"):
            selected = queue_instance.get(
                ra=0,
                dec=0,
                utime=utime,
                collection_deadline=deadline,
            )

        assert selected == target
        target.calc_slewtime.assert_called_once_with(0, 0)

    def test_score_bound_skips_candidate_that_cannot_beat_best(self, queue_instance):
        """Candidates whose optimistic score cannot win should skip slew scoring."""
        utime = 1762924800.0
        queue_instance.slew_distance_weight = 1.0

        best_target = queue_instance.targets[0]
        best_target.merit = 100
        best_target.visible.return_value = [utime, utime + 1000]

        beaten_target = queue_instance.targets[1]
        beaten_target.merit = 90
        beaten_target.visible.return_value = [utime, utime + 1000]

        for target in queue_instance.targets[2:]:
            target.merit = 80
            target.visible.return_value = [utime, utime + 1000]

        estimator = Mock(return_value=TargetSlewEstimate(slewtime=0.0, slewdist=0.0))

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(
                ra=0,
                dec=0,
                utime=utime,
                slew_estimator=estimator,
            )

        assert target == best_target
        estimator.assert_called_once_with(best_target)
        beaten_target.calc_slewtime.assert_not_called()

    def test_score_bound_still_scores_candidate_that_could_beat_best(
        self, queue_instance
    ):
        """Collection reward upper bound must keep possible winners in scoring."""
        utime = 1762924800.0
        queue_instance.slew_distance_weight = 1.0
        queue_instance.collection_time_weight = 10.0

        short_target = queue_instance.targets[0]
        short_target.merit = 100
        short_target.ss_max = 60
        short_target.exptime = 60
        short_target.visible.return_value = [utime, utime + 60]

        long_target = queue_instance.targets[1]
        long_target.merit = 90
        long_target.ss_max = 300
        long_target.exptime = 300
        long_target.visible.return_value = [utime, utime + 300]

        for target in queue_instance.targets[2:]:
            target.visible.return_value = False

        estimator = Mock(return_value=TargetSlewEstimate(slewtime=0.0, slewdist=0.0))

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(
                ra=0,
                dec=0,
                utime=utime,
                slew_estimator=estimator,
            )

        assert target == long_target
        estimator.assert_any_call(short_target)
        estimator.assert_any_call(long_target)

    def test_score_bound_pruning_is_disabled_for_negative_weights(self, queue_instance):
        """Negative weights can reward costs, so they must keep full scoring."""
        utime = 1762924800.0
        queue_instance.slew_distance_weight = -1.0

        nominal_target = queue_instance.targets[0]
        nominal_target.merit = 100
        nominal_target.visible.return_value = [utime, utime + 1000]

        rewarded_target = queue_instance.targets[1]
        rewarded_target.merit = 90
        rewarded_target.visible.return_value = [utime, utime + 1000]

        for target in queue_instance.targets[2:]:
            target.visible.return_value = False

        def estimate(target):
            if target is rewarded_target:
                return TargetSlewEstimate(slewtime=0.0, slewdist=20.0)
            return TargetSlewEstimate(slewtime=0.0, slewdist=0.0)

        estimator = Mock(side_effect=estimate)

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(
                ra=0,
                dec=0,
                utime=utime,
                slew_estimator=estimator,
            )

        assert target == rewarded_target
        estimator.assert_any_call(nominal_target)
        estimator.assert_any_call(rewarded_target)

    def test_slew_time_weight_penalizes_longer_slews(self, queue_instance):
        """Slew time can be scored directly as an opportunity cost."""
        utime = 1762924800.0
        queue_instance.slew_time_weight = 10.0

        quick_target = queue_instance.targets[0]
        quick_target.merit = 100
        quick_target.calc_slewtime.return_value = 30

        slow_target = queue_instance.targets[1]
        slow_target.merit = 100
        slow_target.calc_slewtime.return_value = 300

        for target in queue_instance.targets[2:]:
            target.visible.return_value = False

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(ra=0, dec=0, utime=utime)

        assert target == quick_target

    def test_slew_estimator_drives_scored_slew_cost(self, queue_instance):
        """Selection should use caller-provided attitude-aware slew estimates."""
        utime = 1762924800.0
        queue_instance.slew_distance_weight = 1.0
        queue_instance.slew_time_weight = 1.0

        attitude_expensive_target = queue_instance.targets[0]
        attitude_expensive_target.merit = 100

        attitude_cheap_target = queue_instance.targets[1]
        attitude_cheap_target.merit = 100

        for target in queue_instance.targets[2:]:
            target.visible.return_value = False

        def estimate(target):
            if target is attitude_expensive_target:
                return TargetSlewEstimate(slewtime=300.0, slewdist=100.0)
            return TargetSlewEstimate(slewtime=30.0, slewdist=10.0)

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(
                ra=0,
                dec=0,
                utime=utime,
                slew_estimator=estimate,
            )

        assert target == attitude_cheap_target
        attitude_expensive_target.calc_slewtime.assert_not_called()
        attitude_cheap_target.calc_slewtime.assert_not_called()

    def test_slew_estimator_is_skipped_when_scoring_disabled(self, queue_instance):
        """The default fast path should not pay for attitude-aware estimates."""
        utime = 1762924800.0
        target = queue_instance.targets[0]
        target.merit = 100

        estimator = Mock(return_value=TargetSlewEstimate(slewtime=300.0, slewdist=0.0))

        with patch.object(queue_instance, "meritsort"):
            selected = queue_instance.get(
                ra=0,
                dec=0,
                utime=utime,
                slew_estimator=estimator,
            )

        assert selected == target
        estimator.assert_not_called()
        target.calc_slewtime.assert_called_once()


class TestQueueSelectionBehavior:
    def test_get_without_star_trackers_preserves_existing_behavior(
        self, queue_instance
    ):
        """Queue selection should behave as before when no ST subsystem is configured."""
        utime = 1762924800.0
        queue_instance.config.spacecraft_bus.star_trackers = None

        with patch.object(queue_instance, "meritsort"):
            target = queue_instance.get(ra=0, dec=0, utime=utime)

        assert target == queue_instance.targets[0]

    def test_radiator_exposure_penalty_prefers_cooler_target(self, queue_instance):
        """Radiator exposure weights should steer selection toward lower-exposure targets."""
        utime = 1762924800.0
        queue_instance.slew_distance_weight = 0.0
        queue_instance.radiator_sun_exposure_weight = 100.0
        queue_instance.radiator_earth_exposure_weight = 0.0

        # Make merit identical so only radiator penalty drives selection.
        queue_instance.targets[0].merit = 100
        queue_instance.targets[1].merit = 100
        queue_instance.targets[0].ra = 0.0
        queue_instance.targets[1].ra = 1.0

        radiators = Mock()
        radiators.num_radiators.return_value = 2

        def _metrics(ra_deg, **kwargs):
            if ra_deg == 0.0:
                return {"sun_exposure": 0.9, "earth_exposure": 0.0}
            return {"sun_exposure": 0.1, "earth_exposure": 0.0}

        radiators.exposure_metrics.side_effect = _metrics
        queue_instance.config.spacecraft_bus.radiators = radiators

        with patch.object(queue_instance, "meritsort"):
            chosen = queue_instance.get(ra=0, dec=0, utime=utime)

        assert chosen == queue_instance.targets[1]

    def test_zero_radiator_weights_use_fast_path(self, queue_instance):
        """When both radiator weights are 0, exposure_metrics must not be called."""
        utime = 1762924800.0
        queue_instance.slew_distance_weight = 0.0
        queue_instance.radiator_sun_exposure_weight = 0.0
        queue_instance.radiator_earth_exposure_weight = 0.0

        radiators = Mock()
        radiators.num_radiators.return_value = 2
        queue_instance.config.spacecraft_bus.radiators = radiators

        with patch.object(queue_instance, "meritsort"):
            queue_instance.get(ra=0, dec=0, utime=utime)

        radiators.exposure_metrics.assert_not_called()

    def test_earth_exposure_weight_influences_selection(self, queue_instance):
        """Earth exposure weight should penalise targets with high earth exposure."""
        utime = 1762924800.0
        queue_instance.slew_distance_weight = 0.0
        queue_instance.radiator_sun_exposure_weight = 0.0
        queue_instance.radiator_earth_exposure_weight = 100.0

        queue_instance.targets[0].merit = 100
        queue_instance.targets[1].merit = 100
        queue_instance.targets[0].ra = 0.0
        queue_instance.targets[1].ra = 1.0

        radiators = Mock()
        radiators.num_radiators.return_value = 1

        def _metrics(ra_deg, **kwargs):
            if ra_deg == 0.0:
                return {"sun_exposure": 0.0, "earth_exposure": 0.8}
            return {"sun_exposure": 0.0, "earth_exposure": 0.1}

        radiators.exposure_metrics.side_effect = _metrics
        queue_instance.config.spacecraft_bus.radiators = radiators

        with patch.object(queue_instance, "meritsort"):
            chosen = queue_instance.get(ra=0, dec=0, utime=utime)

        assert chosen == queue_instance.targets[1]

    def test_radiator_scoring_skipped_when_no_radiators(self, queue_instance):
        """When num_radiators() == 0, exposure_metrics should not be called."""
        utime = 1762924800.0
        queue_instance.slew_distance_weight = 0.0
        queue_instance.radiator_sun_exposure_weight = 50.0
        queue_instance.radiator_earth_exposure_weight = 50.0

        radiators = Mock()
        radiators.num_radiators.return_value = 0
        queue_instance.config.spacecraft_bus.radiators = radiators

        with patch.object(queue_instance, "meritsort"):
            queue_instance.get(ra=0, dec=0, utime=utime)

        radiators.exposure_metrics.assert_not_called()
