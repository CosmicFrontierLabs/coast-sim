from unittest.mock import patch

from conops import Queue


class TestQueueInitAndAppend:
    def test_queue_init_targets_empty(self, mock_config):
        queue = Queue(config=mock_config)
        assert queue.targets == []

    def test_queue_init_ephem_none(self, mock_config):
        queue = Queue(config=mock_config)
        assert queue.ephem is None

    def test_queue_init_utime_none(self, mock_config):
        queue = Queue(config=mock_config)
        assert queue.utime == 0.0

    def test_queue_init_gs_none(self, mock_config):
        queue = Queue(config=mock_config)
        assert queue.gs is None

    def test_queue_append_len(self, mock_target, mock_config):
        queue = Queue(config=mock_config)
        queue.append(mock_target)
        assert len(queue.targets) == 1

    def test_queue_append_target_equals(self, mock_target, mock_config):
        queue = Queue(config=mock_config)
        queue.append(mock_target)
        assert queue.targets[0] == mock_target


class TestQueueBasicOps:
    def test_queue_len(self, queue_instance):
        """Test the length of the queue."""
        assert len(queue_instance) == 5

    def test_queue_getitem(self, queue_instance, mock_targets):
        """Test getting an item from the queue."""
        assert queue_instance[2] == mock_targets[2]

    def test_queue_reset_reset_called(self, queue_instance):
        """Test resetting the queue calls reset() on each target."""
        for target in queue_instance.targets:
            target.done = True

        queue_instance.reset()

        for target in queue_instance.targets:
            target.reset.assert_called_once()

    def test_queue_reset_targets_done_false(self, queue_instance):
        """Test resetting the queue clears done flag on each target."""
        for target in queue_instance.targets:
            target.done = True

        queue_instance.reset()

        for target in queue_instance.targets:
            assert not target.done


class TestMeritsort:
    @patch("numpy.random.random", side_effect=[0.1, 0.2, 0.3, 0.4, 0.5])
    def test_meritsort_invisible_target_merit(self, mock_random, queue_instance):
        """The invisible target should get -900 + random penalty as merit."""
        invisible_target = queue_instance.targets[1]
        invisible_target.visible.return_value = False

        queue_instance.meritsort()

        assert invisible_target.merit == -900 + 0.2

    @patch("numpy.random.random", side_effect=[0.1, 0.2, 0.3, 0.4, 0.5])
    def test_meritsort_sorted_descending(self, mock_random, queue_instance):
        """Check that the targets are sorted by merit descending."""
        queue_instance.meritsort()
        for i in range(len(queue_instance.targets) - 1):
            assert (
                queue_instance.targets[i].merit >= queue_instance.targets[i + 1].merit
            )

    @patch("numpy.random.random", side_effect=[0.1, 0.2, 0.3, 0.4, 0.5])
    def test_meritsort_invisible_target_last(self, mock_random, queue_instance):
        """Invisible target should be last after sort (lowest merit)."""
        invisible_target = queue_instance.targets[1]
        invisible_target.visible.return_value = False

        queue_instance.meritsort()

        assert queue_instance.targets[-1] is invisible_target


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
