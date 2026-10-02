from unittest.mock import Mock

from conops.config import ObservationTiming


def test_candidate_scoring_excludes_all_observation_overheads(queue_instance):
    queue_instance.observation_timing = ObservationTiming(
        setup_seconds=43, cleanup_seconds=2, handoff_seconds=10
    )
    target = Mock(slewtime=50, ss_max=1000, exptime=1000)
    assert queue_instance._candidate_collection_seconds(target, [100, 500], 100) == 295
    assert (
        queue_instance._candidate_collection_seconds(
            target, [100, 500], 100, deadline=400
        )
        == 195
    )


def test_zero_slew_filter_reserves_payload_overheads(queue_instance):
    queue_instance.observation_timing = ObservationTiming(
        setup_seconds=43, cleanup_seconds=2, handoff_seconds=10
    )
    target = Mock(slewtime=0, ss_min=300, ss_max=1000, exptime=1000)
    target.visible.return_value = [100, 450]
    assert not queue_instance._can_fit_min_snapshot_with_zero_slew(target, 100, 1000)
    target.visible.return_value = [100, 455]
    assert queue_instance._can_fit_min_snapshot_with_zero_slew(target, 100, 1000)
