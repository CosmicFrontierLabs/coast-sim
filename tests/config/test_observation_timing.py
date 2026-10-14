import pytest
from pydantic import ValidationError

from conops.common import ObsType
from conops.config import MissionConfig, ObservationTiming
from conops.targets import PlanEntry


@pytest.mark.parametrize(
    "field", ["setup_seconds", "cleanup_seconds", "handoff_seconds"]
)
@pytest.mark.parametrize("value", [-1, float("nan"), float("inf")])
def test_observation_timing_requires_finite_nonnegative_budgets(field, value):
    with pytest.raises(ValidationError):
        ObservationTiming(**{field: value})


def test_observation_timing_defaults_and_roundtrip():
    config = MissionConfig()
    assert config.payload.observation_timing.total_seconds == 0
    config.payload.observation_timing = ObservationTiming(
        setup_seconds=43, cleanup_seconds=2, handoff_seconds=10
    )
    restored = MissionConfig.model_validate_json(config.model_dump_json())
    assert restored.payload.observation_timing == config.payload.observation_timing


def test_science_window_excludes_payload_overheads_without_readding_acs_settle():
    entry = PlanEntry(begin=100, end=1180, slewtime=50)
    timing = ObservationTiming(setup_seconds=43, cleanup_seconds=2, handoff_seconds=10)
    entry.set_collection_window(timing)
    assert entry.begin == 100
    assert entry.end == 1180
    assert entry.collection_begin == 193
    assert entry.collection_end == 1168
    assert entry.exposure == 975
    assert entry.collection_seconds_between(160, 220) == 27
    assert entry.collection_seconds_between(1150, 1210) == 18
    restored = PlanEntry.model_validate_json(entry.model_dump_json())
    assert restored.collection_begin == 193
    assert restored.collection_end == 1168
    assert restored.exposure == 975


def test_science_window_shorter_than_overheads_is_empty():
    entry = PlanEntry(begin=100, end=150, slewtime=50)
    entry.set_collection_window(
        ObservationTiming(setup_seconds=43, cleanup_seconds=2, handoff_seconds=10)
    )
    assert entry.exposure == 0
    assert entry.collection_seconds_between(100, 150) == 0


@pytest.mark.parametrize("obstype", [ObsType.GSP, ObsType.CHARGE, ObsType.SAFE])
def test_observation_timing_does_not_change_non_science_entries(obstype):
    entry = PlanEntry(begin=100, end=500, slewtime=50, obstype=obstype)
    old_exposure = entry.exposure
    entry.set_collection_window(
        ObservationTiming(setup_seconds=43, cleanup_seconds=2, handoff_seconds=10)
    )
    assert entry.collection_begin is None
    assert entry.collection_end is None
    assert entry.exposure == old_exposure


@pytest.mark.parametrize(
    "bounds",
    [
        {"collection_begin": 150},
        {"collection_end": 350},
        {"collection_begin": 90, "collection_end": 350},
        {"collection_begin": 150, "collection_end": 410},
        {"collection_begin": 350, "collection_end": 150},
        {"collection_begin": 150, "collection_end": float("nan")},
    ],
)
def test_invalid_serialized_collection_windows_are_rejected(bounds):
    with pytest.raises(ValidationError):
        PlanEntry(begin=100, end=400, **bounds)


def test_dumb_scheduler_rejects_unsupported_timing_budgets():
    from conops import DumbScheduler

    config = MissionConfig()
    config.payload.observation_timing.setup_seconds = 10
    with pytest.raises(ValueError, match="require QueueDITL"):
        DumbScheduler(config)
