"""Tests for the dynamic merit model."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from conops.config import MissionConfig, TargetConfig
from conops.config.observation_categories import (
    ObservationCategories,
    ObservationCategory,
)
from conops.targets import PlanEntry, Pointing
from conops.targets.merit import MeritBreakdown, MeritModel

NOW = 1_000_000.0


def _config(**weights: float) -> MissionConfig:
    config = MissionConfig()
    config.targets = TargetConfig(**weights)
    config.observation_categories = ObservationCategories(
        categories=[
            ObservationCategory(
                name="ToO", obsid_min=1000, obsid_max=2000, tier=2, program="Rapid"
            ),
            ObservationCategory(
                name="Monitor",
                obsid_min=2000,
                obsid_max=3000,
                cadence_seconds=3600.0,
                time_share=0.5,
            ),
            ObservationCategory(
                name="Survey", obsid_min=3000, obsid_max=4000, time_share=0.25
            ),
        ]
    )
    return config


def _target(obsid: int = 5000, merit: float = 10.0, **fields: object) -> Pointing:
    return Pointing(obsid=obsid, merit=merit, fom=merit, **fields)


class TestMeritBreakdown:
    def test_value_and_score_sum_their_terms(self) -> None:
        breakdown = MeritBreakdown(
            tier=1,
            base=10.0,
            urgency=2.0,
            cadence=1.0,
            completion_deficit=-0.5,
            slew_distance=-3.0,
            slew_time=-1.0,
            collection=4.0,
            radiator=-0.25,
        )

        assert breakdown.value == pytest.approx(12.5)
        assert breakdown.score == pytest.approx(12.25)
        assert breakdown.rank == (1, pytest.approx(12.25))
        assert breakdown.value_rank == (1, pytest.approx(12.5))

    def test_tier_outranks_any_score(self) -> None:
        low_tier = MeritBreakdown(tier=0, base=1e9)
        high_tier = MeritBreakdown(tier=1, base=-1e9)

        assert high_tier.rank > low_tier.rank

    def test_describe_lists_nonzero_terms_only(self) -> None:
        text = MeritBreakdown(tier=0, base=10.0, urgency=2.5).describe()

        assert "urgency=+2.500" in text
        assert "cadence" not in text
        assert "value=12.500" in text


class TestTier:
    def test_tier_comes_from_category(self) -> None:
        model = MeritModel(_config())

        assert model.tier(_target(obsid=1500)) == 2
        assert model.tier(_target(obsid=5000)) == 0


class TestZeroWeights:
    def test_value_is_base_merit(self) -> None:
        model = MeritModel(_config())
        target = _target(obsid=2500, merit=42.0, deadline=NOW + 60)

        breakdown = model.value_terms(target, NOW)

        assert not model.is_dynamic
        assert breakdown.value == 42.0
        assert breakdown.urgency == breakdown.cadence == 0.0
        assert breakdown.completion_deficit == 0.0


class TestUrgency:
    @pytest.fixture
    def model(self) -> MeritModel:
        return MeritModel(
            _config(urgency_weight=50.0, urgency_timescale_seconds=3600.0)
        )

    def test_no_deadline_no_urgency(self, model: MeritModel) -> None:
        assert model.value_terms(_target(), NOW).urgency == 0.0

    def test_full_urgency_within_timescale(self, model: MeritModel) -> None:
        target = _target(deadline=NOW + 1800)

        assert model.value_terms(target, NOW).urgency == 50.0

    def test_urgency_falls_off_with_time_to_close(self, model: MeritModel) -> None:
        target = _target(deadline=NOW + 4 * 3600)

        assert model.value_terms(target, NOW).urgency == pytest.approx(12.5)

    def test_last_window_before_deadline_sets_the_close(
        self, model: MeritModel
    ) -> None:
        """No later window before the deadline: the current window is the last chance."""
        target = _target(deadline=NOW + 10 * 3600)
        target.windows = [[NOW - 600, NOW + 1800]]

        breakdown = model.value_terms(
            target, NOW, visibility_window=[NOW - 600, NOW + 1800]
        )

        assert breakdown.urgency == 50.0

    def test_later_window_before_deadline_keeps_the_deadline(
        self, model: MeritModel
    ) -> None:
        target = _target(deadline=NOW + 4 * 3600)
        target.windows = [[NOW - 600, NOW + 1800], [NOW + 7200, NOW + 9000]]

        breakdown = model.value_terms(
            target, NOW, visibility_window=[NOW - 600, NOW + 1800]
        )

        assert breakdown.urgency == pytest.approx(12.5)


class TestCadence:
    @pytest.fixture
    def model(self) -> MeritModel:
        return MeritModel(_config(cadence_weight=20.0))

    def test_never_visited_is_due(self, model: MeritModel) -> None:
        assert model.value_terms(_target(obsid=2500), NOW).cadence == 20.0

    @pytest.mark.parametrize(
        ("elapsed", "expected"), [(0.0, 0.0), (1800.0, 10.0), (7200.0, 20.0)]
    )
    def test_rises_with_elapsed_fraction_of_interval(
        self, model: MeritModel, elapsed: float, expected: float
    ) -> None:
        target = _target(obsid=2500)
        target.record_collection(NOW - elapsed, 60.0)

        assert model.value_terms(target, NOW).cadence == pytest.approx(expected)

    def test_category_without_cadence(self, model: MeritModel) -> None:
        assert model.value_terms(_target(obsid=3500), NOW).cadence == 0.0


class TestCompletionDeficit:
    @pytest.fixture
    def model(self) -> MeritModel:
        return MeritModel(_config(completion_deficit_weight=10.0))

    def test_delivered_shares_by_program(self, model: MeritModel) -> None:
        monitor, survey, rapid = _target(2500), _target(3500), _target(1500)
        monitor.record_collection(NOW, 300.0)
        survey.record_collection(NOW, 100.0)

        shares = model.delivered_shares([monitor, survey, rapid])

        assert shares == {"Monitor": 0.75, "Survey": 0.25}

    def test_nothing_delivered_yet(self, model: MeritModel) -> None:
        assert model.delivered_shares([_target(2500)]) == {}

    def test_deficit_is_allocated_minus_delivered(self, model: MeritModel) -> None:
        shares = {"Monitor": 0.75, "Survey": 0.25}

        behind = model.value_terms(_target(3500), NOW, delivered_shares={})
        on_track = model.value_terms(_target(3500), NOW, delivered_shares=shares)
        ahead = model.value_terms(_target(2500), NOW, delivered_shares=shares)

        assert behind.completion_deficit == pytest.approx(2.5)
        assert on_track.completion_deficit == pytest.approx(0.0)
        assert ahead.completion_deficit == pytest.approx(-2.5)

    def test_program_without_share(self, model: MeritModel) -> None:
        assert model.value_terms(_target(1500), NOW).completion_deficit == 0.0


class TestMaxDynamicValue:
    def test_bounds_the_dynamic_terms(self) -> None:
        model = MeritModel(
            _config(
                urgency_weight=5.0, cadence_weight=3.0, completion_deficit_weight=2.0
            )
        )

        assert model.max_dynamic_value == 10.0


class TestConfigValidation:
    @pytest.mark.parametrize(
        "field", ["urgency_weight", "cadence_weight", "completion_deficit_weight"]
    )
    def test_negative_weights_rejected(self, field: str) -> None:
        with pytest.raises(ValidationError):
            TargetConfig(**{field: -1.0})

    @pytest.mark.parametrize("share", [-0.1, 1.1])
    def test_time_share_bounded(self, share: float) -> None:
        with pytest.raises(ValidationError):
            ObservationCategory(name="X", obsid_min=0, obsid_max=1, time_share=share)

    def test_program_defaults_to_category_name(self) -> None:
        category = ObservationCategory(name="X", obsid_min=0, obsid_max=1)

        assert category.program_name == "X"


class TestCollectionCredit:
    def test_record_collection_reduces_exposure_and_tracks_visit(self) -> None:
        entry = PlanEntry(obsid=1)
        entry.exptime = 600

        entry.record_collection(NOW, 200.0)

        assert entry.exptime == 400
        assert entry.collected_seconds == 200.0
        assert entry.last_collection_time == NOW

    def test_no_collection_is_not_a_visit(self) -> None:
        entry = PlanEntry(obsid=1)

        entry.record_collection(NOW, 0.0)

        assert entry.last_collection_time is None

    def test_reset_clears_collection(self) -> None:
        target = _target()
        target.exptime = 600
        target.record_collection(NOW, 200.0)

        target.reset()

        assert target.collected_seconds == 0.0
        assert target.last_collection_time is None
        assert target.exptime == 600


class TestDeadlineField:
    def test_accepts_datetime_and_serializes_iso(self) -> None:
        deadline = datetime(2026, 1, 1, tzinfo=timezone.utc)
        target = _target(deadline=deadline)

        assert target.deadline == deadline.timestamp()
        assert target.model_dump(mode="json")["deadline"] == deadline.isoformat()

    def test_naive_datetime_is_utc(self) -> None:
        target = _target(deadline=datetime(2026, 1, 1))

        assert target.deadline == datetime(2026, 1, 1, tzinfo=timezone.utc).timestamp()

    def test_absent_deadline_is_not_serialized(self) -> None:
        assert "deadline" not in _target().model_dump(mode="json")


class TestEarliestStartField:
    def test_accepts_datetime_and_serializes_iso(self) -> None:
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)
        target = _target(earliest_start=start)

        assert target.earliest_start == start.timestamp()
        assert target.model_dump(mode="json")["earliest_start"] == start.isoformat()

    def test_absent_earliest_start_is_not_serialized(self) -> None:
        assert "earliest_start" not in _target().model_dump(mode="json")

    def test_may_equal_the_deadline(self) -> None:
        target = _target(earliest_start=1000.0, deadline=1000.0)

        assert target.earliest_start == target.deadline

    def test_must_not_be_after_the_deadline(self) -> None:
        with pytest.raises(ValidationError, match="earliest_start"):
            _target(earliest_start=2000.0, deadline=1000.0)
