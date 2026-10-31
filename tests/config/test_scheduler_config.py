"""The scheduler section of a mission configuration."""

from datetime import timedelta
from pathlib import Path
from typing import Literal

import pytest
from pydantic import ValidationError

from conops.config import (
    MissionConfig,
    ReplanSettings,
    SchedulerConfig,
    SchedulerKind,
    SchedulerMode,
)


class TestDefaults:
    def test_default_is_queue_dispatch(self) -> None:
        scheduler = MissionConfig().scheduler

        assert scheduler.mode is SchedulerMode.DISPATCH
        assert scheduler.kind is SchedulerKind.PRIORITY

    def test_unset_settings_use_scheduler_defaults(self) -> None:
        assert SchedulerConfig().scheduler_options() == {"successor_retries": 3}

    def test_unlimited_successor_retries_tries_every_start(self) -> None:
        options = SchedulerConfig(successor_retries="unlimited").scheduler_options()

        # The schedulers take None for no limit.
        assert options["successor_retries"] is None


class TestSchedulerOptions:
    def test_local_search_settings_become_keyword_arguments(self) -> None:
        settings = SchedulerConfig(
            kind=SchedulerKind.LOCAL_SEARCH,
            time_limit=5.0,
            seed=3,
            earliness_weight=0.2,
        )

        assert settings.scheduler_options() == {
            "successor_retries": 3,
            "seed": 3,
            "time_limit": 5.0,
            "earliness_weight": 0.2,
        }

    def test_include_passes_is_not_a_scheduler_option(self) -> None:
        settings = SchedulerConfig(include_passes=False)

        assert "include_passes" not in settings.scheduler_options()

    def test_chunk_seconds_becomes_a_timedelta(self) -> None:
        settings = SchedulerConfig(
            kind=SchedulerKind.CP_SAT, solver_time_limit=10.0, chunk_seconds=3600
        )

        options = settings.scheduler_options()

        assert options["chunk"] == timedelta(hours=1)
        assert options["solver_time_limit"] == 10.0
        assert "chunk_seconds" not in options

    def test_the_deterministic_budget_is_a_scheduler_option(self) -> None:
        settings = SchedulerConfig(
            kind=SchedulerKind.CP_SAT, solver_deterministic_time=20.0, workers=1, seed=7
        )

        options = settings.scheduler_options()

        assert options["solver_deterministic_time"] == 20.0
        assert options["workers"] == 1 and options["seed"] == 7

    @pytest.mark.parametrize(
        ("kind", "field"),
        [
            (SchedulerKind.PRIORITY, "time_limit"),
            (SchedulerKind.PRIORITY, "solver_time_limit"),
            # The priority scheduler has no search to seed; it breaks ties
            # with random_seed, and fails if given a seed.
            (SchedulerKind.PRIORITY, "seed"),
            (SchedulerKind.LOCAL_SEARCH, "workers"),
            (SchedulerKind.LOCAL_SEARCH, "solver_deterministic_time"),
        ],
    )
    def test_settings_for_another_scheduler_are_rejected(
        self, kind: SchedulerKind, field: str
    ) -> None:
        with pytest.raises(ValidationError, match=field):
            SchedulerConfig.model_validate({"kind": kind, field: 1})

    @pytest.mark.parametrize(
        "settings",
        [
            {"kind": "local_search", "earliness_weight": 1.5},
            {"kind": "local_search", "time_limit": -1},
            {"kind": "cp_sat", "solver_time_limit": 0},
            {"kind": "cp_sat", "solver_deterministic_time": 0},
            {"successor_retries": -1},
            {"successor_retries": "all"},
            # Null would mean no limit here but the default once saved as YAML,
            # which leaves nulls out.
            {"successor_retries": None},
            {"kind": "unknown"},
            {"mode": "unknown"},
        ],
    )
    def test_invalid_settings_are_rejected(self, settings: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            SchedulerConfig.model_validate(settings)


class TestReplanSettings:
    @pytest.mark.parametrize(
        "settings",
        [
            {"horizon_seconds": 0},
            {"replan_interval_seconds": -1},
            {"commit_lead_time_seconds": -1},
        ],
    )
    def test_invalid_timing_is_rejected(self, settings: dict[str, float]) -> None:
        with pytest.raises(ValidationError):
            ReplanSettings.model_validate(settings)


class TestSerialization:
    @pytest.fixture
    def config(self) -> MissionConfig:
        return MissionConfig(
            scheduler=SchedulerConfig(
                mode=SchedulerMode.ROLLING,
                kind=SchedulerKind.CP_SAT,
                solver_time_limit=8.0,
                solver_deterministic_time=16.0,
                workers=2,
                replanning=ReplanSettings(commit_lead_time_seconds=1800),
            )
        )

    def test_json_round_trip(self, config: MissionConfig, tmp_path: Path) -> None:
        path = tmp_path / "mission.json"
        config.to_json_file(str(path))

        loaded = MissionConfig.from_json_file(str(path))

        assert loaded.scheduler == config.scheduler

    def test_yaml_round_trip(self, config: MissionConfig, tmp_path: Path) -> None:
        path = tmp_path / "mission.yaml"
        config.to_yaml_file(str(path))

        loaded = MissionConfig.from_yaml_file(str(path))

        assert loaded.scheduler == config.scheduler

    @pytest.mark.parametrize("retries", [0, "unlimited"])
    def test_yaml_round_trip_keeps_successor_retries(
        self, retries: int | Literal["unlimited"], tmp_path: Path
    ) -> None:
        config = MissionConfig(scheduler=SchedulerConfig(successor_retries=retries))
        path = tmp_path / "mission.yaml"
        config.to_yaml_file(str(path))

        loaded = MissionConfig.from_yaml_file(str(path))

        assert loaded.scheduler.successor_retries == retries

    def test_mode_and_kind_are_written_as_names(self, config: MissionConfig) -> None:
        dumped = config.model_dump(mode="json")["scheduler"]

        assert dumped["mode"] == "rolling"
        assert dumped["kind"] == "cp_sat"
