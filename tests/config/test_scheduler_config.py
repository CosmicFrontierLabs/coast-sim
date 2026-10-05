"""The scheduler section of a mission configuration."""

from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from conops.config import (
    MissionConfig,
    PlannerKind,
    PlannerSettings,
    ReplanSettings,
    SchedulerConfig,
    SchedulerMode,
)


class TestDefaults:
    def test_default_is_queue_dispatch(self) -> None:
        scheduler = MissionConfig().scheduler

        assert scheduler.mode is SchedulerMode.DISPATCH
        assert scheduler.planner.kind is PlannerKind.PRIORITY

    def test_unset_settings_use_planner_defaults(self) -> None:
        assert PlannerSettings().options() == {"include_passes": True}


class TestPlannerOptions:
    def test_local_search_settings_become_keyword_arguments(self) -> None:
        settings = PlannerSettings(
            kind=PlannerKind.LOCAL_SEARCH, time_limit=5.0, seed=3, earliness_weight=0.2
        )

        assert settings.options() == {
            "include_passes": True,
            "seed": 3,
            "time_limit": 5.0,
            "earliness_weight": 0.2,
        }

    def test_chunk_seconds_becomes_a_timedelta(self) -> None:
        settings = PlannerSettings(
            kind=PlannerKind.CP_SAT, solver_time_limit=10.0, chunk_seconds=3600
        )

        options = settings.options()

        assert options["chunk"] == timedelta(hours=1)
        assert options["solver_time_limit"] == 10.0
        assert "chunk_seconds" not in options

    @pytest.mark.parametrize(
        ("kind", "field"),
        [
            (PlannerKind.PRIORITY, "time_limit"),
            (PlannerKind.PRIORITY, "solver_time_limit"),
            # The priority planner has no search to seed; it breaks ties with
            # random_seed. Passing seed to it used to fail when it was built.
            (PlannerKind.PRIORITY, "seed"),
            (PlannerKind.LOCAL_SEARCH, "workers"),
        ],
    )
    def test_settings_for_another_planner_are_rejected(
        self, kind: PlannerKind, field: str
    ) -> None:
        with pytest.raises(ValidationError, match=field):
            PlannerSettings(kind=kind, **{field: 1})

    @pytest.mark.parametrize(
        "settings",
        [
            {"kind": "local_search", "earliness_weight": 1.5},
            {"kind": "local_search", "time_limit": -1},
            {"kind": "cp_sat", "solver_time_limit": 0},
            {"kind": "unknown"},
        ],
    )
    def test_invalid_settings_are_rejected(self, settings: dict[str, object]) -> None:
        with pytest.raises(ValidationError):
            PlannerSettings.model_validate(settings)


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
                planner=PlannerSettings(
                    kind=PlannerKind.CP_SAT, solver_time_limit=8.0, workers=2
                ),
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

    def test_mode_and_kind_are_written_as_names(self, config: MissionConfig) -> None:
        dumped = config.model_dump(mode="json")["scheduler"]

        assert dumped["mode"] == "rolling"
        assert dumped["planner"]["kind"] == "cp_sat"
