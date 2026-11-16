"""Momentum wheels: their configuration and the total capacity they give."""

import numpy as np
import pytest
from pydantic import ValidationError

from conops.config import (
    AttitudeControlSystem,
    MissionConfig,
    MomentumWheel,
    SpacecraftBus,
    StoredMomentumConfig,
    ThrusterDumpConfig,
    wheel_array_capacity_n_m_s,
)
from conops.schedulers.momentum import MomentumForecast


def _orthogonal(capacity: float = 4.0) -> list[MomentumWheel]:
    return [
        MomentumWheel(name=f"RW{i + 1}", spin_axis=axis, capacity_n_m_s=capacity)
        for i, axis in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    ]


def _pyramid(capacity: float = 4.0, cant_deg: float = 35.0) -> list[MomentumWheel]:
    cant = np.radians(cant_deg)
    return [
        MomentumWheel(
            name=f"RW{i + 1}",
            spin_axis=(
                np.sin(cant),
                np.cos(cant) * np.cos(angle),
                np.cos(cant) * np.sin(angle),
            ),
            capacity_n_m_s=capacity,
        )
        for i, angle in enumerate(np.radians([45.0, 135.0, 225.0, 315.0]))
    ]


def _least_reach(wheels: list[MomentumWheel]) -> float:
    """The wheels' least reach over many directions, by brute force."""
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(200_000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    axes = np.array([w.spin_axis for w in wheels])
    capacities = np.array([w.capacity_n_m_s for w in wheels])
    return float(np.min(np.abs(directions @ axes.T) @ capacities))


class TestMomentumWheel:
    def test_spin_axis_is_normalised(self) -> None:
        wheel = MomentumWheel(spin_axis=(0.0, 3.0, 4.0), capacity_n_m_s=1.0)
        assert wheel.spin_axis == pytest.approx((0.0, 0.6, 0.8))

    @pytest.mark.parametrize(
        "axis", [(0.0, 0.0, 0.0), (float("nan"), 0.0, 1.0), (float("inf"), 0.0, 0.0)]
    )
    def test_spin_axis_must_be_a_direction(
        self, axis: tuple[float, float, float]
    ) -> None:
        with pytest.raises(ValidationError, match="spin_axis"):
            MomentumWheel(spin_axis=axis, capacity_n_m_s=1.0)

    @pytest.mark.parametrize("capacity", [0.0, -1.0, float("inf")])
    def test_capacity_must_be_positive_and_finite(self, capacity: float) -> None:
        with pytest.raises(ValidationError):
            MomentumWheel(spin_axis=(1, 0, 0), capacity_n_m_s=capacity)

    def test_position_must_be_finite(self) -> None:
        with pytest.raises(ValidationError, match="position_m"):
            MomentumWheel(
                spin_axis=(1, 0, 0),
                capacity_n_m_s=1.0,
                position_m=(0.0, float("nan"), 0.0),
            )


class TestWheelArrayCapacity:
    def test_orthogonal_wheels_hold_one_wheel_about_every_axis(self) -> None:
        assert wheel_array_capacity_n_m_s(_orthogonal(4.0)) == pytest.approx(4.0)

    def test_the_weakest_orthogonal_wheel_sets_the_capacity(self) -> None:
        wheels = _orthogonal(4.0)
        wheels[1].capacity_n_m_s = 1.5
        assert wheel_array_capacity_n_m_s(wheels) == pytest.approx(1.5)

    @pytest.mark.parametrize("cant_deg", [20.0, 35.26, 54.74])
    def test_pyramid_matches_the_least_reach_in_any_direction(
        self, cant_deg: float
    ) -> None:
        wheels = _pyramid(4.0, cant_deg)
        capacity = wheel_array_capacity_n_m_s(wheels)
        least = _least_reach(wheels)
        # No direction reaches less, and some direction reaches no more.
        assert capacity <= least + 1e-9
        assert capacity == pytest.approx(least, rel=1e-2)

    def test_a_fourth_wheel_adds_capacity(self) -> None:
        assert wheel_array_capacity_n_m_s(_pyramid(4.0)) > 4.0

    def test_capacity_scales_with_the_wheels(self) -> None:
        assert wheel_array_capacity_n_m_s(_pyramid(8.0)) == pytest.approx(
            2 * wheel_array_capacity_n_m_s(_pyramid(4.0))
        )

    @pytest.mark.parametrize(
        "axes",
        [
            [],
            [(1, 0, 0)],
            [(1, 0, 0), (-1, 0, 0)],
            [(1, 0, 0), (0, 1, 0)],
            [(1, 0, 0), (0, 1, 0), (1, 1, 0)],
        ],
    )
    def test_wheels_that_do_not_span_three_axes_hold_nothing(
        self, axes: list[tuple[float, float, float]]
    ) -> None:
        wheels = [MomentumWheel(spin_axis=a, capacity_n_m_s=4.0) for a in axes]
        assert wheel_array_capacity_n_m_s(wheels) == 0.0


class TestStoredMomentumCapacity:
    def test_no_wheels_and_no_capacity_sets_none(self) -> None:
        config = StoredMomentumConfig()
        assert config.wheel_capacity_n_m_s is None
        assert config.total_capacity_n_m_s is None

    def test_explicit_capacity_alone_is_unchanged(self) -> None:
        config = StoredMomentumConfig(capacity_n_m_s=5.0)
        assert config.wheel_capacity_n_m_s is None
        assert config.total_capacity_n_m_s == 5.0

    def test_wheels_give_the_total_capacity(self) -> None:
        config = StoredMomentumConfig(wheels=_pyramid(4.0))
        assert config.capacity_n_m_s is None
        assert config.total_capacity_n_m_s == pytest.approx(
            wheel_array_capacity_n_m_s(_pyramid(4.0))
        )

    def test_explicit_capacity_takes_precedence_over_the_wheels(self) -> None:
        config = StoredMomentumConfig(wheels=_pyramid(4.0), capacity_n_m_s=2.0)
        assert config.total_capacity_n_m_s == 2.0
        assert config.wheel_capacity_n_m_s == pytest.approx(
            wheel_array_capacity_n_m_s(_pyramid(4.0))
        )

    def test_total_capacity_follows_a_changed_wheel(self) -> None:
        config = StoredMomentumConfig(wheels=_orthogonal(4.0))
        config.wheels[0].capacity_n_m_s = 1.0
        assert config.total_capacity_n_m_s == pytest.approx(1.0)

    def test_wheels_that_do_not_span_three_axes_need_a_capacity(self) -> None:
        planar = _orthogonal()[:2]
        with pytest.raises(ValidationError, match="do not span three dimensions"):
            StoredMomentumConfig(wheels=planar)
        assert (
            StoredMomentumConfig(wheels=planar, capacity_n_m_s=3.0).total_capacity_n_m_s
            == 3.0
        )

    def test_wheels_satisfy_dumps_and_slew_momentum(self) -> None:
        config = StoredMomentumConfig(
            wheels=_orthogonal(4.0),
            thruster_dump=ThrusterDumpConfig(),
            slew_momentum_enabled=True,
            enforce_capacity=True,
            planning_level=0.5,
        )
        assert config.dump_level_n_m_s == pytest.approx(2.0)
        assert config.capacity_enabled
        assert config.enabled

    @pytest.mark.parametrize(
        "settings",
        [{"thruster_dump": ThrusterDumpConfig()}, {"slew_momentum_enabled": True}],
    )
    def test_dumps_and_slew_momentum_still_need_some_capacity(
        self, settings: dict[str, object]
    ) -> None:
        with pytest.raises(ValidationError, match="capacity_n_m_s or wheels"):
            StoredMomentumConfig(**settings)  # type: ignore[arg-type]

    def test_wheels_round_trip_through_json(self) -> None:
        config = StoredMomentumConfig(wheels=_pyramid(4.0))
        again = StoredMomentumConfig.model_validate_json(config.model_dump_json())
        assert again.capacity_n_m_s is None
        assert [w.name for w in again.wheels] == [w.name for w in config.wheels]
        for restored, original in zip(again.wheels, config.wheels, strict=True):
            assert restored.spin_axis == pytest.approx(original.spin_axis)
        assert again.total_capacity_n_m_s == pytest.approx(config.total_capacity_n_m_s)

    def test_planners_see_the_wheels_capacity(self) -> None:
        config = MissionConfig(
            spacecraft_bus=SpacecraftBus(
                attitude_control=AttitudeControlSystem(
                    stored_momentum=StoredMomentumConfig(
                        wheels=_orthogonal(4.0),
                        thruster_dump=ThrusterDumpConfig(),
                    )
                )
            )
        )
        forecast = MomentumForecast.__new__(MomentumForecast)
        forecast.config = config
        assert forecast.capacity_n_m_s == pytest.approx(4.0)
        assert forecast.level_n_m_s == pytest.approx(0.8 * 4.0)
