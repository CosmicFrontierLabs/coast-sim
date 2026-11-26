"""Momentum wheels: their configuration and the total capacity they give."""

import numpy as np
import pytest
from pydantic import ValidationError

from conops.common import attitude_to_quat
from conops.common.quaternion_curve import QuaternionHermite
from conops.config import (
    AttitudeControlSystem,
    MissionConfig,
    MomentumWheel,
    SpacecraftBus,
    StoredMomentumConfig,
    ThrusterDumpConfig,
    WheelArray,
    wheel_array_capacity_n_m_s,
)
from conops.config import momentum as momentum_config
from conops.schedulers.momentum import MomentumForecast
from conops.simulation.momentum import StoredMomentumTracker, tracker_from_config


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


def _shares(wheels: list[MomentumWheel], momentum: np.ndarray) -> np.ndarray:
    """Each wheel's momentum, by the least-squares solution for the wheels'
    fractions of their capacities: one row a momentum."""
    axes = np.array([w.spin_axis for w in wheels])
    capacities = np.array([w.capacity_n_m_s for w in wheels])
    fractions = np.linalg.pinv(axes.T * capacities) @ np.atleast_2d(momentum).T
    return (fractions * capacities[:, None]).T


def _least_reach(wheels: list[MomentumWheel]) -> float:
    """The least momentum, over many directions, at which a wheel is full."""
    rng = np.random.default_rng(0)
    directions = rng.normal(size=(200_000, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    capacities = np.array([w.capacity_n_m_s for w in wheels])
    fullest = np.abs(_shares(wheels, directions) / capacities).max(axis=1)
    return float(1.0 / fullest.max())


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
    def test_wheel_geometry_is_built_once_across_capacity_reads(self, monkeypatch):
        from unittest.mock import Mock

        build = Mock(wraps=WheelArray)
        monkeypatch.setattr(momentum_config, "WheelArray", build)
        config = StoredMomentumConfig(wheels=_orthogonal())
        array = config.wheel_array
        for _ in range(10):
            assert config.wheel_array is array
            assert config.wheel_limits is array
            assert config.wheel_capacity_n_m_s == pytest.approx(4.0)
            assert config.total_capacity_n_m_s == pytest.approx(4.0)
            assert not config.capacity_enabled
        build.assert_called_once()

    @pytest.mark.parametrize(
        "change", ["capacity", "axis", "append", "replace", "name"]
    )
    def test_wheel_cache_follows_nested_and_list_edits(self, change):
        config = StoredMomentumConfig(wheels=_orthogonal())
        before = config.wheel_array
        if change == "capacity":
            config.wheels[0].capacity_n_m_s = 1.0
        elif change == "axis":
            config.wheels[0].spin_axis = (1.0, 1.0, 0.0)
        elif change == "append":
            config.wheels.append(MomentumWheel(spin_axis=(1, 1, 1), capacity_n_m_s=8))
        elif change == "replace":
            config.wheels = _pyramid()
        else:
            config.wheels[0].name = "Renamed"
        array = config.wheel_array
        assert array is not None and array is not before
        expected = WheelArray(config.wheels)
        assert array.names == expected.names
        assert array.distribution == pytest.approx(expected.distribution)
        assert config.total_capacity_n_m_s == pytest.approx(expected.capacity_n_m_s)

    def test_invalid_geometry_is_cached_and_recovers_after_an_edit(self, monkeypatch):
        from unittest.mock import Mock

        build = Mock(wraps=WheelArray)
        monkeypatch.setattr(momentum_config, "WheelArray", build)
        config = StoredMomentumConfig(wheels=_orthogonal()[:2], capacity_n_m_s=2)
        for _ in range(10):
            assert config.wheel_array is None
            assert config.wheel_capacity_n_m_s == 0.0
        build.assert_called_once()
        config.wheels.append(_orthogonal()[2])
        assert config.wheel_array is not None
        assert build.call_count == 2
        config.wheels.clear()
        assert config.wheel_array is None
        assert config.wheel_capacity_n_m_s is None

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


class TestWheelArray:
    @pytest.mark.parametrize("wheels", [_orthogonal(), _pyramid(), _pyramid(4.0, 20.0)])
    def test_the_wheels_shares_make_up_the_momentum(
        self, wheels: list[MomentumWheel]
    ) -> None:
        array = WheelArray(wheels)
        momentum = np.array([0.7, -1.3, 2.1])
        shares = array.wheel_momenta_n_m_s(momentum)
        axes = np.array([w.spin_axis for w in wheels])
        assert shares @ axes == pytest.approx(momentum)
        assert shares == pytest.approx(_shares(wheels, momentum)[0])

    def test_orthogonal_wheels_each_hold_their_own_axis(self) -> None:
        array = WheelArray(_orthogonal(4.0))
        assert array.wheel_momenta_n_m_s((1.0, -2.0, 3.0)) == pytest.approx(
            (1.0, -2.0, 3.0)
        )
        assert array.utilisation((1.0, -2.0, 3.0)) == pytest.approx(0.75)

    def test_a_larger_wheel_takes_more_of_a_shared_axis(self) -> None:
        wheels = _orthogonal(4.0) + [
            MomentumWheel(name="Big", spin_axis=(1, 0, 0), capacity_n_m_s=8.0)
        ]
        shares = WheelArray(wheels).wheel_momenta_n_m_s((5.0, 0.0, 0.0))
        assert shares == pytest.approx((1.0, 0.0, 0.0, 4.0))
        assert shares[0] / 4.0 < shares[3] / 8.0

    def test_load_is_the_norm_only_along_the_weakest_direction(self) -> None:
        array = WheelArray(_orthogonal(4.0))
        assert array.capacity_n_m_s == pytest.approx(4.0)
        # Along a wheel's axis the load is the norm; between axes it is less.
        assert array.load_n_m_s((4.0, 0.0, 0.0)) == pytest.approx(4.0)
        assert array.load_n_m_s((4.0, 4.0, 4.0)) == pytest.approx(4.0)
        assert np.linalg.norm((4.0, 4.0, 4.0)) > 6.9

    @pytest.mark.parametrize("wheels", [_orthogonal(), _pyramid(), _pyramid(2.0, 50.0)])
    def test_load_never_exceeds_the_norm_nor_changes_faster(
        self, wheels: list[MomentumWheel]
    ) -> None:
        array = WheelArray(wheels)
        rng = np.random.default_rng(1)
        first = rng.normal(size=(500, 3))
        second = first + rng.normal(scale=0.3, size=(500, 3))
        for a, b in zip(first, second, strict=True):
            assert array.load_n_m_s(a) <= np.linalg.norm(a) * (1 + 1e-12)
            assert abs(array.load_n_m_s(a) - array.load_n_m_s(b)) <= np.linalg.norm(
                a - b
            ) * (1 + 1e-12)

    def test_load_reaches_capacity_exactly_when_a_wheel_is_full(self) -> None:
        wheels = _pyramid(4.0)
        array = WheelArray(wheels)
        direction = np.array([0.2, 0.9, -0.4])
        direction /= np.linalg.norm(direction)
        full = direction / array.utilisation(direction)
        assert array.load_n_m_s(full) == pytest.approx(array.capacity_n_m_s)
        assert np.abs(array.wheel_momenta_n_m_s(full)).max() == pytest.approx(4.0)

    def test_wheels_must_span_three_axes(self) -> None:
        with pytest.raises(ValueError, match="do not span three dimensions"):
            WheelArray(_orthogonal()[:2])
        planar = [
            MomentumWheel(spin_axis=a, capacity_n_m_s=1.0)
            for a in [(1, 0, 0), (0, 1, 0), (1, 1, 0)]
        ]
        with pytest.raises(ValueError, match="do not span three dimensions"):
            WheelArray(planar)


class TestWheelLimitsInTheConfig:
    def test_wheels_alone_set_the_limits(self) -> None:
        config = StoredMomentumConfig(wheels=_orthogonal())
        assert config.wheel_array is not None
        assert config.wheel_limits is not None

    def test_an_explicit_capacity_replaces_the_wheels_limits(self) -> None:
        config = StoredMomentumConfig(wheels=_orthogonal(), capacity_n_m_s=2.0)
        assert config.wheel_array is not None
        assert config.wheel_limits is None

    def test_wheels_that_do_not_span_three_axes_are_not_shared_among(self) -> None:
        config = StoredMomentumConfig(wheels=_orthogonal()[:2], capacity_n_m_s=2.0)
        assert config.wheel_array is None
        assert config.wheel_limits is None

    def test_no_wheels(self) -> None:
        config = StoredMomentumConfig(capacity_n_m_s=2.0)
        assert config.wheel_array is None
        assert config.wheel_limits is None


def _sample(tracker: StoredMomentumTracker, attitude=(0.0, 0.0, 0.0), rate=None):
    return tracker.update(
        utime=0.0,
        position_eci_km=(7000.0, 0.0, 0.0),
        attitude_quaternion_eci_to_body=attitude_to_quat(*attitude),
        angular_velocity_body_rad_s=rate,
    )


class TestTrackerWheelLoads:
    INERTIA = np.diag([100.0, 100.0, 100.0])

    def _tracker(
        self, initial=(3.0, 3.0, 3.0), *, limit: bool = True, wheels: bool = True
    ) -> StoredMomentumTracker:
        return StoredMomentumTracker(
            self.INERTIA,
            initial,
            sources=[],
            wheels=WheelArray(_orthogonal(4.0)) if wheels else None,
            limit_by_wheels=limit,
        )

    def test_without_wheels_the_load_is_the_norm(self) -> None:
        tracker = self._tracker(wheels=False)
        assert tracker.stored_momentum_load_n_m_s == pytest.approx(np.sqrt(27.0))
        sample = _sample(tracker)
        assert sample.wheel_momentum_n_m_s is None
        assert sample.stored_momentum_load_n_m_s == sample.stored_momentum_norm_n_m_s

    def test_wheels_without_limits_report_shares_and_keep_the_norm(self) -> None:
        sample = _sample(self._tracker(limit=False))
        assert sample.wheel_momentum_n_m_s == pytest.approx((3.0, 3.0, 3.0))
        assert sample.stored_momentum_load_n_m_s == pytest.approx(np.sqrt(27.0))

    def test_wheel_limits_load_the_fullest_wheel(self) -> None:
        tracker = self._tracker()
        assert tracker.stored_momentum_load_n_m_s == pytest.approx(3.0)
        sample = _sample(tracker)
        assert sample.stored_momentum_norm_n_m_s == pytest.approx(np.sqrt(27.0))
        assert sample.stored_momentum_load_n_m_s == pytest.approx(3.0)
        assert tracker.stored_momentum_load_n_m_s == pytest.approx(3.0)
        assert tracker.actuator_momentum_load_n_m_s == pytest.approx(3.0)

    def test_the_load_follows_the_attitude(self) -> None:
        tracker = self._tracker(initial=(3.0, 0.0, 0.0))
        _sample(tracker)
        assert tracker.stored_momentum_load_n_m_s == pytest.approx(3.0)
        # Turned 45 degrees about the boresight-normal axis, two wheels share it.
        turned = tracker.update(
            utime=1.0,
            position_eci_km=(7000.0, 0.0, 0.0),
            attitude_quaternion_eci_to_body=attitude_to_quat(45.0, 0.0, 0.0),
        )
        assert turned.stored_momentum_norm_n_m_s == pytest.approx(3.0)
        assert turned.stored_momentum_load_n_m_s == pytest.approx(3.0 / np.sqrt(2.0))
        assert sorted(np.abs(turned.wheel_momentum_n_m_s)) == pytest.approx(
            (0.0, 3.0 / np.sqrt(2.0), 3.0 / np.sqrt(2.0)), abs=1e-12
        )

    def test_the_bodys_own_momentum_comes_out_of_the_wheels(self) -> None:
        tracker = self._tracker(initial=(0.0, 0.0, 0.0))
        sample = _sample(tracker, rate=(0.01, 0.0, 0.0))
        # The wheels start empty, and the spacecraft's total is the body's.
        assert sample.actuator_momentum_load_n_m_s == pytest.approx(0.0, abs=1e-12)
        later = tracker.update(
            utime=1.0,
            position_eci_km=(7000.0, 0.0, 0.0),
            attitude_quaternion_eci_to_body=attitude_to_quat(0.0, 0.0, 0.0),
            angular_velocity_body_rad_s=(0.0, 0.0, 0.0),
        )
        assert later.wheel_momentum_n_m_s == pytest.approx((1.0, 0.0, 0.0))
        assert later.actuator_momentum_load_n_m_s == pytest.approx(1.0)
        assert tracker.actuator_momentum_load_n_m_s == pytest.approx(1.0)

    def test_unloading_scales_the_load(self) -> None:
        tracker = self._tracker()
        _sample(tracker)
        tracker.unload(0.5)
        assert tracker.stored_momentum_load_n_m_s == pytest.approx(1.5)
        tracker.unload()
        assert tracker.stored_momentum_load_n_m_s == 0.0

    def test_the_norm_stands_in_until_a_sample_gives_the_attitude(self) -> None:
        tracker = self._tracker()
        _sample(tracker)
        state = tracker.state
        other = self._tracker()
        other.restore(state)
        assert other.stored_momentum_load_n_m_s == pytest.approx(np.sqrt(27.0))
        started = self._tracker()
        started.start_from_eci((3.0, 3.0, 3.0))
        assert started.stored_momentum_load_n_m_s == pytest.approx(np.sqrt(27.0))

    def test_copies_and_assignment_carry_the_attitude(self) -> None:
        tracker = self._tracker()
        _sample(tracker)
        assert tracker.copy().stored_momentum_load_n_m_s == pytest.approx(3.0)
        other = self._tracker(initial=(0.0, 0.0, 0.0))
        other.assign(tracker)
        assert other.stored_momentum_load_n_m_s == pytest.approx(3.0)

    @pytest.mark.parametrize("explicit", [None, 2.0])
    def test_the_configuration_builds_the_tracker(self, explicit: float | None) -> None:
        config = MissionConfig(
            spacecraft_bus=SpacecraftBus(
                inertia_tensor_body_kg_m2=self.INERTIA.tolist(),
                attitude_control=AttitudeControlSystem(
                    stored_momentum=StoredMomentumConfig(
                        wheels=_orthogonal(4.0),
                        capacity_n_m_s=explicit,
                        gravity_gradient_enabled=True,
                    )
                ),
            )
        )
        tracker = tracker_from_config(config)
        assert tracker is not None
        assert tracker.wheels is not None
        assert tracker.limit_by_wheels is (explicit is None)


class TestTrackingCurveWheelLimits:
    def _curve(self) -> QuaternionHermite:
        return QuaternionHermite(
            first=tuple(attitude_to_quat(0, 0, 0)),
            last=tuple(attitude_to_quat(0.005, 0, 0)),
            first_rate=(0, 0, 0),
            last_rate=(0, 0, 0),
            duration=0.2,
        )

    def test_wheel_limits_admit_what_the_norm_rejects(self) -> None:
        # 0.8 along each of two wheels: within each wheel's 1.0, beyond a
        # 1.0 sphere.
        momentum = np.array([[0.0, 0.8, 0.8], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        inertia = np.eye(3)
        rows = WheelArray(_orthogonal(1.0)).fractions
        curve = self._curve()
        assert not curve.within_momentum_limit(0.0, 1.0, inertia, momentum, 1.0)
        assert curve.within_momentum_limit(0.0, 1.0, inertia, momentum, 1.0, rows)

    def test_wheel_limits_reject_a_full_wheel(self) -> None:
        momentum = np.array([[0.0, 0.0, 1.05], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        rows = WheelArray(_orthogonal(1.0)).fractions
        assert not self._curve().within_momentum_limit(
            0.0, 1.0, np.eye(3), momentum, 1.0, rows
        )
