"""Runs with momentum wheels: each wheel's telemetry and its own limit."""

from datetime import timedelta
from types import SimpleNamespace

import numpy as np
import pytest

from conops import create_ditl
from conops.common import ACSMode
from conops.config import (
    AttitudeControlSystem,
    MissionConfig,
    MomentumWheel,
    SpacecraftBus,
    StoredMomentumConfig,
    ThrusterDumpConfig,
)
from conops.simulation.attitude import AttitudeExecutor
from conops.simulation.momentum_budget import MomentumBudget, MomentumLimitError

from .test_momentum_dumps import (
    BEGIN,
    END,
    _capacity_faults,
    _config,
    _held,
    _momentum_events,
)


def _wheels(capacity: float = 1.0) -> list[MomentumWheel]:
    return [
        MomentumWheel(name=f"RW{i + 1}", spin_axis=axis, capacity_n_m_s=capacity)
        for i, axis in enumerate([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    ]


def _run(**momentum: object):
    config = _config(thruster_dump=ThrusterDumpConfig(), **momentum)
    ditl = create_ditl(config, [_held(config)], BEGIN, END)
    assert ditl.calc()
    return ditl


def _dumps(ditl) -> int:
    modes = list(ditl.mode)
    return sum(
        later == ACSMode.DESAT and earlier != ACSMode.DESAT
        for earlier, later in zip(modes, modes[1:], strict=False)
    )


class TestWheelTelemetry:
    def test_each_wheels_momentum_is_recorded(self) -> None:
        records = _run(wheels=_wheels()).telemetry.housekeeping
        shares = records.wheel_momentum_n_m_s
        held = records.actuator_momentum_body_n_m_s
        stored = records.stored_momentum_body_n_m_s
        assert any(share is not None for share in shares)
        for share, actuator, whole in zip(shares, held, stored, strict=True):
            if share is None:
                continue
            assert len(share) == 3
            # Orthogonal wheels along the body axes hold a component each.
            assert share == pytest.approx(actuator if actuator is not None else whole)

    def test_no_wheels_records_none(self) -> None:
        records = _run(capacity_n_m_s=1.0).telemetry.housekeeping
        assert all(share is None for share in records.wheel_momentum_n_m_s)

    def test_an_explicit_capacity_still_records_the_wheels(self) -> None:
        records = _run(wheels=_wheels(), capacity_n_m_s=1.0).telemetry.housekeeping
        assert any(share is not None for share in records.wheel_momentum_n_m_s)


class TestWheelLimitsInARun:
    def test_planning_message_reports_utilisation_and_actual_stored_norm(self):
        ditl = _run(wheels=_wheels(1.0))
        events = [
            e for e in ditl.log.events if "has reached the planning" in e.description
        ]
        assert events
        for event in events:
            assert event.description.startswith("Fullest wheel at ")
            assert "planning level 80.0%" in event.description
            # Dispatch reads the tracker before this tick's housekeeping update.
            sample = next(
                hk
                for hk in ditl.telemetry.housekeeping
                if hk.timestamp.timestamp() == event.time - ditl.step_size
            )
            norm = sample.stored_momentum_norm_n_m_s
            assert f"stored momentum {norm:.3f} N m s" in event.description

    @pytest.mark.parametrize("slew_momentum", [False, True])
    def test_emergency_reports_real_norm_and_wheel_utilisation(self, slew_momentum):
        # Distinct components make the norm differ from the fullest wheel.
        config = _config(
            wheels=_wheels(),
            thruster_dump=ThrusterDumpConfig(),
            slew_momentum_enabled=slew_momentum,
        )
        ditl = create_ditl(config, [_held(config)], BEGIN, END)
        tracker = ditl._stored_momentum_tracker
        assert tracker is not None
        tracker.update(
            utime=BEGIN.timestamp(),
            position_eci_km=(7000, 0, 0),
            attitude_quaternion_eci_to_body=(1, 0, 0, 0),
        )
        tracker.start_from_eci((1.1, 0.9, 0.0))
        tracker.update(
            utime=BEGIN.timestamp(),
            position_eci_km=(7000, 0, 0),
            attitude_quaternion_eci_to_body=(1, 0, 0, 0),
            angular_velocity_body_rad_s=(0.0001, 0, 0),
        )
        stored = tracker.stored_momentum_norm_n_m_s
        demand = ditl._momentum_demand_n_m_s(tracker)
        assert stored != pytest.approx(demand)
        ditl._call_for_desaturation(BEGIN.timestamp())
        fault = _capacity_faults(ditl)[0]
        assert fault.metadata["stored_n_m_s"] == pytest.approx(stored)
        assert fault.metadata["wheel_utilisation"] == pytest.approx(demand)
        if slew_momentum:
            assert fault.metadata["actuator_n_m_s"] == pytest.approx(
                tracker.actuator_momentum_norm_n_m_s
            )
        assert f"Fullest wheel at {demand:.1%}" in fault.cause
        assert f"stored momentum {stored:.3f} N m s" in fault.cause
        emergency = next(e for e in _momentum_events(ditl) if "emergency" in e)
        assert f"Fullest wheel at {demand:.1%}" in emergency
        assert f"stored momentum {stored:.3f} N m s" in emergency

    def test_dumps_keep_every_wheel_under_its_capacity(self) -> None:
        ditl = _run(wheels=_wheels(1.0))
        assert _dumps(ditl) >= 1
        assert not _capacity_faults(ditl)
        stored = np.array(
            [
                s
                for s in ditl.telemetry.housekeeping.stored_momentum_body_n_m_s
                if s is not None
            ]
        )
        assert np.abs(stored).max() < 1.0

    def test_the_limit_depends_on_the_momentums_direction(self) -> None:
        """Momentum shared between wheels may exceed what any one holds: the
        norm passes the planning level a single figure would dump at."""
        wheels = _run(wheels=_wheels(1.0)).telemetry.housekeeping
        scalar = _run(capacity_n_m_s=1.0).telemetry.housekeeping
        peak = max(n for n in wheels.stored_momentum_norm_n_m_s if n is not None)
        single = max(n for n in scalar.stored_momentum_norm_n_m_s if n is not None)
        assert single < 1.0
        assert peak > single

    def test_an_explicit_capacity_overrides_the_wheels(self) -> None:
        both = _run(wheels=_wheels(1.0), capacity_n_m_s=1.0)
        scalar = _run(capacity_n_m_s=1.0)
        assert both.telemetry.housekeeping.stored_momentum_norm_n_m_s == (
            scalar.telemetry.housekeeping.stored_momentum_norm_n_m_s
        )
        assert _dumps(both) == _dumps(scalar)

    def test_enforcement_holds_every_wheel_within_its_capacity(self) -> None:
        ditl = _run(
            wheels=_wheels(5.0), enforce_capacity=True, slew_momentum_enabled=True
        )
        shares = np.array(
            [
                s
                for s in ditl.telemetry.housekeeping.wheel_momentum_n_m_s
                if s is not None
            ]
        )
        assert np.abs(shares).max() <= 5.0 * (1 + 1e-9)


def _fixed_orbit(timestamp, position=(7000.0, 0.0, 0.0)):
    """An ephemeris double at rest at ``position``, with the Sun along +x."""
    at_rest = SimpleNamespace(
        position=np.array([position], dtype=float), velocity=np.zeros((1, 3))
    )
    sun = SimpleNamespace(position=np.array([[1.5e8, 0.0, 0.0]]))
    return SimpleNamespace(
        timestamp=timestamp,
        gcrs_pv=SimpleNamespace(position=np.tile(position, (len(timestamp), 1))),
        gcrs_pv_at=lambda time: at_rest,
        itrs_pv_at=lambda time: at_rest,
        sun_pv_at=lambda time: sun,
    )


def _budget(initial: tuple[float, float, float], **momentum: object) -> MomentumBudget:
    epoch = BEGIN
    ephem = _fixed_orbit([epoch + timedelta(seconds=i) for i in range(0, 601, 2)])
    config = MissionConfig(
        spacecraft_bus=SpacecraftBus(
            inertia_tensor_body_kg_m2=np.diag((10.0, 10.0, 10.0)),
            attitude_control=AttitudeControlSystem(
                max_slew_rate=2.0,
                slew_acceleration=1.0,
                stored_momentum=StoredMomentumConfig(
                    gravity_gradient_enabled=False,
                    enforce_capacity=True,
                    initial_momentum_body_n_m_s=initial,
                    **momentum,  # type: ignore[arg-type]
                ),
            ),
        )
    )
    executor = AttitudeExecutor(epoch.timestamp(), (0.0, 0.0, 0.0))
    return MomentumBudget(config, ephem, executor.state)


class TestEnforcementBudgetWheelLimits:
    @pytest.mark.parametrize("wheel_limits", [False, True])
    def test_initial_coast_and_forecasts_share_the_complete_signature(
        self, wheel_limits
    ):
        settings = {"wheels": _wheels()} if wheel_limits else {"capacity_n_m_s": 1.0}
        budget = _budget((0.3, 0.4, 0.0), **settings)
        path = budget.current.path
        assert path is not None
        assert len(path.signature) == 4
        assert (path.signature[3] is not None) is wheel_limits
        forecast = budget.forecast(
            path.state_at, budget.current.utime + 7, trajectory=None
        )
        assert forecast.path is path
        assert budget.current.utime == BEGIN.timestamp()
        # A change to the wheel rows alone must invalidate the cached proof.
        if wheel_limits:
            budget.wheel_limit_rows = budget.wheel_limit_rows.copy() * 1.01
            changed = budget.forecast(path.state_at, forecast.utime, trajectory=None)
            assert changed.path is not path
            assert changed.path.signature[3] == budget.wheel_limit_rows.tobytes()

    def test_the_budget_takes_its_limits_from_the_wheels(self) -> None:
        budget = _budget((0.0, 0.0, 0.0), wheels=_wheels(1.0))
        assert budget.capacity == pytest.approx(1.0)
        assert budget.wheels is not None
        assert budget.live.limit_by_wheels

    def test_shared_momentum_fits_the_wheels_but_not_one_figure(self) -> None:
        shared = (0.8, 0.8, 0.0)
        budget = _budget(shared, wheels=_wheels(1.0))
        assert budget.current.first_violation is None
        assert budget.current.peak_n_m_s == pytest.approx(0.8)
        with pytest.raises(MomentumLimitError):
            _budget(shared, capacity_n_m_s=1.0)

    def test_a_full_wheel_is_a_violation(self) -> None:
        with pytest.raises(MomentumLimitError):
            _budget((0.0, 1.05, 0.0), wheels=_wheels(1.0))

    def test_an_explicit_capacity_overrides_the_wheels(self) -> None:
        budget = _budget((0.0, 0.0, 0.0), wheels=_wheels(1.0), capacity_n_m_s=3.0)
        assert budget.capacity == 3.0
        assert budget.wheels is None
        assert not budget.live.limit_by_wheels
