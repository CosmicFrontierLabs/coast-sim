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

from .test_momentum_dumps import BEGIN, END, _capacity_faults, _config, _held


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


def _budget(initial: tuple[float, float, float], **momentum: object) -> MomentumBudget:
    epoch = BEGIN
    ephem = SimpleNamespace(
        timestamp=[epoch + timedelta(seconds=i) for i in range(0, 601, 2)],
        gcrs_pv=SimpleNamespace(position=np.tile([7000.0, 0.0, 0.0], (301, 1))),
    )
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
