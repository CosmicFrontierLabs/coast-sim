"""The momentum the body carries while it slews, I·ω, and the slew rate
fractions a slew is flown slower at where that momentum does not fit."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from pydantic import ValidationError
from rust_ephem import TLEEphemeris

from conops import AttitudeControlSystem, MissionConfig, SpacecraftBus
from conops.common.vector import attitude_to_quat
from conops.config import StoredMomentumConfig
from conops.simulation.attitude import AttitudeTrajectory, slew_angular_velocity
from conops.simulation.momentum import StoredMomentumTracker
from conops.simulation.slew import Slew
from conops.targets import Pointing

BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
END = BEGIN + timedelta(hours=1)


def _config(**momentum: object) -> MissionConfig:
    config = MissionConfig(
        spacecraft_bus=SpacecraftBus(
            inertia_tensor_body_kg_m2=np.diag([1000.0, 1000.0, 1000.0]),
            attitude_control=AttitudeControlSystem(
                max_slew_rate=0.5,
                slew_acceleration=0.25,
                settle_time=30.0,
                stored_momentum=StoredMomentumConfig(**momentum),  # type: ignore[arg-type]
            ),
        )
    )
    config.constraint.ephem = TLEEphemeris(
        tle="examples/example.tle", begin=BEGIN, end=END, step_size=10
    )
    return config


class TestTracker:
    def test_the_wheels_hold_the_stored_momentum_less_the_bodys_own(self) -> None:
        """The momentum tracked is the whole spacecraft's; the body turning
        at ω carries I·ω of it, and the wheels hold the rest as a vector
        difference, not a sum of norms."""
        tracker = StoredMomentumTracker(np.eye(3) * 1000, (0, 0, 5), sources=[])
        q = attitude_to_quat(0, 0, 0)
        sample = None
        for time, rate in [(0, (0, 0, 0)), (1, (0, 0, 0.01))]:
            sample = tracker.update(
                utime=time,
                position_eci_km=[7000, 0, 0],
                attitude_quaternion_eci_to_body=q,
                angular_velocity_body_rad_s=rate,
            )
        assert sample is not None
        assert sample.stored_momentum_norm_n_m_s == pytest.approx(5)
        assert sample.actuator_momentum_norm_n_m_s == pytest.approx(5)
        assert sample.actuator_momentum_body_n_m_s == pytest.approx((0, 0, -5))
        assert tracker.actuator_momentum_norm_n_m_s == pytest.approx(5)

    def test_a_body_already_turning_carries_its_own_momentum(self) -> None:
        """The configured initial momentum is the wheels'; a body turning
        at the first sample adds I·ω to the whole, so the wheels still hold
        what was configured."""
        tracker = StoredMomentumTracker(np.eye(3) * 1000, (1, 2, 3), sources=[])
        sample = tracker.update(
            utime=0,
            position_eci_km=[7000, 0, 0],
            attitude_quaternion_eci_to_body=attitude_to_quat(20, 30, 40),
            angular_velocity_body_rad_s=[0.01, 0.02, 0.03],
        )
        assert sample.actuator_momentum_body_n_m_s == pytest.approx((1, 2, 3))
        assert sample.stored_momentum_norm_n_m_s == pytest.approx(
            np.linalg.norm([11, 22, 33])
        )

    def test_without_a_rate_the_wheels_hold_it_all(self) -> None:
        tracker = StoredMomentumTracker(np.eye(3) * 1000, (0, 0, 5), sources=[])
        sample = tracker.update(
            utime=0,
            position_eci_km=[7000, 0, 0],
            attitude_quaternion_eci_to_body=attitude_to_quat(0, 0, 0),
        )
        assert sample.actuator_momentum_body_n_m_s is None
        assert sample.actuator_momentum_norm_n_m_s is None
        assert tracker.actuator_momentum_norm_n_m_s == pytest.approx(5)

    def test_a_dump_and_a_restore_drop_the_last_rate(self) -> None:
        tracker = StoredMomentumTracker(np.eye(3) * 1000, (0, 0, 5), sources=[])
        tracker.update(
            utime=0,
            position_eci_km=[7000, 0, 0],
            attitude_quaternion_eci_to_body=attitude_to_quat(0, 0, 0),
            angular_velocity_body_rad_s=(0, 0, 0.01),
        )
        state = tracker.state
        assert tracker.actuator_momentum_norm_n_m_s == pytest.approx(5)
        tracker.restore(state)
        assert tracker.actuator_momentum_norm_n_m_s == pytest.approx(15)
        tracker.unload()
        assert tracker.actuator_momentum_norm_n_m_s == 0.0


class TestConfiguration:
    def test_slew_momentum_needs_a_capacity_and_an_inertia(self) -> None:
        with pytest.raises(ValidationError, match="capacity_n_m_s"):
            StoredMomentumConfig(slew_momentum_enabled=True)
        with pytest.raises(ValidationError, match="inertia_tensor_body_kg_m2"):
            SpacecraftBus(
                attitude_control=AttitudeControlSystem(
                    stored_momentum=StoredMomentumConfig(
                        slew_momentum_enabled=True, capacity_n_m_s=10.0
                    )
                )
            )
        config = StoredMomentumConfig(slew_momentum_enabled=True, capacity_n_m_s=10.0)
        assert config.enabled
        assert not StoredMomentumConfig(capacity_n_m_s=10.0).enabled

    @pytest.mark.parametrize(
        "fractions", [(), (0.5, 0.25), (1.0, 1.0), (1.0, 0.5, 0.75), (1.0, 0.0)]
    )
    def test_slew_rate_fractions_start_at_one_and_descend(
        self, fractions: tuple[float, ...]
    ) -> None:
        with pytest.raises(ValidationError, match="slew_rate_fractions"):
            StoredMomentumConfig(slew_rate_fractions=fractions)
        assert StoredMomentumConfig(slew_rate_fractions=(1.0,)).slew_rate_fractions == (
            1.0,
        )

    def test_limits_at_a_rate_fraction_scale_the_rates_only(self) -> None:
        acs = AttitudeControlSystem(
            max_slew_rate=0.5,
            slew_acceleration=0.25,
            settle_time=30.0,
            max_slew_rate_body=(0.4, 0.5, 0.6),
        )
        assert acs.at_rate_fraction(1.0) is acs
        half = acs.at_rate_fraction(0.5)
        assert half.max_slew_rate == 0.25
        assert half.max_slew_rate_body == (0.2, 0.25, 0.3)
        assert half.slew_acceleration == 0.25
        assert half.settle_time == 30.0
        with pytest.raises(ValueError, match="fraction"):
            acs.at_rate_fraction(0.0)


class TestSlew:
    def _slew(self, config: MissionConfig, fraction: float) -> Slew:
        slew = Slew(config=config)
        slew.slewstart = BEGIN.timestamp()
        slew.startra, slew.startdec, slew.startroll = 0.0, 0.0, 0.0
        slew.endra, slew.enddec, slew.endroll = 60.0, 0.0, 0.0
        slew.rate_fraction = fraction
        slew.calc_slewtime()
        return slew

    def test_a_slower_slew_takes_longer_along_the_same_path(self) -> None:
        config = _config()
        full, half = self._slew(config, 1.0), self._slew(config, 0.5)
        assert half.slewdist == pytest.approx(full.slewdist)
        assert half.slewtime > full.slewtime
        # The motion time doubles less than the whole, which holds the settling.
        settle = config.spacecraft_bus.attitude_control.settle_time
        assert half.slewtime - settle == pytest.approx(
            2 * (full.slewtime - settle), abs=4
        )
        midway = full.slewstart + (full.slewtime - settle) / 2
        assert full.attitude(midway)[0] == pytest.approx(30.0, abs=1.0)
        assert half.attitude(midway)[0] < 20.0

    def test_the_body_rate_along_a_slew_peaks_at_the_fraction(self) -> None:
        """The rate the tracker is given along a slew is the ACS's for it:
        zero before and after, the capped rate at its fastest."""
        config = _config()
        for fraction in (1.0, 0.5):
            slew = self._slew(config, fraction)
            rate = slew_angular_velocity(slew)
            assert rate(slew.slewstart - 1.0) == pytest.approx((0, 0, 0))
            assert rate(slew.slewend) == pytest.approx((0, 0, 0))
            times = np.linspace(slew.slewstart, slew.slewend, 500)
            peak = max(float(np.linalg.norm(rate(t))) for t in times)
            assert peak == pytest.approx(np.deg2rad(0.5 * fraction), rel=1e-6)
            trajectory = AttitudeTrajectory.from_slew(slew, slew.motion_limits)
            assert trajectory.end == pytest.approx(slew.slewend)

    def test_a_slew_that_goes_nowhere_has_no_rate(self) -> None:
        config = _config()
        slew = self._slew(config, 1.0)
        slew.endra = 0.0
        slew.calc_slewtime()
        assert slew.slewtime == 0
        assert slew_angular_velocity(slew)(slew.slewstart) == pytest.approx((0, 0, 0))

    def test_a_plan_entry_estimates_its_slew_at_its_fraction(self) -> None:
        config = _config()
        target = Pointing(config=config, obsid=1, ra=60.0, dec=0.0, roll=0.0)
        full = target.calc_slewtime(0.0, 0.0, 0.0)
        target.slew_rate_fraction = 0.5
        assert target.calc_slewtime(0.0, 0.0, 0.0) > full
        assert target.model_dump()["slew_rate_fraction"] == 0.5
        target.slew_rate_fraction = 1.0
        assert "slew_rate_fraction" not in target.model_dump()
