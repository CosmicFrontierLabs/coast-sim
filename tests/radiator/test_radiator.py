"""Tests for radiator configuration and exposure modelling."""

from unittest.mock import Mock, patch

import pytest

from conops.config import (
    Constraint,
    Radiator,
    RadiatorConfiguration,
    RadiatorOrientation,
)


class TestRadiatorOrientation:
    def test_normal_must_be_unit_vector(self) -> None:
        with pytest.raises(ValueError):
            RadiatorOrientation(normal=(2.0, 0.0, 0.0))


class TestRadiatorConfiguration:
    def test_hard_violation_count(self) -> None:
        hard_constraint = Constraint()
        rad = Radiator(
            name="R1",
            orientation=RadiatorOrientation(normal=(1.0, 0.0, 0.0)),
            hard_constraint=hard_constraint,
        )
        cfg = RadiatorConfiguration(radiators=[rad])
        with patch.object(Constraint, "in_constraint", return_value=True):
            assert cfg.radiators_violating_hard_constraints(0.0, 0.0, 1000.0, 0.0) == 1

    def test_exposure_metrics_and_heat(self) -> None:
        ephem = Mock()
        ephem.sun_ra_deg = [0.0]
        ephem.sun_dec_deg = [0.0]
        ephem.earth_ra_deg = [90.0]
        ephem.earth_dec_deg = [0.0]
        ephem.index = Mock(return_value=0)

        rad = Radiator(
            name="R1",
            width_m=1.0,
            height_m=1.0,
            orientation=RadiatorOrientation(normal=(1.0, 0.0, 0.0)),
            efficiency=1.0,
            emissivity=1.0,
            dissipation_coefficient_w_per_m2=100.0,
            absorptivity=1.0,
            solar_constant_w_per_m2=1361.0,
            sun_loading_factor=1.0,
            earth_loading_factor=0.0,
        )

        # Keep sun exposure active for test determinism.
        rad._eclipse_constraint = Mock()
        rad._eclipse_constraint.in_constraint = Mock(return_value=False)

        cfg = RadiatorConfiguration(radiators=[rad])
        metrics = cfg.exposure_metrics(
            ra_deg=0.0,
            dec_deg=0.0,
            utime=1000.0,
            ephem=ephem,
            roll_deg=0.0,
        )

        sun_exp = metrics["sun_exposure"]
        earth_exp = metrics["earth_exposure"]
        heat = metrics["heat_dissipation_w"]
        assert isinstance(sun_exp, float)
        assert isinstance(earth_exp, float)
        assert isinstance(heat, float)
        assert sun_exp == pytest.approx(1.0)
        assert earth_exp == pytest.approx(0.0)
        # Emitted flux is capped to 100 W/m^2 while absorbed solar flux is 1361 W/m^2.
        assert heat == pytest.approx(-1261.0)
