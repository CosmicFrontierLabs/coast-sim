"""Tests for radiator configuration and exposure modelling."""

from unittest.mock import Mock, PropertyMock, patch

import numpy as np
import pytest

from conops.config import (
    Constraint,
    Radiator,
    RadiatorConfiguration,
    RadiatorOrientation,
)
from conops.config.radiator import _normal_to_euler_deg


class TestRadiatorOrientation:
    def test_normal_must_be_unit_vector(self) -> None:
        with pytest.raises(ValueError):
            RadiatorOrientation(normal=(2.0, 0.0, 0.0))


class TestRadiator:
    def test_normal_to_euler_converts_expected_axes(self) -> None:
        roll, pitch, yaw = _normal_to_euler_deg((0.0, 1.0, 0.0))
        assert roll == pytest.approx(0.0)
        assert pitch == pytest.approx(0.0)
        assert yaw == pytest.approx(90.0)

    def test_set_ephem_no_hard_constraint_noop(self) -> None:
        rad = Radiator()
        rad.set_ephem(Mock())
        assert rad.hard_constraint is None

    def test_hard_constraint_with_offset_none_when_no_hard_constraint(self) -> None:
        rad = Radiator()
        assert rad._hard_constraint_with_offset is None

    def test_hard_constraint_with_offset_none_when_base_constraint_missing(
        self,
    ) -> None:
        hard_constraint = Constraint()
        hard_constraint.constraint = None
        rad = Radiator(hard_constraint=hard_constraint)
        assert rad._hard_constraint_with_offset is None

    def test_in_hard_constraint_false_when_no_constraint(self) -> None:
        rad = Radiator()
        assert rad.in_hard_constraint(10.0, -5.0, 1000.0, 7.0) is False

    def test_in_hard_constraint_uses_fallback_constraint_when_no_offset(self) -> None:
        hard_constraint = Constraint()
        hard_constraint.constraint = None
        rad = Radiator(hard_constraint=hard_constraint)

        with patch.object(Constraint, "in_constraint", return_value=True) as patched:
            assert rad.in_hard_constraint(10.0, -5.0, 1000.0, 12.0) is True
            patched.assert_called_once_with(
                10.0,
                -5.0,
                1000.0,
                target_roll=12.0,
            )

    def test_in_hard_constraint_requires_ephem_with_offset_constraint(self) -> None:
        hard_constraint = Constraint(ephem=None)
        hard_constraint.constraint = Mock()
        hard_constraint.constraint.boresight_offset = Mock(return_value=Mock())
        rad = Radiator(
            orientation=RadiatorOrientation(normal=(0.0, 1.0, 0.0)),
            hard_constraint=hard_constraint,
        )

        with pytest.raises(AssertionError, match="Ephemeris must be set"):
            rad.in_hard_constraint(0.0, 0.0, 1000.0)

    def test_in_hard_constraint_with_offset_path(self) -> None:
        base_constraint = Mock()
        offset_constraint = Mock()
        offset_constraint.in_constraint = Mock(return_value=True)
        base_constraint.boresight_offset = Mock(return_value=offset_constraint)

        hard_constraint = Constraint()
        object.__setattr__(hard_constraint, "ephem", Mock())
        hard_constraint.constraint = base_constraint

        rad = Radiator(
            orientation=RadiatorOrientation(normal=(0.0, 1.0, 0.0)),
            hard_constraint=hard_constraint,
        )

        assert rad.in_hard_constraint(1.0, 2.0, 1000.0, 3.0) is True
        offset_constraint.in_constraint.assert_called_once()

    def test_exposure_factors_zeroes_sun_when_in_eclipse(self) -> None:
        ephem = Mock()
        ephem.sun_ra_deg = [0.0]
        ephem.sun_dec_deg = [0.0]
        ephem.earth_ra_deg = [90.0]
        ephem.earth_dec_deg = [0.0]
        ephem.index = Mock(return_value=0)

        rad = Radiator(orientation=RadiatorOrientation(normal=(1.0, 0.0, 0.0)))
        rad._eclipse_constraint = Mock()
        rad._eclipse_constraint.in_constraint = Mock(return_value=True)

        sun_exp, earth_exp = rad.exposure_factors(
            ra_deg=0.0,
            dec_deg=0.0,
            utime=1000.0,
            ephem=ephem,
            roll_deg=0.0,
        )

        assert sun_exp == pytest.approx(0.0)
        assert earth_exp == pytest.approx(0.0)

    def test_exposure_factors_zero_sun_norm(self) -> None:
        """When scbodyvector returns a zero sun vector, sun_exposure stays 0."""
        ephem = Mock()
        ephem.sun_ra_deg = [0.0]
        ephem.sun_dec_deg = [0.0]
        ephem.earth_ra_deg = [90.0]
        ephem.earth_dec_deg = [0.0]
        ephem.index = Mock(return_value=0)

        rad = Radiator(orientation=RadiatorOrientation(normal=(1.0, 0.0, 0.0)))
        rad._eclipse_constraint = Mock()
        rad._eclipse_constraint.in_constraint = Mock(return_value=False)

        zero = np.zeros(3)
        normal_body = np.array([0.0, 0.0, 1.0])
        with patch(
            "conops.config.radiator.scbodyvector", side_effect=[zero, normal_body]
        ):
            sun_exp, earth_exp = rad.exposure_factors(
                ra_deg=0.0, dec_deg=0.0, utime=1000.0, ephem=ephem, roll_deg=0.0
            )

        assert sun_exp == pytest.approx(0.0)
        # earth still contributes (normal dot [0,0,1] = 0 here, but path is covered)
        assert isinstance(earth_exp, float)

    def test_exposure_factors_zero_earth_norm(self) -> None:
        """When scbodyvector returns a zero earth vector, earth_exposure stays 0."""
        ephem = Mock()
        ephem.sun_ra_deg = [0.0]
        ephem.sun_dec_deg = [0.0]
        ephem.earth_ra_deg = [90.0]
        ephem.earth_dec_deg = [0.0]
        ephem.index = Mock(return_value=0)

        rad = Radiator(orientation=RadiatorOrientation(normal=(1.0, 0.0, 0.0)))
        rad._eclipse_constraint = Mock()
        rad._eclipse_constraint.in_constraint = Mock(return_value=False)

        sun_body = np.array([1.0, 0.0, 0.0])
        zero = np.zeros(3)
        with patch("conops.config.radiator.scbodyvector", side_effect=[sun_body, zero]):
            sun_exp, earth_exp = rad.exposure_factors(
                ra_deg=0.0, dec_deg=0.0, utime=1000.0, ephem=ephem, roll_deg=0.0
            )

        assert earth_exp == pytest.approx(0.0)
        assert isinstance(sun_exp, float)


class TestRadiatorConfiguration:
    def test_default_radiator_orientation_normal(self) -> None:
        rad = Radiator()
        assert rad.orientation.normal == (0.0, 1.0, 0.0)

    def test_set_ephem_and_num_radiators_and_lookup(self) -> None:
        c1 = Constraint()
        c2 = Constraint()
        r1 = Radiator(name="R1", hard_constraint=c1)
        r2 = Radiator(name="R2", hard_constraint=c2)
        cfg = RadiatorConfiguration(radiators=[r1, r2])
        ephem = Mock()

        cfg.set_ephem(ephem)

        assert cfg.num_radiators() == 2
        assert c1.ephem is ephem
        assert c2.ephem is ephem
        assert cfg.get_radiator_by_name("R2") is r2
        assert cfg.get_radiator_by_name("missing") is None

    def test_pointing_valid_for_empty_and_non_empty_configs(self) -> None:
        empty_cfg = RadiatorConfiguration(radiators=[])
        assert empty_cfg.is_pointing_valid(0.0, 0.0, 1000.0, 0.0) is True

        rad = Radiator(name="R1")
        cfg = RadiatorConfiguration(radiators=[rad])
        with patch.object(
            RadiatorConfiguration,
            "radiators_violating_hard_constraints",
            return_value=1,
        ):
            assert cfg.is_pointing_valid(0.0, 0.0, 1000.0, 0.0) is False

    def test_radiator_hard_constraint_combines_offsets(self) -> None:
        c1 = Constraint()
        c2 = Constraint()
        offset1 = Mock()
        offset2 = Mock()
        combined = Mock()

        base1 = Mock()
        base1.boresight_offset = Mock(return_value=offset1)
        c1.constraint = base1

        base2 = Mock()
        base2.boresight_offset = Mock(return_value=offset2)
        c2.constraint = base2

        offset1.__or__ = Mock(return_value=combined)

        cfg = RadiatorConfiguration(
            radiators=[Radiator(hard_constraint=c1), Radiator(hard_constraint=c2)]
        )
        assert cfg.radiator_hard_constraint is combined

    def test_radiator_hard_constraint_skips_missing_offsets(self) -> None:
        c1 = Constraint()
        c1.constraint = None

        c2 = Constraint()
        offset2 = Mock()
        base2 = Mock()
        base2.boresight_offset = Mock(return_value=offset2)
        c2.constraint = base2

        cfg = RadiatorConfiguration(
            radiators=[Radiator(hard_constraint=c1), Radiator(hard_constraint=c2)]
        )
        assert cfg.radiator_hard_constraint is offset2

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

    def test_hard_violation_count_multi_radiator_loop(self) -> None:
        """Two radiators: first no violation, second violation — forces the for-loop back-edge."""
        rad1 = Radiator(name="R1")
        rad2 = Radiator(name="R2")
        cfg = RadiatorConfiguration(radiators=[rad1, rad2])
        side_effects = [False, True]
        with patch.object(Radiator, "in_hard_constraint", side_effect=side_effects):
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

    def test_exposure_metrics_empty_radiators(self) -> None:
        cfg = RadiatorConfiguration(radiators=[])
        metrics = cfg.exposure_metrics(
            ra_deg=0.0,
            dec_deg=0.0,
            utime=1000.0,
            ephem=Mock(),
            roll_deg=0.0,
        )
        assert metrics == {
            "sun_exposure": 0.0,
            "earth_exposure": 0.0,
            "heat_dissipation_w": 0.0,
            "per_radiator": [],
        }

    def test_exposure_metrics_with_zero_area_guard(self) -> None:
        ephem = Mock()
        ephem.sun_ra_deg = [0.0]
        ephem.sun_dec_deg = [0.0]
        ephem.earth_ra_deg = [0.0]
        ephem.earth_dec_deg = [0.0]
        ephem.index = Mock(return_value=0)

        rad = Radiator(name="R1")
        cfg = RadiatorConfiguration(radiators=[rad])
        with (
            patch.object(Radiator, "area_m2", new_callable=PropertyMock) as area_m2,
            patch.object(
                Radiator,
                "exposure_factors",
                return_value=(0.5, 0.25),
            ),
            patch.object(
                Radiator,
                "heat_dissipation_w",
                return_value=10.0,
            ),
            patch.object(
                Radiator,
                "in_hard_constraint",
                return_value=False,
            ),
        ):
            area_m2.return_value = 0.0
            metrics = cfg.exposure_metrics(0.0, 0.0, 1000.0, ephem, 0.0)

        assert metrics["sun_exposure"] == pytest.approx(0.0)
        assert metrics["earth_exposure"] == pytest.approx(0.0)
        assert metrics["heat_dissipation_w"] == pytest.approx(10.0)
