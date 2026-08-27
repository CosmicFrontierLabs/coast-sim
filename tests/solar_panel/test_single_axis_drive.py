from datetime import datetime, timezone
from unittest.mock import Mock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from conops import (
    IncidenceLossPoint,
    SingleAxisSolarArrayDrive,
    SolarPanel,
    SolarPanelSet,
    optimum_roll,
)


def _drive(**overrides: object) -> SingleAxisSolarArrayDrive:
    values: dict[str, object] = {
        "rotation_axis": (0.0, 0.0, 1.0),
        "min_angle_deg": -165.0,
        "max_angle_deg": 165.0,
        "max_rate_deg_per_s": 1.0,
        "initial_angle_deg": 0.0,
    }
    values.update(overrides)
    return SingleAxisSolarArrayDrive(**values)


def _ephem_with_sun(sun_vector: tuple[float, float, float]) -> Mock:
    ephem = Mock()
    ephem.index = Mock(return_value=0)
    ephem.sun_pv.position = np.asarray([sun_vector], dtype=float)
    ephem.gcrs_pv.position = np.zeros((1, 3), dtype=float)
    return ephem


class TestSingleAxisSolarArrayDrive:
    def test_positive_rotation_uses_right_hand_rule(self) -> None:
        normal = _drive().normal_at_angle((1.0, 0.0, 0.0), 90.0)

        assert normal == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)

    def test_optimal_angle_respects_finite_travel(self) -> None:
        angle = _drive().optimal_angle(
            (1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), reference_angle_deg=0.0
        )

        assert abs(angle) == pytest.approx(165.0)
        normal = _drive().normal_at_angle((1.0, 0.0, 0.0), angle)
        assert np.dot(normal, (-1.0, 0.0, 0.0)) == pytest.approx(
            np.cos(np.deg2rad(15.0))
        )

    def test_step_toward_applies_rate_and_travel_limits(self) -> None:
        drive = _drive(max_rate_deg_per_s=2.0)

        assert drive.step_toward(0.0, 100.0, 10.0) == 20.0
        assert drive.step_toward(160.0, 200.0, 10.0) == 165.0

    @pytest.mark.parametrize(
        "overrides",
        [
            {"rotation_axis": (0.0, 0.0, 0.0)},
            {"min_angle_deg": 10.0, "max_angle_deg": 10.0},
            {"min_angle_deg": -181.0, "max_angle_deg": 181.0},
            {"initial_angle_deg": 170.0},
            {"max_rate_deg_per_s": 0.0},
        ],
    )
    def test_invalid_drive_configuration_is_rejected(
        self, overrides: dict[str, object]
    ) -> None:
        with pytest.raises(ValidationError):
            _drive(**overrides)


class TestDrivenPanelRuntime:
    def test_configuration_round_trips_without_runtime_state(self) -> None:
        panel = SolarPanel(
            normal=(1.0, 0.0, 0.0),
            single_axis_drive=_drive(initial_angle_deg=15.0),
            incidence_loss_curve=[
                IncidenceLossPoint(incidence_angle_deg=0.0, power_factor=1.0),
                IncidenceLossPoint(incidence_angle_deg=90.0, power_factor=0.5),
            ],
        )
        panel.illumination_from_sun_body(
            60.0, (0.0, 1.0, 0.0), advance_drive_state=True
        )
        panel.illumination_from_sun_body(
            90.0, (0.0, 1.0, 0.0), advance_drive_state=True
        )

        restored = SolarPanel.model_validate(panel.model_dump())

        assert restored.model_dump() == panel.model_dump()
        assert panel.drive_angle_deg == pytest.approx(45.0)
        assert restored.drive_angle_deg == pytest.approx(15.0)

    def test_drive_rate_limits_executed_motion(self) -> None:
        panel = SolarPanel(normal=(1.0, 0.0, 0.0), single_axis_drive=_drive())
        panel.reset_drive_state()

        initial, _ = panel.illumination_from_sun_body(
            0.0, (0.0, 1.0, 0.0), advance_drive_state=True
        )
        after_30_s, _ = panel.illumination_from_sun_body(
            30.0, (0.0, 1.0, 0.0), advance_drive_state=True
        )

        assert initial == pytest.approx(0.0, abs=1e-12)
        assert panel.drive_angle_deg == pytest.approx(30.0)
        assert after_30_s == pytest.approx(0.5)

    def test_candidate_preview_does_not_mutate_drive_state(self) -> None:
        panel = SolarPanel(normal=(1.0, 0.0, 0.0), single_axis_drive=_drive())
        panel.illumination_from_sun_body(0.0, (0.0, 1.0, 0.0), advance_drive_state=True)

        preview, _ = panel.illumination_from_sun_body(
            60.0, (0.0, 1.0, 0.0), advance_drive_state=False
        )

        assert preview == pytest.approx(np.sin(np.deg2rad(60.0)))
        assert panel.drive_angle_deg == pytest.approx(0.0)

    def test_vectorized_candidate_preview_matches_rate_limited_geometry(self) -> None:
        panel = SolarPanel(normal=(1.0, 0.0, 0.0), single_axis_drive=_drive())
        panel.illumination_from_sun_body(0.0, (0.0, 1.0, 0.0), advance_drive_state=True)

        factors = panel.preview_power_factors_from_sun_body(
            30.0,
            np.asarray(
                [
                    (0.0, 1.0, 0.0),
                    (1.0, 0.0, 0.0),
                ]
            ),
        )

        assert factors == pytest.approx((0.5, 1.0))
        assert panel.drive_angle_deg == pytest.approx(0.0)

    def test_executed_samples_cannot_advance_backward_in_time(self) -> None:
        panel = SolarPanel(normal=(1.0, 0.0, 0.0), single_axis_drive=_drive())
        panel.illumination_from_sun_body(
            60.0, (0.0, 1.0, 0.0), advance_drive_state=True
        )

        with pytest.raises(ValueError, match="backward in time"):
            panel.illumination_from_sun_body(
                30.0, (0.0, 1.0, 0.0), advance_drive_state=True
            )

    def test_legacy_ideal_gimbal_cannot_also_use_finite_drive(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            SolarPanel(gimbled=True, single_axis_drive=_drive())


class TestIncidenceLoss:
    def test_curve_reduces_power_beyond_cosine_incidence(self) -> None:
        panel = SolarPanel(
            normal=(1.0, 0.0, 0.0),
            incidence_loss_curve=[
                IncidenceLossPoint(incidence_angle_deg=0.0, power_factor=1.0),
                IncidenceLossPoint(incidence_angle_deg=60.0, power_factor=0.8),
                IncidenceLossPoint(incidence_angle_deg=90.0, power_factor=0.5),
            ],
        )

        illumination, power_factor = panel.illumination_from_sun_body(
            0.0, (0.5, np.sqrt(3.0) / 2.0, 0.0)
        )

        assert illumination == pytest.approx(0.5)
        assert power_factor == pytest.approx(0.4)

    def test_curve_angles_must_be_strictly_increasing(self) -> None:
        with pytest.raises(ValidationError, match="strictly increasing"):
            SolarPanel(
                incidence_loss_curve=[
                    IncidenceLossPoint(incidence_angle_deg=60.0, power_factor=0.8),
                    IncidenceLossPoint(incidence_angle_deg=30.0, power_factor=0.9),
                ]
            )

    def test_curve_power_factors_must_not_increase_with_incidence(self) -> None:
        with pytest.raises(ValidationError, match="must be non-increasing"):
            SolarPanel(
                incidence_loss_curve=[
                    IncidenceLossPoint(incidence_angle_deg=0.0, power_factor=0.8),
                    IncidenceLossPoint(incidence_angle_deg=60.0, power_factor=0.9),
                ]
            )


class TestDrivenPanelSet:
    def test_power_calculation_advances_and_reports_drive_angle(self) -> None:
        panel_set = SolarPanelSet(
            panels=[
                SolarPanel(
                    normal=(1.0, 0.0, 0.0),
                    max_power=100.0,
                    conversion_efficiency=1.0,
                    single_axis_drive=_drive(),
                )
            ]
        )
        ephem = _ephem_with_sun((0.0, 1.0, 0.0))
        eclipse = Mock()
        eclipse.in_constraint.return_value = False

        with patch(
            "conops.config.solar_panel._get_eclipse_constraint",
            return_value=eclipse,
        ):
            panel_set.illumination_and_power(
                time=datetime(2026, 1, 1, tzinfo=timezone.utc),
                ra=0.0,
                dec=0.0,
                ephem=ephem,
                advance_drive_state=True,
            )
            illumination, power = panel_set.illumination_and_power(
                time=datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc),
                ra=0.0,
                dec=0.0,
                ephem=ephem,
                advance_drive_state=True,
            )

        assert panel_set.drive_angles_deg == [pytest.approx(30.0)]
        assert illumination == pytest.approx(0.5)
        assert power == pytest.approx(50.0)

    def test_roll_search_previews_drive_without_mutating_it(self) -> None:
        panel = SolarPanel(
            normal=(1.0, 0.0, 0.0),
            max_power=100.0,
            single_axis_drive=_drive(),
        )
        panel_set = SolarPanelSet(panels=[panel])
        panel.illumination_from_sun_body(0.0, (0.0, 1.0, 0.0), advance_drive_state=True)
        ephem = _ephem_with_sun((0.0, 1.0, 0.0))

        roll = optimum_roll(0.0, 0.0, 30.0, ephem, panel_set)

        assert roll == pytest.approx(0.0)
        assert panel.drive_angle_deg == pytest.approx(0.0)
