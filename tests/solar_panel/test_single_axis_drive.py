from datetime import datetime, timedelta, timezone
from unittest.mock import Mock, patch

import numpy as np
import pytest
from pydantic import ValidationError

from conops import (
    ACSMode,
    IncidenceLossPoint,
    SingleAxisSolarArrayDrive,
    SolarArrayDriveControl,
    SolarPanel,
    SolarPanelSet,
    optimum_roll,
)
from conops.config import PanelGeometry


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

    @pytest.mark.parametrize(
        ("rotation_axis", "normal", "sun_body"),
        [
            ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            ((0.0, 0.0, 1.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        ],
    )
    def test_optimal_angle_holds_when_rotation_cannot_change_illumination(
        self,
        rotation_axis: tuple[float, float, float],
        normal: tuple[float, float, float],
        sun_body: tuple[float, float, float],
    ) -> None:
        drive = _drive(rotation_axis=rotation_axis, initial_angle_deg=73.0)

        angle = drive.optimal_angle(
            normal,
            sun_body,
            reference_angle_deg=73.0,
        )

        assert angle == pytest.approx(73.0)

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
    def test_drive_holds_without_an_explicit_tracking_command(self) -> None:
        panel = SolarPanel(normal=(1.0, 0.0, 0.0), single_axis_drive=_drive())

        panel.illumination_from_sun_body(0.0, (0.0, 1.0, 0.0), advance_drive_state=True)
        illumination, _ = panel.illumination_from_sun_body(
            60.0, (0.0, 1.0, 0.0), advance_drive_state=True
        )

        assert panel.drive_angle_deg == pytest.approx(0.0)
        assert illumination == pytest.approx(0.0, abs=1e-12)

    def test_control_policy_is_explicitly_mode_and_eclipse_dependent(self) -> None:
        panel = SolarPanel(
            single_axis_drive=_drive(),
            drive_control=SolarArrayDriveControl(sun_tracking_modes=[ACSMode.SCIENCE]),
        )

        assert panel.tracks_sun(ACSMode.SCIENCE, in_eclipse=False)
        assert not panel.tracks_sun(ACSMode.PASS, in_eclipse=False)
        assert not panel.tracks_sun(ACSMode.SCIENCE, in_eclipse=True)
        assert not SolarPanel(single_axis_drive=_drive()).tracks_sun(
            ACSMode.SCIENCE, in_eclipse=False
        )

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
            60.0,
            (0.0, 1.0, 0.0),
            track_sun=True,
            advance_drive_state=True,
        )
        panel.illumination_from_sun_body(
            90.0,
            (0.0, 1.0, 0.0),
            track_sun=True,
            advance_drive_state=True,
        )

        restored = SolarPanel.model_validate(panel.model_dump())

        assert restored.model_dump() == panel.model_dump()
        assert panel.drive_angle_deg == pytest.approx(45.0)
        assert restored.drive_angle_deg == pytest.approx(15.0)

    def test_drive_rate_limits_executed_motion(self) -> None:
        panel = SolarPanel(normal=(1.0, 0.0, 0.0), single_axis_drive=_drive())
        panel.reset_drive_state()

        initial, _ = panel.illumination_from_sun_body(
            0.0,
            (0.0, 1.0, 0.0),
            track_sun=True,
            advance_drive_state=True,
        )
        after_30_s, _ = panel.illumination_from_sun_body(
            30.0,
            (0.0, 1.0, 0.0),
            track_sun=True,
            advance_drive_state=True,
        )

        assert initial == pytest.approx(0.0, abs=1e-12)
        assert panel.drive_angle_deg == pytest.approx(30.0)
        assert after_30_s == pytest.approx(0.5)

    def test_candidate_preview_does_not_mutate_drive_state(self) -> None:
        panel = SolarPanel(normal=(1.0, 0.0, 0.0), single_axis_drive=_drive())
        panel.illumination_from_sun_body(
            0.0,
            (0.0, 1.0, 0.0),
            track_sun=True,
            advance_drive_state=True,
        )

        preview, _ = panel.illumination_from_sun_body(
            60.0,
            (0.0, 1.0, 0.0),
            track_sun=True,
            advance_drive_state=False,
        )

        assert preview == pytest.approx(np.sin(np.deg2rad(60.0)))
        assert panel.drive_angle_deg == pytest.approx(0.0)

    def test_vectorized_candidate_preview_matches_rate_limited_geometry(self) -> None:
        panel = SolarPanel(normal=(1.0, 0.0, 0.0), single_axis_drive=_drive())
        panel.illumination_from_sun_body(
            0.0,
            (0.0, 1.0, 0.0),
            track_sun=True,
            advance_drive_state=True,
        )

        factors = panel.preview_power_factors_from_sun_body(
            np.asarray(
                [
                    (0.0, 1.0, 0.0),
                    (1.0, 0.0, 0.0),
                ]
            ),
            track_sun=True,
            elapsed_seconds=30.0,
        )

        assert factors == pytest.approx((0.5, 1.0))
        assert panel.drive_angle_deg == pytest.approx(0.0)

    def test_executed_samples_cannot_advance_backward_in_time(self) -> None:
        panel = SolarPanel(normal=(1.0, 0.0, 0.0), single_axis_drive=_drive())
        panel.illumination_from_sun_body(
            60.0,
            (0.0, 1.0, 0.0),
            track_sun=True,
            advance_drive_state=True,
        )

        with pytest.raises(ValueError, match="backward in time"):
            panel.illumination_from_sun_body(
                30.0, (0.0, 1.0, 0.0), advance_drive_state=True
            )

    def test_legacy_ideal_gimbal_cannot_also_use_finite_drive(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            SolarPanel(gimbled=True, single_axis_drive=_drive())

    def test_articulated_geometry_rotates_about_explicit_axis_line(self) -> None:
        panel = SolarPanel(
            normal=(1.0, 0.0, 0.0),
            geometry=PanelGeometry(
                center_m=(1.0, 0.0, 0.0),
                u=(0.0, 1.0, 0.0),
                v=(0.0, 0.0, 1.0),
                width_m=2.0,
                height_m=1.0,
            ),
            single_axis_drive=_drive(rotation_origin_m=(0.0, 0.0, 0.0)),
        )

        geometry = panel.geometry_at_drive_angle(90.0)

        assert geometry is not None
        assert geometry.center_m == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)
        assert geometry.u == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)
        assert geometry.v == pytest.approx((0.0, 0.0, 1.0), abs=1e-12)

    def test_articulated_geometry_requires_axis_origin(self) -> None:
        with pytest.raises(ValidationError, match="rotation_origin_m"):
            SolarPanel(
                geometry=PanelGeometry(
                    u=(1.0, 0.0, 0.0),
                    v=(0.0, 0.0, 1.0),
                ),
                single_axis_drive=_drive(),
            )


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
    def test_shadow_geometry_previews_the_same_angle_execution_commits(self) -> None:
        panel = SolarPanel(
            name="Wing",
            normal=(1.0, 0.0, 0.0),
            geometry=PanelGeometry(
                center_m=(1.0, 0.0, 0.0),
                u=(0.0, 1.0, 0.0),
                v=(0.0, 0.0, 1.0),
            ),
            single_axis_drive=_drive(rotation_origin_m=(0.0, 0.0, 0.0)),
            drive_control=SolarArrayDriveControl(sun_tracking_modes=[ACSMode.SCIENCE]),
        )
        panel_set = SolarPanelSet(panels=[panel])
        ephem = _ephem_with_sun((0.0, 1.0, 0.0))
        eclipse = Mock()
        eclipse.in_constraint.return_value = False
        start = datetime(2026, 1, 1, tzinfo=timezone.utc)

        with patch(
            "conops.config.solar_panel._get_eclipse_constraint",
            return_value=eclipse,
        ):
            panel_set.illumination_and_power(
                time=start,
                ra=0.0,
                dec=0.0,
                ephem=ephem,
                acs_mode=ACSMode.SCIENCE,
                advance_drive_state=True,
            )
            projected = panel_set.shadow_geometries(
                time_s=(start + timedelta(seconds=30)).timestamp(),
                ra=0.0,
                dec=0.0,
                roll=0.0,
                ephem=ephem,
                acs_mode=ACSMode.SCIENCE,
                in_eclipse=False,
            )["Wing"]
            assert panel.drive_angle_deg == pytest.approx(0.0)

            panel_set.illumination_and_power(
                time=start + timedelta(seconds=30),
                ra=0.0,
                dec=0.0,
                ephem=ephem,
                acs_mode=ACSMode.SCIENCE,
                advance_drive_state=True,
            )

        committed = panel.geometry_at_drive_angle()
        assert committed is not None
        assert projected.center_m == pytest.approx(committed.center_m)
        assert projected.u == pytest.approx(committed.u)
        assert projected.v == pytest.approx(committed.v)

    def test_drive_holds_in_eclipse_without_accumulating_motion_time(self) -> None:
        panel = SolarPanel(
            normal=(1.0, 0.0, 0.0),
            single_axis_drive=_drive(),
            drive_control=SolarArrayDriveControl(sun_tracking_modes=[ACSMode.SCIENCE]),
        )
        panel_set = SolarPanelSet(panels=[panel])
        ephem = _ephem_with_sun((0.0, 1.0, 0.0))
        eclipse = Mock()
        eclipse.in_constraint.side_effect = [False, True, False]

        with patch(
            "conops.config.solar_panel._get_eclipse_constraint",
            return_value=eclipse,
        ):
            for seconds in (0, 30, 60):
                _, power = panel_set.illumination_and_power(
                    time=datetime(2026, 1, 1, tzinfo=timezone.utc)
                    + timedelta(seconds=seconds),
                    ra=0.0,
                    dec=0.0,
                    ephem=ephem,
                    acs_mode=ACSMode.SCIENCE,
                    advance_drive_state=True,
                )
                if seconds == 30:
                    assert power == pytest.approx(0.0)

        assert panel.drive_angle_deg == pytest.approx(30.0)

    def test_power_calculation_advances_and_reports_drive_angle(self) -> None:
        panel_set = SolarPanelSet(
            panels=[
                SolarPanel(
                    normal=(1.0, 0.0, 0.0),
                    max_power=100.0,
                    conversion_efficiency=1.0,
                    single_axis_drive=_drive(),
                    drive_control=SolarArrayDriveControl(
                        sun_tracking_modes=[ACSMode.SCIENCE]
                    ),
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
                acs_mode=ACSMode.SCIENCE,
                advance_drive_state=True,
            )
            illumination, power = panel_set.illumination_and_power(
                time=datetime(2026, 1, 1, 0, 0, 30, tzinfo=timezone.utc),
                ra=0.0,
                dec=0.0,
                ephem=ephem,
                acs_mode=ACSMode.SCIENCE,
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
            drive_control=SolarArrayDriveControl(sun_tracking_modes=[ACSMode.SCIENCE]),
        )
        panel_set = SolarPanelSet(panels=[panel])
        panel.illumination_from_sun_body(
            0.0,
            (0.0, 1.0, 0.0),
            track_sun=True,
            advance_drive_state=True,
        )
        ephem = _ephem_with_sun((0.0, 1.0, 0.0))

        roll = optimum_roll(0.0, 0.0, 30.0, ephem, panel_set, acs_mode=ACSMode.SCIENCE)

        assert roll == pytest.approx(0.0)
        assert panel.drive_angle_deg == pytest.approx(0.0)

    def test_roll_search_requires_explicit_candidate_drive_motion(self) -> None:
        panel = SolarPanel(
            normal=(0.0, 1.0, 0.0),
            max_power=100.0,
            single_axis_drive=_drive(rotation_axis=(1.0, 0.0, 0.0)),
            drive_control=SolarArrayDriveControl(sun_tracking_modes=[ACSMode.SCIENCE]),
        )
        panel_set = SolarPanelSet(panels=[panel])
        panel.illumination_from_sun_body(
            0.0,
            (1.0, 0.0, 1.0),
            track_sun=False,
            advance_drive_state=True,
        )
        ephem = _ephem_with_sun((1.0, 0.0, 1.0))

        held_roll = optimum_roll(
            0.0,
            0.0,
            30.0,
            ephem,
            panel_set,
            acs_mode=ACSMode.SCIENCE,
        )
        moving_roll = optimum_roll(
            0.0,
            0.0,
            30.0,
            ephem,
            panel_set,
            acs_mode=ACSMode.SCIENCE,
            in_eclipse=False,
            drive_preview_seconds=30.0,
        )

        assert held_roll == pytest.approx(90.0)
        assert moving_roll != held_roll
        assert panel.drive_angle_deg == pytest.approx(0.0)

    def test_roll_search_requires_eclipse_state_for_candidate_drive_motion(
        self,
    ) -> None:
        panel_set = SolarPanelSet(
            panels=[
                SolarPanel(
                    normal=(0.0, 1.0, 0.0),
                    single_axis_drive=_drive(rotation_axis=(1.0, 0.0, 0.0)),
                )
            ]
        )

        with pytest.raises(ValueError, match="in_eclipse must be provided"):
            optimum_roll(
                0.0,
                0.0,
                30.0,
                _ephem_with_sun((1.0, 0.0, 1.0)),
                panel_set,
                acs_mode=ACSMode.SCIENCE,
                drive_preview_seconds=30.0,
            )

    def test_roll_search_respects_eclipse_drive_hold_policy(self) -> None:
        panel = SolarPanel(
            normal=(0.0, 1.0, 0.0),
            max_power=100.0,
            single_axis_drive=_drive(rotation_axis=(1.0, 0.0, 0.0)),
            drive_control=SolarArrayDriveControl(sun_tracking_modes=[ACSMode.SCIENCE]),
        )
        panel_set = SolarPanelSet(panels=[panel])
        ephem = _ephem_with_sun((1.0, 0.0, 1.0))

        sunlit_roll = optimum_roll(
            0.0,
            0.0,
            30.0,
            ephem,
            panel_set,
            acs_mode=ACSMode.SCIENCE,
            in_eclipse=False,
            drive_preview_seconds=30.0,
        )
        eclipse_roll = optimum_roll(
            0.0,
            0.0,
            30.0,
            ephem,
            panel_set,
            acs_mode=ACSMode.SCIENCE,
            in_eclipse=True,
            drive_preview_seconds=30.0,
        )

        assert eclipse_roll == pytest.approx(90.0)
        assert sunlit_roll != eclipse_roll
        assert panel.drive_angle_deg == pytest.approx(0.0)
