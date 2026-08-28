from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError

from conops import (
    ACSMode,
    SingleAxisSolarArrayDrive,
    SolarArrayDriveControl,
    SolarPanel,
    SolarPanelSet,
    optimum_roll,
)

_START = datetime(2026, 1, 1, tzinfo=timezone.utc)


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


def _sample(
    panel: SolarPanel,
    time_s: float,
    sun: tuple[float, float, float] = (0.0, 1.0, 0.0),
    *,
    track_sun: bool = True,
    advance_drive_state: bool = True,
) -> float:
    return panel.illumination_from_sun_body(
        time_s,
        sun,
        track_sun=track_sun,
        advance_drive_state=advance_drive_state,
    )


def _tracking_panel(
    *,
    normal: tuple[float, float, float] = (1.0, 0.0, 0.0),
    rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    conversion_efficiency: float | None = None,
) -> SolarPanel:
    """Build the common finite-drive panel used by integration tests."""
    return SolarPanel(
        name="Wing",
        normal=normal,
        max_power=100.0,
        conversion_efficiency=conversion_efficiency,
        single_axis_drive=_drive(
            rotation_axis=rotation_axis,
        ),
        drive_control=SolarArrayDriveControl(sun_tracking_modes=[ACSMode.SCIENCE]),
    )


def _execute(
    panel_set: SolarPanelSet, ephem: Mock, seconds: float = 0.0
) -> tuple[float | np.ndarray, float | np.ndarray]:
    """Execute one common science-mode power sample."""
    return panel_set.illumination_and_power(
        time=_START + timedelta(seconds=seconds),
        ra=0.0,
        dec=0.0,
        ephem=ephem,
        acs_mode=ACSMode.SCIENCE,
        advance_drive_state=True,
    )


def _roll(
    panel_set: SolarPanelSet,
    ephem: Mock,
    *,
    in_eclipse: bool | None = None,
    drive_preview_seconds: float = 0.0,
) -> float:
    """Run the common roll search used by drive-preview tests."""
    return optimum_roll(
        0.0,
        0.0,
        30.0,
        ephem,
        panel_set,
        acs_mode=ACSMode.SCIENCE,
        in_eclipse=in_eclipse,
        drive_preview_seconds=drive_preview_seconds,
    )


@pytest.fixture
def eclipse(monkeypatch: pytest.MonkeyPatch) -> Mock:
    """Install a controllable eclipse result for set-level calculations."""
    constraint = Mock()
    constraint.in_constraint.return_value = False
    monkeypatch.setattr(
        "conops.config.solar_panel._get_eclipse_constraint", lambda: constraint
    )
    return constraint


class TestSingleAxisSolarArrayDrive:
    def test_positive_rotation_uses_right_hand_rule(self) -> None:
        normal = _drive().normals_at_angles((1.0, 0.0, 0.0), np.asarray([90.0]))[0]

        assert normal == pytest.approx((0.0, 1.0, 0.0), abs=1e-12)

    def test_optimal_angle_respects_finite_travel(self) -> None:
        drive = _drive()
        angle = float(
            drive.optimal_angles(
                (1.0, 0.0, 0.0),
                np.asarray([(-1.0, 0.0, 0.0)]),
                reference_angle_deg=0.0,
            )[0]
        )

        assert abs(angle) == pytest.approx(165.0)
        normal = drive.normals_at_angles((1.0, 0.0, 0.0), np.asarray([angle]))[0]
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

        angle = drive.optimal_angles(
            normal,
            np.asarray([sun_body]),
            reference_angle_deg=73.0,
        )[0]

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

        _sample(panel, 0.0, track_sun=False)
        illumination = _sample(panel, 60.0, track_sun=False)

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
        )
        _sample(panel, 60.0)
        _sample(panel, 90.0)

        restored = SolarPanel.model_validate(panel.model_dump())

        assert restored.model_dump() == panel.model_dump()
        assert panel.drive_angle_deg == pytest.approx(45.0)
        assert restored.drive_angle_deg == pytest.approx(15.0)

    def test_drive_rate_limits_executed_motion(self) -> None:
        panel = SolarPanel(normal=(1.0, 0.0, 0.0), single_axis_drive=_drive())
        panel.reset_drive_state()

        initial = _sample(panel, 0.0)
        after_30_s = _sample(panel, 30.0)

        assert initial == pytest.approx(0.0, abs=1e-12)
        assert panel.drive_angle_deg == pytest.approx(30.0)
        assert after_30_s == pytest.approx(0.5)

    def test_candidate_preview_does_not_mutate_drive_state(self) -> None:
        panel = SolarPanel(normal=(1.0, 0.0, 0.0), single_axis_drive=_drive())
        _sample(panel, 0.0)

        preview = _sample(panel, 60.0, advance_drive_state=False)

        assert preview == pytest.approx(np.sin(np.deg2rad(60.0)))
        assert panel.drive_angle_deg == pytest.approx(0.0)

    def test_vectorized_candidate_preview_matches_rate_limited_geometry(self) -> None:
        panel = SolarPanel(normal=(1.0, 0.0, 0.0), single_axis_drive=_drive())
        _sample(panel, 0.0)

        illumination = panel.preview_illumination_from_sun_body(
            np.asarray(
                [
                    (0.0, 1.0, 0.0),
                    (1.0, 0.0, 0.0),
                ]
            ),
            track_sun=True,
            elapsed_seconds=30.0,
        )

        assert illumination == pytest.approx((0.5, 1.0))
        assert panel.drive_angle_deg == pytest.approx(0.0)

    def test_executed_samples_cannot_advance_backward_in_time(self) -> None:
        panel = SolarPanel(normal=(1.0, 0.0, 0.0), single_axis_drive=_drive())
        _sample(panel, 60.0)

        with pytest.raises(ValueError, match="backward in time"):
            _sample(panel, 30.0)

    def test_legacy_ideal_gimbal_cannot_also_use_finite_drive(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            SolarPanel(gimbled=True, single_axis_drive=_drive())


@pytest.mark.usefixtures("eclipse")
class TestDrivenPanelSet:
    def test_drive_holds_in_eclipse_without_accumulating_motion_time(
        self, eclipse: Mock
    ) -> None:
        panel = _tracking_panel()
        panel_set = SolarPanelSet(panels=[panel])
        ephem = _ephem_with_sun((0.0, 1.0, 0.0))
        eclipse.in_constraint.side_effect = [False, True, False]

        for seconds in (0.0, 30.0, 60.0):
            _, power = _execute(panel_set, ephem, seconds)
            if seconds == 30.0:
                assert power == pytest.approx(0.0)

        assert panel.drive_angle_deg == pytest.approx(30.0)

    def test_power_calculation_advances_and_reports_drive_angle(self) -> None:
        panel_set = SolarPanelSet(panels=[_tracking_panel(conversion_efficiency=1.0)])
        ephem = _ephem_with_sun((0.0, 1.0, 0.0))

        _execute(panel_set, ephem)
        illumination, power = _execute(panel_set, ephem, 30.0)

        assert panel_set.drive_angles_deg == [pytest.approx(30.0)]
        assert illumination == pytest.approx(0.5)
        assert power == pytest.approx(50.0)

    def test_roll_search_previews_drive_without_mutating_it(self) -> None:
        panel = _tracking_panel()
        panel_set = SolarPanelSet(panels=[panel])
        _sample(panel, 0.0)
        ephem = _ephem_with_sun((0.0, 1.0, 0.0))

        roll = _roll(panel_set, ephem)

        assert roll == pytest.approx(0.0)
        assert panel.drive_angle_deg == pytest.approx(0.0)

    def test_roll_search_requires_explicit_candidate_drive_motion(self) -> None:
        panel = _tracking_panel(
            normal=(0.0, 1.0, 0.0),
            rotation_axis=(1.0, 0.0, 0.0),
        )
        panel_set = SolarPanelSet(panels=[panel])
        _sample(panel, 0.0, (1.0, 0.0, 1.0), track_sun=False)
        ephem = _ephem_with_sun((1.0, 0.0, 1.0))

        held_roll = _roll(panel_set, ephem)
        moving_roll = _roll(
            panel_set,
            ephem,
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
                _tracking_panel(
                    normal=(0.0, 1.0, 0.0),
                    rotation_axis=(1.0, 0.0, 0.0),
                )
            ]
        )

        with pytest.raises(ValueError, match="in_eclipse must be provided"):
            _roll(
                panel_set,
                _ephem_with_sun((1.0, 0.0, 1.0)),
                drive_preview_seconds=30.0,
            )

    def test_roll_search_respects_eclipse_drive_hold_policy(self) -> None:
        panel = _tracking_panel(
            normal=(0.0, 1.0, 0.0),
            rotation_axis=(1.0, 0.0, 0.0),
        )
        panel_set = SolarPanelSet(panels=[panel])
        ephem = _ephem_with_sun((1.0, 0.0, 1.0))

        sunlit_roll = _roll(
            panel_set,
            ephem,
            in_eclipse=False,
            drive_preview_seconds=30.0,
        )
        eclipse_roll = _roll(
            panel_set,
            ephem,
            in_eclipse=True,
            drive_preview_seconds=30.0,
        )

        assert eclipse_roll == pytest.approx(90.0)
        assert sunlit_roll != eclipse_roll
        assert panel.drive_angle_deg == pytest.approx(0.0)
