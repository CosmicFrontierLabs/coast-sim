from datetime import datetime, timedelta, timezone
from unittest.mock import Mock

import numpy as np
import pytest
from pydantic import ValidationError

from conops import (
    ACSMode,
    SingleAxisSolarArrayDrive,
    SolarArrayDriveControl,
    SolarArrayDriveState,
    SolarPanel,
    SolarPanelSet,
    optimum_instrument_roll,
    optimum_roll,
)
from conops.common import scbodyvector
from conops.config import DTOR, Telescope
from conops.config.geometry import PanelGeometry

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


def _tracking_panel(
    *,
    normal: tuple[float, float, float] = (1.0, 0.0, 0.0),
    rotation_axis: tuple[float, float, float] = (0.0, 0.0, 1.0),
    conversion_efficiency: float | None = 1.0,
) -> SolarPanel:
    return SolarPanel(
        name="Wing",
        normal=normal,
        max_power=100.0,
        conversion_efficiency=conversion_efficiency,
        single_axis_drive=_drive(rotation_axis=rotation_axis),
        drive_control=SolarArrayDriveControl(sun_tracking_modes=[ACSMode.SCIENCE]),
    )


def _ephem_with_sun(sun_vector: tuple[float, float, float]) -> Mock:
    ephem = Mock()
    ephem.index.return_value = 0
    ephem.sun_pv.position = np.asarray([sun_vector], dtype=float)
    ephem.gcrs_pv.position = np.zeros((1, 3), dtype=float)
    return ephem


@pytest.fixture
def eclipse(monkeypatch: pytest.MonkeyPatch) -> Mock:
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


class TestSolarArrayDriveState:
    def test_configuration_contains_no_runtime_state(self) -> None:
        panel = _tracking_panel()
        before = panel.model_dump()
        panel_set = SolarPanelSet(panels=[panel])
        state = panel_set.advance_drive_state(
            0.0,
            (0.0, 1.0, 0.0),
            panel_set.initial_drive_state(),
            acs_mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )
        panel_set.advance_drive_state(
            30.0,
            (0.0, 1.0, 0.0),
            state,
            acs_mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )
        assert panel.model_dump() == before
        assert SolarPanel.model_validate(before).model_dump() == before

    def test_state_is_aligned_with_configured_panel_indices(self) -> None:
        panel_set = SolarPanelSet(
            panels=[SolarPanel(), _tracking_panel(), SolarPanel(gimbled=True)]
        )
        state = panel_set.initial_drive_state()
        assert state.angles_deg == (None, 0.0, None)
        assert state.driven_angles_deg == [0.0]

    def test_drive_transition_is_rate_limited_and_immutable(self) -> None:
        panel_set = SolarPanelSet(panels=[_tracking_panel()])
        initial = panel_set.initial_drive_state()
        at_zero = panel_set.advance_drive_state(
            0.0,
            (0.0, 1.0, 0.0),
            initial,
            acs_mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )
        after_30_s = panel_set.advance_drive_state(
            30.0,
            (0.0, 1.0, 0.0),
            at_zero,
            acs_mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )
        assert initial.angles_deg == (0.0,)
        assert at_zero.angles_deg == (0.0,)
        assert after_30_s.angles_deg == pytest.approx((30.0,))
        assert after_30_s.revision == 2

    def test_uncommanded_and_eclipse_intervals_hold_without_accumulating_time(
        self,
    ) -> None:
        panel_set = SolarPanelSet(panels=[_tracking_panel()])
        state = panel_set.advance_drive_state(
            0.0,
            (0.0, 1.0, 0.0),
            panel_set.initial_drive_state(),
            acs_mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )
        state = panel_set.advance_drive_state(
            30.0,
            (0.0, 1.0, 0.0),
            state,
            acs_mode=ACSMode.SCIENCE,
            in_eclipse=True,
        )
        state = panel_set.advance_drive_state(
            60.0,
            (0.0, 1.0, 0.0),
            state,
            acs_mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )
        assert state.angles_deg == pytest.approx((30.0,))

    def test_transition_rejects_backward_time_and_misaligned_state(self) -> None:
        panel_set = SolarPanelSet(panels=[_tracking_panel()])
        state = panel_set.advance_drive_state(
            60.0,
            (0.0, 1.0, 0.0),
            panel_set.initial_drive_state(),
            acs_mode=ACSMode.SCIENCE,
            in_eclipse=False,
        )
        with pytest.raises(ValueError, match="backward in time"):
            panel_set.advance_drive_state(
                30.0,
                (0.0, 1.0, 0.0),
                state,
                acs_mode=ACSMode.SCIENCE,
                in_eclipse=False,
            )
        with pytest.raises(ValueError, match="one entry per configured panel"):
            panel_set.power_from_normalized_sun_body(
                np.asarray([(1.0, 0.0, 0.0)]),
                drive_state=SolarArrayDriveState(angles_deg=()),
            )

    def test_legacy_ideal_gimbal_cannot_also_use_finite_drive(self) -> None:
        with pytest.raises(ValidationError, match="mutually exclusive"):
            SolarPanel(gimbled=True, single_axis_drive=_drive())

    def test_static_shadow_geometry_cannot_use_finite_drive(self) -> None:
        geometry = PanelGeometry(u=(1.0, 0.0, 0.0), v=(0.0, 0.0, 1.0))
        with pytest.raises(ValidationError, match="articulated shadow transforms"):
            SolarPanel(geometry=geometry, single_axis_drive=_drive())


@pytest.mark.usefixtures("eclipse")
class TestExecutedEvaluation:
    def test_endpoint_power_and_next_state_are_returned_together(self) -> None:
        panel_set = SolarPanelSet(panels=[_tracking_panel()])
        ephem = _ephem_with_sun((0.0, 1.0, 0.0))
        _, _, state = panel_set.evaluate_executed_attitude(
            _START,
            0.0,
            0.0,
            ephem,
            panel_set.initial_drive_state(),
            acs_mode=ACSMode.SCIENCE,
        )
        illumination, power, next_state = panel_set.evaluate_executed_attitude(
            _START + timedelta(seconds=30.0),
            0.0,
            0.0,
            ephem,
            state,
            acs_mode=ACSMode.SCIENCE,
        )
        assert next_state.driven_angles_deg == pytest.approx([30.0])
        assert illumination == pytest.approx(0.5)
        assert power == pytest.approx(50.0)

    def test_scoring_is_pure_and_uses_the_supplied_state(self) -> None:
        panel_set = SolarPanelSet(panels=[_tracking_panel()])
        state = SolarArrayDriveState((30.0,), updated_at_s=30.0, revision=1)
        before = state
        scores = panel_set.power_from_normalized_sun_body(
            np.asarray([(0.0, 1.0, 0.0), (1.0, 0.0, 0.0)]),
            drive_state=state,
        )
        assert scores == pytest.approx((50.0, 100.0 * np.cos(np.deg2rad(30.0))))
        assert state is before


class TestDriveAwareRollSelection:
    def test_fixed_panel_scoring_normalizes_each_panel_independently(self) -> None:
        panel_set = SolarPanelSet(
            panels=[
                SolarPanel(
                    normal=(0.0, 2.0, 0.0),
                    max_power=1.0,
                    conversion_efficiency=1.0,
                ),
                SolarPanel(
                    normal=(0.0, 0.0, 3.0),
                    max_power=1.0,
                    conversion_efficiency=1.0,
                ),
            ],
            conversion_efficiency=1.0,
        )
        diagonal = 1.0 / np.sqrt(2.0)
        scores = panel_set.power_from_normalized_sun_body(
            np.asarray([(0.0, 1.0, 0.0), (0.0, diagonal, diagonal)])
        )
        assert scores == pytest.approx((1.0, np.sqrt(2.0)))

    def test_roll_search_reads_state_without_advancing_it(self) -> None:
        panel_set = SolarPanelSet(
            panels=[
                _tracking_panel(
                    normal=(0.0, 1.0, 0.0),
                    rotation_axis=(1.0, 0.0, 0.0),
                )
            ]
        )
        state = SolarArrayDriveState((30.0,), updated_at_s=30.0, revision=4)
        ephem = _ephem_with_sun((1.0, 0.0, 1.0))
        roll = optimum_roll(
            0.0,
            0.0,
            30.0,
            ephem,
            panel_set,
            drive_state=state,
        )
        assert roll == pytest.approx(60.0)
        assert state == SolarArrayDriveState((30.0,), updated_at_s=30.0, revision=4)

    def test_ideal_gimbal_is_flat_in_fast_path(self) -> None:
        panel_set = SolarPanelSet(
            panels=[SolarPanel(gimbled=True, normal=(0.0, 1.0, 0.0))]
        )
        ephem = _ephem_with_sun((0.0, 1.0, 0.0))
        assert optimum_roll(0.0, 0.0, 0.0, ephem, panel_set) == 0.0
        assert (
            optimum_roll(
                0.0,
                0.0,
                0.0,
                ephem,
                panel_set,
                reference_roll=90.0,
                max_roll_delta=180.0,
            )
            == 90.0
        )

    def test_mounted_instrument_scores_finite_drive_in_body_frame(self) -> None:
        panel_set = SolarPanelSet(panels=[_tracking_panel()])
        state = SolarArrayDriveState((30.0,), updated_at_s=30.0, revision=1)
        telescope = Telescope(boresight=(0.0, 1.0, 0.0))
        ephem = _ephem_with_sun((0.2, 0.7, 1.0))

        selected = optimum_instrument_roll(
            15.0,
            -20.0,
            30.0,
            ephem,
            telescope,
            panel_set,
            drive_state=state,
        )

        sun_eci = ephem.sun_pv.position[0] - ephem.gcrs_pv.position[0]
        scores = []
        for roll in range(360):
            body_attitude = telescope.target_body_attitude(15.0, -20.0, float(roll))
            sun_body = scbodyvector(
                body_attitude[0] * DTOR,
                body_attitude[1] * DTOR,
                body_attitude[2] * DTOR,
                sun_eci,
            )
            scores.append(
                panel_set.power_from_normalized_sun_body(
                    np.asarray([sun_body]), drive_state=state
                )[0]
            )
        assert scores[int(selected)] == pytest.approx(max(scores))
