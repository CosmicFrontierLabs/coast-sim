"""The body frame the mission writes: which body axes the boresight and the
Sun's side lie along, declared once on the bus."""

import math
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from conops.common.vector import (
    quaternion_to_rotation_matrix,
    rotation_matrix_to_quaternion,
)
from conops.config import (
    InstrumentMounting,
    MissionConfig,
    Payload,
    SpacecraftBus,
    SpacecraftFrame,
    Telescope,
    create_solar_panel_vector,
    create_star_tracker_vector,
)

Z_BORESIGHT = SpacecraftFrame(boresight=(0.0, 0.0, 1.0), sun_side=(0.0, -1.0, 0.0))


class TestFrame:
    def test_the_default_is_the_canonical_frame(self) -> None:
        frame = SpacecraftFrame()
        assert frame.is_identity
        assert frame.quaternion_wxyz == (1.0, 0.0, 0.0, 0.0)
        assert frame.body_vector((0.0, 1.0, 0.0)) == (0.0, 1.0, 0.0)
        assert frame.mounting.is_identity

    def test_the_axes_become_the_frame_s_columns(self) -> None:
        matrix = Z_BORESIGHT.body_from_canonical
        assert matrix[:, 0].tolist() == [0.0, 0.0, 1.0]
        assert matrix[:, 1].tolist() == [0.0, -1.0, 0.0]
        assert matrix[:, 2].tolist() == [1.0, 0.0, 0.0]
        assert Z_BORESIGHT.body_vector((1.0, 0.0, 0.0)) == (0.0, 0.0, 1.0)
        assert Z_BORESIGHT.body_vector((0.0, 1.0, 0.0)) == (0.0, -1.0, 0.0)
        assert Z_BORESIGHT.canonical_vector((0.0, 0.0, 1.0)) == (1.0, 0.0, 0.0)

    def test_the_mounting_points_a_telescope_along_the_boresight(self) -> None:
        mounting = Z_BORESIGHT.mounting
        assert isinstance(mounting, InstrumentMounting)
        assert mounting.boresight_body == pytest.approx((0.0, 0.0, 1.0))
        assert mounting.roll_reference_body == pytest.approx((1.0, 0.0, 0.0), abs=1e-12)

    @pytest.mark.parametrize(
        "axes",
        [
            {"boresight": (0.0, 0.0, 2.0)},
            {"sun_side": (1.0, 0.0, 0.0)},
            {"boresight": (0.0, 0.0, float("nan"))},
        ],
    )
    def test_axes_must_be_unit_and_perpendicular(self, axes: dict) -> None:
        with pytest.raises(ValidationError):
            SpacecraftFrame(**axes)

    def test_axes_within_the_tolerance_give_an_exact_rotation(self) -> None:
        """Validation allows a cosine of 1e-6 between the axes; the rotation
        takes the Sun's side perpendicular, so the quaternion, the mounting
        and a configuration built on the frame never fail."""
        frame = SpacecraftFrame(sun_side=(5e-7, 1.0, 0.0))
        matrix = frame.body_from_canonical
        assert np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-12)
        assert frame.mounting.boresight_body == pytest.approx((1.0, 0.0, 0.0))
        config = MissionConfig(
            spacecraft_bus=SpacecraftBus(frame=frame),
            payload=Payload(instruments=[Telescope(name="T")]),
        )
        assert isinstance(config.payload.instruments[0], Telescope)

    def test_a_tilted_frame_round_trips_through_its_quaternion(self) -> None:
        angle = math.radians(30.0)
        frame = SpacecraftFrame(
            boresight=(math.cos(angle), math.sin(angle), 0.0),
            sun_side=(-math.sin(angle), math.cos(angle), 0.0),
        )
        matrix = quaternion_to_rotation_matrix(frame.quaternion_wxyz)
        assert np.allclose(matrix, frame.body_from_canonical)
        assert np.allclose(
            rotation_matrix_to_quaternion(matrix), np.asarray(frame.quaternion_wxyz)
        )


class TestTelescopes:
    def _config(self, *instruments: Telescope) -> MissionConfig:
        return MissionConfig(
            spacecraft_bus=SpacecraftBus(frame=Z_BORESIGHT),
            payload=Payload(instruments=list(instruments)),
        )

    def test_a_telescope_without_a_mounting_takes_the_frame(self) -> None:
        config = self._config(Telescope(name="T"))
        telescope = config.payload.instruments[0]
        assert isinstance(telescope, Telescope)
        assert telescope.boresight == pytest.approx((0.0, 0.0, 1.0))
        assert telescope.mounting == Z_BORESIGHT.mounting

    def test_a_telescope_with_its_own_mounting_or_boresight_keeps_it(self) -> None:
        config = self._config(
            Telescope(name="side", boresight=(0.0, 1.0, 0.0)),
            Telescope(name="explicit", mounting=InstrumentMounting()),
        )
        side, explicit = config.payload.instruments
        assert isinstance(side, Telescope) and isinstance(explicit, Telescope)
        assert side.boresight == pytest.approx((0.0, 1.0, 0.0))
        assert explicit.mounting.is_identity

    def test_the_canonical_frame_leaves_telescopes_alone(self) -> None:
        config = MissionConfig(payload=Payload(instruments=[Telescope(name="T")]))
        telescope = config.payload.instruments[0]
        assert isinstance(telescope, Telescope)
        assert telescope.mounting.is_identity

    def test_the_frame_s_attitude_is_the_equivalent_mounting_s(self) -> None:
        from_frame = self._config(Telescope(name="T")).payload.instruments[0]
        mounted = Telescope(name="M", mounting=Z_BORESIGHT.mounting)
        assert isinstance(from_frame, Telescope)
        assert from_frame.target_body_attitude(10.0, 20.0, 30.0) == pytest.approx(
            mounted.target_body_attitude(10.0, 20.0, 30.0)
        )

    def test_the_frame_and_the_mounting_survive_the_yaml_round_trip(
        self, tmp_path: Path
    ) -> None:
        config = self._config(Telescope(name="T"))
        path = tmp_path / "mission.yaml"
        config.to_yaml_file(str(path))
        loaded = MissionConfig.from_yaml_file(str(path))
        assert loaded.spacecraft_bus.frame == Z_BORESIGHT
        telescope = loaded.payload.instruments[0]
        assert isinstance(telescope, Telescope)
        assert telescope.boresight == pytest.approx((0.0, 0.0, 1.0))


class TestHelpers:
    def test_panel_mounts_follow_the_frame(self) -> None:
        assert create_solar_panel_vector("boresight", frame=Z_BORESIGHT) == (
            pytest.approx((0.0, 0.0, 1.0))
        )
        assert create_solar_panel_vector("sidemount", frame=Z_BORESIGHT) == (
            pytest.approx((0.0, -1.0, 0.0))
        )
        assert create_solar_panel_vector("aftmount", frame=Z_BORESIGHT) == (
            pytest.approx((0.0, 0.0, -1.0))
        )
        assert create_solar_panel_vector(azimuth_deg=90.0, frame=Z_BORESIGHT) == (
            pytest.approx((1.0, 0.0, 0.0))
        )

    def test_a_canted_panel_is_canted_in_the_frame(self) -> None:
        canonical = create_solar_panel_vector("sidemount", cant_perp=15.0)
        in_frame = create_solar_panel_vector(
            "sidemount", cant_perp=15.0, frame=Z_BORESIGHT
        )
        assert in_frame == pytest.approx(Z_BORESIGHT.body_vector(canonical))

    def test_without_a_frame_the_helpers_are_unchanged(self) -> None:
        assert create_solar_panel_vector("boresight") == (1.0, 0.0, 0.0)
        assert create_star_tracker_vector() == (1.0, 0.0, 0.0)

    def test_star_trackers_follow_the_frame(self) -> None:
        assert create_star_tracker_vector(frame=Z_BORESIGHT) == (
            pytest.approx((0.0, 0.0, 1.0))
        )
        assert create_star_tracker_vector(pitch_deg=90.0, frame=Z_BORESIGHT) == (
            pytest.approx((1.0, 0.0, 0.0))
        )
