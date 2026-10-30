"""The atmosphere models and the geodetic geometry they stand on."""

import numpy as np
import pytest

from conops.common.geodesy import (
    EARTH_ROTATION_RATE_RAD_S,
    WGS84_EQUATORIAL_RADIUS_KM,
    WGS84_POLAR_RADIUS_KM,
    ecef_to_eci_rotation,
    ecef_to_geodetic,
    enu_to_ecef_rotation,
)
from conops.config import AtmosphereConfig
from conops.simulation.atmosphere import (
    ExponentialAtmosphere,
    MsisAtmosphere,
    atmosphere_from_config,
    exponential_density_kg_m3,
)

UTIME = 1.76e9


class TestGeodesy:
    def test_the_equator_and_the_pole(self) -> None:
        assert ecef_to_geodetic(
            [WGS84_EQUATORIAL_RADIUS_KM, 0.0, 0.0]
        ) == pytest.approx((0.0, 0.0, 0.0))
        latitude, _, height = ecef_to_geodetic(
            [0.0, 0.0, WGS84_POLAR_RADIUS_KM + 500.0]
        )
        assert latitude == pytest.approx(90.0)
        assert height == pytest.approx(500.0)

    def test_a_height_above_the_ellipsoid_round_trips(self) -> None:
        latitude, longitude, height = 45.0, -120.0, 600.0
        lat, lon = np.radians(latitude), np.radians(longitude)
        e2 = 1.0 - (WGS84_POLAR_RADIUS_KM / WGS84_EQUATORIAL_RADIUS_KM) ** 2
        n = WGS84_EQUATORIAL_RADIUS_KM / np.sqrt(1.0 - e2 * np.sin(lat) ** 2)
        ecef = [
            (n + height) * np.cos(lat) * np.cos(lon),
            (n + height) * np.cos(lat) * np.sin(lon),
            (n * (1.0 - e2) + height) * np.sin(lat),
        ]
        assert ecef_to_geodetic(ecef) == pytest.approx(
            (latitude, longitude, height), abs=1e-4
        )

    def test_the_local_frame_is_orthonormal_and_up_is_radial_at_the_equator(
        self,
    ) -> None:
        rotation = enu_to_ecef_rotation(0.0, 90.0)
        assert np.allclose(rotation @ rotation.T, np.eye(3))
        assert np.allclose(rotation @ [0.0, 0.0, 1.0], [0.0, 1.0, 0.0])

    def test_the_earth_rotation_takes_the_fixed_position_to_the_inertial_one(
        self,
    ) -> None:
        ecef = np.array([7000.0, 0.0, 0.0])
        eci = np.array([0.0, 7000.0, 0.0])
        assert np.allclose(ecef_to_eci_rotation(ecef, eci) @ ecef, eci)
        assert EARTH_ROTATION_RATE_RAD_S == pytest.approx(2 * np.pi / 86164.1, rel=1e-4)


class TestExponentialAtmosphere:
    def test_the_table_decays_with_height(self) -> None:
        assert exponential_density_kg_m3(0.0) == pytest.approx(1.225)
        assert exponential_density_kg_m3(400.0) == pytest.approx(3.725e-12)
        assert exponential_density_kg_m3(600.0) == pytest.approx(1.454e-13)
        between = exponential_density_kg_m3(450.0)
        assert (
            exponential_density_kg_m3(500.0)
            < between
            < exponential_density_kg_m3(400.0)
        )
        assert exponential_density_kg_m3(1500.0) < exponential_density_kg_m3(1000.0)
        with pytest.raises(ValueError):
            exponential_density_kg_m3(float("nan"))

    def test_the_model_uses_the_geodetic_height_and_scales(self) -> None:
        ecef = np.array([WGS84_EQUATORIAL_RADIUS_KM + 600.0, 0.0, 0.0])
        eci = ecef.copy()
        plain = ExponentialAtmosphere().density_kg_m3(UTIME, eci, ecef)
        assert plain == pytest.approx(exponential_density_kg_m3(600.0))
        assert ExponentialAtmosphere(2.0).density_kg_m3(
            UTIME, eci, ecef
        ) == pytest.approx(2.0 * plain)
        # Without an Earth-fixed position the geocentric height stands in.
        assert ExponentialAtmosphere().density_kg_m3(UTIME, eci, None) > 0.0
        with pytest.raises(ValueError):
            ExponentialAtmosphere(0.0)


class TestMsisAtmosphere:
    def test_the_density_follows_the_solar_activity(self) -> None:
        pytest.importorskip("pymsis")
        ecef = np.array([WGS84_EQUATORIAL_RADIUS_KM + 600.0, 0.0, 0.0])
        eci = ecef.copy()
        moderate = MsisAtmosphere().density_kg_m3(UTIME, eci, ecef)
        quiet = MsisAtmosphere(f107=70.0, f107a=70.0).density_kg_m3(UTIME, eci, ecef)
        active = MsisAtmosphere(f107=250.0, f107a=250.0, ap=20.0).density_kg_m3(
            UTIME, eci, ecef
        )
        assert quiet < moderate < active
        assert active / quiet > 10.0
        # Of the order of the table at the same height.
        assert 0.1 < moderate / exponential_density_kg_m3(600.0) < 10.0
        assert MsisAtmosphere(density_scale=3.0).density_kg_m3(
            UTIME, eci, ecef
        ) == pytest.approx(3.0 * moderate)
        with pytest.raises(ValueError):
            MsisAtmosphere(f107=-1.0)

    def test_the_configuration_selects_the_model(self) -> None:
        assert isinstance(
            atmosphere_from_config(AtmosphereConfig()), ExponentialAtmosphere
        )
        pytest.importorskip("pymsis")
        model = atmosphere_from_config(
            AtmosphereConfig(model="msis", f107=120.0, f107a=110.0, ap=8.0)
        )
        assert isinstance(model, MsisAtmosphere)
        assert (model.f107, model.f107a, model.ap) == (120.0, 110.0, 8.0)
