"""The tilted dipole geomagnetic field."""

import numpy as np
import pytest

from conops.simulation.geomagnetic import (
    DIPOLE_SURFACE_FIELD_T,
    EARTH_RADIUS_KM,
    TiltedDipoleField,
)


def _untilted() -> TiltedDipoleField:
    """A dipole along the rotation axis, so magnetic and geographic agree."""
    return TiltedDipoleField(pole_deg=(90.0, 0.0))


class TestTiltedDipoleField:
    def test_the_equatorial_surface_field_points_north(self) -> None:
        field = _untilted()
        position = np.array([EARTH_RADIUS_KM, 0.0, 0.0])
        b = field.field_eci_t(position, position)
        # On the magnetic equator the field is horizontal, towards the
        # geographic north pole, at the surface strength.
        assert b == pytest.approx([0.0, 0.0, DIPOLE_SURFACE_FIELD_T], rel=1e-12)

    def test_the_polar_field_is_twice_the_equatorial_and_points_down(self) -> None:
        field = _untilted()
        north = np.array([0.0, 0.0, EARTH_RADIUS_KM])
        b = field.field_eci_t(north, north)
        # Field lines enter the northern hemisphere: downwards at the north pole.
        assert b == pytest.approx([0.0, 0.0, -2.0 * DIPOLE_SURFACE_FIELD_T], rel=1e-12)
        south = -north
        assert field.field_eci_t(south, south) == pytest.approx(
            [0.0, 0.0, -2.0 * DIPOLE_SURFACE_FIELD_T], rel=1e-12
        )

    def test_the_field_falls_with_the_cube_of_the_radius(self) -> None:
        field = _untilted()
        position = np.array([2.0 * EARTH_RADIUS_KM, 0.0, 0.0])
        b = field.field_eci_t(position, position)
        assert np.linalg.norm(b) == pytest.approx(
            DIPOLE_SURFACE_FIELD_T / 8.0, rel=1e-12
        )

    def test_the_earth_rotation_is_taken_from_the_two_positions(self) -> None:
        """The same Earth-fixed point seen a quarter turn later in inertial
        coordinates gives the same field rotated by that quarter turn."""
        field = TiltedDipoleField()
        ecef = np.array([EARTH_RADIUS_KM + 600.0, 0.0, 0.0])
        eci_now = ecef.copy()
        eci_later = np.array([0.0, EARTH_RADIUS_KM + 600.0, 0.0])
        b_now = field.field_eci_t(eci_now, ecef)
        b_later = field.field_eci_t(eci_later, ecef)
        turned = np.array([-b_now[1], b_now[0], b_now[2]])
        assert b_later == pytest.approx(turned, rel=1e-12)

    def test_the_tilt_moves_the_pole(self) -> None:
        """At the geomagnetic pole the field is vertical and twice the
        equatorial strength, though it is not the geographic pole."""
        field = TiltedDipoleField()
        latitude, longitude = np.deg2rad((80.65, -72.68))
        direction = np.array(
            [
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            ]
        )
        position = EARTH_RADIUS_KM * direction
        b = field.field_eci_t(position, position)
        assert np.linalg.norm(b) == pytest.approx(
            2.0 * DIPOLE_SURFACE_FIELD_T, rel=1e-12
        )
        assert b @ direction == pytest.approx(-np.linalg.norm(b), rel=1e-12)

    def test_invalid_inputs_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            TiltedDipoleField(surface_field_t=0.0)
        with pytest.raises(ValueError, match="nonzero"):
            _untilted().field_eci_t(np.zeros(3), np.zeros(3))


class TestIGRFField:
    def test_the_igrf_is_the_dipole_to_a_third(self) -> None:
        """At 600 km over the equator the IGRF and the tilted dipole agree
        in strength to a third and in direction to forty degrees, the South
        Atlantic anomaly's longitude included: the dipole is the field's
        leading term."""
        pytest.importorskip("ppigrf")
        from conops.simulation.geomagnetic import IGRFField, field_from_name

        igrf = field_from_name("igrf")
        assert isinstance(igrf, IGRFField) and igrf.name == "igrf"
        dipole = TiltedDipoleField()
        for longitude in (0.0, 90.0, 180.0, 270.0):
            lon = np.radians(longitude)
            ecef = np.array([np.cos(lon), np.sin(lon), 0.0]) * (EARTH_RADIUS_KM + 600.0)
            eci = ecef.copy()
            real = igrf.field_eci_t(eci, ecef, 1.76e9)
            model = dipole.field_eci_t(eci, ecef)
            assert np.linalg.norm(real) == pytest.approx(
                np.linalg.norm(model), rel=0.35
            )
            cosine = float(
                real @ model / (np.linalg.norm(real) * np.linalg.norm(model))
            )
            assert cosine > np.cos(np.radians(40.0))
        with pytest.raises(ValueError):
            field_from_name("other")

    def test_the_dipole_is_the_default_name(self) -> None:
        from conops.simulation.geomagnetic import field_from_name

        assert field_from_name("dipole").name == "dipole"
