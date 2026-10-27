"""Earth-fixed geometry the environment models share: the Earth's rotation,
the WGS84 ellipsoid, geodetic coordinates and the local frame at a point."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from .numerics import POLE_NORM

EARTH_ROTATION_RATE_RAD_S = 7.2921159e-5
"""The Earth's sidereal rotation rate."""
WGS84_EQUATORIAL_RADIUS_KM = 6378.137
WGS84_FLATTENING = 1.0 / 298.257223563
WGS84_POLAR_RADIUS_KM = WGS84_EQUATORIAL_RADIUS_KM * (1.0 - WGS84_FLATTENING)
_ECCENTRICITY_SQUARED = WGS84_FLATTENING * (2.0 - WGS84_FLATTENING)
_SECOND_ECCENTRICITY_SQUARED = _ECCENTRICITY_SQUARED / (1.0 - _ECCENTRICITY_SQUARED)


MEAN_EARTH_RADIUS_KM = 6371.0
"""The radius of a sphere of the Earth's mean radius."""


def ecef_to_geodetic(position_ecef_km: npt.ArrayLike) -> tuple[float, float, float]:
    """Geodetic latitude and longitude in degrees and height above the WGS84
    ellipsoid in kilometres of an Earth-fixed position, by Bowring's
    closed form, exact to a millimetre at any height a spacecraft flies."""
    x, y, z = (float(c) for c in np.asarray(position_ecef_km, dtype=np.float64))
    distance = float(np.hypot(x, y))
    a, b = WGS84_EQUATORIAL_RADIUS_KM, WGS84_POLAR_RADIUS_KM
    if distance <= 0.0:
        latitude = np.pi / 2 if z >= 0.0 else -np.pi / 2
        return float(np.degrees(latitude)), 0.0, abs(z) - b
    reduced = np.arctan2(z * a, distance * b)
    latitude = np.arctan2(
        z + _SECOND_ECCENTRICITY_SQUARED * b * np.sin(reduced) ** 3,
        distance - _ECCENTRICITY_SQUARED * a * np.cos(reduced) ** 3,
    )
    longitude = np.arctan2(y, x)
    prime_vertical = a / np.sqrt(1.0 - _ECCENTRICITY_SQUARED * np.sin(latitude) ** 2)
    cosine = float(np.cos(latitude))
    if abs(cosine) > POLE_NORM:
        height = distance / cosine - prime_vertical
    else:
        height = abs(z) - b
    return float(np.degrees(latitude)), float(np.degrees(longitude)), float(height)


def enu_to_ecef_rotation(
    latitude_deg: float, longitude_deg: float
) -> npt.NDArray[np.float64]:
    """The rotation taking a vector in the local east, north, up frame at a
    point to the Earth-fixed frame."""
    latitude, longitude = np.radians(latitude_deg), np.radians(longitude_deg)
    sin_lat, cos_lat = np.sin(latitude), np.cos(latitude)
    sin_lon, cos_lon = np.sin(longitude), np.cos(longitude)
    return np.array(
        [
            [-sin_lon, -sin_lat * cos_lon, cos_lat * cos_lon],
            [cos_lon, -sin_lat * sin_lon, cos_lat * sin_lon],
            [0.0, cos_lat, sin_lat],
        ],
        dtype=np.float64,
    )


def ecef_to_eci_rotation(
    position_ecef_km: npt.ArrayLike, position_eci_km: npt.ArrayLike
) -> npt.NDArray[np.float64]:
    """The rotation about the pole taking the Earth-fixed position to the
    inertial one: the Earth's rotation at that instant. The precession and
    nutation it leaves out move a direction by a tenth of a degree, far
    inside what the environment models themselves get wrong."""
    ecef = np.asarray(position_ecef_km, dtype=np.float64)
    eci = np.asarray(position_eci_km, dtype=np.float64)
    angle = float(np.arctan2(eci[1], eci[0]) - np.arctan2(ecef[1], ecef[0]))
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array(
        [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def geocentric_height_km(position_eci_km: npt.ArrayLike) -> float:
    """Height above a sphere of the Earth's mean radius, for a position
    whose Earth-fixed counterpart is unknown."""
    return (
        float(np.linalg.norm(np.asarray(position_eci_km, dtype=np.float64)))
        - MEAN_EARTH_RADIUS_KM
    )
