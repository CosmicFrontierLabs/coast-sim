"""The Earth's magnetic field as a tilted dipole.

Magnetorquers unload the wheels against the field, and a residual dipole on
the spacecraft is torqued by it. A tilted dipole is enough for either at
planning level: it gives the field's strength and direction to ten or twenty
per cent of the real field, which is what the torques' order of magnitude
needs. A full field model (IGRF) can replace it behind the same interface.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

EARTH_RADIUS_KM = 6371.2
DIPOLE_SURFACE_FIELD_T = 3.12e-5
"""The dipole field at the magnetic equator on the Earth's surface."""
GEOMAGNETIC_NORTH_POLE_DEG = (80.65, -72.68)
"""Latitude and longitude of the geomagnetic north pole (IGRF-13, 2020)."""


def _unit(vector: npt.ArrayLike, name: str) -> npt.NDArray[np.float64]:
    array = np.asarray(vector, dtype=np.float64)
    if array.shape != (3,) or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain three finite values")
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        raise ValueError(f"{name} must have nonzero magnitude")
    return array / norm


class TiltedDipoleField:
    """The geomagnetic field as a dipole at the Earth's centre, tilted to the
    geomagnetic pole, evaluated in inertial coordinates.

    The dipole axis is fixed in the Earth-fixed frame. At a sample it is
    rotated into inertial coordinates by the rotation about the pole that
    takes the spacecraft's Earth-fixed position to its inertial one; the
    precession and nutation that rotation leaves out move the axis by a
    tenth of a degree, far inside what a dipole gets wrong anyway.
    """

    def __init__(
        self,
        surface_field_t: float = DIPOLE_SURFACE_FIELD_T,
        pole_deg: tuple[float, float] = GEOMAGNETIC_NORTH_POLE_DEG,
    ) -> None:
        if not np.isfinite(surface_field_t) or surface_field_t <= 0.0:
            raise ValueError("surface field must be finite and positive")
        self.surface_field_t = float(surface_field_t)
        latitude, longitude = np.deg2rad(pole_deg)
        # The field lines leave the southern hemisphere and enter the northern,
        # so the dipole moment points from the geomagnetic north pole through
        # the centre to the south.
        self.moment_direction_ecef = -np.array(
            [
                np.cos(latitude) * np.cos(longitude),
                np.cos(latitude) * np.sin(longitude),
                np.sin(latitude),
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _ecef_to_eci(
        position_ecef_km: npt.NDArray[np.float64],
        position_eci_km: npt.NDArray[np.float64],
    ) -> npt.NDArray[np.float64]:
        """The rotation about the pole taking the Earth-fixed position to the
        inertial one: the Earth's rotation at that instant."""
        angle = float(
            np.arctan2(position_eci_km[1], position_eci_km[0])
            - np.arctan2(position_ecef_km[1], position_ecef_km[0])
        )
        cosine, sine = np.cos(angle), np.sin(angle)
        return np.array(
            [[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )

    def field_eci_t(
        self,
        position_eci_km: npt.ArrayLike,
        position_ecef_km: npt.ArrayLike,
    ) -> npt.NDArray[np.float64]:
        """The field at the spacecraft in inertial coordinates, in tesla."""
        position = np.asarray(position_eci_km, dtype=np.float64)
        radius_km = float(np.linalg.norm(position))
        if radius_km <= 0.0:
            raise ValueError("position must have nonzero magnitude")
        radial = position / radius_km
        rotation = self._ecef_to_eci(
            np.asarray(position_ecef_km, dtype=np.float64), position
        )
        moment = rotation @ self.moment_direction_ecef
        scale = self.surface_field_t * (EARTH_RADIUS_KM / radius_km) ** 3
        return np.asarray(
            scale * (3.0 * float(moment @ radial) * radial - moment), dtype=np.float64
        )
