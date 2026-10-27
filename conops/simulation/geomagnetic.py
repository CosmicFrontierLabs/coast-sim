"""The Earth's magnetic field, for the magnetorquers and a residual dipole.

Two models behind one interface. The built-in one is a tilted dipole, which
gives the field's strength and direction to ten or twenty per cent of the
real field: enough for the torques' order of magnitude. The other is the
IGRF through the optional ``ppigrf`` package, for a run that wants the real
field, its non-dipole terms and its drift with the epoch included.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ..common.geodesy import (
    ecef_to_eci_rotation,
    ecef_to_geodetic,
    enu_to_ecef_rotation,
)

EARTH_RADIUS_KM = 6371.2
DIPOLE_SURFACE_FIELD_T = 3.12e-5
"""The dipole field at the magnetic equator on the Earth's surface."""
GEOMAGNETIC_NORTH_POLE_DEG = (80.65, -72.68)
"""Latitude and longitude of the geomagnetic north pole (IGRF-13, 2020)."""


class GeomagneticField(Protocol):
    """The field at the spacecraft in inertial coordinates, in tesla."""

    name: str

    def field_eci_t(
        self,
        position_eci_km: npt.ArrayLike,
        position_ecef_km: npt.ArrayLike,
        utime: float | None = None,
    ) -> npt.NDArray[np.float64]: ...


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

    name = "dipole"

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
        return ecef_to_eci_rotation(position_ecef_km, position_eci_km)

    def field_eci_t(
        self,
        position_eci_km: npt.ArrayLike,
        position_ecef_km: npt.ArrayLike,
        utime: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """The field at the spacecraft in inertial coordinates, in tesla; the
        dipole does not change with the epoch, so ``utime`` is not used."""
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


class IGRFField:
    """The International Geomagnetic Reference Field through ``ppigrf``,
    at the sample's epoch, evaluated in inertial coordinates."""

    name = "igrf"

    def __init__(self) -> None:
        try:
            import ppigrf  # type: ignore[import-untyped]
        except ImportError as exc:  # pragma: no cover - exercised without the extra
            raise ImportError(
                "the 'igrf' field model needs the ppigrf package: "
                "pip install 'coast-sim[momentum]'"
            ) from exc
        self._igrf = ppigrf.igrf

    def field_eci_t(
        self,
        position_eci_km: npt.ArrayLike,
        position_ecef_km: npt.ArrayLike,
        utime: float | None = None,
    ) -> npt.NDArray[np.float64]:
        """The field at the spacecraft in inertial coordinates, in tesla, at
        the epoch ``utime``, which the IGRF needs; the IGRF-2020 epoch
        where none is given."""
        ecef = np.asarray(position_ecef_km, dtype=np.float64)
        eci = np.asarray(position_eci_km, dtype=np.float64)
        latitude, longitude, height = ecef_to_geodetic(ecef)
        when = (
            datetime.fromtimestamp(float(utime), tz=timezone.utc).replace(tzinfo=None)
            if utime is not None
            else datetime(2020, 1, 1)
        )
        east, north, up = self._igrf(longitude, latitude, height, when)
        enu_nt = np.array(
            [float(np.squeeze(east)), float(np.squeeze(north)), float(np.squeeze(up))],
            dtype=np.float64,
        )
        field_ecef = enu_to_ecef_rotation(latitude, longitude) @ enu_nt
        return np.asarray(ecef_to_eci_rotation(ecef, eci) @ field_ecef * 1e-9)


def field_from_name(name: str) -> GeomagneticField:
    """The field model a configuration names: ``dipole`` or ``igrf``."""
    if name == "igrf":
        return IGRFField()
    if name == "dipole":
        return TiltedDipoleField()
    raise ValueError(f"unknown geomagnetic field model {name!r}")
