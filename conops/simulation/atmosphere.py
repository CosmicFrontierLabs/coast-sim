"""The upper atmosphere's density, for aerodynamic drag.

Two models behind one interface. The built-in one is a piecewise
exponential profile by height (Vallado's table for moderate solar activity),
which gives the density to within a factor of a few at any height a
spacecraft flies and needs no data. The other is NRLMSIS 2.1 through the
optional ``pymsis`` package, which takes the solar flux and geomagnetic
activity the run assumes, and is the model to use where the density matters,
since it varies by a factor of twenty between solar minimum and maximum.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Protocol

import numpy as np
import numpy.typing as npt

from ..common.geodesy import ecef_to_geodetic, geocentric_height_km

if TYPE_CHECKING:
    from ..config.momentum import AtmosphereConfig


class AtmosphereModel(Protocol):
    """The atmosphere's mass density at a point and time."""

    name: str

    def density_kg_m3(
        self,
        utime: float,
        position_eci_km: npt.NDArray[np.float64],
        position_ecef_km: npt.NDArray[np.float64] | None,
    ) -> float: ...


_EXPONENTIAL_TABLE: tuple[tuple[float, float, float], ...] = (
    # base height km, density at it kg/m³, scale height km
    (0.0, 1.225, 7.249),
    (25.0, 3.899e-2, 6.349),
    (30.0, 1.774e-2, 6.682),
    (40.0, 3.972e-3, 7.554),
    (50.0, 1.057e-3, 8.382),
    (60.0, 3.206e-4, 7.714),
    (70.0, 8.770e-5, 6.549),
    (80.0, 1.905e-5, 5.799),
    (90.0, 3.396e-6, 5.382),
    (100.0, 5.297e-7, 5.877),
    (110.0, 9.661e-8, 7.263),
    (120.0, 2.438e-8, 9.473),
    (130.0, 8.484e-9, 12.636),
    (140.0, 3.845e-9, 16.149),
    (150.0, 2.070e-9, 22.523),
    (180.0, 5.464e-10, 29.740),
    (200.0, 2.789e-10, 37.105),
    (250.0, 7.248e-11, 45.546),
    (300.0, 2.418e-11, 53.628),
    (350.0, 9.518e-12, 53.298),
    (400.0, 3.725e-12, 58.515),
    (450.0, 1.585e-12, 60.828),
    (500.0, 6.967e-13, 63.822),
    (600.0, 1.454e-13, 71.835),
    (700.0, 3.614e-14, 88.667),
    (800.0, 1.170e-14, 124.64),
    (900.0, 5.245e-15, 181.05),
    (1000.0, 3.019e-15, 268.00),
)
"""Vallado, Fundamentals of Astrodynamics and Applications, table 8-4:
a piecewise exponential atmosphere for moderate solar activity."""


def exponential_density_kg_m3(height_km: float) -> float:
    """The built-in table's density at a geodetic height, in kg/m³: the
    base density of the band the height falls in, decayed by the band's
    scale height; above the last band the last band's decay continues."""
    if not np.isfinite(height_km):
        raise ValueError("height must be finite")
    if height_km <= 0.0:
        return _EXPONENTIAL_TABLE[0][1]
    base, density, scale = _EXPONENTIAL_TABLE[0]
    for row in _EXPONENTIAL_TABLE:
        if row[0] > height_km:
            break
        base, density, scale = row
    return float(density * np.exp(-(height_km - base) / scale))


def _height_km(
    position_eci_km: npt.NDArray[np.float64],
    position_ecef_km: npt.NDArray[np.float64] | None,
) -> float:
    if position_ecef_km is not None:
        return ecef_to_geodetic(position_ecef_km)[2]
    return geocentric_height_km(position_eci_km)


class ExponentialAtmosphere:
    """The built-in piecewise exponential atmosphere, scaled by a factor
    for sensitivity studies (twenty covers quiet Sun to solar maximum)."""

    name = "table"

    def __init__(self, density_scale: float = 1.0) -> None:
        if not np.isfinite(density_scale) or density_scale <= 0.0:
            raise ValueError("density_scale must be finite and positive")
        self.density_scale = float(density_scale)

    def density_kg_m3(
        self,
        utime: float,
        position_eci_km: npt.NDArray[np.float64],
        position_ecef_km: npt.NDArray[np.float64] | None,
    ) -> float:
        return self.density_scale * exponential_density_kg_m3(
            _height_km(position_eci_km, position_ecef_km)
        )


class MsisAtmosphere:
    """NRLMSIS 2.1 through ``pymsis``, at the solar flux and geomagnetic
    activity given, so a run assumes one level of activity rather than
    fetching the indices of its dates. Needs an Earth-fixed position for
    the geodetic coordinates; without one the built-in table is used."""

    name = "msis"

    def __init__(
        self,
        f107: float = 150.0,
        f107a: float = 150.0,
        ap: float = 4.0,
        density_scale: float = 1.0,
        version: float = 2.1,
    ) -> None:
        try:
            import pymsis
        except ImportError as exc:  # pragma: no cover - exercised without the extra
            raise ImportError(
                "the 'msis' atmosphere needs the pymsis package: "
                "pip install 'coast-sim[momentum]'"
            ) from exc
        self._calculate = pymsis.calculate
        for name, value in (("f107", f107), ("f107a", f107a), ("ap", ap)):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and not negative")
        if not np.isfinite(density_scale) or density_scale <= 0.0:
            raise ValueError("density_scale must be finite and positive")
        self.f107 = float(f107)
        self.f107a = float(f107a)
        self.ap = float(ap)
        self.density_scale = float(density_scale)
        self.version = version
        self._fallback = ExponentialAtmosphere(density_scale)

    def density_kg_m3(
        self,
        utime: float,
        position_eci_km: npt.NDArray[np.float64],
        position_ecef_km: npt.NDArray[np.float64] | None,
    ) -> float:
        if position_ecef_km is None:
            return self._fallback.density_kg_m3(utime, position_eci_km, None)
        latitude, longitude, height = ecef_to_geodetic(position_ecef_km)
        when = datetime.fromtimestamp(float(utime), tz=timezone.utc).replace(
            tzinfo=None
        )
        output = self._calculate(
            np.array([np.datetime64(when)]),
            [longitude],
            [latitude],
            [height],
            f107s=[self.f107],
            f107as=[self.f107a],
            aps=[[self.ap] * 7],
            version=self.version,
        )
        density = float(np.asarray(output).reshape(-1)[0])
        if not np.isfinite(density):
            # MSIS returns NaN above its range; nothing is there to drag on.
            return 0.0
        return self.density_scale * density


def atmosphere_from_config(config: AtmosphereConfig) -> AtmosphereModel:
    """The atmosphere a configuration selects."""
    if config.model == "msis":
        return MsisAtmosphere(
            f107=config.f107,
            f107a=config.f107a,
            ap=config.ap,
            density_scale=config.density_scale,
        )
    return ExponentialAtmosphere(config.density_scale)
