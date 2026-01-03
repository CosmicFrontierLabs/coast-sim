from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from pymsis import calculate  # type: ignore[import-untyped]
from pymsis.msis import Variable  # type: ignore[import-untyped]

from ..common import dtutcfromtimestamp


@dataclass
class MsisConfig:
    use_msis_density: bool = False
    msis_f107: float = 200.0
    msis_f107a: float = 180.0
    msis_ap: float = 12.0
    msis_density_scale: float = 1.0


class MsisAdapter:
    """Lightweight MSIS wrapper to normalize output shape/units and apply sanity checks."""

    _MIN_DENSITY = 1e-18
    _MAX_DENSITY = 1e-10

    def __init__(self, cfg: MsisConfig) -> None:
        self.cfg = cfg
        self._logger = logging.getLogger(__name__)
        self.last_density_out_of_range: float | None = None

    def density(
        self,
        utime: float,
        alt_m: float,
        lat_deg: float,
        lon_deg: float,
        table_density: float,
    ) -> float | None:
        """Return MSIS mass density (kg/m^3), or None to signal fallback."""
        if not self.cfg.use_msis_density:
            return None

        date = dtutcfromtimestamp(utime)
        dates = np.array([date])
        alts = np.array([max(0.0, alt_m / 1000.0)])
        lats = np.array([lat_deg])
        lons = np.array([lon_deg])
        f107 = np.array([self.cfg.msis_f107])
        f107a = np.array([self.cfg.msis_f107a])
        ap = np.array([[self.cfg.msis_ap] * 7])
        dens = calculate(
            dates,
            lons,
            lats,
            alts,
            f107s=f107,
            f107as=f107a,
            aps=ap,
            version="00",
        )
        dens_arr = np.asarray(dens, dtype=float)
        idx = int(Variable.MASS_DENSITY)
        if dens_arr.size == 0:
            return None
        if dens_arr.ndim == 1 and dens_arr.size > idx:
            rho_val = dens_arr[idx]
        elif dens_arr.shape[-1] > idx:
            rho_val = dens_arr[..., idx]
        else:
            rho_val = dens_arr
        rho = float(np.ravel(rho_val)[0]) * float(self.cfg.msis_density_scale)
        if not np.isfinite(rho) or rho <= 0:
            return None
        if rho < self._MIN_DENSITY or rho > self._MAX_DENSITY:
            self.last_density_out_of_range = rho
            self._logger.warning(
                "MSIS density %.3e kg/m^3 outside sanity range [%.0e, %.0e]; "
                "falling back to table density %.3e kg/m^3",
                rho,
                self._MIN_DENSITY,
                self._MAX_DENSITY,
                table_density,
            )
            return None
        return rho
