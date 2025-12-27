from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from pymsis import calculate  # type: ignore[import-untyped]

from ..common import dtutcfromtimestamp


@dataclass
class DisturbanceConfig:
    magnetorquer_bfield_T: float = 3e-5
    cp_offset_body: tuple[float, float, float] = (0.0, 0.0, 0.0)
    residual_magnetic_moment: tuple[float, float, float] = (0.0, 0.0, 0.0)
    drag_area_m2: float = 0.0
    drag_coeff: float = 2.2
    solar_area_m2: float = 0.0
    solar_reflectivity: float = 1.0
    use_msis_density: bool = False
    msis_f107: float = 200.0
    msis_f107a: float = 180.0
    msis_ap: float = 12.0


class DisturbanceModel:
    """Centralized disturbance torque model (GG/drag/SRP/mag + MSIS handling)."""

    RE_M = 6378e3
    MU_EARTH = 3.986004418e14  # m^3/s^2
    B0_T = 3.12e-5  # equatorial field at surface (T)
    AU_M = 1.495978707e11
    SRP_P0 = 4.56e-6  # N/m^2 at 1 AU

    def __init__(self, ephem: Any, cfg: DisturbanceConfig) -> None:
        self.ephem = ephem
        self.cfg = cfg

    @staticmethod
    def _inertial_to_body(vec: np.ndarray, ra_deg: float, dec_deg: float) -> np.ndarray:
        """Rotate inertial vector into body frame assuming Z points at RA/Dec."""
        vec_arr = np.array(vec, dtype=float)
        try:
            ra_r = np.deg2rad(float(ra_deg))
            dec_r = np.deg2rad(float(dec_deg))
            z_b = np.array(
                [
                    np.cos(dec_r) * np.cos(ra_r),
                    np.cos(dec_r) * np.sin(ra_r),
                    np.sin(dec_r),
                ]
            )
            x_candidate = np.array([1.0, 0.0, 0.0])
            x_b = x_candidate - np.dot(x_candidate, z_b) * z_b
            if np.linalg.norm(x_b) < 1e-6:
                x_candidate = np.array([0.0, 1.0, 0.0])
                x_b = x_candidate - np.dot(x_candidate, z_b) * z_b
            x_b = x_b / np.linalg.norm(x_b)
            y_b = np.cross(z_b, x_b)
            y_b = y_b / np.linalg.norm(y_b)
            r_ib = np.vstack([x_b, y_b, z_b])
            return np.asarray(r_ib.dot(vec_arr), dtype=float)
        except Exception:
            return vec_arr

    @staticmethod
    def _build_inertia(moi_cfg: Any) -> np.ndarray:
        try:
            if isinstance(moi_cfg, (list, tuple)):
                arr = np.array(moi_cfg, dtype=float)
                if arr.shape == (3, 3):
                    return arr
                if arr.shape == (3,):
                    return np.diag(arr)
                val = float(np.mean(arr))
                return np.diag([val, val, val])
            val = float(moi_cfg)
            return np.diag([val, val, val])
        except Exception:
            return np.diag([1.0, 1.0, 1.0])

    def local_bfield_vector(
        self, utime: float, ra_deg: float, dec_deg: float
    ) -> tuple[np.ndarray, float]:
        """Estimate local magnetic field vector in body frame."""
        b_const = np.array([0.0, 0.0, self.cfg.magnetorquer_bfield_T], dtype=float)
        try:
            idx = self.ephem.index(dtutcfromtimestamp(utime))
        except Exception:
            return b_const, float(np.linalg.norm(b_const))

        try:
            lat_deg = float(getattr(self.ephem, "latitude_deg", [0.0])[idx])
            lon_deg = float(getattr(self.ephem, "longitude_deg", [0.0])[idx])
        except Exception:
            lat_deg = 0.0
            lon_deg = 0.0

        try:
            pos = np.array(self.ephem.gcrs_pv.position[idx], dtype=float)
            rmag = float(np.linalg.norm(pos))
            if rmag < 1e5:
                rmag *= 1000.0
            alt_m = float(max(0.0, rmag - self.RE_M))
        except Exception:
            alt_m = 500e3

        lat = np.deg2rad(lat_deg)
        lon = np.deg2rad(lon_deg)
        r = float(self.RE_M + max(0.0, alt_m))
        scale = (self.RE_M / r) ** 3

        br = -2 * self.B0_T * scale * np.sin(lat)
        btheta = -self.B0_T * scale * np.cos(lat)
        r_hat = np.array(
            [np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)]
        )
        theta_hat = np.array(
            [-np.sin(lat) * np.cos(lon), -np.sin(lat) * np.sin(lon), np.cos(lat)]
        )
        b_ecef = br * r_hat + btheta * theta_hat
        b_body = self._inertial_to_body(b_ecef, ra_deg, dec_deg)
        return b_body, float(np.linalg.norm(b_body))

    def atmospheric_density(
        self, utime: float, alt_m: float, lat_deg: float | None, lon_deg: float | None
    ) -> float:
        """Lookup/interpolate atmospheric density (kg/m^3) vs altitude."""
        if self.cfg.use_msis_density:
            date = dtutcfromtimestamp(utime)
            dates = np.array([date])
            alts = np.array([max(0.0, alt_m / 1000.0)])
            lats = np.array([lat_deg if lat_deg is not None else 0.0])
            lons = np.array([lon_deg if lon_deg is not None else 0.0])
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
            rho = float(dens[0, 0, 0, 0, -1])
            if rho > 0:
                return rho

        table = [
            (0, 1.225),
            (25, 3.9e-2),
            (50, 1.0e-3),
            (75, 3.2e-5),
            (100, 5.6e-7),
            (125, 3.3e-9),
            (150, 9.7e-11),
            (175, 1.7e-12),
            (200, 2.5e-13),
            (250, 5.0e-14),
            (300, 9.7e-15),
            (350, 2.5e-15),
            (400, 4.0e-16),
            (450, 8.7e-17),
            (500, 1.9e-17),
            (600, 4.5e-18),
            (700, 8.0e-19),
            (800, 1.8e-19),
            (900, 4.0e-20),
            (1000, 1.0e-20),
        ]
        alt_km = max(0.0, alt_m / 1000.0)
        if alt_km <= table[0][0]:
            return table[0][1]
        for i in range(1, len(table)):
            if alt_km <= table[i][0]:
                alt0, rho0 = table[i - 1]
                alt1, rho1 = table[i]
                frac = (alt_km - alt0) / (alt1 - alt0)
                return float(rho0 * np.exp(np.log(rho1 / rho0) * frac))
        return table[-1][1]

    def compute(
        self,
        utime: float,
        ra_deg: float,
        dec_deg: float,
        in_eclipse: bool,
        moi_cfg: Any,
    ) -> tuple[np.ndarray, dict[str, float | list[float]]]:
        """Compute aggregate disturbance torque in body frame."""
        torque = np.zeros(3, dtype=float)
        i_mat = self._build_inertia(moi_cfg)

        r_vec = None
        v_vec = None
        try:
            idx = self.ephem.index(dtutcfromtimestamp(utime))
            r_vec = np.array(self.ephem.gcrs_pv.position[idx])
            v_vec = np.array(self.ephem.gcrs_pv.velocity[idx])
        except Exception:
            pass

        def _normalize(vec: np.ndarray) -> np.ndarray:
            n = np.linalg.norm(vec)
            if n == 0:
                return vec
            if n < 1e6:
                return vec * 1000.0
            return vec

        if r_vec is not None:
            r_vec = _normalize(r_vec)
        if v_vec is not None:
            v_vec = _normalize(v_vec)

        gg_mag = drag_mag = srp_mag = mag_mag = 0.0

        if r_vec is not None:
            r_body = self._inertial_to_body(r_vec, ra_deg, dec_deg)
            r_norm = np.linalg.norm(r_body)
            if r_norm > 0:
                r_hat = r_body / r_norm
                torque_gg = (
                    3 * self.MU_EARTH / (r_norm**3) * np.cross(r_hat, i_mat.dot(r_hat))
                )
                torque += torque_gg
                gg_mag = float(np.linalg.norm(torque_gg))

        lat_deg = lon_deg = None
        try:
            idx_latlon = self.ephem.index(dtutcfromtimestamp(utime))
            lat_seq = getattr(self.ephem, "lat", None)
            lon_seq = getattr(self.ephem, "long", None)
            if lat_seq is not None and lon_seq is not None:
                lat_deg = float(lat_seq[idx_latlon])
                lon_deg = float(lon_seq[idx_latlon])
        except Exception:
            pass

        if self.cfg.drag_area_m2 > 0 and v_vec is not None:
            v_body = self._inertial_to_body(v_vec, ra_deg, dec_deg)
            v_mag = np.linalg.norm(v_body)
            if v_mag > 0 and r_vec is not None:
                alt = float(max(0.0, float(np.linalg.norm(r_vec)) - self.RE_M))
                lat_use = float(lat_deg) if lat_deg is not None else 0.0
                lon_use = float(lon_deg) if lon_deg is not None else 0.0
                rho = self.atmospheric_density(utime, alt, lat_use, lon_use)
                q = 0.5 * rho * v_mag * v_mag
                v_hat = v_body / v_mag
                f_drag = -q * self.cfg.drag_coeff * self.cfg.drag_area_m2 * v_hat
                torque_drag = np.cross(self.cfg.cp_offset_body, f_drag)
                torque += torque_drag
                drag_mag = float(np.linalg.norm(torque_drag))

        if self.cfg.solar_area_m2 > 0 and not in_eclipse:
            try:
                idx = self.ephem.index(dtutcfromtimestamp(utime))
                sun_vec = np.array(self.ephem.sun[idx].cartesian.xyz.to_value())
                if np.linalg.norm(sun_vec) < 1e9:
                    sun_vec = sun_vec * 1000.0
                r_sc = r_vec if r_vec is not None else np.zeros(3, dtype=float)
                r_sun = sun_vec - r_sc
                r_sun_mag = np.linalg.norm(r_sun)
                if r_sun_mag > 0:
                    scale = (self.AU_M / r_sun_mag) ** 2
                    r_sun_body = self._inertial_to_body(r_sun, ra_deg, dec_deg)
                    sun_hat = r_sun_body / np.linalg.norm(r_sun_body)
                    f_srp = (
                        self.SRP_P0
                        * scale
                        * self.cfg.solar_area_m2
                        * self.cfg.solar_reflectivity
                        * sun_hat
                    )
                    torque_srp = np.cross(self.cfg.cp_offset_body, f_srp)
                    torque += torque_srp
                    srp_mag = float(np.linalg.norm(torque_srp))
            except Exception:
                pass

        if np.linalg.norm(self.cfg.residual_magnetic_moment) > 0:
            b_body, _ = self.local_bfield_vector(utime, ra_deg, dec_deg)
            torque_mag = np.cross(self.cfg.residual_magnetic_moment, b_body)
            torque += torque_mag
            mag_mag = float(np.linalg.norm(torque_mag))

        components: dict[str, float | list[float]] = {
            "total": float(np.linalg.norm(torque)),
            "gg": gg_mag,
            "drag": drag_mag,
            "srp": srp_mag,
            "mag": mag_mag,
            "vector": [float(v) for v in torque.tolist()],
        }

        return torque, components
