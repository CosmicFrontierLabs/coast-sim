from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..common import dtutcfromtimestamp
from .msis_adapter import MsisAdapter, MsisConfig

_logger = logging.getLogger(__name__)


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
    msis_density_scale: float = 1.0
    # Ephemeris position/velocity units: "m", "km", or "auto" (detect from magnitude)
    ephemeris_units: str = "auto"

    @classmethod
    def from_acs_config(cls, acs_cfg: Any) -> DisturbanceConfig:
        """Create DisturbanceConfig from an ACS config object with safe defaults.

        Args:
            acs_cfg: ACS configuration object (e.g., AttitudeControlSystem).

        Returns:
            DisturbanceConfig with values parsed from acs_cfg.
        """

        def safe_float(attr: str, default: float) -> float:
            try:
                val = getattr(acs_cfg, attr, default)
                return float(val) if val is not None else default
            except (TypeError, ValueError):
                return default

        def safe_vec3(
            attr: str, default: tuple[float, float, float]
        ) -> tuple[float, float, float]:
            try:
                val = getattr(acs_cfg, attr, None)
                if val is None:
                    return default
                arr = np.array(val, dtype=float)
                if len(arr) >= 3:
                    return (float(arr[0]), float(arr[1]), float(arr[2]))
                return default
            except (TypeError, ValueError):
                return default

        return cls(
            magnetorquer_bfield_T=safe_float("magnetorquer_bfield_T", 3e-5),
            cp_offset_body=safe_vec3("cp_offset_body", (0.0, 0.0, 0.0)),
            residual_magnetic_moment=safe_vec3(
                "residual_magnetic_moment", (0.0, 0.0, 0.0)
            ),
            drag_area_m2=safe_float("drag_area_m2", 0.0),
            drag_coeff=safe_float("drag_coeff", 2.2),
            solar_area_m2=safe_float("solar_area_m2", 0.0),
            solar_reflectivity=safe_float("solar_reflectivity", 1.0),
            use_msis_density=bool(getattr(acs_cfg, "use_msis_density", False)),
            msis_f107=safe_float("msis_f107", 200.0),
            msis_f107a=safe_float("msis_f107a", 180.0),
            msis_ap=safe_float("msis_ap", 12.0),
            msis_density_scale=safe_float("msis_density_scale", 1.0),
        )


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
        self._msis = MsisAdapter(
            MsisConfig(
                use_msis_density=cfg.use_msis_density,
                msis_f107=cfg.msis_f107,
                msis_f107a=cfg.msis_f107a,
                msis_ap=cfg.msis_ap,
                msis_density_scale=cfg.msis_density_scale,
            )
        )

    @staticmethod
    def _build_rotation_matrix(ra_deg: float, dec_deg: float) -> np.ndarray:
        """Build rotation matrix from inertial to body frame for given RA/Dec."""
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
            return np.vstack([x_b, y_b, z_b])
        except Exception:
            return np.eye(3)

    @staticmethod
    def _inertial_to_body(
        vec: np.ndarray,
        ra_deg: float,
        dec_deg: float,
        r_ib: np.ndarray | None = None,
    ) -> np.ndarray:
        """Rotate inertial vector into body frame assuming Z points at RA/Dec.

        Args:
            vec: Vector in inertial frame
            ra_deg: Right ascension in degrees
            dec_deg: Declination in degrees
            r_ib: Optional precomputed rotation matrix for efficiency
        """
        vec_arr = np.array(vec, dtype=float)
        try:
            if r_ib is None:
                r_ib = DisturbanceModel._build_rotation_matrix(ra_deg, dec_deg)
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

    def _to_meters(self, vec: np.ndarray, vec_type: str) -> np.ndarray:
        """Convert position/velocity vector to meters.

        Args:
            vec: Position or velocity vector
            vec_type: "position" or "velocity" for context in warnings

        Uses cfg.ephemeris_units:
            - "m": Already in meters, no conversion
            - "km": Multiply by 1000
            - "auto": Detect based on physical plausibility
        """
        units = self.cfg.ephemeris_units.lower()
        mag = float(np.linalg.norm(vec))

        if mag == 0:
            return vec

        if units == "m":
            return vec
        elif units == "km":
            return vec * 1000.0
        else:  # "auto"
            # Use physical constraints to detect units
            if vec_type == "position":
                # Earth radius is ~6378 km = 6.378e6 m
                # LEO: 6678-7078 km, GEO: ~42164 km, lunar: ~384400 km
                # If magnitude < 1e5, assume km (plausible for any Earth orbit in km)
                # If magnitude > 1e5, assume meters (LEO in meters is ~6.7e6)
                if mag < 1e5:
                    _logger.debug(
                        "Auto-detected %s units as km (mag=%.2e < 1e5), converting to m",
                        vec_type,
                        mag,
                    )
                    return vec * 1000.0
                # Validate result is plausible (> Earth radius, < lunar distance * 2)
                if mag < self.RE_M * 0.9:
                    _logger.warning(
                        "Position magnitude %.2e m is below Earth radius - check units",
                        mag,
                    )
            else:  # velocity
                # LEO velocity: ~7.5 km/s = 7500 m/s
                # If < 100, assume km/s; otherwise assume m/s
                if mag < 100:
                    _logger.debug(
                        "Auto-detected %s units as km/s (mag=%.2e < 100), converting to m/s",
                        vec_type,
                        mag,
                    )
                    return vec * 1000.0
            return vec

    def local_bfield_vector(
        self,
        utime: float,
        ra_deg: float,
        dec_deg: float,
        r_ib: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Estimate local magnetic field vector in body frame.

        Args:
            utime: Unix timestamp
            ra_deg: Right ascension in degrees
            dec_deg: Declination in degrees
            r_ib: Optional precomputed rotation matrix for efficiency
        """
        b_const = np.array([0.0, 0.0, self.cfg.magnetorquer_bfield_T], dtype=float)
        try:
            idx = self.ephem.index(dtutcfromtimestamp(utime))
        except Exception:
            return b_const, float(np.linalg.norm(b_const))

        lat_deg = lon_deg = None
        try:
            lat_deg = float(getattr(self.ephem, "latitude_deg", [0.0])[idx])
            lon_deg = float(getattr(self.ephem, "longitude_deg", [0.0])[idx])
        except Exception:
            pass

        pos = None
        try:
            pos = np.array(self.ephem.gcrs_pv.position[idx], dtype=float)
            rmag = float(np.linalg.norm(pos))
            if rmag < 1e5:
                rmag *= 1000.0
            alt_m = float(max(0.0, rmag - self.RE_M))
        except Exception:
            alt_m = 500e3

        if (lat_deg is None or lon_deg is None) and pos is not None:
            try:
                x, y, z = pos
                lon_deg = float(np.rad2deg(np.arctan2(y, x)))
                lat_deg = float(np.rad2deg(np.arctan2(z, np.sqrt(x * x + y * y))))
            except Exception:
                lat_deg = lat_deg if lat_deg is not None else 0.0
                lon_deg = lon_deg if lon_deg is not None else 0.0

        lat = np.deg2rad(lat_deg if lat_deg is not None else 0.0)
        lon = np.deg2rad(lon_deg if lon_deg is not None else 0.0)
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
        b_body = self._inertial_to_body(b_ecef, ra_deg, dec_deg, r_ib)
        return b_body, float(np.linalg.norm(b_body))

    def atmospheric_density(
        self, utime: float, alt_m: float, lat_deg: float | None, lon_deg: float | None
    ) -> float:
        """Lookup/interpolate atmospheric density (kg/m^3) vs altitude.

        Uses exponential atmosphere model with altitude-dependent scale heights.
        Table covers 0-1000+ km. Above 400 km, solar activity significantly
        affects actual density; enable MSIS for improved accuracy.
        """
        alt_km = alt_m / 1000.0
        if alt_km > 1000 and not self.cfg.use_msis_density:
            _logger.debug(
                "Altitude %.0f km > 1000 km; extrapolating with final scale height",
                alt_km,
            )
        elif alt_km > 400 and not self.cfg.use_msis_density:
            _logger.debug(
                "Altitude %.0f km > 400 km; density estimate may vary with solar activity",
                alt_km,
            )

        rho_table = self._table_density(alt_m)
        lat_use = float(lat_deg) if lat_deg is not None else 0.0
        lon_use = float(lon_deg) if lon_deg is not None else 0.0
        rho_msis = self._msis.density(utime, alt_m, lat_use, lon_use, rho_table)
        if rho_msis is not None:
            return rho_msis
        return rho_table

    @staticmethod
    def _table_density(alt_m: float) -> float:
        """Lookup atmospheric density from altitude table.

        Uses exponential atmosphere model: ρ(h) = ρ₀ × exp(-(h - h₀) / H)
        where h₀ is the reference altitude, ρ₀ is the reference density,
        and H is the scale height for each altitude band.

        Table based on US Standard Atmosphere 1976 with extended data
        for LEO altitudes up to 1000+ km.

        For high-fidelity applications above 400 km, enable MSIS density
        model via use_msis_density=True in DisturbanceConfig, as solar
        activity significantly affects density at those altitudes.

        Args:
            alt_m: Altitude in meters

        Returns:
            Atmospheric density in kg/m³
        """
        # Table format: (h0_km, rho0_kg_m3, scale_height_km)
        # Each entry is valid from h0 up to the next entry's h0
        table = [
            (0, 1.225, 8.44),
            (25, 3.899e-2, 6.49),
            (30, 1.774e-2, 6.75),
            (35, 8.279e-3, 7.07),
            (40, 3.972e-3, 7.47),
            (45, 1.995e-3, 7.83),
            (50, 1.057e-3, 7.95),
            (55, 5.821e-4, 7.73),
            (60, 3.206e-4, 7.29),
            (65, 1.718e-4, 6.81),
            (70, 8.770e-5, 6.33),
            (75, 4.178e-5, 6.00),
            (80, 1.905e-5, 5.70),
            (85, 8.337e-6, 5.41),
            (90, 3.396e-6, 5.38),
            (95, 1.343e-6, 5.74),
            (100, 5.297e-7, 6.15),
            (110, 9.661e-8, 8.06),
            (120, 2.438e-8, 11.6),
            (130, 8.484e-9, 16.1),
            (140, 3.845e-9, 20.6),
            (150, 2.070e-9, 24.6),
            (160, 1.224e-9, 26.3),
            (180, 5.464e-10, 33.2),
            (200, 2.789e-10, 38.5),
            (250, 7.248e-11, 46.9),
            (300, 2.418e-11, 52.5),
            (350, 9.158e-12, 56.4),
            (400, 3.725e-12, 59.4),
            (450, 1.585e-12, 62.2),
            (500, 6.967e-13, 65.8),
            (600, 1.454e-13, 79.0),
            (700, 3.614e-14, 109.0),
            (800, 1.170e-14, 164.0),
            (900, 5.245e-15, 225.0),
            (1000, 3.019e-15, 268.0),
        ]
        alt_km = max(0.0, alt_m / 1000.0)

        # Find the appropriate altitude band
        band_idx = 0
        for i in range(len(table) - 1, -1, -1):
            if alt_km >= table[i][0]:
                band_idx = i
                break

        h0, rho0, scale_h = table[band_idx]
        # Exponential atmosphere: ρ = ρ₀ × exp(-(h - h₀) / H)
        return float(rho0 * np.exp(-(alt_km - h0) / scale_h))

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

        # Build rotation matrix once for reuse in all coordinate transforms
        r_ib = self._build_rotation_matrix(ra_deg, dec_deg)

        # Compute ephemeris index once and reuse for all lookups
        idx = None
        r_vec = None
        v_vec = None
        try:
            idx = self.ephem.index(dtutcfromtimestamp(utime))
            r_vec = np.array(self.ephem.gcrs_pv.position[idx])
            v_vec = np.array(self.ephem.gcrs_pv.velocity[idx])
        except Exception:
            pass

        # Convert position/velocity to meters based on configured units
        if r_vec is not None:
            r_vec = self._to_meters(r_vec, "position")
        if v_vec is not None:
            v_vec = self._to_meters(v_vec, "velocity")

        gg_mag = drag_mag = srp_mag = mag_mag = 0.0

        if r_vec is not None:
            r_body = self._inertial_to_body(r_vec, ra_deg, dec_deg, r_ib)
            r_norm = np.linalg.norm(r_body)
            if r_norm > 0:
                r_hat = r_body / r_norm
                torque_gg = (
                    3 * self.MU_EARTH / (r_norm**3) * np.cross(r_hat, i_mat.dot(r_hat))
                )
                torque += torque_gg
                gg_mag = float(np.linalg.norm(torque_gg))

        lat_deg = lon_deg = None
        if idx is not None:
            try:
                lat_seq = getattr(self.ephem, "lat", None)
                lon_seq = getattr(self.ephem, "long", None)
                if lat_seq is not None and lon_seq is not None:
                    lat_deg = float(lat_seq[idx])
                    lon_deg = float(lon_seq[idx])
            except Exception:
                pass
        if (lat_deg is None or lon_deg is None) and r_vec is not None:
            try:
                x, y, z = r_vec
                lon_deg = float(np.rad2deg(np.arctan2(y, x)))
                lat_deg = float(np.rad2deg(np.arctan2(z, np.sqrt(x * x + y * y))))
            except Exception:
                lat_deg = lat_deg if lat_deg is not None else 0.0
                lon_deg = lon_deg if lon_deg is not None else 0.0

        if self.cfg.drag_area_m2 > 0 and v_vec is not None:
            v_body = self._inertial_to_body(v_vec, ra_deg, dec_deg, r_ib)
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

        if self.cfg.solar_area_m2 > 0 and not in_eclipse and idx is not None:
            try:
                # Use sun_pv.position for fast array access (rust-ephem 0.3.0+)
                sun_vec = np.array(self.ephem.sun_pv.position[idx])
                # Sun distance is ~1 AU = 1.496e11 m = 1.496e8 km
                # If magnitude < 1e9, assume km; otherwise assume meters
                sun_mag = float(np.linalg.norm(sun_vec))
                if sun_mag > 0 and sun_mag < 1e9:
                    sun_vec = sun_vec * 1000.0
                r_sc = r_vec if r_vec is not None else np.zeros(3, dtype=float)
                r_sun = sun_vec - r_sc
                r_sun_mag = np.linalg.norm(r_sun)
                if r_sun_mag > 0:
                    scale = (self.AU_M / r_sun_mag) ** 2
                    r_sun_body = self._inertial_to_body(r_sun, ra_deg, dec_deg, r_ib)
                    sun_hat = r_sun_body / np.linalg.norm(r_sun_body)
                    # SRP force pushes away from sun (negative sun_hat direction)
                    # solar_reflectivity is (1 + r) where r is surface reflectivity:
                    #   1.0 = absorbing surface, 2.0 = perfectly reflective
                    f_srp = (
                        -self.SRP_P0
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
            b_body, _ = self.local_bfield_vector(utime, ra_deg, dec_deg, r_ib)
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
