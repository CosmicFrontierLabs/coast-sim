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
    _MAX_EPHEM_CACHE = 200000

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
        # Cache rotation matrix across calls when pointing unchanged
        self._cached_r_ib: np.ndarray | None = None
        self._cached_ra: float | None = None
        self._cached_dec: float | None = None
        self._cached_inertia: np.ndarray | None = None
        self._cached_moi_key: tuple[float, ...] | float | None = None
        self._ephem_cache_ready = False
        self._ephem_cache_failed = False
        self._ephem_pos_m: np.ndarray | None = None
        self._ephem_vel_m: np.ndarray | None = None
        self._ephem_sun_m: np.ndarray | None = None
        self._ephem_lat: np.ndarray | None = None
        self._ephem_lon: np.ndarray | None = None
        self._ephem_r_norm: np.ndarray | None = None
        self._ephem_alt_m: np.ndarray | None = None
        # Cache whether we have residual magnetic moment (avoid recomputing every call)
        rmm = cfg.residual_magnetic_moment
        self._has_residual_magnetic = (rmm[0] ** 2 + rmm[1] ** 2 + rmm[2] ** 2) > 0

    @staticmethod
    def _build_rotation_matrix(ra_deg: float, dec_deg: float) -> np.ndarray:
        """Build rotation matrix from inertial to body frame for given RA/Dec.

        Uses celestial pole (Z-axis) as Gram-Schmidt reference, which places
        the singularity at Dec=±90° (celestial poles) rather than at RA=0°/180°.
        This is preferable since astronomical targets are rarely at the poles.
        """
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
            # Use celestial pole as reference (singularity at Dec=±90°)
            pole = np.array([0.0, 0.0, 1.0])
            y_b = pole - np.dot(pole, z_b) * z_b
            if np.linalg.norm(y_b) < 1e-6:
                # Fallback for polar pointing (Dec ≈ ±90°)
                x_candidate = np.array([1.0, 0.0, 0.0])
                y_b = np.cross(z_b, x_candidate)
            y_b = y_b / np.linalg.norm(y_b)
            x_b = np.cross(y_b, z_b)
            x_b = x_b / np.linalg.norm(x_b)
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

    @staticmethod
    def _moi_key(moi_cfg: Any) -> tuple[float, ...] | float | None:
        """Create a hashable key for inertia caching."""
        try:
            if isinstance(moi_cfg, (list, tuple, np.ndarray)):
                arr = np.array(moi_cfg, dtype=float).reshape(-1)
                return tuple(float(v) for v in arr)
            return float(moi_cfg)
        except Exception:
            return None

    def _infer_scale(self, vecs: np.ndarray, vec_type: str) -> float:
        units = self.cfg.ephemeris_units.lower()
        if units == "m":
            return 1.0
        if units == "km":
            return 1000.0
        mags = np.linalg.norm(vecs, axis=1)
        if vec_type == "position":
            return 1000.0 if float(np.nanmedian(mags)) < 1e5 else 1.0
        return 1000.0 if float(np.nanmedian(mags)) < 100.0 else 1.0

    def _scale_ephem_vectors(self, vecs: np.ndarray, vec_type: str) -> np.ndarray:
        scale = self._infer_scale(vecs, vec_type)
        return vecs if scale == 1.0 else vecs * scale

    def _scale_sun_vectors(self, vecs: np.ndarray) -> np.ndarray:
        units = self.cfg.ephemeris_units.lower()
        if units == "m":
            return vecs
        if units == "km":
            return vecs * 1000.0
        mags = np.linalg.norm(vecs, axis=1)
        scale = 1000.0 if float(np.nanmedian(mags)) < 1e9 else 1.0
        return vecs if scale == 1.0 else vecs * scale

    def _ephem_lat_lon_arrays(
        self, n: int, pos_m: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        for lat_attr, lon_attr in (("lat", "long"), ("latitude_deg", "longitude_deg")):
            try:
                lat_seq = getattr(self.ephem, lat_attr, None)
                lon_seq = getattr(self.ephem, lon_attr, None)
                if lat_seq is None or lon_seq is None:
                    continue
                lat_arr = np.asarray(lat_seq, dtype=float)
                lon_arr = np.asarray(lon_seq, dtype=float)
                if lat_arr.shape[0] == n and lon_arr.shape[0] == n:
                    return lat_arr, lon_arr
            except Exception:
                continue

        x = pos_m[:, 0]
        y = pos_m[:, 1]
        z = pos_m[:, 2]
        lon = np.rad2deg(np.arctan2(y, x))
        lat = np.rad2deg(np.arctan2(z, np.sqrt(x * x + y * y)))
        return lat, lon

    def _prepare_ephem_cache(self) -> None:
        if self._ephem_cache_ready or self._ephem_cache_failed:
            return

        try:
            pos = np.asarray(self.ephem.gcrs_pv.position, dtype=float)
            vel = np.asarray(self.ephem.gcrs_pv.velocity, dtype=float)
        except Exception:
            self._ephem_cache_failed = True
            return

        if pos.ndim != 2 or vel.ndim != 2 or pos.shape[1] < 3 or vel.shape[1] < 3:
            self._ephem_cache_failed = True
            return

        n = pos.shape[0]
        if n == 0 or vel.shape[0] != n or n > self._MAX_EPHEM_CACHE:
            self._ephem_cache_failed = True
            return

        pos = pos[:, :3]
        vel = vel[:, :3]
        pos_m = self._scale_ephem_vectors(pos, "position")
        vel_m = self._scale_ephem_vectors(vel, "velocity")

        r_norm = np.linalg.norm(pos_m, axis=1)
        alt_m = np.maximum(0.0, r_norm - self.RE_M)
        lat_arr, lon_arr = self._ephem_lat_lon_arrays(n, pos_m)

        self._ephem_pos_m = pos_m
        self._ephem_vel_m = vel_m
        self._ephem_r_norm = r_norm
        self._ephem_alt_m = alt_m
        self._ephem_lat = lat_arr
        self._ephem_lon = lon_arr

        if self.cfg.solar_area_m2 > 0:
            try:
                sun = np.asarray(self.ephem.sun_pv.position, dtype=float)
                if sun.ndim == 2 and sun.shape[0] == n and sun.shape[1] >= 3:
                    self._ephem_sun_m = self._scale_sun_vectors(sun[:, :3])
            except Exception:
                self._ephem_sun_m = None

        self._ephem_cache_ready = True

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

    def _bfield_from_geodetic(
        self,
        alt_m: float,
        lat_deg: float,
        lon_deg: float,
        ra_deg: float,
        dec_deg: float,
        r_ib: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float]:
        """Compute magnetic field from pre-computed geodetic coordinates.

        This is the inner computation, avoiding duplicate ephemeris lookups.

        Args:
            alt_m: Altitude in meters
            lat_deg: Geodetic latitude in degrees
            lon_deg: Geodetic longitude in degrees
            ra_deg: Right ascension in degrees (for body frame rotation)
            dec_deg: Declination in degrees (for body frame rotation)
            r_ib: Optional precomputed rotation matrix for efficiency
        """
        import math

        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        r = self.RE_M + max(0.0, alt_m)
        scale = (self.RE_M / r) ** 3

        cos_lat = math.cos(lat)
        sin_lat = math.sin(lat)
        cos_lon = math.cos(lon)
        sin_lon = math.sin(lon)

        br = -2 * self.B0_T * scale * sin_lat
        btheta = -self.B0_T * scale * cos_lat

        # Build field vector in ECEF
        r_hat = np.array([cos_lat * cos_lon, cos_lat * sin_lon, sin_lat])
        theta_hat = np.array([-sin_lat * cos_lon, -sin_lat * sin_lon, cos_lat])
        b_ecef = br * r_hat + btheta * theta_hat

        b_body = self._inertial_to_body(b_ecef, ra_deg, dec_deg, r_ib)
        return b_body, float(np.linalg.norm(b_body))

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

        Note: For better performance when ephemeris data is already available,
        use _bfield_from_geodetic() directly with pre-computed coordinates.
        """
        b_const = np.array([0.0, 0.0, self.cfg.magnetorquer_bfield_T], dtype=float)
        if not self._ephem_cache_failed:
            self._prepare_ephem_cache()
        try:
            idx = self.ephem.index(dtutcfromtimestamp(utime))
        except Exception:
            return b_const, float(np.linalg.norm(b_const))

        lat_deg_val: float | None = None
        lon_deg_val: float | None = None
        if self._ephem_cache_ready:
            try:
                alt_m = (
                    float(self._ephem_alt_m[idx])
                    if self._ephem_alt_m is not None
                    else 500e3
                )
                lat_deg_val = (
                    float(self._ephem_lat[idx]) if self._ephem_lat is not None else 0.0
                )
                lon_deg_val = (
                    float(self._ephem_lon[idx]) if self._ephem_lon is not None else 0.0
                )
                return self._bfield_from_geodetic(
                    alt_m, lat_deg_val, lon_deg_val, ra_deg, dec_deg, r_ib
                )
            except Exception:
                pass

        try:
            lat_deg_val = float(getattr(self.ephem, "latitude_deg", [0.0])[idx])
            lon_deg_val = float(getattr(self.ephem, "longitude_deg", [0.0])[idx])
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

        if (lat_deg_val is None or lon_deg_val is None) and pos is not None:
            try:
                x, y, z = pos
                lon_deg_val = float(np.rad2deg(np.arctan2(y, x)))
                lat_deg_val = float(np.rad2deg(np.arctan2(z, np.sqrt(x * x + y * y))))
            except Exception:
                lat_deg_val = 0.0
                lon_deg_val = 0.0

        if lat_deg_val is None:
            lat_deg_val = 0.0
        if lon_deg_val is None:
            lon_deg_val = 0.0

        return self._bfield_from_geodetic(
            alt_m, lat_deg_val, lon_deg_val, ra_deg, dec_deg, r_ib
        )

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
        moi_key = self._moi_key(moi_cfg)
        if (
            moi_key is not None
            and self._cached_inertia is not None
            and self._cached_moi_key == moi_key
        ):
            i_mat = self._cached_inertia
        else:
            i_mat = self._build_inertia(moi_cfg)
            if moi_key is not None:
                self._cached_inertia = i_mat
                self._cached_moi_key = moi_key

        # Reuse cached rotation matrix if pointing unchanged
        if (
            self._cached_r_ib is not None
            and self._cached_ra == ra_deg
            and self._cached_dec == dec_deg
        ):
            r_ib = self._cached_r_ib
        else:
            r_ib = self._build_rotation_matrix(ra_deg, dec_deg)
            self._cached_r_ib = r_ib
            self._cached_ra = ra_deg
            self._cached_dec = dec_deg

        if not self._ephem_cache_failed:
            self._prepare_ephem_cache()

        # Compute ephemeris index once and reuse for all lookups
        idx = None
        r_vec: np.ndarray | None = None
        v_vec: np.ndarray | None = None
        r_vec_norm: float | None = None
        alt_m: float | None = None
        lat_deg: float | None = None
        lon_deg: float | None = None
        r_vec_from_cache = False
        v_vec_from_cache = False
        try:
            idx = self.ephem.index(dtutcfromtimestamp(utime))
        except Exception:
            idx = None

        if idx is not None and self._ephem_cache_ready:
            try:
                if self._ephem_pos_m is not None:
                    r_vec = self._ephem_pos_m[idx]
                    r_vec_from_cache = True
                if self._ephem_vel_m is not None:
                    v_vec = self._ephem_vel_m[idx]
                    v_vec_from_cache = True
                if self._ephem_r_norm is not None:
                    r_vec_norm = float(self._ephem_r_norm[idx])
                if self._ephem_alt_m is not None:
                    alt_m = float(self._ephem_alt_m[idx])
                if self._ephem_lat is not None:
                    lat_deg = float(self._ephem_lat[idx])
                if self._ephem_lon is not None:
                    lon_deg = float(self._ephem_lon[idx])
            except Exception:
                r_vec = None
                v_vec = None
                r_vec_norm = None
                alt_m = None
                lat_deg = None
                lon_deg = None
                r_vec_from_cache = False
                v_vec_from_cache = False

        if idx is not None and (r_vec is None or v_vec is None):
            try:
                r_vec = np.asarray(self.ephem.gcrs_pv.position[idx], dtype=float)
                v_vec = np.asarray(self.ephem.gcrs_pv.velocity[idx], dtype=float)
            except Exception:
                r_vec = None
                v_vec = None

        # Convert position/velocity to meters based on configured units
        if r_vec is not None and not r_vec_from_cache:
            r_vec = self._to_meters(r_vec, "position")
        if v_vec is not None and not v_vec_from_cache:
            v_vec = self._to_meters(v_vec, "velocity")

        gg_mag = drag_mag = srp_mag = mag_mag = 0.0

        if r_vec is not None:
            r_body = self._inertial_to_body(r_vec, ra_deg, dec_deg, r_ib)
            r_norm = np.linalg.norm(r_body)
            # Compute altitude once for reuse in drag and magnetic field
            if r_vec_norm is None:
                r_vec_norm = float(np.linalg.norm(r_vec))
            if alt_m is None:
                alt_m = max(0.0, r_vec_norm - self.RE_M)
            if r_norm > 0:
                r_hat = r_body / r_norm
                torque_gg = (
                    3 * self.MU_EARTH / (r_norm**3) * np.cross(r_hat, i_mat.dot(r_hat))
                )
                torque += torque_gg
                gg_mag = float(np.linalg.norm(torque_gg))

        if (lat_deg is None or lon_deg is None) and idx is not None:
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
                lat_use = float(lat_deg) if lat_deg is not None else 0.0
                lon_use = float(lon_deg) if lon_deg is not None else 0.0
                alt_use = alt_m if alt_m is not None else 0.0
                rho = self.atmospheric_density(utime, alt_use, lat_use, lon_use)
                q = 0.5 * rho * v_mag * v_mag
                v_hat = v_body / v_mag
                f_drag = -q * self.cfg.drag_coeff * self.cfg.drag_area_m2 * v_hat
                torque_drag = np.cross(self.cfg.cp_offset_body, f_drag)
                torque += torque_drag
                drag_mag = float(np.linalg.norm(torque_drag))

        if self.cfg.solar_area_m2 > 0 and not in_eclipse and idx is not None:
            try:
                # Use sun_pv.position for fast array access (rust-ephem 0.3.0+)
                if self._ephem_cache_ready and self._ephem_sun_m is not None:
                    sun_vec = self._ephem_sun_m[idx]
                else:
                    sun_vec = np.asarray(self.ephem.sun_pv.position[idx], dtype=float)
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

        if self._has_residual_magnetic and r_vec is not None:
            lat_use = float(lat_deg) if lat_deg is not None else 0.0
            lon_use = float(lon_deg) if lon_deg is not None else 0.0
            alt_use = alt_m if alt_m is not None else 0.0
            b_body, _ = self._bfield_from_geodetic(
                alt_use, lat_use, lon_use, ra_deg, dec_deg, r_ib
            )
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
