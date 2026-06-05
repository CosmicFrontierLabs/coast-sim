from __future__ import annotations

import math
from typing import Any, Protocol

SECONDS_PER_DAY = 86400.0
WGS72_EARTH_MU_M3_S2 = 398600.8e9


class TLELine2Record(Protocol):
    """Minimal TLE record shape needed for mean element extraction."""

    line2: str


def _normalize_degrees(value: float) -> float:
    return value % 360.0


def _solve_eccentric_anomaly(mean_anomaly_rad: float, eccentricity: float) -> float:
    if not 0.0 <= eccentricity < 1.0:
        raise ValueError(
            f"Only elliptical TLE eccentricities are supported; got {eccentricity}"
        )

    eccentric_anomaly = mean_anomaly_rad if eccentricity < 0.8 else math.pi
    for _ in range(50):
        denominator = 1.0 - eccentricity * math.cos(eccentric_anomaly)
        delta = (
            eccentric_anomaly
            - eccentricity * math.sin(eccentric_anomaly)
            - mean_anomaly_rad
        ) / denominator
        eccentric_anomaly -= delta
        if abs(delta) < 1e-14:
            break
    return eccentric_anomaly


def true_anomaly_from_mean_anomaly(
    mean_anomaly_deg: float, eccentricity: float
) -> float:
    """Convert mean anomaly to true anomaly for an elliptical orbit."""
    mean_anomaly_rad = math.radians(_normalize_degrees(mean_anomaly_deg))
    eccentric_anomaly = _solve_eccentric_anomaly(mean_anomaly_rad, eccentricity)
    true_anomaly_rad = math.atan2(
        math.sqrt(1.0 - eccentricity**2) * math.sin(eccentric_anomaly),
        math.cos(eccentric_anomaly) - eccentricity,
    )
    return _normalize_degrees(math.degrees(true_anomaly_rad))


def classical_elements_from_tle(
    tle_record: TLELine2Record, mu_m3_s2: float = WGS72_EARTH_MU_M3_S2
) -> dict[str, Any]:
    """Return TLE mean classical elements at the TLE epoch.

    TLE line 2 carries inclination, RAAN, eccentricity, argument of perigee,
    mean anomaly, and mean motion. Semimajor axis and true anomaly are derived
    from those mean elements so downstream tools can compare against the exact
    element set used to seed the ephemeris.
    """
    fields = tle_record.line2.split()
    if len(fields) < 8 or fields[0] != "2":
        raise ValueError(f"Could not parse TLE line 2: {tle_record.line2!r}")

    eccentricity = float(f"0.{fields[4]}")
    mean_anomaly_deg = float(fields[6])
    mean_motion_rev_per_day = float(fields[7])
    mean_motion_rad_s = mean_motion_rev_per_day * 2.0 * math.pi / SECONDS_PER_DAY
    semimajor_axis_m = (mu_m3_s2 / mean_motion_rad_s**2) ** (1.0 / 3.0)

    return {
        "SemimajorAxis_m": semimajor_axis_m,
        "Eccentricity": eccentricity,
        "Inclination_deg": float(fields[2]),
        "RightAscension_deg": float(fields[3]),
        "ArgPeriapsis_deg": float(fields[5]),
        "TrueAnomaly_deg": true_anomaly_from_mean_anomaly(
            mean_anomaly_deg, eccentricity
        ),
        "MeanAnomaly_deg": mean_anomaly_deg,
        "MeanMotion_rev_per_day": mean_motion_rev_per_day,
        "GravitationalParameter_m3_s2": mu_m3_s2,
    }
