from __future__ import annotations

from types import SimpleNamespace

import pytest

from conops.common.orbit import (
    WGS72_EARTH_MU_M3_S2,
    classical_elements_from_tle,
    true_anomaly_from_mean_anomaly,
)


def test_true_anomaly_from_mean_anomaly_circular_orbit() -> None:
    assert true_anomaly_from_mean_anomaly(725.0, 0.0) == pytest.approx(5.0)


def test_true_anomaly_from_mean_anomaly_rejects_non_elliptical_orbit() -> None:
    with pytest.raises(ValueError, match="Only elliptical TLE eccentricities"):
        true_anomaly_from_mean_anomaly(10.0, 1.0)


def test_classical_elements_from_tle_mean_elements() -> None:
    tle_record = SimpleNamespace(
        line2="2 99999  97.7898  39.6457 0016466  83.3495 116.0254 15.13083683    01"
    )

    elements = classical_elements_from_tle(tle_record)

    assert elements["SemimajorAxis_m"] == pytest.approx(6904941.542146514)
    assert elements["Eccentricity"] == pytest.approx(0.0016466)
    assert elements["Inclination_deg"] == pytest.approx(97.7898)
    assert elements["RightAscension_deg"] == pytest.approx(39.6457)
    assert elements["ArgPeriapsis_deg"] == pytest.approx(83.3495)
    assert elements["TrueAnomaly_deg"] == pytest.approx(116.19480034509827)
    assert elements["MeanAnomaly_deg"] == pytest.approx(116.0254)
    assert elements["MeanMotion_rev_per_day"] == pytest.approx(15.13083683)
    assert elements["GravitationalParameter_m3_s2"] == WGS72_EARTH_MU_M3_S2


def test_classical_elements_from_tle_rejects_invalid_line2() -> None:
    tle_record = SimpleNamespace(line2="not a TLE line 2")

    with pytest.raises(ValueError, match="Could not parse TLE line 2"):
        classical_elements_from_tle(tle_record)
