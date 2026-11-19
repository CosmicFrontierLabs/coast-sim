#!/usr/bin/env python
"""Compare the orbit between ephemeris samples, interpolated two ways, for
stored-momentum tracking.

The momentum integrators sample the orbit between ephemeris samples: along a
slew, and wherever a motion's boundary falls. ``OrbitSampler`` asks a Rust
ephemeris for its own cubic Hermite interpolation there; the alternative, and
what a test double gets, is linear interpolation of each table.

This holds one inertial attitude for a few orbits, sampling the orbit every
10 s from an ephemeris at a coarser step, and compares each interpolation
against an ephemeris propagated at the 10 s samples themselves: the position
error, the torque each source gives, the stored momentum at the end, and the
time each takes.

Example:
    uv run python scripts/benchmark_orbit_interpolation.py --step 60
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from rust_ephem import TLEEphemeris

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from conops.common.ephemeris import OrbitSampler, OrbitState  # noqa: E402
from conops.common.vector import attitude_to_quat  # noqa: E402
from conops.config import (  # noqa: E402
    AttitudeControlSystem,
    MissionConfig,
    SpacecraftBus,
    StoredMomentumConfig,
)
from conops.simulation.momentum import tracker_from_config  # noqa: E402

TLE = REPO_ROOT / "examples" / "example.tle"
BEGIN = datetime(2025, 11, 1, tzinfo=timezone.utc)
SAMPLE_S = 10


class _LinearEphemeris:
    """An ephemeris ``OrbitSampler`` does not know, so that it interpolates
    its tables linearly."""

    def __init__(self, ephem: TLEEphemeris) -> None:
        self.timestamp = ephem.timestamp
        self.gcrs_pv = ephem.gcrs_pv
        self.sun_pv = ephem.sun_pv
        self.itrs_pv = ephem.itrs_pv


def _config(ephem: TLEEphemeris) -> MissionConfig:
    config = MissionConfig(
        spacecraft_bus=SpacecraftBus(
            inertia_tensor_body_kg_m2=(
                (1000.0, 0.0, 0.0),
                (0.0, 1200.0, 0.0),
                (0.0, 0.0, 800.0),
            ),
            attitude_control=AttitudeControlSystem(
                stored_momentum=StoredMomentumConfig(
                    gravity_gradient_enabled=True,
                    aerodynamic_drag_enabled=True,
                    residual_dipole_body_a_m2=(1.0, 0.5, 0.2),
                )
            ),
        )
    )
    config.constraint.ephem = ephem
    return config


def _integrate(
    config: MissionConfig, utimes: np.ndarray, orbit_at: Callable[[float], OrbitState]
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, float]:
    """Positions, each source's torque and the stored momentum along
    ``utimes``, and the seconds spent sampling the orbit."""
    tracker = tracker_from_config(config)
    assert tracker is not None
    quaternion = attitude_to_quat(80.0, -20.0, 0.0)
    positions = []
    torques: dict[str, list[tuple[float, float, float]]] = {}
    momentum = []
    spent = 0.0
    for utime in utimes:
        began = time.perf_counter()
        orbit = orbit_at(float(utime))
        spent += time.perf_counter() - began
        assert orbit.sun_position_eci_km is not None
        sample = tracker.update(
            utime=float(utime),
            position_eci_km=orbit.position_eci_km,
            attitude_quaternion_eci_to_body=quaternion,
            sun_eci_km=orbit.sun_position_eci_km - orbit.position_eci_km,
            sunlit=True,
            velocity_eci_km_s=orbit.velocity_eci_km_s,
            position_ecef_km=orbit.position_ecef_km,
        )
        positions.append(orbit.position_eci_km)
        for name, torque in sample.torques_body_n_m.items():
            torques.setdefault(name, []).append(torque)
        momentum.append(sample.stored_momentum_body_n_m_s)
    return (
        np.array(positions),
        {name: np.array(rows) for name, rows in torques.items()},
        np.array(momentum),
        spent,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--step", type=int, default=60, help="ephemeris step, s")
    parser.add_argument("--hours", type=float, default=6.0)
    args = parser.parse_args()
    if args.step % SAMPLE_S:
        parser.error(f"--step must be a multiple of {SAMPLE_S}")

    end = BEGIN + timedelta(hours=args.hours)
    coarse = TLEEphemeris(tle=str(TLE), begin=BEGIN, end=end, step_size=args.step)
    fine = TLEEphemeris(tle=str(TLE), begin=BEGIN, end=end, step_size=SAMPLE_S)
    utimes = np.array([t.timestamp() for t in fine.timestamp], dtype=np.float64)
    config = _config(coarse)

    reference = _integrate(config, utimes, OrbitSampler(fine).at)
    contenders = {
        "linear": OrbitSampler(_LinearEphemeris(coarse)),  # type: ignore[arg-type]
        "hermite": OrbitSampler(coarse),
    }
    assert contenders["hermite"].hermite and not contenders["linear"].hermite

    print(
        f"{args.hours:g} h held attitude, ephemeris every {args.step} s, "
        f"sampled every {SAMPLE_S} s ({len(utimes)} samples); errors against "
        f"an ephemeris propagated at every sample\n"
    )
    names = list(reference[1])
    header = f"{'':10}{'position':>12}" + "".join(f"{n:>22}" for n in names)
    print(header + f"{'stored momentum':>20}{'per sample':>12}")
    print(
        f"{'':10}{'max, km':>12}"
        + "".join(f"{'max torque error':>22}" for _ in names)
        + f"{'error at end':>20}{'us':>12}"
    )
    stored = float(np.linalg.norm(reference[2][-1]))
    for name, sampler in contenders.items():
        positions, torques, momentum, spent = _integrate(config, utimes, sampler.at)
        row = f"{name:10}{np.linalg.norm(positions - reference[0], axis=1).max():12.4f}"
        for source in names:
            error = np.linalg.norm(torques[source] - reference[1][source], axis=1)
            scale = np.linalg.norm(reference[1][source], axis=1).max()
            row += f"{100 * error.max() / scale:21.4f}%"
        final = float(np.linalg.norm(momentum[-1] - reference[2][-1]))
        row += f"{100 * final / stored:19.4f}%{1e6 * spent / len(utimes):12.1f}"
        print(row)
    print(
        f"\nTorque errors are a share of each source's greatest torque; the "
        f"stored momentum at the end is {stored:.4f} N m s."
    )


if __name__ == "__main__":
    main()
