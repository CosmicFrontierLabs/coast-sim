"""Whole-hold power and attitude selection shared by planning and fault SAFE."""

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING

import erfa  # type: ignore[import-untyped]
import numpy as np

from ..common import ACSMode
from ..common.numerics import POLE_DEC_DEG
from ..config.activity_attitude import ActivityAttitude
from ..config.constraint import attitude_constraint_name_for_scopes

if TYPE_CHECKING:
    from ..config import MissionConfig

Attitude = tuple[float, float, float]


class SafeHoldUnavailableError(ValueError):
    """No sampled candidate satisfies the requested whole safe hold."""


def ecliptic_poles() -> list[tuple[float, float]]:
    """Mean J2000 ecliptic poles in the equatorial axes used by the simulator."""
    north = erfa.ecm06(2451545.0, 0.0)[2]
    return [
        (
            float(np.degrees(np.arctan2(v[1], v[0]))) % 360,
            float(np.degrees(np.arcsin(v[2]))),
        )
        for v in (north, -north)
    ]


def power_positive_attitudes(
    config: "MissionConfig",
    request: ActivityAttitude,
    begin: float,
    end: float,
    mode: ACSMode,
    instrument_modes: Mapping[str, str] | None = None,
) -> Iterator[Attitude]:
    """Yield geometry-safe poses positive at every sunlit sample of the hold.

    Fixed body-mounted arrays are supported. A finite drive needs an ingress
    drive-state forecast, so it is rejected rather than credited with power
    before it has physically reached a useful angle.
    """
    ephem = config.constraint.ephem
    if ephem is None:
        raise ValueError("power-positive attitude selection needs an ephemeris")
    if any(p.single_axis_drive is not None for p in config.solar_panel.panels):
        raise ValueError(
            "power-positive whole-hold selection requires body-mounted panels"
        )
    times = [
        begin,
        *(t.timestamp() for t in ephem.timestamp if begin < t.timestamp() < end),
    ]
    if end <= begin or end > ephem.timestamp[-1].timestamp():
        raise SafeHoldUnavailableError(
            "safe-hold interval is empty or outside the ephemeris"
        )
    lit = [t for t in times if not config.constraint.in_eclipse(0, 0, t)]
    if not lit:
        raise SafeHoldUnavailableError(
            "no sunlit samples to verify a power-positive safe hold"
        )
    load = config.spacecraft_bus.power(mode, in_eclipse=False) + config.payload.power(
        mode,
        in_eclipse=False,
        instrument_modes=instrument_modes,
    )
    if not config.solar_panel.panels:
        raise SafeHoldUnavailableError("no solar panels supply the safe-hold load")
    scopes = config.attitude_constraint_scopes_for_mode(mode)
    mounting = config.spacecraft_bus.frame.mounting
    poles = [
        mounting.body_attitude(ra, dec, float(roll))
        for ra, dec in ecliptic_poles()
        for roll in range(360)
    ]
    grid = [
        (float(ra), float(dec), float(roll))
        for dec in range(-90, 91, 30)
        for ra in (range(0, 360, 30) if abs(dec) != POLE_DEC_DEG else [0])
        for roll in range(0, 360, 15)
    ]

    def power(pose: Attitude, at: float) -> float:
        return float(
            config.solar_panel.power(at, pose[0], pose[1], ephem, roll=pose[2])
        )

    groups = [poles, grid] if request.prefer_ecliptic_poles else [[*grid, *poles]]
    found = False
    for group in groups:
        ranked = sorted(group, key=lambda a: power(a, lit[0]), reverse=True)
        with config.constraint.held_over(times):
            for pose in ranked:
                if power(pose, lit[0]) - load <= request.minimum_margin_w:
                    continue
                if any(
                    attitude_constraint_name_for_scopes(
                        config.constraint,
                        scopes,
                        *pose[:2],
                        t,
                        target_roll=pose[2],
                        acs_mode=mode,
                    )
                    is not None
                    for t in times
                ):
                    continue
                if any(power(pose, t) - load <= request.minimum_margin_w for t in lit):
                    continue
                found = True
                yield pose
    if not found:
        raise SafeHoldUnavailableError(
            f"no candidate stays safe and exceeds the {load:g} W load by "
            f"{request.minimum_margin_w:g} W at every sunlit sample of the full hold"
        )
