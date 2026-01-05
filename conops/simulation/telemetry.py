from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WheelReading:
    name: str
    momentum: float
    max_momentum: float
    momentum_fraction: float
    momentum_fraction_raw: float
    torque_command: float
    torque_applied: float
    max_torque: float


@dataclass
class WheelTelemetrySnapshot:
    utime: float
    wheels: list[WheelReading]
    max_momentum_fraction: float
    max_momentum_fraction_raw: float
    max_torque_fraction: float
    saturated: bool
    t_actual_mag: float
    hold_torque_target_mag: float
    hold_torque_actual_mag: float
    pass_tracking_rate_deg_s: float
    pass_torque_target_mag: float
    pass_torque_actual_mag: float
    mtq_proj_max: float
    mtq_torque_mag: float
    mtq_bleed_torque_mag: float
    mtq_power_w: float
    body_momentum: tuple[float, float, float]
    external_impulse: tuple[float, float, float]
