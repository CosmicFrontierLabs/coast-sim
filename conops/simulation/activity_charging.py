"""Forecast whether a scheduled hold can finish before emergency recharge."""

from collections.abc import Mapping

from ..common import ACSMode, ObsType
from ..common.numerics import ROUNDOFF
from ..config import MissionConfig, SolarArrayDriveState
from ..ditl.ditl_log import DITLLog
from ..schedulers.context import SchedulingContext
from .emergency_charging import EmergencyCharging
from .slew import Slew


def can_finish_before_recharge(
    config: MissionConfig,
    context: SchedulingContext,
    *,
    now: float,
    hold_begin: float,
    hold_end: float,
    attitude: tuple[float, float, float],
    ingress: Slew | None,
    mode: ACSMode,
    instrument_modes: Mapping[str, str],
    charge_wh: float,
    drive_state: SolarArrayDriveState,
) -> bool:
    """Check the remaining hold, any eclipse wait, and the real charge ingress.

    Use the physical discharge floor, not the alert threshold that requested
    recharge. Clone drive state; this forecast cannot change the flown array.
    """
    capacity = float(config.battery.watthour)
    floor = config.battery.minimum_charge_level * capacity
    if charge_wh < floor - ROUNDOFF:
        return False
    end = min(hold_end, context.uend)
    state = drive_state.model_copy()

    def fly(begin: float, stop: float, *, charge_slew: Slew | None = None) -> bool:
        nonlocal charge_wh, state
        for sample in context.steps(begin, stop):
            at = float(sample)
            seconds = min(at + context.step_size, stop) - at
            moving = charge_slew or (
                ingress if ingress is not None and at < ingress.slewend else None
            )
            pose = moving.attitude(at) if moving is not None else attitude
            step_mode = (
                ACSMode.SLEWING
                if moving is not None
                else mode
                if at < end and at >= hold_begin
                else ACSMode.IDLE
            )
            named = instrument_modes if moving is None and at < end else None
            eclipse = config.constraint.in_eclipse(0, 0, at)
            watts = config.spacecraft_bus.power(
                step_mode, in_eclipse=eclipse
            ) + config.payload.power(
                step_mode, in_eclipse=eclipse, instrument_modes=named
            )
            _, generated, state = config.solar_panel.evaluate_executed_attitude(
                at,
                pose[0],
                pose[1],
                context.ephem,
                state,
                roll=pose[2],
                acs_mode=step_mode,
            )
            charge_wh = min(
                capacity,
                max(0.0, charge_wh - watts * seconds / 3600)
                + float(generated) * seconds / 3600,
            )
            if charge_wh < floor - ROUNDOFF:
                return False
        return True

    if not fly(now, end):
        return False
    if end >= context.uend:
        return True
    departure = context.ceil_step(end)
    while departure < context.uend and config.constraint.in_eclipse(0, 0, departure):
        departure += context.step_size
    departure = min(departure, context.uend)
    if context.first_hold_violation(attitude, end, departure, ACSMode.IDLE) is not None:
        return False
    if not fly(end, departure):
        return False
    if departure >= context.uend:
        return True
    charger = EmergencyCharging(config, log=DITLLog())
    pointing = charger.create_charging_pointing(
        departure,
        context.ephem,
        attitude[0],
        attitude[1],
        drive_state=state,
        lastroll=attitude[2],
    )
    if pointing is None:
        return False
    slew = context.slew(
        attitude,
        (pointing.ra, pointing.dec, pointing.roll),
        departure,
        obstype=ObsType.CHARGE,
    )
    arrival = context.ceil_step(float(slew.slewend))
    if (
        arrival > context.uend
        or context.first_slew_violation(slew, ACSMode.SLEWING) is not None
    ):
        return False
    if not fly(departure, arrival, charge_slew=slew):
        return False
    if arrival >= context.uend:
        return True
    if config.constraint.in_eclipse(0, 0, arrival):
        return False
    pose = (pointing.ra, pointing.dec, pointing.roll)
    if (
        context.first_hold_violation(
            pose,
            arrival,
            min(arrival + context.step_size, context.uend),
            ACSMode.CHARGING,
        )
        is not None
    ):
        return False
    _, generated, _ = config.solar_panel.evaluate_executed_attitude(
        arrival,
        *pose[:2],
        context.ephem,
        state,
        roll=pose[2],
        acs_mode=ACSMode.CHARGING,
    )
    draw = config.spacecraft_bus.power(ACSMode.CHARGING) + config.payload.power(
        ACSMode.CHARGING
    )
    return float(generated) > draw
