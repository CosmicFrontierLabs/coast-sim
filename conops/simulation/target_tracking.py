"""The moving-target guidance shared by execution and offline planning."""

from typing import TYPE_CHECKING

from ..config import AttitudeControlSystem
from .attitude import AttitudeState, AttitudeTrajectory

if TYPE_CHECKING:
    from ..targets import PlanEntry

TRACK_SAMPLE_SECONDS = 60.0


def target_tracking(
    target: "PlanEntry",
    arrival: AttitudeState,
    instrument_roll: float,
    limits: AttitudeControlSystem,
    sample_seconds: float = TRACK_SAMPLE_SECONDS,
) -> AttitudeTrajectory | None:
    """Follow a target from arrival through its end, then brake to rest.

    Every positive observation gets a final interval, including observations
    shorter than the guidance sampling interval.
    """
    end = float(target.end or 0.0)
    if not target.moves or end <= arrival.utime:
        return None
    samples = [(arrival.utime, arrival.attitude)]
    # Acquisition ends at rest. A full guidance interval to pick up the
    # track can lag a fast target beyond the pointing tolerance. Use a
    # shorter first interval to acquire its rate, then the normal cadence.
    rate = target.track_rate_deg_per_s(arrival.utime)
    first_interval = (
        min(sample_seconds, limits.slew_accuracy / (2.0 * rate))
        if rate > 0.0
        else sample_seconds
    )
    time = arrival.utime + first_interval
    while time < end:
        samples.append((time, target.target_body_attitude(instrument_roll, at=time)))
        time += sample_seconds
    samples.append((end, target.target_body_attitude(instrument_roll, at=end)))
    return AttitudeTrajectory.tracking(samples, limits, initial_state=arrival)
