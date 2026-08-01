import math
from datetime import datetime, timezone
from typing import Any, cast

import matplotlib.pyplot as plt
import rust_ephem
from pydantic import BaseModel, ConfigDict

from conops.common.enums import ACSMode
from conops.common.vector import quaternion_attitude_distance
from conops.config.groundstation import GroundStation

from ..config import MissionConfig
from ..simulation.acs import ACS
from ..simulation.passes import Pass, PassTimes
from ..targets import Plan, PlanEntry
from .telemetry import Telemetry

ATTITUDE_RATE_NUMERICAL_TOLERANCE_DEG = 1e-9


class _AttitudeRateViolation(BaseModel):
    """An adjacent attitude sample pair that exceeds the configured slew rate."""

    model_config = ConfigDict(frozen=True)

    previous_index: int
    index: int
    previous_utime: float
    utime: float
    elapsed_seconds: float
    distance_deg: float
    allowed_distance_deg: float
    max_rate_deg_per_s: float
    previous_mode: str | None
    mode: str | None
    obsid: int | None
    reason: str = "rate_limit_exceeded"

    @property
    def actual_rate_deg_per_s(self) -> float:
        """Return the average angular rate over the sample interval."""
        if not math.isfinite(self.elapsed_seconds) or self.elapsed_seconds <= 0:
            return math.inf
        return self.distance_deg / self.elapsed_seconds

    def __str__(self) -> str:
        """Return a concise diagnostic suitable for plan-generation errors."""
        return (
            f"attitude_rate_violation: samples {self.previous_index}->{self.index} "
            f"at {self.previous_utime:.3f}->{self.utime:.3f}, "
            f"modes {self.previous_mode}->{self.mode}, reason {self.reason}, "
            f"rotation {self.distance_deg:.6f} deg over "
            f"{self.elapsed_seconds:.3f} s "
            f"({self.actual_rate_deg_per_s:.6f} deg/s), allowed "
            f"{self.allowed_distance_deg:.6f} deg "
            f"({self.max_rate_deg_per_s:.6f} deg/s)"
        )


class AttitudeRateContinuityError(RuntimeError):
    """Raised when adjacent executed attitudes violate the configured slew rate."""


class DITLMixin:
    """Shared initialization, plotting, and data-management logic for DITL simulations."""

    ppt: PlanEntry | None
    ra: list[float]
    dec: list[float]
    roll: list[float]
    mode: list[int]
    panel: list[float]
    power: list[float]
    in_eclipse: list[bool]
    begin: datetime
    end: datetime
    step_size: int
    panel_power: list[float]
    batterylevel: list[float]
    charge_state: list[int]
    obsid: list[int]
    plan: Plan
    utime: list[float]
    ephem: rust_ephem.Ephemeris
    # Subsystem power tracking
    power_bus: list[float]
    power_payload: list[float]
    # Data recorder tracking
    recorder_volume_gb: list[float]
    recorder_fill_fraction: list[float]
    recorder_alert: list[int]
    data_generated_gb: list[float]
    data_downlinked_gb: list[float]
    # Telemetry container
    telemetry: Telemetry
    calculate_field_of_regard: bool

    def __init__(
        self,
        config: MissionConfig,
        ephem: rust_ephem.Ephemeris | None = None,
        begin: datetime | None = None,
        end: datetime | None = None,
        plan: Plan | None = None,
        calculate_field_of_regard: bool = False,
    ) -> None:
        """Initialize shared DITL state, ephemeris, and subsystems from config."""
        # Initialize mixin
        self.config = config
        self.calculate_field_of_regard = calculate_field_of_regard

        # Initialize telemetry container
        self.telemetry = Telemetry()

        # Set ephemeris if provided
        if ephem is not None:
            self.ephem = ephem
            self.config.constraint.ephem = ephem
            # Also set ephemeris on star tracker constraints
            self.config.spacecraft_bus.star_trackers.set_ephem(ephem)
            self.config.spacecraft_bus.radiators.set_ephem(ephem)
        else:
            assert config.constraint.ephem is not None, (
                "Ephemeris must be set in Config Constraint"
            )
            self.ephem = config.constraint.ephem
            # Also set ephemeris on star tracker constraints
            self.config.spacecraft_bus.star_trackers.set_ephem(config.constraint.ephem)
            self.config.spacecraft_bus.radiators.set_ephem(config.constraint.ephem)

        # Keep mission-level planning/FOR constraints synchronized with star-tracker
        # hard exclusions.
        self.config.constraint.star_tracker_hard_constraint = (
            self.config.spacecraft_bus.star_trackers.startracker_hard_constraint
        )
        self.config.constraint.star_tracker_soft_constraint = (
            self.config.spacecraft_bus.star_trackers.startracker_constraint
        )
        self.config.constraint.radiator_hard_constraint = (
            self.config.spacecraft_bus.radiators.radiator_hard_constraint
        )
        self.config.constraint.invalidate_combined_constraint_cache()

        # Override begin/end if provided, else use limits of ephemeris
        if begin is not None:
            self.begin = begin
        else:
            self.begin = self.ephem.timestamp[0]
        if end is not None:
            self.end = end
        else:
            self.end = self.ephem.timestamp[-1]

        self.ra = []
        self.dec = []
        self.roll = []
        self.utime = []
        self.mode = []
        self.obsid = []
        # Defining when the model is run
        self.step_size = 60  # seconds
        self.ustart = 0.0  # Calculate these
        self.uend = 0.0  # later
        self.plan = plan if plan is not None else Plan()
        self.saa = None
        self.passes = PassTimes(config=config)
        self.executed_passes = PassTimes(config=config)

        # Set up event based ACS
        assert self.config.constraint.ephem is not None, (
            "Ephemeris must be set in Config Constraint"
        )
        # Note: log will be set by subclass (DITL/QueueDITL) before use
        # For now, create ACS without log (will be set later)
        self.acs = ACS(config=self.config, log=None)

        # Current target
        self.ppt = None

        # Initialize common subsystems (can be overridden by subclasses)
        self._init_subsystems()

    def _init_subsystems(self) -> None:
        """Initialize subsystems from config. Can be overridden by subclasses."""
        self.constraint = self.config.constraint
        self.battery = self.config.battery
        self.spacecraft_bus = self.config.spacecraft_bus
        self.payload = self.config.payload
        self.recorder = self.config.recorder

    @staticmethod
    def _attitude_mode_name(mode: ACSMode | int | None) -> str | None:
        """Return the mode's name, coercing an int/None to a name or None."""
        if mode is None:
            return None
        if isinstance(mode, ACSMode):
            return mode.name
        try:
            return ACSMode(int(mode)).name
        except (TypeError, ValueError):
            return str(mode)

    @staticmethod
    def _timestamp_to_utc(timestamp: Any) -> datetime:
        if isinstance(timestamp, datetime):
            dt = timestamp
        else:
            dt = datetime.fromtimestamp(float(timestamp), tz=timezone.utc)
        if dt.tzinfo is None:
            return dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)

    def _attitude_rate_violations(self) -> list[_AttitudeRateViolation]:
        """Validate the housekeeping samples used for attitude-timeseries export."""
        max_rate = float(self.config.spacecraft_bus.attitude_control.max_slew_rate)
        if not math.isfinite(max_rate) or max_rate < 0:
            raise ValueError(
                f"max_slew_rate must be a finite, non-negative value, got {max_rate}"
            )

        violations = []
        samples = self.telemetry.housekeeping
        for index in range(1, len(samples)):
            previous_index = index - 1
            previous_sample = samples[previous_index]
            sample = samples[index]
            previous_utime = self._timestamp_to_utc(
                previous_sample.timestamp
            ).timestamp()
            utime = self._timestamp_to_utc(sample.timestamp).timestamp()
            elapsed_seconds = utime - previous_utime
            attitude_values = (
                previous_sample.ra,
                previous_sample.dec,
                previous_sample.roll,
                sample.ra,
                sample.dec,
                sample.roll,
            )

            timestamps_are_finite = math.isfinite(previous_utime) and math.isfinite(
                utime
            )
            reason = "rate_limit_exceeded"
            if not timestamps_are_finite:
                distance_deg = math.inf
                reason = "non_finite_timestamp"
            elif any(value is None for value in attitude_values):
                distance_deg = math.inf
                reason = "missing_attitude"
            else:
                attitudes = cast(
                    tuple[float, float, float, float, float, float],
                    attitude_values,
                )
                if not all(math.isfinite(value) for value in attitudes):
                    distance_deg = math.inf
                    reason = "non_finite_attitude"

            if reason == "rate_limit_exceeded":
                distance_deg = quaternion_attitude_distance(*attitudes)

            allowed_distance_deg = (
                max_rate * elapsed_seconds
                if math.isfinite(elapsed_seconds) and elapsed_seconds > 0
                else 0.0
            )
            if timestamps_are_finite and elapsed_seconds <= 0:
                reason = "non_increasing_timestamp"

            if (
                not timestamps_are_finite
                or elapsed_seconds <= 0
                or not math.isfinite(distance_deg)
                or distance_deg
                > allowed_distance_deg + ATTITUDE_RATE_NUMERICAL_TOLERANCE_DEG
            ):
                violations.append(
                    _AttitudeRateViolation(
                        previous_index=previous_index,
                        index=index,
                        previous_utime=previous_utime,
                        utime=utime,
                        elapsed_seconds=elapsed_seconds,
                        distance_deg=distance_deg,
                        allowed_distance_deg=allowed_distance_deg,
                        max_rate_deg_per_s=max_rate,
                        previous_mode=self._attitude_mode_name(
                            previous_sample.acs_mode
                        ),
                        mode=self._attitude_mode_name(sample.acs_mode),
                        obsid=sample.obsid,
                        reason=reason,
                    )
                )
        return violations

    def _assert_attitude_rate_continuity(self) -> None:
        """Fail plan generation when adjacent attitudes exceed the slew-rate limit."""
        violations = self._attitude_rate_violations()
        if not violations:
            return
        examples = "; ".join(str(violation) for violation in violations[:5])
        raise AttitudeRateContinuityError(
            f"Attitude rate validation failed with {len(violations)} "
            f"violation(s): {examples}"
        )

    def _attach_execution_timeseries_to_plan(self) -> None:
        """Attach executed attitude and orbit-state timelines to the current plan."""
        self._attach_attitude_timeseries_to_plan()
        self._attach_orbit_state_timeseries_to_plan()

    def _attach_attitude_timeseries_to_plan(self) -> None:
        """Attach the executed attitude timeline to the current plan for export."""
        from ..targets import AttitudeSampleSchema, AttitudeTimeseriesSchema

        samples = []
        for hk in self.telemetry.housekeeping:
            if hk.timestamp.tzinfo is None:
                timestamp = hk.timestamp.replace(tzinfo=timezone.utc)
            else:
                timestamp = hk.timestamp.astimezone(timezone.utc)
            samples.append(
                AttitudeSampleSchema(
                    utime=timestamp.timestamp(),
                    timestamp=timestamp.isoformat(),
                    ra=hk.ra,
                    dec=hk.dec,
                    roll=hk.roll,
                    mode=self._attitude_mode_name(hk.acs_mode),
                    obsid=hk.obsid,
                    quat_w=hk.quat_w,
                    quat_x=hk.quat_x,
                    quat_y=hk.quat_y,
                    quat_z=hk.quat_z,
                )
            )

        self.plan.attitude_timeseries = AttitudeTimeseriesSchema(samples=samples)

    def _attach_orbit_state_timeseries_to_plan(self) -> None:
        """Attach GCRS spacecraft position/velocity samples to the current plan."""
        from ..targets import (
            OrbitStateSampleSchema,
            OrbitStateTimeseriesSchema,
            attach_osculating_elements_metadata,
        )

        pv = getattr(self.ephem, "gcrs_pv", None)
        positions = getattr(pv, "position", None)
        velocities = getattr(pv, "velocity", None)
        timestamps = getattr(self.ephem, "timestamp", None)
        if positions is None or velocities is None or timestamps is None:
            return

        try:
            timestamp_count = len(timestamps)
            position_count = len(positions)
            velocity_count = len(velocities)
        except TypeError:
            return
        if not timestamp_count == position_count == velocity_count:
            raise ValueError(
                "Orbit state timeseries inputs must have matching lengths "
                f"(timestamps={timestamp_count}, positions={position_count}, "
                f"velocities={velocity_count})"
            )

        samples = []
        for i in range(timestamp_count):
            timestamp = self._timestamp_to_utc(timestamps[i])
            position = positions[i]
            velocity = velocities[i]
            samples.append(
                OrbitStateSampleSchema(
                    utime=timestamp.timestamp(),
                    timestamp=timestamp.isoformat(),
                    position_km=(
                        float(position[0]),
                        float(position[1]),
                        float(position[2]),
                    ),
                    velocity_km_s=(
                        float(velocity[0]),
                        float(velocity[1]),
                        float(velocity[2]),
                    ),
                )
            )

        self.plan.orbit_state_timeseries = OrbitStateTimeseriesSchema(samples=samples)
        attach_osculating_elements_metadata(
            self.plan,
            self.ephem,
            self.begin,
        )

    def plot(self) -> None:
        """Plot DITL timeline.

        .. deprecated::
            Use :func:`conops.visualization.plot_ditl_telemetry` instead.
            This method is maintained for backward compatibility.
        """
        from ..visualization import plot_ditl_telemetry

        plot_ditl_telemetry(self, config=self.config.visualization)
        plt.show()

    def _find_current_pass(self, utime: float) -> Pass | None:
        """Find the current pass at the given time.

        Args:
            utime: Unix timestamp to check.

        Returns:
            Pass object if currently in a pass, None otherwise.
        """
        # Check in ACS passrequests (scheduled passes)
        if self.acs.passrequests.passes:
            for pass_obj in self.acs.passrequests.passes:
                if pass_obj.in_pass(utime):
                    return pass_obj

        # Fallback to executed_passes for backwards compatibility
        if self.executed_passes.passes:
            for pass_obj in self.executed_passes.passes:
                if pass_obj.in_pass(utime):
                    return pass_obj

        return None

    def _process_data_management(
        self, utime: float, mode: ACSMode, step_size: int
    ) -> tuple[float, float]:
        """Process data generation and downlink for a single timestep.

        Args:
            utime: Unix timestamp for current timestep.
            mode: Current ACS mode.
            step_size: Time step in seconds.

        Returns:
            Tuple of (data_generated, data_downlinked) in Gb for this timestep.
        """
        data_generated = 0.0
        data_downlinked = 0.0

        # Generate data during SCIENCE mode
        if mode == ACSMode.SCIENCE:
            data_generated = self.payload.data_generated(step_size)
            self.recorder.add_data(data_generated)

        # Downlink data during PASS mode
        if mode == ACSMode.PASS:
            current_pass = self._find_current_pass(utime)
            if current_pass is not None:
                station = self.config.ground_stations.get(current_pass.station)

                # Determine actual data rate based on both ground station and spacecraft capabilities
                effective_rate_mbps = self._get_effective_data_rate(station)

                if effective_rate_mbps is not None and effective_rate_mbps > 0:
                    # Convert Mbps to Gb per step: Mbps * seconds / 1000 / 8 = Gb
                    megabits_per_step = effective_rate_mbps * step_size
                    data_to_downlink = megabits_per_step / 1000.0 / 8.0  # Convert to Gb
                    data_downlinked = self.recorder.remove_data(data_to_downlink)

        return data_generated, data_downlinked

    def _get_effective_data_rate(self, station: GroundStation) -> float | None:
        """Calculate effective downlink data rate based on ground station and spacecraft capabilities.

        The effective rate is, per band, min(GS downlink rate, SC downlink rate);
        we take the maximum of this across all common bands.

        Args:
            station: GroundStation object with antenna capabilities

        Returns:
            Effective data rate in Mbps, or None if no compatible bands/rates
        """
        # If pass has no comms config, use GS overall maximum across bands
        if self.config.spacecraft_bus.communications is None:
            return station.get_overall_max_downlink()

        # If GS has no per-band capabilities, no defined rate
        gs_bands = set(station.supported_bands()) if station.bands else set()
        if not gs_bands:
            # No bands defined on ground station
            return None

        # Compute effective rate per common band
        best_effective = 0.0
        for band in gs_bands:
            gs_rate = station.get_downlink_rate(band) or 0.0
            sc_rate = (
                self.config.spacecraft_bus.communications.get_downlink_rate(band) or 0.0
            )
            if gs_rate > 0.0 and sc_rate > 0.0:
                effective = min(gs_rate, sc_rate)
                if effective > best_effective:
                    best_effective = effective

        return best_effective if best_effective > 0.0 else None
