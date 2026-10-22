import math
from datetime import datetime, timezone
from typing import Any, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import rust_ephem
from pydantic import BaseModel, ConfigDict

from conops.common.enums import ACSMode, ObsType
from conops.common.vector import quaternion_attitude_delta
from conops.config.groundstation import GroundStation, effective_downlink_rate_mbps

from ..config import FaultManagementRun, MissionConfig, SolarArrayDriveState
from ..config.observation_timing import ObservationTiming, timing_for
from ..simulation.acs import ACS
from ..simulation.coverage import StationCoverage
from ..simulation.momentum import MomentumSample, StoredMomentumTracker
from ..simulation.passes import Pass, PassTimes
from ..targets import Plan, PlanEntry
from ..targets.plan_entry import extra_data_gb, instrument_modes_of
from .ditl_log import DITLLog
from .telemetry import SolarArrayDriveAngle, Telemetry

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
    allowed_distance_deg: float | None
    max_rate_deg_per_s: float | None
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
        if self.allowed_distance_deg is None or self.max_rate_deg_per_s is None:
            allowed = "unknown (maneuver axis unavailable)"
        else:
            allowed = (
                f"{self.allowed_distance_deg:.6f} deg "
                f"({self.max_rate_deg_per_s:.6f} deg/s)"
            )
        return (
            f"attitude_rate_violation: samples {self.previous_index}->{self.index} "
            f"at {self.previous_utime:.3f}->{self.utime:.3f}, "
            f"modes {self.previous_mode}->{self.mode}, reason {self.reason}, "
            f"rotation {self.distance_deg:.6f} deg over "
            f"{self.elapsed_seconds:.3f} s "
            f"({self.actual_rate_deg_per_s:.6f} deg/s), allowed {allowed}"
        )


class AttitudeRateContinuityError(RuntimeError):
    """Raised when adjacent executed attitudes violate the configured slew rate."""


class DITLMixin:
    def _instrument_modes_telemetry(
        self, mode: ACSMode | int, utime: float | None = None
    ) -> dict[str, str] | None:
        """The mode each instrument with modes is in at this step, by name,
        resolved as its power is: the observation's for the instruments it
        uses, and IDLE or a mode named like the ACS mode for the rest. None
        when no instrument defines modes."""
        payload = getattr(getattr(self, "config", None), "payload", None)
        instruments = getattr(payload, "instruments", None)
        # A test double's payload has no instruments.
        if not isinstance(instruments, list):
            return None
        with_modes = [i for i in instruments if getattr(i, "modes", None)]
        if not with_modes:
            return None
        try:
            acs_mode = int(mode)
        except (TypeError, ValueError):
            acs_mode = None
        named = self._instrument_modes_now(mode, utime) or {}
        return {
            instrument.name: instrument.mode_in(acs_mode, named.get(instrument.name))
            for instrument in with_modes
        }

    def _instrument_modes_now(
        self, mode: ACSMode | int, utime: float | None = None
    ) -> dict[str, str] | None:
        """The mode each instrument is in at this step: the observation's
        while ACS is in SCIENCE mode, a calibration step's own during a
        setup or cleanup step at ``utime``, a scheduled activity's while ACS
        holds for it in IDLE mode, else None, when every instrument idles or
        holds a mode named like the ACS mode."""
        try:
            acs_mode = ACSMode(int(mode))
        except (TypeError, ValueError):
            return None
        if acs_mode is ACSMode.IDLE:
            # Dispatch holds the activity itself; a plan holds its entry.
            active = getattr(self, "_active_activity", None)
            modes = getattr(active, "instrument_modes", None)
            if isinstance(modes, dict):
                return dict(modes)
            ppt = getattr(self, "ppt", None)
            if getattr(ppt, "obstype", None) == ObsType.ACTIVITY:
                modes = getattr(ppt, "instrument_modes", None)
                return dict(modes) if isinstance(modes, dict) else None
            return None
        ppt = getattr(self, "ppt", None)
        if acs_mode is not ACSMode.SCIENCE or ppt is None:
            return None
        by_time = getattr(ppt, "instrument_modes_at", None)
        if utime is not None and callable(by_time):
            resolved = by_time(utime)
            return resolved if isinstance(resolved, dict) else None
        modes = getattr(ppt, "effective_instrument_modes", None)
        # A test double's observation names no modes.
        resolved = modes() if callable(modes) else None
        return resolved if isinstance(resolved, dict) else None

    """Shared initialization, plotting, and data-management logic for DITL simulations."""

    ppt: PlanEntry | None
    log: DITLLog
    repoints_in_observations: list[tuple[float, int, str]]
    _data_observation: tuple[int, float] | None
    _data_collected: float
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
    uend: float
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
    _stored_momentum_tracker: StoredMomentumTracker | None

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
        self.plan.bind_runtime(self.config, self.ephem)
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
        self.fault_runs: list[FaultManagementRun] = []
        self._run_started = False

        # Current target
        self.ppt = None
        # The observation whose data is being generated, and its collection so far
        self._data_observation = None
        self._data_collected = 0.0

        # Initialize common subsystems (can be overridden by subclasses)
        self._init_subsystems()
        self._stored_momentum_tracker = self._build_stored_momentum_tracker()

    @property
    def fault_management(self) -> FaultManagementRun:
        """Current run's fault evaluator/report, shared with the ACS."""
        return self.acs.fault_management

    def _begin_fault_run(self) -> None:
        """Start an independent fault interval, retaining all earlier reports.

        This is not a replay/reset of battery, recorder or target consumption.
        A fresh simulation instance is needed to replay the entire scenario.
        """
        if self._run_started:
            # Commands, SAFE slews and mode/roll latches belong to the old run.
            # Retain the actual attitude as this interval's boundary condition.
            ra, dec, roll = self.acs.ra, self.acs.dec, self.acs.roll
            self.acs = ACS(config=self.config, log=self.acs.log)
            self.acs.ra, self.acs.dec, self.acs.roll = ra, dec, roll
            self.acs._hold_idle_attitude(ra, dec, roll, self.begin.timestamp())
            self.ppt = None
            self._data_observation = None
            self._data_collected = 0.0
            self.telemetry = Telemetry()
            self.executed_passes = PassTimes(config=self.config)
            self.repoints_in_observations = []
            # Do not concatenate samples from separate runs (possibly covering
            # identical dates) into a single attitude audit or plotting timeline.
            for name in (
                "ra",
                "dec",
                "roll",
                "mode",
                "obsid",
                "panel",
                "panel_power",
                "batterylevel",
                "charge_state",
                "power",
                "power_bus",
                "power_payload",
                "recorder_volume_gb",
                "recorder_fill_fraction",
                "recorder_alert",
                "data_generated_gb",
                "data_downlinked_gb",
                "in_eclipse",
            ):
                setattr(self, name, [])
        self.acs.fault_management = self.config.fault_management.new_run()
        self.fault_runs.append(self.fault_management)
        self._run_started = True

    def _init_subsystems(self) -> None:
        """Initialize subsystems from config. Can be overridden by subclasses."""
        self.constraint = self.config.constraint
        self.battery = self.config.battery
        self.spacecraft_bus = self.config.spacecraft_bus
        self.payload = self.config.payload
        self.recorder = self.config.recorder
        self.coverage: StationCoverage | None = StationCoverage.for_config(self.config)
        """Station coverage for an antenna that needs no attitude, which
        downlinks whenever a station is in view; None for a body-fixed
        antenna, which downlinks during ground passes."""
        self.repoints_in_observations = []
        """Each antenna repoint that fell in an observation whose category
        reports or forbids them: when, the observation's obsid, and the
        category's setting."""

    def _timing_for(self, entry: PlanEntry) -> ObservationTiming:
        """The setup, cleanup and handoff budgets of ``entry`` under the run's
        configuration: its category's where that sets them, else the payload's."""
        return timing_for(self.config, int(getattr(entry, "obsid", 0) or 0))

    def _antenna_repoints(self, target: PlanEntry) -> str:
        """The category setting for antenna repoints during ``target``."""
        categories = getattr(self.config, "observation_categories", None)
        setting = getattr(categories, "antenna_repoints", None)
        if not callable(setting):
            return "allowed"
        result = setting(int(target.obsid))
        return result if isinstance(result, str) else "allowed"

    def _build_stored_momentum_tracker(self) -> StoredMomentumTracker | None:
        attitude_control = self.config.spacecraft_bus.attitude_control
        momentum_config = getattr(attitude_control, "stored_momentum", None)
        if getattr(momentum_config, "gravity_gradient_enabled", False) is not True:
            return None
        assert momentum_config is not None
        inertia = getattr(self.config.spacecraft_bus, "inertia_tensor_body_kg_m2", None)
        if inertia is None:
            raise ValueError(
                "inertia_tensor_body_kg_m2 is required when gravity-gradient "
                "momentum tracking is enabled"
            )
        # A coupled-axis ellipsoid cannot exceed its fastest principal rate.
        # Bound possible motion, not just endpoint separation: a complete slew
        # can occur between samples even when both endpoint torques are zero.
        fastest_rate = (
            max(attitude_control.max_slew_rate_body)
            if attitude_control.max_slew_rate_body is not None
            else attitude_control.max_slew_rate
        )
        if not math.isfinite(fastest_rate) or fastest_rate <= 0.0:
            raise ValueError(
                "momentum tracking requires a finite positive maximum slew rate"
            )
        return StoredMomentumTracker(
            inertia_tensor_body_kg_m2=inertia,
            initial_momentum_body_n_m_s=(momentum_config.initial_momentum_body_n_m_s),
            max_sample_interval_s=min(
                momentum_config.max_sample_interval_s,
                momentum_config.max_attitude_step_deg / fastest_rate,
            ),
        )

    def _reset_stored_momentum_tracker(self) -> None:
        """Reload run inputs and fail before execution if sampling is too coarse."""
        self._stored_momentum_tracker = self._build_stored_momentum_tracker()
        if self._stored_momentum_tracker is not None:
            self._stored_momentum_tracker.validate_sample_interval(self.step_size)
            self._stored_momentum_tracker.validate_sample_interval(self.ephem.step_size)

    def _update_stored_momentum(
        self,
        utime: float,
        position_eci_km: npt.NDArray[np.float64],
        attitude_quaternion_eci_to_body: npt.NDArray[np.float64],
    ) -> MomentumSample | None:
        """Advance the tracker using values already computed for housekeeping."""
        if self._stored_momentum_tracker is None:
            return None
        return self._stored_momentum_tracker.update(
            utime=utime,
            position_eci_km=position_eci_km,
            attitude_quaternion_eci_to_body=attitude_quaternion_eci_to_body,
        )

    def _solar_array_drive_telemetry(self) -> list[SolarArrayDriveAngle] | None:
        """Return executed drive angles with stable configured panel identity."""
        state = self.acs.solar_array_drive_state
        if not isinstance(state, SolarArrayDriveState):
            return None
        entries = [
            SolarArrayDriveAngle(
                panel_index=index,
                panel_name=panel.name,
                angle_deg=angle,
            )
            for index, (panel, angle) in enumerate(
                zip(self.config.solar_panel.panels, state.angles_deg)
            )
            if angle is not None
        ]
        return entries or None

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
        acs_config = self.config.spacecraft_bus.attitude_control
        fallback_max_rate = (
            None
            if acs_config.max_slew_rate_body is not None
            else acs_config.effective_max_slew_rate()
        )
        if fallback_max_rate is not None and (
            not math.isfinite(fallback_max_rate) or fallback_max_rate < 0
        ):
            raise ValueError(
                "configured slew-rate limit must be finite and non-negative, "
                f"got {fallback_max_rate}"
            )

        violations = []
        samples = self.telemetry.housekeeping
        for index in range(1, len(samples)):
            max_rate = fallback_max_rate
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
                distance_deg, rotation_axis_body = quaternion_attitude_delta(*attitudes)
                if distance_deg > 0.0:
                    max_rate = acs_config.effective_max_slew_rate(rotation_axis_body)

            allowed_distance_deg = (
                max_rate * elapsed_seconds
                if max_rate is not None
                and math.isfinite(elapsed_seconds)
                and elapsed_seconds > 0
                else None
            )
            if timestamps_are_finite and elapsed_seconds <= 0:
                reason = "non_increasing_timestamp"

            exceeds_rate_limit = (
                math.isfinite(distance_deg)
                and allowed_distance_deg is not None
                and distance_deg
                > allowed_distance_deg + ATTITUDE_RATE_NUMERICAL_TOLERANCE_DEG
            )
            if (
                not timestamps_are_finite
                or elapsed_seconds <= 0
                or not math.isfinite(distance_deg)
                or exceeds_rate_limit
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
            attach_earth_orientation_metadata,
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
        attach_earth_orientation_metadata(self.plan)

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

    def _collection_seconds_for_step(
        self, utime: float, mode: ACSMode, obsid: int | None = None
    ) -> float:
        """Intersect the planned collection window with executed slew/setup readiness."""
        start, stop = self._collection_interval_for_step(utime, mode, obsid)
        return stop - start

    def _collection_interval_for_step(
        self, utime: float, mode: ACSMode, obsid: int | None = None
    ) -> tuple[float, float]:
        """Return the (start, stop) of science collected this step; empty when equal."""
        if (
            self.ppt is None
            or self.ppt.collection_end is None
            or mode not in (ACSMode.SCIENCE, ACSMode.SLEWING)
            or (obsid is not None and obsid != self.ppt.obsid)
        ):
            return utime, utime
        start = utime
        slew = self.acs.current_slew if mode == ACSMode.SLEWING else self.acs.last_slew
        if slew is not None:
            if slew.obsid != self.ppt.obsid:
                return utime, utime
            assert self.ppt.collection_begin is not None
            # Infer setup from the saved window, not a possibly changed replay config.
            setup = max(
                0.0, self.ppt.collection_begin - self.ppt.begin - self.ppt.slewtime
            )
            start = max(start, slew.slewend + setup)
        elif mode == ACSMode.SLEWING:
            return utime, utime
        return self.ppt.collection_interval_between(
            start, min(utime + self.step_size, self.uend)
        )

    def _process_data_management(
        self,
        utime: float,
        mode: ACSMode,
        step_size: int,
        *,
        collection_seconds: float,
    ) -> tuple[float, float]:
        """Process data generation and downlink for a single timestep.

        Args:
            utime: Unix timestamp for current timestep.
            mode: Current ACS mode.
            step_size: Time step in seconds.
            collection_seconds: Useful collection in this step, computed from the
                observation window rather than inferred from the ACS mode.

        Returns:
            Tuple of (data_generated, data_downlinked) in Gb for this timestep.
        """
        data_downlinked = 0.0

        # Housekeeping data for the whole step, in its ACS mode.
        interval = max(0.0, min(utime + step_size, self.uend) - utime)
        data_generated = self.config.housekeeping_generated(interval, mode=mode)

        # Science data for the time collected, which can be less than the
        # step: ACS science pointing can include payload setup and cleanup.
        # What an observation has generated is its rate for the seconds it
        # has collected plus its fixed amount, so the step adds the increase.
        if collection_seconds > 0:
            science = self.ppt.science_data if self.ppt is not None else None
            observation = (
                (id(self.ppt), float(self.ppt.begin)) if self.ppt is not None else None
            )
            before = (
                self._data_collected if observation == self._data_observation else 0.0
            )
            after = before + collection_seconds
            modes = instrument_modes_of(self.ppt)
            data_generated += self.config.science_generated(
                after, science=science, instrument_modes=modes
            )
            if before > 0:
                data_generated -= self.config.science_generated(
                    before, science=science, instrument_modes=modes
                )
            else:
                # Once per observation: its calibration steps' frames.
                data_generated += extra_data_gb(self.ppt)
            self._data_observation = observation
            self._data_collected = after

        current_pass = self.acs.current_pass
        contact_start = contact_stop = utime
        data_to_downlink = 0.0
        if self.coverage is not None:
            # An antenna that needs no attitude downlinks whenever a station
            # is in view, in whatever mode ACS is in. A gimballed antenna
            # reaches the station only within its range from the attitude
            # flown, taken as the attitude at the start of the step.
            attitude = (
                (self.acs.ra, self.acs.dec, self.acs.roll)
                if self.coverage.gimballed
                else None
            )
            contact_start, contact_stop, data_to_downlink = self.coverage.contact_in(
                utime, utime + interval, attitude
            )
            self._log_repoints(utime, utime + interval, collection_seconds)
        elif mode == ACSMode.PASS and current_pass is not None:
            # Downlink data during PASS mode, once ACS has acquired the
            # contact: the ingress slew also reports PASS, but carries no link.
            # A contact can end partway through a step, and the run can end
            # before the step does; only the time in contact is credited.
            contact_start = max(utime, float(current_pass.begin))
            contact_stop = max(
                contact_start,
                min(utime + step_size, float(current_pass.end), self.uend),
            )
            station = self.config.ground_stations.get(current_pass.station)

            # Determine actual data rate based on both ground station and spacecraft capabilities
            effective_rate_mbps = self._get_effective_data_rate(station)
            if effective_rate_mbps is not None and effective_rate_mbps > 0:
                # Mbps * seconds = megabits; the recorder counts gigabits.
                megabits = effective_rate_mbps * (contact_stop - contact_start)
                data_to_downlink = megabits / 1000.0

        if data_to_downlink <= 0:
            if data_generated > 0:
                self.recorder.add_data(data_generated)
            return data_generated, data_downlinked

        # Generation runs on through the contact, so the step is taken in
        # pieces: what is generated before the contact is stored, what is
        # generated during it meets the downlink at the recorder, and what is
        # generated after it is stored. The contact never sends data generated
        # after it ends, and nothing is lost to a full recorder while the link
        # empties it faster than data arrives.
        rate = data_generated / interval
        self.recorder.add_data(rate * (contact_start - utime))
        data_downlinked = self.recorder.transfer(
            rate * (contact_stop - contact_start), data_to_downlink
        )
        self.recorder.add_data(rate * (utime + interval - contact_stop))
        return data_generated, data_downlinked

    def _log_repoints(
        self, begin: float, end: float, collection_seconds: float
    ) -> None:
        """Log the antenna repoints in ``[begin, end)``, and note each that
        falls in an observation whose category reports or forbids them."""
        assert self.coverage is not None
        for repoint in self.coverage.repoints_in(begin, end):
            description = f"Antenna repoint to {repoint.station} ({repoint.kind}"
            if repoint.maneuver_deg is not None:
                description += f", {repoint.maneuver_deg:.1f} deg"
            description += ")"
            if self.ppt is not None and collection_seconds > 0:
                setting = self._antenna_repoints(self.ppt)
                if setting in ("reported", "forbidden"):
                    obsid = int(self.ppt.obsid)
                    self.repoints_in_observations.append((repoint.time, obsid, setting))
                    description += f" during observation {obsid} ({setting})"
            self.log.log_event(
                utime=repoint.time,
                event_type="ANTENNA",
                description=description,
                obsid=int(self.ppt.obsid) if self.ppt is not None else None,
                acs_mode=self.acs.acsmode,
            )

    def _get_effective_data_rate(self, station: GroundStation) -> float | None:
        """The downlink rate in Mbps the spacecraft gets from ``station``, or None.

        See :func:`~conops.config.groundstation.effective_downlink_rate_mbps`.
        """
        return effective_downlink_rate_mbps(
            station, self.config.spacecraft_bus.communications
        )
