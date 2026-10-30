import math
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, NamedTuple, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import rust_ephem
from pydantic import BaseModel, ConfigDict

from conops.common.enums import ACSMode, ObsType
from conops.common.vector import attitude_to_quat, quaternion_attitude_delta
from conops.config.groundstation import GroundStation, effective_downlink_rate_mbps

from ..common.ephemeris import position_vectors, velocity_vectors
from ..config import (
    FaultManagementRun,
    MissionConfig,
    OnboardRecorder,
    SolarArrayDriveState,
)
from ..config.data_storage import DataStorage, RecorderSummary, RecorderView
from ..config.observation_timing import ObservationTiming, timing_for
from ..schedulers.recorder import contact_downlink_gb
from ..simulation.acs import ACS
from ..simulation.attitude import slew_angular_velocity
from ..simulation.coverage import StationCoverage
from ..simulation.momentum import (
    MomentumSample,
    StoredMomentumTracker,
    build_torque_sources,
    slew_sample_interval_s,
    tracker_from_config,
)
from ..simulation.momentum_budget import MomentumBudget, MomentumLimitError
from ..simulation.passes import Pass, PassTimes
from ..simulation.slew import Slew
from ..targets import Plan, PlanEntry
from ..targets.plan_entry import extra_data_gb, instrument_modes_of
from .ditl_log import DITLLog
from .telemetry import SolarArrayDriveAngle, Telemetry

ATTITUDE_RATE_NUMERICAL_TOLERANCE_DEG = 1e-9


class _MomentumInputs:
    """The orbit at one momentum sample: what the tracker was given at a
    step, kept so the orbit can be interpolated along a slew within the next."""

    def __init__(
        self,
        utime: float,
        position_eci_km: npt.NDArray[np.float64],
        sun_eci_km: npt.NDArray[np.float64] | None,
        position_ecef_km: npt.NDArray[np.float64] | None,
        velocity_eci_km_s: npt.NDArray[np.float64] | None = None,
    ) -> None:
        self.utime = float(utime)
        self.position_eci_km = position_eci_km
        self.sun_eci_km = sun_eci_km
        self.position_ecef_km = position_ecef_km
        self.velocity_eci_km_s = velocity_eci_km_s

    def between(self, later: "_MomentumInputs", fraction: float) -> "_MomentumInputs":
        """The orbit ``fraction`` of the way from this sample to ``later``."""

        def lerp(
            a: npt.NDArray[np.float64] | None, b: npt.NDArray[np.float64] | None
        ) -> npt.NDArray[np.float64] | None:
            if a is None or b is None:
                return None
            return a + fraction * (b - a)

        return _MomentumInputs(
            self.utime + fraction * (later.utime - self.utime),
            np.asarray(lerp(self.position_eci_km, later.position_eci_km)),
            lerp(self.sun_eci_km, later.sun_eci_km),
            lerp(self.position_ecef_km, later.position_ecef_km),
            lerp(self.velocity_eci_km_s, later.velocity_eci_km_s),
        )


class MomentumProfile(NamedTuple):
    """A momentum forecast: the sample times, the stored momentum's norm at
    each, what the wheels hold at each (the stored momentum less the body's
    own along a slew, where slew momentum counts; the stored momentum
    otherwise) and the momentum at the end in inertial coordinates."""

    times: npt.NDArray[np.float64]
    norms: npt.NDArray[np.float64]
    demands: npt.NDArray[np.float64]
    end_eci_n_m_s: npt.NDArray[np.float64] | None


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
        holds for it in IDLE mode, or in DESAT for a thruster dump, else
        None, when every instrument idles or holds a mode named like the
        ACS mode."""
        try:
            acs_mode = ACSMode(int(mode))
        except (TypeError, ValueError):
            return None
        if acs_mode in (ACSMode.IDLE, ACSMode.DESAT):
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
    _activity_data_gb: float
    _store_activity_data_gb: dict[str, float]
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
    _store_data_totals: dict[str, dict[str, float]]
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
        # Resolve hardware inventory before creating the ephemeris or subsystems.
        config.recorder_views(refresh=True)
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
        # What activities put on the recorder since the last data step
        self._activity_data_gb = 0.0
        self._store_activity_data_gb = {}

        # Initialize common subsystems (can be overridden by subclasses)
        self._init_subsystems()
        self._stored_momentum_tracker = self._build_stored_momentum_tracker()

    @property
    def fault_management(self) -> FaultManagementRun:
        """Current run's fault evaluator/report, shared with the ACS."""
        return self.acs.fault_management

    def _report_momentum_failure(self, error: MomentumLimitError) -> None:
        # A rejected command can use ACS fault recovery. If advancing the
        # installed physical trajectory itself fails, terminate the run:
        # freezing attitude or disabling the guard cannot provide a safehold.
        executor = self.acs._executor
        self.fault_management.report_fault(
            utime=executor.state.utime
            if executor is not None
            else self.begin.timestamp(),
            name="attitude_execution",
            cause=str(error),
            metadata={"momentum_capacity": True},
            acs=self.acs,
        )

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
            self._activity_data_gb = 0.0
            self._store_activity_data_gb = {}
            self.telemetry = Telemetry()
            self._store_data_totals = {}
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
        self._store_data_totals = {}
        self._store_activity_data_gb = {}
        self._activity_data_gb = 0.0
        self.recorder = (
            RecorderSummary(self.config.recorder_views())
            if isinstance(self.config.data_storage, DataStorage)
            else self.config.recorder
        )
        self.coverage: StationCoverage | None = StationCoverage.for_config(
            self.config, scheduled_passes=self.plan.ground_contacts
        )
        """Usable links during passes without spacecraft tracking; None
        when passes require tracking activities."""
        if self.coverage is not None:
            self.passes = self.coverage.pass_times
        self.plan.ground_contacts = self._booked_ground_contacts()
        self.repoints_in_observations = []
        """Each antenna repoint that fell in an observation whose category
        reports or forbids them: when, the observation's obsid, and the
        category's setting."""

    def _booked_ground_contacts(self) -> list[tuple[float, float, str]] | None:
        """Station reservations to preserve in the executed plan."""
        return (
            [
                (span.begin, span.end, span.station)
                for span in self.coverage.scheduled_passes
            ]
            if self.coverage is not None
            else None
        )

    def _timing_for(self, entry: PlanEntry) -> ObservationTiming:
        """The setup, cleanup and handoff budgets of ``entry``: its own where
        it has them, else its category's under the run's configuration, else
        the payload's."""
        own = getattr(entry, "own_timing", None)
        if isinstance(own, ObservationTiming):
            return own
        return timing_for(self.config, int(getattr(entry, "obsid", 0) or 0))

    def _antenna_repoints(self, target: PlanEntry) -> str:
        """The category setting for antenna repoints during ``target``."""
        return self.config.observation_categories.antenna_repoints(int(target.obsid))

    def _build_stored_momentum_tracker(self) -> StoredMomentumTracker | None:
        attitude_control = self.config.spacecraft_bus.attitude_control
        momentum_config = getattr(attitude_control, "stored_momentum", None)
        if getattr(momentum_config, "enabled", False) is not True:
            return None
        assert momentum_config is not None
        inertia = getattr(self.config.spacecraft_bus, "inertia_tensor_body_kg_m2", None)
        if inertia is None:
            raise ValueError(
                "inertia_tensor_body_kg_m2 is required when stored-momentum "
                "tracking is enabled"
            )
        self._momentum_slew_sample_s = slew_sample_interval_s(self.config)
        return StoredMomentumTracker(
            inertia_tensor_body_kg_m2=inertia,
            initial_momentum_body_n_m_s=(momentum_config.initial_momentum_body_n_m_s),
            sources=build_torque_sources(self.config),
        )

    def _reset_stored_momentum_tracker(self) -> None:
        """Reload the run's inputs. Where the capacity is enforced, the
        executor's budget integrates into this tracker, so the ACS is given
        it before the budget exists, or the budget adopts it."""
        tracker = self._build_stored_momentum_tracker()
        self._stored_momentum_tracker = tracker
        self._momentum_previous: _MomentumInputs | None = None
        acs = getattr(self, "acs", None)
        if acs is None:
            return
        acs.momentum_tracker = tracker
        budget = getattr(acs, "momentum_budget", None)
        executor = getattr(acs, "_executor", None)
        if tracker is not None and isinstance(budget, MomentumBudget) and executor:
            budget.adopt(tracker, executor.state)

    def _update_stored_momentum(
        self,
        utime: float,
        position_eci_km: npt.NDArray[np.float64],
        attitude_quaternion_eci_to_body: npt.NDArray[np.float64],
        index: int | None = None,
    ) -> MomentumSample | None:
        """Advance the tracker to this step using values already computed for
        housekeeping; ``index`` is the ephemeris sample, for the Sun vector
        and eclipse.

        The step is the ephemeris resolution, which may be far coarser than
        a slew. Where the attitude moved during the step, the slew flown is
        sampled from the ACS at the configured attitude step, with the orbit
        interpolated between the step's ends, so the torque along the slew is
        integrated whatever the resolution. Where the capacity is enforced,
        the executor's budget integrates into the same tracker instead, at
        the motion's own boundaries, and its last sample is reported.
        """
        if (
            self.config.spacecraft_bus.attitude_control.stored_momentum.capacity_enabled
            is True
        ):
            budget = self.acs.momentum_budget
            if budget is None:
                raise ValueError("Momentum enforcement has not been initialized")
            return budget.current.sample
        tracker = self._stored_momentum_tracker
        if tracker is None:
            return None
        sun_eci_km: npt.NDArray[np.float64] | None = None
        position_ecef_km: npt.NDArray[np.float64] | None = None
        velocity_eci_km_s: npt.NDArray[np.float64] | None = None
        sunlit = True
        if index is not None:
            velocities = velocity_vectors(self.ephem)
            if velocities is not None and index < len(velocities):
                velocity_eci_km_s = np.asarray(velocities[index], dtype=np.float64)
            sun_positions = position_vectors(self.ephem, "sun")
            if index < len(sun_positions):
                sun_eci_km = (
                    np.asarray(sun_positions[index], dtype=np.float64) - position_eci_km
                )
            sunlit = not bool(getattr(self.acs, "in_eclipse", False))
            # A test double's ephemeris may have no Earth-fixed positions.
            ecef = getattr(getattr(self.ephem, "itrs_pv", None), "position", None)
            if ecef is not None and index < len(ecef):
                position_ecef_km = np.asarray(ecef[index], dtype=np.float64)
        inputs = _MomentumInputs(
            utime, position_eci_km, sun_eci_km, position_ecef_km, velocity_eci_km_s
        )
        previous = getattr(self, "_momentum_previous", None)
        self._momentum_previous = inputs
        elapsed = utime - previous.utime if previous is not None else 0.0
        if (
            previous is not None
            and elapsed > self._momentum_slew_sample_s
            and self.acs.attitude_moved(previous.utime, utime)
        ):
            count = math.ceil(elapsed / self._momentum_slew_sample_s)
            for k in range(1, count):
                fraction = k / count
                between = previous.between(inputs, fraction)
                ra, dec, roll = self.acs.executed_attitude(between.utime)
                tracker.update(
                    utime=between.utime,
                    position_eci_km=between.position_eci_km,
                    attitude_quaternion_eci_to_body=attitude_to_quat(ra, dec, roll),
                    sun_eci_km=between.sun_eci_km,
                    sunlit=sunlit,
                    velocity_eci_km_s=between.velocity_eci_km_s,
                    position_ecef_km=between.position_ecef_km,
                    angular_velocity_body_rad_s=self._body_rate_rad_s(between.utime),
                )
        return tracker.update(
            utime=utime,
            position_eci_km=position_eci_km,
            attitude_quaternion_eci_to_body=attitude_quaternion_eci_to_body,
            sun_eci_km=sun_eci_km,
            sunlit=sunlit,
            velocity_eci_km_s=velocity_eci_km_s,
            position_ecef_km=position_ecef_km,
            angular_velocity_body_rad_s=self._body_rate_rad_s(utime),
        )

    def _body_rate_rad_s(self, utime: float) -> npt.NDArray[np.float64]:
        """The body's angular velocity as executed at ``utime``, in radians
        per second, for the momentum it carries while it turns. At rest
        where the ACS gives none (a test double has no motion)."""
        executed = getattr(self.acs, "executed_angular_velocity", None)
        try:
            rate = np.asarray(
                executed(utime) if callable(executed) else None, dtype=np.float64
            )
        except (TypeError, ValueError):
            return np.zeros(3, dtype=np.float64)
        if rate.shape != (3,) or not np.all(np.isfinite(rate)):
            return np.zeros(3, dtype=np.float64)
        return np.deg2rad(rate)

    def _slew_momentum_enabled(self) -> bool:
        """Whether the momentum of slews counts against the capacity."""
        return bool(
            self.config.spacecraft_bus.attitude_control.stored_momentum.slew_momentum_enabled
        )

    def _momentum_demand_n_m_s(self, tracker: StoredMomentumTracker) -> float:
        """What is checked against the capacity: what the wheels hold where
        the momentum of slews counts, the stored momentum otherwise."""
        if self._slew_momentum_enabled():
            return tracker.actuator_momentum_norm_n_m_s
        return tracker.stored_momentum_norm_n_m_s

    def _dump_level_n_m_s(self) -> float | None:
        """The stored momentum at which a thruster dump is called for, if
        the configuration has thrusters and a capacity."""
        momentum_config = self.config.spacecraft_bus.attitude_control.stored_momentum
        return momentum_config.dump_level_n_m_s

    def _unload_stored_momentum(self, utime: float, obsid: int | None) -> None:
        """A thruster dump ends: take the stored momentum out of the wheels."""
        tracker = self._stored_momentum_tracker
        dump = self.config.spacecraft_bus.attitude_control.stored_momentum.thruster_dump
        if tracker is None or dump is None:
            return
        before = tracker.stored_momentum_norm_n_m_s
        tracker.unload(dump.residual_fraction)
        budget = getattr(self.acs, "momentum_budget", None)
        executor = getattr(self.acs, "_executor", None)
        if isinstance(budget, MomentumBudget) and executor is not None:
            # Enforcement forecasts from the emptied wheels from here on.
            budget.unloaded(executor.state)
        self.log.log_event(
            utime=utime,
            event_type="MOMENTUM",
            description=(
                f"Thruster dump unloads {before:.3f} N m s of stored momentum "
                f"to {tracker.stored_momentum_norm_n_m_s:.3f} N m s"
            ),
            obsid=obsid,
            acs_mode=self.acs.acsmode,
        )
        self.fault_management.clear_momentum_capacity(
            utime=utime, stored_n_m_s=tracker.stored_momentum_norm_n_m_s
        )

    def _momentum_at_capacity(self, utime: float, obsid: int | None) -> bool:
        """Whether what the wheels hold has reached their capacity: the
        stored momentum, plus the body's own along a slew where slew
        momentum counts. The first time, fault management records it, with
        the observation under way if any: an emergency dump interrupting an
        observation is what the planning level exists to prevent."""
        tracker = self._stored_momentum_tracker
        capacity = (
            self.config.spacecraft_bus.attitude_control.stored_momentum.capacity_n_m_s
        )
        if tracker is None or capacity is None:
            return False
        stored = self._momentum_demand_n_m_s(tracker)
        if stored < capacity:
            return False
        self.fault_management.report_momentum_capacity(
            utime=utime, stored_n_m_s=stored, capacity_n_m_s=capacity, obsid=obsid
        )
        return True

    def _momentum_ephem_utimes(self) -> npt.NDArray[np.float64]:
        """The ephemeris sample times in Unix seconds, cached per ephemeris."""
        timestamps = self.ephem.timestamp
        cached = getattr(self, "_momentum_utimes_cache", None)
        if cached is not None and cached[0] is timestamps:
            utimes: npt.NDArray[np.float64] = cached[1]
            return utimes
        utimes = np.array([t.timestamp() for t in timestamps], dtype=np.float64)
        self._momentum_utimes_cache = (timestamps, utimes)
        return utimes

    def _forecast_momentum(
        self,
        begin: float,
        end: float,
        attitude_at: Callable[[float], tuple[float, float, float]],
        *,
        start_eci_n_m_s: npt.ArrayLike | None,
        slew: Slew | None = None,
    ) -> MomentumProfile:
        """Forecast the stored momentum from ``begin`` to ``end`` at the
        attitudes ``attitude_at`` gives, from ``start_eci_n_m_s`` or empty
        wheels where None: the sample times, the norm at each, what the
        wheels hold at each, and the momentum at the end in inertial
        coordinates. Held attitudes are sampled at the run's step and
        ``slew`` at the configured attitude step, with the orbit interpolated
        between ephemeris samples, as the run integrates; where the momentum
        of slews counts, the body's rate along ``slew`` is what the wheels
        give up to it. Empty where momentum is not tracked."""
        empty = np.empty(0, dtype=np.float64)
        forecast = tracker_from_config(self.config)
        if forecast is None or end <= begin:
            return MomentumProfile(empty, empty, empty, None)
        if start_eci_n_m_s is not None:
            forecast.start_from_eci(start_eci_n_m_s)
        else:
            forecast.reset()
        samples = np.arange(begin, end, self.step_size)
        if len(samples) == 0 or samples[-1] < end:
            samples = np.append(samples, end)
        slew_sample = getattr(self, "_momentum_slew_sample_s", 0.0)
        rate_at: Callable[[float], npt.NDArray[np.float64]] | None = None
        if slew is not None and slew.slewtime > 0.0 and slew_sample > 0.0:
            along = np.arange(
                max(begin, float(slew.slewstart)),
                min(end, float(slew.slewend)),
                slew_sample,
            )
            samples = np.unique(np.concatenate([samples, along]))
            if self._slew_momentum_enabled():
                rate_at = slew_angular_velocity(slew)
        utimes = self._momentum_ephem_utimes()
        positions = np.asarray(position_vectors(self.ephem, "gcrs"), dtype=np.float64)
        suns = np.asarray(position_vectors(self.ephem, "sun"), dtype=np.float64)
        ecef = getattr(getattr(self.ephem, "itrs_pv", None), "position", None)
        ecefs = np.asarray(ecef, dtype=np.float64) if ecef is not None else None
        velocities = velocity_vectors(self.ephem)

        def interpolate(
            table: npt.NDArray[np.float64], at: float
        ) -> npt.NDArray[np.float64]:
            return np.array(
                [np.interp(at, utimes, table[:, k]) for k in range(3)], dtype=np.float64
            )

        norms: list[float] = []
        demands: list[float] = []
        for at in samples:
            position = interpolate(positions, float(at))
            # Eclipse is known at the ephemeris samples: the nearest one.
            nearest = int(np.clip(np.searchsorted(utimes, at), 0, len(utimes) - 1))
            if nearest > 0 and abs(utimes[nearest - 1] - at) < abs(
                utimes[nearest] - at
            ):
                nearest -= 1
            sample = forecast.update(
                utime=float(at),
                position_eci_km=position,
                attitude_quaternion_eci_to_body=attitude_to_quat(
                    *attitude_at(float(at))
                ),
                sun_eci_km=interpolate(suns, float(at)) - position,
                sunlit=not self.config.constraint.in_eclipse(
                    0.0, 0.0, float(utimes[nearest])
                ),
                velocity_eci_km_s=(
                    interpolate(velocities, float(at))
                    if velocities is not None
                    else None
                ),
                position_ecef_km=(
                    interpolate(ecefs, float(at)) if ecefs is not None else None
                ),
                angular_velocity_body_rad_s=(
                    rate_at(float(at)) if rate_at is not None else None
                ),
            )
            norms.append(sample.stored_momentum_norm_n_m_s)
            demands.append(
                sample.actuator_momentum_norm_n_m_s
                if sample.actuator_momentum_norm_n_m_s is not None
                else sample.stored_momentum_norm_n_m_s
            )
        return MomentumProfile(
            np.asarray(samples, dtype=np.float64),
            np.array(norms, dtype=np.float64),
            np.array(demands, dtype=np.float64),
            forecast.momentum_eci_n_m_s,
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

    def _add_activity_data(self, data_gb: float, entry: object | None = None) -> None:
        """Store an activity's fixed data now and report it on the next step."""
        if data_gb <= 0.0:
            return
        if isinstance(self.config.data_storage, DataStorage):
            for name, view in self.config.recorder_views().items():
                assert isinstance(view, RecorderView)
                amount = (
                    view.extra_data_gb(entry)
                    if entry is not None
                    else (
                        data_gb
                        if name == self.config.data_storage.bus_housekeeping
                        else 0.0
                    )
                )
                before = view.recorder.current_volume_gb
                view.recorder.add_data(amount)
                self._store_activity_data_gb[name] = (
                    self._store_activity_data_gb.get(name, 0.0) + amount
                )
                totals = self._store_data_totals.setdefault(
                    name, {"generated_gb": 0.0, "downlinked_gb": 0.0, "lost_gb": 0.0}
                )
                totals["lost_gb"] += max(
                    0.0, before + amount - view.recorder.current_volume_gb
                )
            return
        cast(OnboardRecorder, self.recorder).add_data(data_gb)
        self._activity_data_gb += data_gb

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
        if self.coverage is not None:
            self._log_repoints(
                utime, min(utime + step_size, self.uend), collection_seconds
            )
        if isinstance(self.config.data_storage, DataStorage):
            observation, collected = self._data_observation, self._data_collected
            generated = downlinked = 0.0
            for view in self.config.recorder_views().values():
                self._data_observation, self._data_collected = observation, collected
                volume_before = view.recorder.current_volume_gb
                activity_data = self._store_activity_data_gb.get(view.name, 0.0)
                incoming, outgoing = self._process_store_data(
                    view, utime, mode, step_size, collection_seconds=collection_seconds
                )
                totals = self._store_data_totals.setdefault(
                    view.name,
                    {"generated_gb": 0.0, "downlinked_gb": 0.0, "lost_gb": 0.0},
                )
                totals["generated_gb"] += incoming
                totals["downlinked_gb"] += outgoing
                totals["lost_gb"] += max(
                    0.0,
                    volume_before
                    + incoming
                    - activity_data
                    - outgoing
                    - view.recorder.current_volume_gb,
                )
                generated += incoming
                downlinked += outgoing
            return generated, downlinked
        return self._process_store_data(
            self.config, utime, mode, step_size, collection_seconds=collection_seconds
        )

    def _process_store_data(
        self,
        config: MissionConfig | RecorderView,
        utime: float,
        mode: ACSMode,
        step_size: int,
        *,
        collection_seconds: float,
    ) -> tuple[float, float]:
        recorder = (
            config.recorder
            if isinstance(config, RecorderView)
            else cast(OnboardRecorder, self.recorder)
        )
        data_downlinked = 0.0

        # Housekeeping data for the whole step, in its ACS mode.
        interval = max(0.0, min(utime + step_size, self.uend) - utime)
        data_generated = config.housekeeping_generated(interval, mode=mode)

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
            if getattr(self.ppt, "calibration", False) is not True:
                data_generated += config.science_generated(
                    after, science=science, instrument_modes=modes
                )
                if before > 0:
                    data_generated -= config.science_generated(
                        before, science=science, instrument_modes=modes
                    )
            if before <= 0:
                # Once per observation: its calibration steps' frames.
                data_generated += extra_data_gb(self.ppt, config)
            self._data_observation = observation
            self._data_collected = after

        current_pass = self.acs.current_pass
        contacts: list[tuple[float, float, float]] = []
        contact_start = contact_stop = utime
        data_to_downlink = 0.0
        if self.coverage is not None:
            # Passes without spacecraft tracking can downlink in any ACS
            # mode. A gimballed antenna
            # reaches the station only within its range from the attitude
            # flown, taken as the attitude at the start of the step.
            attitude = (
                (self.acs.ra, self.acs.dec, self.acs.roll)
                if self.coverage.gimballed
                else None
            )
            spans = self.coverage.rate_spans(
                utime,
                utime + interval,
                config if isinstance(config, RecorderView) else None,
            )
            contacts = [
                (
                    span.begin,
                    span.end,
                    span.rate_mbps * (span.end - span.begin) / 1000.0,
                )
                for span in spans
                if attitude is None
                or self.coverage.contact_at(span.begin, attitude) is not None
            ]
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
            if isinstance(config, RecorderView):
                data_to_downlink = contact_downlink_gb(
                    config, current_pass.station, contact_stop - contact_start
                )
            else:
                rate_mbps = self._get_effective_data_rate(
                    config.ground_stations.get(current_pass.station)
                )
                if rate_mbps is not None and rate_mbps > 0:
                    data_to_downlink = (
                        rate_mbps * (contact_stop - contact_start) / 1000.0
                    )

            if data_to_downlink > 0:
                contacts.append((contact_start, contact_stop, data_to_downlink))

        # Activities have already stored their frames; report them only once.
        if isinstance(config, RecorderView):
            activity_data = self._store_activity_data_gb.pop(config.name, 0.0)
        else:
            activity_data = self._activity_data_gb
            self._activity_data_gb = 0.0

        if not contacts or interval <= 0:
            if data_generated > 0:
                recorder.add_data(data_generated)
            return data_generated + activity_data, data_downlinked

        # Walk each usable span and gap in time order: outages can overflow
        # the recorder even when a later link can send everything it retains.
        rate = data_generated / interval
        cursor = utime
        for contact_start, contact_stop, data_to_downlink in contacts:
            recorder.add_data(rate * (contact_start - cursor))
            data_downlinked += recorder.transfer(
                rate * (contact_stop - contact_start), data_to_downlink
            )
            cursor = contact_stop
        recorder.add_data(rate * (utime + interval - cursor))
        return data_generated + activity_data, data_downlinked

    def _recorder_telemetry(self) -> dict[str, dict[str, float]]:
        if not isinstance(self.config.data_storage, DataStorage):
            return {}
        return {
            name: {
                "volume_gb": view.recorder.current_volume_gb,
                "capacity_gb": view.recorder.capacity_gb,
                "fill_fraction": view.recorder.get_fill_fraction(),
                "alert": float(view.recorder.get_alert_level()),
                **self._store_data_totals.get(
                    name, {"generated_gb": 0.0, "downlinked_gb": 0.0, "lost_gb": 0.0}
                ),
            }
            for name, view in self.config.recorder_views().items()
        }

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
