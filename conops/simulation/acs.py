# ruff: noqa: N806
import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import rust_ephem

from ..common import (
    ACSCommandType,
    ACSMode,
    dtutcfromtimestamp,
    unixtime2date,
    unixtime2yearday,
)
from ..config import MissionConfig
from ..simulation.passes import PassTimes
from .acs_command import ACSCommand
from .command_queue import CommandQueue
from .disturbance import DisturbanceConfig, DisturbanceModel
from .emergency_charging import EmergencyCharging
from .passes import Pass
from .reaction_wheel import ReactionWheel
from .slew import Slew
from .telemetry import WheelReading, WheelTelemetrySnapshot
from .wheel_dynamics import WheelDynamics

_logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from ..ditl.ditl_log import DITLLog
    from ..targets import Pointing


class ACS:
    """
    Queue-driven state machine for spacecraft Attitude Control System (ACS).

    The ACS manages spacecraft pointing through a command queue, where each command
    represents a state transition (slew, pass, return to pointing, etc.). The state
    machine processes commands at their scheduled execution times and maintains
    current pointing state.
    """

    ephem: rust_ephem.Ephemeris
    slew_dists: list[float]
    ra: float
    dec: float
    roll: float
    obstype: str
    acsmode: ACSMode
    _commands: CommandQueue
    current_slew: Slew | None
    last_ppt: Slew | None
    last_slew: Slew | None
    in_eclipse: bool

    def __init__(self, config: MissionConfig, log: "DITLLog | None" = None) -> None:
        """Initialize the Attitude Control System.

        Args:
            constraint: Constraint object with ephemeris.
            config: MissionConfiguration object.
            log: Optional DITLLog for event logging. If None, prints to stdout.
        """
        assert config.constraint is not None, "Constraint must be provided to ACS"
        self.constraint = config.constraint
        self.config = config
        self.log = log

        # Configuration
        assert self.constraint.ephem is not None, "Ephemeris must be set in Constraint"
        self.ephem = self.constraint.ephem

        # Initial pointing derived from ephemeris (opposite Earth vector)
        self.ra = (180 + self.ephem.earth_ra_deg[0]) % 360
        self.dec = -self.ephem.earth_dec_deg[0]
        # Set up initial last_slew pointing (this would have been the
        # last slew to execute before the current DITL), so didn't
        # happen in our simulation, but defines a realistic boundary
        # condition for our simulation.
        self.last_slew = Slew(
            config=config,
        )
        self.last_slew.endra = self.ra
        self.last_slew.enddec = self.dec

        # Current state
        self.roll = 0.0
        self.obstype = "PPT"
        self.acsmode = ACSMode.SCIENCE  # Start in science/pointing mode
        self.in_eclipse = False  # Initialize eclipse state
        self.in_safe_mode = False  # Safe mode flag - once True, cannot be exited

        # Command queue (sorted by execution_time)
        self._commands = CommandQueue()

        # Current and historical state
        self.current_slew = None
        self.last_ppt = None

        self.passrequests = PassTimes(config=config)
        self.current_pass: Pass | None = None
        self.solar_panel = config.solar_panel
        self.slew_dists: list[float] = []
        self.saa = None
        # Reaction wheel instances (optional)
        acs_cfg = self.config.spacecraft_bus.attitude_control
        # store ACS config for use by Slew and other helpers
        self.acs_config = acs_cfg
        self.reaction_wheels = []
        self.magnetorquers = []
        self._last_wheel_snapshot: WheelTelemetrySnapshot | None = None
        self._last_pointing_time: float | None = None
        # Momentum management stats
        self.headroom_rejects = 0
        self.desat_requests = 0
        self.desat_events = 0
        self._desat_active = False
        self._desat_end = 0.0
        self._desat_use_mtq = False
        self._mtq_bleed_in_science = False
        self._last_desat_request: float = 0.0
        self._last_desat_end: float = 0.0
        self.mtq_power_w: float = 0.0
        # Allow some configurable buffer below absolute wheel momentum capacity for allocation
        # Keep some buffer below absolute capacity so we avoid hard saturation;
        # adjust in tests/notebooks if needed for what-if sweeps.
        self._wheel_mom_margin = 0.9
        # Only enable legacy single-wheel when wheel_enabled is explicitly True
        if getattr(acs_cfg, "wheel_enabled", False) is True:
            # legacy single-wheel support
            self.reaction_wheels.append(
                ReactionWheel(
                    max_torque=acs_cfg.wheel_max_torque,
                    max_momentum=acs_cfg.wheel_max_momentum,
                    orientation=(1.0, 0.0, 0.0),
                    name="wheel0",
                )
            )
        # Multi-wheel config - be defensive: tests may supply a Mock for acs_cfg.wheels
        wheels_val = getattr(acs_cfg, "wheels", None)
        wheels_iter: list[Any] = []
        if wheels_val:
            try:
                wheels_iter = list(wheels_val)
            except TypeError:
                wheels_iter = []

        for i, w in enumerate(wheels_iter):
            # Support both WheelSpec objects and dict[str, Any]
            if hasattr(w, "orientation"):
                # WheelSpec or similar object
                orient = tuple(getattr(w, "orientation", (1.0, 0.0, 0.0)))
                mt = float(getattr(w, "max_torque", acs_cfg.wheel_max_torque or 0.0))
                mm = float(
                    getattr(w, "max_momentum", acs_cfg.wheel_max_momentum or 0.0)
                )
                name = getattr(w, "name", "") or f"wheel{i}"
                idle_power = float(getattr(w, "idle_power_w", 5.0))
                torque_coeff = float(getattr(w, "torque_power_coeff", 50.0))
            else:
                # Dict-style config (legacy)
                orient = tuple(w.get("orientation", (1.0, 0.0, 0.0)))
                mt = float(w.get("max_torque", acs_cfg.wheel_max_torque or 0.0))
                mm = float(w.get("max_momentum", acs_cfg.wheel_max_momentum or 0.0))
                name = w.get("name", "") or f"wheel{i}"
                idle_power = float(w.get("idle_power_w", 5.0))
                torque_coeff = float(w.get("torque_power_coeff", 50.0))
            self.reaction_wheels.append(
                ReactionWheel(
                    max_torque=mt,
                    max_momentum=mm,
                    orientation=orient,
                    name=name,
                    idle_power_w=idle_power,
                    torque_power_coeff=torque_coeff,
                )
            )

        # Validate wheel configuration for controllability
        self._validate_wheel_configuration()

        # Magnetorquers (optional) for finite desat
        mtq_iter_raw = getattr(acs_cfg, "magnetorquers", None)
        mtq_iter: list[Any] = []
        if mtq_iter_raw:
            try:
                mtq_iter = list(mtq_iter_raw)
            except TypeError:
                mtq_iter = []
        for i, m in enumerate(mtq_iter):
            # Support both MagnetorquerSpec objects and dict[str, Any]
            if hasattr(m, "orientation") and hasattr(m, "dipole_strength"):
                # MagnetorquerSpec or similar object
                orient = tuple(getattr(m, "orientation", (1.0, 0.0, 0.0)))
                dipole = float(getattr(m, "dipole_strength", 0.0))
                power = float(getattr(m, "power_draw", 0.0))
                name = getattr(m, "name", "") or f"mtq{i}"
            else:
                # Dict-style config (legacy)
                try:
                    orient = tuple(m.get("orientation", (1.0, 0.0, 0.0)))
                except (TypeError, AttributeError):
                    orient = (1.0, 0.0, 0.0)
                try:
                    dipole = float(m.get("dipole_strength", 0.0))
                except (TypeError, ValueError):
                    dipole = 0.0
                try:
                    power = float(m.get("power_draw", 0.0))
                except (TypeError, ValueError):
                    power = 0.0
                name = m.get("name", "") if hasattr(m, "get") else ""
                name = name or f"mtq{i}"
            self.magnetorquers.append(
                {
                    "orientation": orient,
                    "dipole": dipole,
                    "power_draw": power,
                    "name": name,
                }
            )
        self._mtq_bleed_in_science = bool(
            getattr(acs_cfg, "mtq_bleed_in_science", False)
        )
        # Create disturbance model from ACS config
        self.disturbance_model = DisturbanceModel(
            self.ephem,
            DisturbanceConfig.from_acs_config(acs_cfg),
        )

        # module logger for optional debug tracing
        self._logger = logging.getLogger(__name__)
        self._last_disturbance_components: dict[str, float | list[float]] = {
            "total": 0.0,
            "gg": 0.0,
            "drag": 0.0,
            "srp": 0.0,
            "mag": 0.0,
        }

        # Build inertia matrix for WheelDynamics
        moi_cfg = acs_cfg.spacecraft_moi
        inertia_matrix = DisturbanceModel._build_inertia(moi_cfg)

        # Create WheelDynamics subsystem for momentum/torque management
        self.wheel_dynamics = WheelDynamics(
            wheels=self.reaction_wheels,
            inertia_matrix=inertia_matrix,
            magnetorquers=self.magnetorquers,
            disturbance_model=self.disturbance_model,
            momentum_margin=self._wheel_mom_margin,
            budget_margin=0.85,
            conservation_tolerance=0.1,
        )

        # Slew state tracking (still needed for operational logic)
        self._was_slewing = False
        self._last_slew_for_verification: Slew | None = None

        # Momentum warnings (operational tracking, not physics)
        self._momentum_warnings: list[str] = []

    def _log_or_print(self, utime: float, event_type: str, description: str) -> None:
        """Log an event to DITLLog if available, otherwise print to stdout.

        Args:
            utime: Unix timestamp.
            event_type: Event category (ACS, SLEW, PASS, etc.).
            description: Human-readable description.
        """
        if self.log is not None:
            self.log.log_event(
                utime=utime,
                event_type=event_type,
                description=description,
                obsid=getattr(self.last_slew, "obsid", None)
                if self.last_slew
                else None,
                acs_mode=self.acsmode if hasattr(self, "acsmode") else None,
            )
        else:
            # Fallback to print if no log available
            print(description)

    # -------------------------------------------------------------------------
    # Backward Compatibility Properties (delegate to WheelDynamics)
    # -------------------------------------------------------------------------

    @property
    def spacecraft_body_momentum(self) -> np.ndarray:
        """Spacecraft body momentum (delegates to WheelDynamics)."""
        return self.wheel_dynamics.body_momentum

    @spacecraft_body_momentum.setter
    def spacecraft_body_momentum(self, value: np.ndarray) -> None:
        """Set spacecraft body momentum (delegates to WheelDynamics)."""
        self.wheel_dynamics.body_momentum = np.array(value, dtype=float)

    # -------------------------------------------------------------------------
    # Backward Compatibility Properties (delegate to CommandQueue)
    # -------------------------------------------------------------------------

    @property
    def command_queue(self) -> list[ACSCommand]:
        """Pending commands (delegates to CommandQueue)."""
        return self._commands.pending

    @property
    def executed_commands(self) -> list[ACSCommand]:
        """Executed commands (delegates to CommandQueue)."""
        return self._commands.executed

    # -------------------------------------------------------------------------
    # Momentum Bookkeeping Methods (delegate to WheelDynamics)
    # -------------------------------------------------------------------------

    def _get_total_wheel_momentum(self) -> np.ndarray:
        """Compute total wheel angular momentum vector in body frame."""
        return self.wheel_dynamics.get_total_wheel_momentum()

    def _get_wheel_headroom_along_axis(self, axis: np.ndarray) -> float:
        """Compute available momentum headroom projected onto an axis."""
        return self.wheel_dynamics.get_headroom_along_axis(axis)

    def _get_total_system_momentum(self) -> np.ndarray:
        """Compute total angular momentum of the system (wheels + body)."""
        return self.wheel_dynamics.get_total_system_momentum()

    def _apply_wheel_torques_conserving(
        self, taus_allowed: np.ndarray, dt: float
    ) -> np.ndarray:
        """Apply wheel torques while conserving total angular momentum."""
        return self.wheel_dynamics.apply_wheel_torques(taus_allowed, dt)

    def _apply_external_torque(
        self, external_torque: np.ndarray, dt: float, source: str = "external"
    ) -> None:
        """Apply external torque to the system (changes total momentum)."""
        self.wheel_dynamics.apply_external_torque(external_torque, dt, source)

    def _check_momentum_conservation(self, utime: float) -> bool:
        """Verify momentum conservation."""
        return self.wheel_dynamics.check_conservation(utime)

    def get_body_momentum_magnitude(self) -> float:
        """Return magnitude of spacecraft body momentum."""
        return self.wheel_dynamics.get_body_momentum_magnitude()

    def _compute_slew_peak_momentum(self, slew: "Slew") -> tuple[float, np.ndarray]:
        """Compute peak angular momentum required for a slew.

        For a bang-bang slew profile, peak angular velocity occurs at the midpoint.
        Delegates to WheelDynamics for the physics computation.

        Args:
            slew: The slew object with geometry and profile parameters.

        Returns:
            Tuple of (peak_momentum_magnitude, rotation_axis).
            Peak momentum is in N·m·s, axis is a unit vector.
        """
        # Get rotation axis
        axis = np.array(getattr(slew, "rotation_axis", (0.0, 0.0, 1.0)), dtype=float)
        nrm = np.linalg.norm(axis)
        if nrm <= 0:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis / nrm

        # Get slew parameters
        angle_deg = float(getattr(slew, "slewdist", 0.0))
        if angle_deg <= 0:
            return 0.0, axis

        # Get acceleration (use override if available)
        accel_override = getattr(slew, "_accel_override", None)
        if accel_override is not None:
            accel_deg = float(accel_override)
        else:
            accel_deg = float(getattr(self.acs_config, "slew_acceleration", 0.5))
        if accel_deg <= 0:
            return 0.0, axis

        # Delegate to WheelDynamics
        h_peak = self.wheel_dynamics.compute_slew_peak_momentum(
            angle_deg, axis, accel_deg
        )
        return h_peak, axis

    def _check_slew_momentum_budget(
        self, slew: "Slew", utime: float
    ) -> tuple[bool, str]:
        """Check if wheels have sufficient momentum headroom for a slew.

        This is a pre-slew feasibility check that verifies the wheels can
        accommodate the peak momentum required during the slew, including
        a safety margin and estimated disturbance accumulation.

        Args:
            slew: The slew to check.
            utime: Current time (for disturbance estimation).

        Returns:
            Tuple of (feasible, message). If not feasible, message explains why.
        """
        if not self.reaction_wheels:
            return True, "No wheels configured"

        # Get slew parameters
        axis = np.array(getattr(slew, "rotation_axis", (0.0, 0.0, 1.0)), dtype=float)
        nrm = np.linalg.norm(axis)
        if nrm > 0:
            axis = axis / nrm

        angle_deg = float(getattr(slew, "slewdist", 0.0))

        accel_override = getattr(slew, "_accel_override", None)
        if accel_override is not None:
            accel_deg = float(accel_override)
        else:
            accel_deg = float(getattr(self.acs_config, "slew_acceleration", 0.5))

        slew_time = float(getattr(slew, "slewtime", 0.0))
        if slew_time <= 0:
            slew_time = 60.0  # Default estimate

        # Get disturbance torque magnitude (rough estimate)
        try:
            dist_torque = self._compute_disturbance_torque(utime)
            dist_mag = float(np.linalg.norm(dist_torque))
        except Exception:
            dist_mag = 1e-6  # Default small disturbance

        # Delegate to WheelDynamics
        return self.wheel_dynamics.check_slew_momentum_budget(
            angle_deg, axis, accel_deg, slew_time, dist_mag
        )

    def _record_slew_start_momentum(self, slew: "Slew") -> None:
        """Record wheel momentum state at the start of a slew for later validation."""
        # Delegate to WheelDynamics for momentum tracking
        self.wheel_dynamics.record_slew_start()

    def _verify_slew_end_momentum(self, slew: "Slew", utime: float) -> None:
        """Verify momentum consistency at the end of a slew.

        Compares actual wheel momentum change against expected and logs
        warnings if they diverge significantly.
        """
        # Get slew start momentum from WheelDynamics
        h_start = self.wheel_dynamics._slew_momentum_at_start
        if h_start is None:
            return

        h_end = self.wheel_dynamics.get_total_wheel_momentum()
        delta_h = h_end - h_start

        # For an ideal slew with perfect disturbance rejection,
        # net momentum change should be small (only disturbance accumulation)
        slew_time = float(getattr(slew, "slewtime", 60.0))

        # Expected disturbance accumulation
        try:
            dist_torque = self._compute_disturbance_torque(utime)
            expected_dist_h = float(np.linalg.norm(dist_torque)) * slew_time
        except Exception:
            expected_dist_h = 0.0

        # Actual momentum change magnitude
        actual_delta_h = float(np.linalg.norm(delta_h))

        # Check if momentum change is reasonable
        # Allow for disturbance accumulation plus some tolerance
        tolerance = max(expected_dist_h * 2, 0.01)  # At least 0.01 N·m·s tolerance

        if actual_delta_h > tolerance:
            # Get peak momentum for context
            h_peak, _ = self._compute_slew_peak_momentum(slew)
            warning = (
                f"Momentum consistency warning at t={utime:.1f}: "
                f"ΔH_wheels={actual_delta_h:.4f} N·m·s (expected ~{expected_dist_h:.4f}), "
                f"slew peak was {h_peak:.4f} N·m·s"
            )
            self._momentum_warnings.append(warning)
            self._logger.warning(warning)

        # Clear slew tracking state in WheelDynamics
        self.wheel_dynamics._slew_momentum_at_start = None

    def get_momentum_warnings(self) -> list[str]:
        """Return list of momentum consistency warnings accumulated during simulation."""
        return self._momentum_warnings.copy()

    def clear_momentum_warnings(self) -> None:
        """Clear accumulated momentum warnings."""
        self._momentum_warnings.clear()

    def get_momentum_summary(self) -> dict[str, Any]:
        """Return a summary of current momentum state for diagnostics.

        Returns:
            Dictionary with wheel momentum vector, per-wheel details, and headroom.
        """
        # Get physics summary from WheelDynamics
        summary = self.wheel_dynamics.get_momentum_summary()
        # Add ACS-specific operational data
        summary["num_warnings"] = len(self._momentum_warnings)
        return summary

    # -------------------------------------------------------------------------
    # End Momentum Bookkeeping Methods
    # -------------------------------------------------------------------------

    def _validate_wheel_configuration(self) -> None:
        """Validate that wheel orientations form a controllable configuration.

        Checks:
        - At least 3 wheels needed for full 3-axis control
        - Wheel orientation matrix should have rank 3 (or close to it)
        - Warns if configuration is singular or under-actuated
        """
        if not self.reaction_wheels:
            return  # No wheels configured, nothing to validate

        n_wheels = len(self.reaction_wheels)

        # Build wheel orientation matrix E (3 x N)
        orientations = []
        for w in self.reaction_wheels:
            v = np.array(w.orientation, dtype=float)
            norm = np.linalg.norm(v)
            if norm > 0:
                v = v / norm
            else:
                v = np.array([1.0, 0.0, 0.0])
            orientations.append(v)

        if not orientations:
            return

        E = np.column_stack(orientations)

        # Check matrix rank
        try:
            rank = np.linalg.matrix_rank(E, tol=1e-6)
        except Exception:
            rank = 0

        # Store controllability info for diagnostics
        self._wheel_config_rank = rank
        self._wheel_config_n_wheels = n_wheels

        # Check for strict validation mode
        strict = getattr(
            self.config.spacecraft_bus.attitude_control,
            "strict_wheel_validation",
            False,
        )

        # Log warnings or raise errors for problematic configurations
        if n_wheels < 3:
            msg = (
                f"ACS: Only {n_wheels} reaction wheel(s) configured; full 3-axis "
                "control requires at least 3 wheels with linearly independent axes."
            )
            if strict and n_wheels > 0:
                raise ValueError(msg)
            _logger.warning(msg)
        elif rank < 3:
            msg = (
                f"ACS: Wheel orientation matrix has rank {rank} < 3; spacecraft "
                "does not have full 3-axis controllability. Check wheel orientations."
            )
            if strict:
                raise ValueError(msg)
            _logger.warning(msg)

        # Check for nearly-singular configuration (condition number)
        if rank >= min(3, n_wheels):
            try:
                # Use SVD to get condition number
                _, s, _ = np.linalg.svd(E)
                if len(s) >= 2 and s[-1] > 0:
                    cond = s[0] / s[-1]
                    if cond > 100:
                        _logger.warning(
                            "ACS: Wheel configuration is poorly conditioned "
                            "(condition number %.1f). Torque allocation may be "
                            "numerically unstable.",
                            cond,
                        )
                    self._wheel_config_condition = cond
            except Exception:
                pass

    def enqueue_command(self, command: ACSCommand) -> None:
        """Add a command to the queue, maintaining time-sorted order.

        Commands cannot be enqueued if safe mode has been entered, except
        for SAFE slews which are part of safe mode entry.
        """
        # Allow SAFE slews to be enqueued even in safe mode (part of safe mode entry)
        is_safe_slew = (
            command.command_type == ACSCommandType.SLEW_TO_TARGET
            and hasattr(command, "slew")
            and command.slew is not None
            and command.slew.obstype == "SAFE"
        )

        # Prevent any commands from being enqueued in safe mode (except SAFE slews)
        if self.in_safe_mode and not is_safe_slew:
            self._log_or_print(
                command.execution_time,
                "ACS",
                f"{unixtime2date(command.execution_time)}: Command {command.command_type.name} rejected - spacecraft is in SAFE MODE",
            )
            return

        self._commands.enqueue(command)
        self._log_or_print(
            command.execution_time,
            "ACS",
            f"{unixtime2date(command.execution_time)}: Enqueued {command.command_type.name} command for execution  (queue size: {len(self._commands)})",
        )

    def _process_commands(self, utime: float) -> None:
        """Process all commands scheduled for execution at or before current time."""
        while (command := self._commands.pop_due(utime)) is not None:
            self._log_or_print(
                utime,
                "ACS",
                f"{unixtime2date(utime)}: Executing {command.command_type.name} command.",
            )

            # Dispatch to appropriate handler based on command type
            handlers: dict[ACSCommandType, Any] = {
                ACSCommandType.SLEW_TO_TARGET: lambda: self._handle_slew_command(
                    command, utime
                ),
                ACSCommandType.START_PASS: lambda: self._start_pass(command, utime),
                ACSCommandType.END_PASS: lambda: self._end_pass(utime),
                ACSCommandType.START_BATTERY_CHARGE: lambda: self._start_battery_charge(
                    command, utime
                ),
                ACSCommandType.END_BATTERY_CHARGE: lambda: self._end_battery_charge(
                    utime
                ),
                ACSCommandType.ENTER_SAFE_MODE: lambda: self._handle_safe_mode_command(
                    utime
                ),
                ACSCommandType.DESAT: lambda: self._start_desat(command, utime),
            }

            handler = handlers.get(command.command_type)
            if handler:
                handler()
            self._commands.mark_executed(command)

    def _handle_slew_command(self, command: ACSCommand, utime: float) -> None:
        """Handle SLEW_TO_TARGET command."""
        if command.slew is not None:
            ok = self._start_slew(command.slew, utime)
            if not ok:
                try:
                    command.slew._rejected = True  # type: ignore[attr-defined]
                    command.slew.slewtime = 0.0
                except Exception:
                    pass
                self._log_or_print(
                    utime,
                    "SLEW",
                    f"{unixtime2date(utime)}: Slew rejected at execution",
                )

    # Handle Ground Station Pass Commands
    def _start_pass(self, command: ACSCommand, utime: float) -> None:
        """Handle START_PASS command to command the start of a groundstation pass."""
        # Fetch the current pass from pass requests
        self.current_pass = self.passrequests.current_pass(utime)
        if self.current_pass is None:
            self._log_or_print(
                utime, "PASS", f"{unixtime2date(utime)}: No active pass found to start."
            )
            return
        self.acsmode = ACSMode.PASS
        self._log_or_print(
            utime,
            "PASS",
            f"{unixtime2date(utime)}: Starting pass over groundstation {self.current_pass.station}.",
        )

    def _end_pass(self, utime: float) -> None:
        """Handle the END_PASS command to command the end of a groundstation pass."""
        self.current_pass = None
        self.acsmode = ACSMode.SCIENCE

        # Clear any stale slew that was issued during the pass. Such slews have
        # start positions from pass tracking that don't reflect where the
        # spacecraft actually is now. Keeping them would cause teleportation.
        if self.last_slew is not None and self.last_slew.slewstart < utime:
            self._log_or_print(
                utime,
                "PASS",
                f"{unixtime2date(utime)}: Clearing stale slew (started {self.last_slew.slewstart:.0f})",
            )
            self.last_slew = None

        self._log_or_print(
            utime,
            "PASS",
            f"{unixtime2date(utime)}: Pass over - returning to last PPT {getattr(self.last_ppt, 'obsid', 'unknown')}",
        )

    # Handle Safe Mode Command
    def _handle_safe_mode_command(self, utime: float) -> None:
        """Handle ENTER_SAFE_MODE command.

        Once safe mode is entered, it cannot be exited. The spacecraft will
        point solar panels at the Sun and obey bus-level constraints.
        """
        self._log_or_print(
            utime, "SAFE", f"{unixtime2date(utime)}: Entering SAFE MODE - irreversible"
        )
        self.in_safe_mode = True
        # Clear command queue to prevent any future commands from executing
        self._commands.clear()
        self._log_or_print(
            utime,
            "SAFE",
            f"{unixtime2date(utime)}: Command queue cleared - no further commands will be executed",
        )

        # Initiate slew to Sun pointing for safe mode
        # Use solar panel optimal pointing to maximize power generation
        if self.solar_panel is not None:
            safe_ra, safe_dec = self.solar_panel.optimal_charging_pointing(
                utime, self.ephem
            )
        else:
            # Fallback: point directly at Sun if no solar panel config
            index = self.ephem.index(dtutcfromtimestamp(utime))
            safe_ra = self.ephem.sun_ra_deg[index]
            safe_dec = self.ephem.sun_dec_deg[index]

        self._log_or_print(
            utime,
            "SAFE",
            f"{unixtime2date(utime)}: Initiating slew to safe mode pointing at RA={safe_ra:.2f} Dec={safe_dec:.2f}",
        )
        # Enqueue slew to safe pointing with a special obsid
        self._enqueue_slew(safe_ra, safe_dec, obsid=-999, utime=utime, obstype="SAFE")

    def _start_desat(self, command: ACSCommand, utime: float) -> None:
        """Start a reaction wheel desaturation/unloading period."""
        duration = float(command.duration or 300.0)
        # Do not start desat during SCIENCE dwell; defer until later
        try:
            current_mode = self.get_mode(utime)
        except Exception:
            current_mode = self.acsmode
        if current_mode == ACSMode.SCIENCE and not self._is_actively_slewing(utime):
            delay = getattr(self.ephem, "step_size", 1) or 1
            command.execution_time = utime + delay
            self._log_or_print(
                utime,
                "DESAT",
                f"{unixtime2date(utime)}: DESAT deferred in SCIENCE mode, rescheduled to {unixtime2date(command.execution_time)}",
            )
            self.enqueue_command(command)
            return

        self._desat_active = True
        self._desat_end = utime + duration
        self._desat_use_mtq = bool(self.magnetorquers)
        if not self._desat_use_mtq:
            # legacy instant unload
            for w in self.reaction_wheels:
                try:
                    w.current_momentum = 0.0
                except Exception:
                    continue
            self._log_or_print(
                utime,
                "DESAT",
                f"{unixtime2date(utime)}: Starting desat for {duration:.0f}s (wheels reset)",
            )
        else:
            self._log_or_print(
                utime,
                "DESAT",
                f"{unixtime2date(utime)}: Starting desat for {duration:.0f}s using magnetorquers",
            )
        self.desat_events += 1
        # Drop any other queued DESAT commands now that one is active
        self._commands.remove_pending_type(ACSCommandType.DESAT)

    def _start_slew(self, slew: Slew, utime: float) -> bool:
        """Start executing a slew.

        The ACS always drives the spacecraft from its current position. When a slew
        command is executed, we set the start position to the current ACS pointing
        and recalculate the slew profile. This ensures continuous motion without
        teleportation, regardless of when commands were originally scheduled.
        """
        # Always start slew from current spacecraft position - ACS drives the spacecraft
        slew.startra = self.ra
        slew.startdec = self.dec
        slew.slewstart = utime
        # Refresh slew geometry now that start point is updated.
        slew.predict_slew()
        # If reaction wheel present, compute an accel override based on torque limits
        accel_override = None
        vmax_override = None
        if self.reaction_wheels:
            # Use aggregate wheel capability along slew rotation axis
            axis = np.array(
                getattr(slew, "rotation_axis", (0.0, 0.0, 1.0)), dtype=float
            )
            # motion time estimate using current accel config
            try:
                motion_time_est = float(self.acs_config.motion_time(slew.slewdist))
            except Exception:
                motion_time_est = 0.0
            achievable_accel = self.wheel_dynamics.get_axis_accel_limit(
                axis, motion_time_est
            )
            if achievable_accel > 0:
                # Derive accel directly from wheel capability (not a fixed external parameter)
                accel_override = achievable_accel
            # Derive a max rate limit from wheel momentum capacity along axis
            vmax_limit = self.wheel_dynamics.get_axis_rate_limit(axis)
            if vmax_limit > 0:
                vmax_override = vmax_limit
            # If we're effectively unable to slew, reject at execution time.
            min_accel = 1e-4  # deg/s^2
            min_rate = 1e-3  # deg/s
            if slew.obstype != "SAFE":
                if achievable_accel <= min_accel or vmax_limit <= min_rate:
                    self._log_or_print(
                        utime,
                        "SLEW",
                        f"{unixtime2date(utime)}: Slew rejected at execution - "
                        f"accel/rate too low (accel={achievable_accel:.3e}, rate={vmax_limit:.3e})",
                    )
                    self.request_desat(utime)
                    return False
        # apply overrides to slew and calc time
        slew._accel_override = accel_override
        slew._vmax_override = vmax_override
        slew.calc_slewtime()

        # Pre-slew momentum budget check (for non-SAFE slews with wheels)
        if self.reaction_wheels and slew.obstype != "SAFE":
            budget_ok, budget_msg = self._check_slew_momentum_budget(slew, utime)
            if not budget_ok:
                self._log_or_print(
                    utime,
                    "SLEW",
                    f"{unixtime2date(utime)}: Slew rejected - {budget_msg}",
                )
                self._momentum_warnings.append(
                    f"Slew rejected at t={utime:.1f}: {budget_msg}"
                )
                self.request_desat(utime)
                return False
            else:
                self._logger.debug("Slew momentum budget: %s", budget_msg)

        self._log_or_print(
            utime,
            "SLEW",
            f"{unixtime2date(utime)}: Starting slew from RA={self.ra:.2f} Dec={self.dec:.2f} "
            f"to RA={slew.endra:.2f} Dec={slew.enddec:.2f} (duration: {slew.slewtime:.1f}s)",
        )

        self.current_slew = slew
        self.last_slew = slew

        # Reset clamp flag for this slew
        self._slew_clamped = False

        # Record momentum state at slew start for bookkeeping
        if self.reaction_wheels:
            self._record_slew_start_momentum(slew)
            self._last_slew_for_verification = slew

        # Update last_ppt if this is a science pointing
        if self._is_science_pointing(slew):
            self.last_ppt = slew
        return True

    def _is_science_pointing(self, slew: Slew) -> bool:
        """Check if slew represents a science pointing (not a pass)."""
        return slew.obstype == "PPT" and isinstance(slew, Slew)

    def _enqueue_slew(
        self, ra: float, dec: float, obsid: int, utime: float, obstype: str = "PPT"
    ) -> bool:
        """Create and enqueue a slew command.

        This is a private helper method used internally by ACS for creating slew
        commands during battery charging operations.
        """
        # Do not accept new slews while a desat is active or scheduled
        if self._desat_active or self._commands.has_pending_type(ACSCommandType.DESAT):
            self._log_or_print(
                utime,
                "SLEW",
                f"{unixtime2date(utime)}: Slew rejected - desat in progress/queued",
            )
            return False

        # Create slew object
        slew = Slew(
            config=self.config,
        )
        slew.ephem = self.ephem
        slew.slewrequest = utime
        slew.endra = ra
        slew.enddec = dec
        slew.obstype = obstype
        slew.obsid = obsid

        # For SAFE mode, skip visibility checking (emergency situation)
        if obstype == "SAFE":
            # Initialize slew positions without target
            is_first_slew = self._initialize_slew_positions(slew, utime)
            slew.at = None  # No visibility constraint in safe mode
            self._compute_slew_distance(slew)
            execution_time = utime  # Execute immediately
        else:
            # Set up target observation request and check visibility
            target_request = self._create_target_request(slew, utime)
            slew.at = target_request

            visstart = target_request.next_vis(utime)
            is_first_slew = self._initialize_slew_positions(slew, utime)
            self._compute_slew_distance(slew)

            # Validate wheel headroom before committing the slew
            if not self._has_wheel_headroom(slew, utime):
                self._log_or_print(
                    utime,
                    "SLEW",
                    f"{unixtime2date(utime)}: Slew rejected - insufficient wheel momentum/torque headroom",
                )
                self.headroom_rejects += 1
                # Request desat to free momentum and avoid deadlock
                self.request_desat(utime)
                return False

            # Validate slew is possible
            if not self._is_slew_valid(visstart, slew.obstype, utime):
                return False

            # Calculate slew timing
            execution_time = self._calculate_slew_timing(
                slew, visstart, utime, is_first_slew
            )

        slew.slewstart = execution_time
        slew.calc_slewtime()
        self.slew_dists.append(slew.slewdist)

        # Enqueue the slew command
        command = ACSCommand(
            command_type=ACSCommandType.SLEW_TO_TARGET,
            execution_time=slew.slewstart,
            slew=slew,
        )
        self.enqueue_command(command)

        if is_first_slew:
            self.last_slew = slew

        return True

    def _create_target_request(self, slew: Slew, utime: float) -> "Pointing":
        """Create and configure a target observation request for visibility checking."""
        from ..targets import Pointing

        target = Pointing(
            config=self.config,
            ra=slew.endra,
            dec=slew.enddec,
            obsid=slew.obsid,
        )
        target.isat = slew.obstype != "PPT"

        year, day = unixtime2yearday(utime)
        target.visibility()
        return target

    def _initialize_slew_positions(self, slew: Slew, utime: float) -> bool:
        """Initialize slew start positions.

        If a previous slew exists, start from current pointing (self.ra/dec).
        If this is the first slew, derive current pointing from ephemeris if
        ra/dec have not yet been initialized (both zero) and use that as start.
        Returns True if this is the first slew (used for accounting/logging).
        """
        if self.last_slew:
            slew.startra = self.ra
            slew.startdec = self.dec
            return False

        slew.startra = self.ra
        slew.startdec = self.dec
        return True

    def _is_slew_valid(self, visstart: float, obstype: str, utime: float) -> bool:
        """Check if the requested slew is valid (target is visible)."""
        if not visstart and obstype == "PPT":
            self._log_or_print(
                utime,
                "SLEW",
                f"{unixtime2date(utime)}: Slew rejected - target not visible",
            )
            return False
        return True

    def _calculate_slew_timing(
        self, slew: Slew, visstart: float, utime: float, is_first_slew: bool
    ) -> float:
        """Calculate when the slew should start, accounting for current slew and constraints."""
        execution_time = utime

        # If desat is active, delay slews until desat completes
        if self._desat_active and self._desat_end > execution_time:
            execution_time = self._desat_end
            self._log_or_print(
                utime,
                "SLEW",
                "%s: Desat active - delaying slew until %s"
                % (
                    unixtime2date(utime),
                    unixtime2date(execution_time),
                ),
            )

        # Wait for current slew to finish if in progress
        if (
            not is_first_slew
            and isinstance(self.last_slew, Slew)
            and self.last_slew.is_slewing(utime)
        ):
            execution_time = self.last_slew.slewstart + self.last_slew.slewtime
            self._log_or_print(
                utime,
                "SLEW",
                "%s: Slewing - delaying next slew until %s"
                % (
                    unixtime2date(utime),
                    unixtime2date(execution_time),
                ),
            )

        # Wait for target visibility if constrained
        if visstart > execution_time and slew.obstype == "PPT":
            self._log_or_print(
                utime,
                "SLEW",
                "%s: Slew delayed by %.1fs"
                % (
                    unixtime2date(utime),
                    visstart - execution_time,
                ),
            )
            execution_time = visstart

        return execution_time

    def _compute_slew_distance(self, slew: Slew) -> float:
        """Ensure slew distance/path are populated and return distance (deg)."""
        try:
            slew.predict_slew()
        except Exception:
            return 0.0
        return float(getattr(slew, "slewdist", 0.0) or 0.0)

    def _check_current_load(self, utime: float | None = None) -> bool:
        """Check if reaction wheels are within acceptable momentum load.

        Returns True if wheels are OK to accept new slews, False if overloaded.
        When overloaded, requests desaturation automatically.
        """
        if not self.reaction_wheels:
            return True

        max_frac = self.wheel_dynamics.get_max_momentum_fraction()
        if max_frac >= 0.6:
            t = utime if utime is not None else 0.0
            self._log_or_print(
                t,
                "SLEW",
                f"{unixtime2date(t)}: Slew rejected - wheels already at {max_frac:.2f} fraction, requesting desat",
            )
            self.request_desat(t)
            return False
        return True

    def _has_wheel_headroom(self, slew: Slew, utime: float | None = None) -> bool:
        """Check if reaction wheels have torque/momentum headroom for this slew."""
        if not self.reaction_wheels:
            return True

        if not self._check_current_load(utime):
            return False

        dist = float(getattr(slew, "slewdist", 0.0) or 0.0)
        if dist <= 0:
            return True

        # Compute requested vmax for this slew (triangular vs trapezoidal)
        try:
            max_rate_req = float(
                getattr(self.acs_config, "max_slew_rate", float("inf"))
            )
        except Exception:
            max_rate_req = float("inf")
        from math import sqrt

        # Requested accel from ACS config (fallback to achievable accel)
        acs_accel = 0.0
        try:
            acs_accel = float(getattr(self.acs_config, "slew_acceleration", 0.0))
        except Exception:
            acs_accel = 0.0

        # Estimate available accel along slew axis using all wheels
        axis = np.array(getattr(slew, "rotation_axis", (0.0, 0.0, 1.0)), dtype=float)
        try:
            motion_time = float(self.acs_config.motion_time(dist))
        except Exception:
            motion_time = 0.0
        achievable_accel = self.wheel_dynamics.get_axis_accel_limit(axis, motion_time)
        # Guard against effectively infeasible slews from headroom-limited torque/rate.
        min_accel = 1e-4  # deg/s^2
        min_rate = 1e-3  # deg/s
        if achievable_accel <= min_accel:
            if utime is None:
                utime = 0.0
            self._log_or_print(
                float(utime),
                "SLEW",
                f"{unixtime2date(float(utime))}: Slew rejected - accel limit too low ({achievable_accel:.3e} deg/s^2)",
            )
            self.request_desat(float(utime))
            return False
        if acs_accel <= 0:
            acs_accel = achievable_accel

        # Triangular profile peak rate if max_rate is high enough
        triangular_vmax = sqrt(max(0.0, acs_accel * dist))
        requested_vmax = min(
            max_rate_req, triangular_vmax if max_rate_req > 0 else triangular_vmax
        )

        if achievable_accel + 1e-6 < acs_accel:
            return False

        # Also ensure momentum headroom allows the required peak rate
        rate_limit = self.wheel_dynamics.get_axis_rate_limit(axis)
        if rate_limit <= min_rate:
            if utime is None:
                utime = 0.0
            self._log_or_print(
                float(utime),
                "SLEW",
                f"{unixtime2date(float(utime))}: Slew rejected - rate limit too low ({rate_limit:.3e} deg/s)",
            )
            self.request_desat(float(utime))
            return False
        # Require generous margin so we desat well before hitting the absolute limit
        margin = 0.5
        if rate_limit * margin + 1e-6 < requested_vmax:
            return False

        # Direct momentum headroom check along slew axis (predict required peak H)
        from math import pi

        i_axis = self.wheel_dynamics.get_inertia_about_axis(axis)
        required_axis_h = requested_vmax * (pi / 180.0) * i_axis

        # Get headroom from WheelDynamics (includes momentum margin)
        total_axis_headroom = self.wheel_dynamics.get_headroom_along_axis(axis)

        # Use the smaller of the two (projection can over/under-estimate vs rate_limit-derived)
        headroom_axis_h = min(total_axis_headroom, rate_limit * i_axis * (pi / 180.0))

        # WheelDynamics already applies margin, so compare directly
        if headroom_axis_h + 1e-9 < required_axis_h:
            return False

        # Predict per-wheel peak momentum during the slew (accel phase)
        utime_val = 0.0 if utime is None else float(utime)
        if not self._predict_wheel_peak_momentum(
            slew,
            axis,
            acs_accel,
            requested_vmax,
            i_axis,
            margin=self._wheel_mom_margin,
            utime=utime_val,
        ):
            self._log_or_print(
                utime_val,
                "SLEW",
                f"{unixtime2date(utime_val)}: Slew rejected - wheel peak momentum would exceed capacity",
            )
            return False

        return True

    def pointing(self, utime: float) -> tuple[float, float, float, int]:
        """
        Calculate ACS pointing for the given time.

        This is the main state machine update method. It:
        1. Checks for upcoming passes and enqueues commands
        2. Processes any commands due for execution
        3. Updates the current ACS mode based on slew/pass state
        4. Calculates current RA/Dec pointing
        """
        # Determine if the spacecraft is currently in eclipse
        self.in_eclipse = self.constraint.in_eclipse(ra=0, dec=0, time=utime)

        # Process any commands scheduled for execution at or before current time
        self._process_commands(utime)

        # Update ACS mode based on current state
        self._update_mode(utime)

        # Check current constraints
        self._check_constraints(utime)

        # Calculate current RA/Dec pointing
        # Also update runtime-coupled wheel momentum while slewing
        self._update_wheel_momentum(utime)
        self._calculate_pointing(utime)

        # Return current pointing
        if self.last_slew is not None:
            return self.ra, self.dec, self.roll, self.last_slew.obsid
        else:
            return self.ra, self.dec, self.roll, 1

    def get_mode(self, utime: float) -> ACSMode:
        """Determine current spacecraft mode based on ACS state and external factors.

        This is the authoritative source for determining spacecraft operational mode,
        considering slewing state, passes, SAA region, battery charging, and safe mode.
        """
        # Safe mode takes absolute priority - once entered, cannot be exited
        if self.in_safe_mode:
            return ACSMode.SAFE

        # Desat holds the spacecraft busy; treat as slewing/maintenance
        if self._desat_active:
            if utime >= self._desat_end:
                self._desat_active = False
            else:
                return ACSMode.SLEWING

        # Check if actively slewing
        if self._is_actively_slewing(utime):
            assert self.current_slew is not None, (
                "Current slew must be set when actively slewing"
            )
            # Check if slewing for charging - but only report CHARGING if in sunlight
            if self.current_slew.obstype == "CHARGE":
                # Check eclipse state - no point being in CHARGING mode during eclipse
                if self.in_eclipse:
                    return ACSMode.SLEWING  # In eclipse, treat as normal slew
                return ACSMode.CHARGING
            if self.current_slew.obstype == "GSP":
                if self.current_pass is None:
                    if not getattr(self.current_slew, "_warned_no_pass", False):
                        self._log_or_print(
                            utime,
                            "PASS",
                            (
                                f"{unixtime2date(utime)}: PASS slew requested without an "
                                "active pass; treating as SLEWING."
                            ),
                        )
                        setattr(self.current_slew, "_warned_no_pass", True)
                    return ACSMode.SLEWING
                return ACSMode.PASS
            return ACSMode.SLEWING

        # Check if dwelling in charging mode (after slew to charge pointing)
        if self._is_in_charging_mode(utime):
            return ACSMode.CHARGING

        # Check if in pass dwell phase (after slew, during communication)
        if self._is_in_pass_dwell(utime):
            return ACSMode.PASS

        # Check if in SAA region
        if self.saa is not None and self.saa.insaa(utime):
            return ACSMode.SAA

        return ACSMode.SCIENCE

    def _is_actively_slewing(self, utime: float) -> bool:
        """Check if spacecraft is currently executing a slew."""
        return self.current_slew is not None and self.current_slew.is_slewing(utime)

    def _is_in_charging_mode(self, utime: float) -> bool:
        """Check if spacecraft is in charging mode (dwelling at charge pointing).

        Charging mode persists after slew completes until END_BATTERY_CHARGE command.
        Returns False during eclipse since charging is not useful without sunlight.
        """
        # Must have completed a CHARGE slew and not be actively slewing
        if not (
            self.last_slew is not None
            and self.last_slew.obstype == "CHARGE"
            and not self._is_actively_slewing(utime)
        ):
            return False

        # Check if spacecraft is in sunlight (not in eclipse)
        if self.ephem is None:
            # No ephemeris, assume sunlight (charging possible)
            return True

        # Only charging mode if NOT in eclipse
        return not self.in_eclipse

    def _is_in_pass_dwell(self, utime: float) -> bool:
        """Check if spacecraft is in pass dwell phase (stationary during groundstation contact)."""
        if self.current_pass is None:
            return False
        if self.current_pass.in_pass(utime):
            return True
        return False

    def _update_mode(self, utime: float) -> None:
        """Update ACS mode based on current slew/pass state."""
        self.acsmode = self.get_mode(utime)

    def _check_constraints(self, utime: float) -> None:
        """Check and log constraint violations for current pointing."""
        if (
            isinstance(self.last_slew, Slew)
            and self.last_slew.at is not None
            and not isinstance(self.last_slew.at, bool)
            and self.last_slew.obstype == "PPT"
            and self.constraint.in_constraint(
                self.last_slew.at.ra, self.last_slew.at.dec, utime
            )
        ):
            assert self.last_slew.at is not None
            # Avoid stale constraint spam: only log while slewing or within the
            # active PPT window, if available.
            if not self.last_slew.is_slewing(utime):
                begin = getattr(self.last_slew.at, "begin", None)
                end = getattr(self.last_slew.at, "end", None)
                if begin is None or end is None:
                    return
                if not (float(begin) <= utime <= float(end)):
                    return

            # Collect only the true constraints
            true_constraints = []
            if self.last_slew.at.in_moon(utime):
                true_constraints.append("Moon")
            if self.last_slew.at.in_sun(utime):
                true_constraints.append("Sun")
            if self.last_slew.at.in_earth(utime):
                true_constraints.append("Earth")
            if self.last_slew.at.in_panel(utime):
                true_constraints.append("Panel")

            # Print only if there are true constraints
            if true_constraints:
                self._log_or_print(
                    utime,
                    "CONSTRAINT",
                    "%s: CONSTRAINT: RA=%s Dec=%s obsid=%s %s"
                    % (
                        unixtime2date(utime),
                        self.last_slew.at.ra,
                        self.last_slew.at.dec,
                        self.last_slew.obsid,
                        " ".join(true_constraints),
                    ),
                )
            # Note: acsmode remains SCIENCE - the DITL will decide if charging is needed

    def _calculate_pointing(self, utime: float) -> None:
        """Calculate current RA/Dec based on slew state or safe mode."""
        # Safe mode overrides all other pointing
        if self.in_safe_mode:
            self._calculate_safe_mode_pointing(utime)
        # If we are in a groundstations pass
        elif self.current_pass is not None:
            self.ra, self.dec = self.current_pass.ra_dec(utime)  # type: ignore[assignment]
        # If we are actively slewing
        elif self.last_slew is not None and self.last_slew.is_slewing(utime):
            self.ra, self.dec = self.last_slew.ra_dec(utime)
        # Slew completed (or no slew in progress): hold last commanded pointing.
        elif self.last_slew is not None:
            self.ra = self.last_slew.endra
            self.dec = self.last_slew.enddec
        else:
            # If there's no slew or pass, maintain current pointing
            pass

    def _calculate_safe_mode_pointing(self, utime: float) -> None:
        """Calculate safe mode pointing - point solar panels at the Sun.

        In safe mode, the spacecraft points to maximize solar panel illumination.
        This may be perpendicular to the Sun for side-mounted panels or directly
        at the Sun for body-mounted panels, following the optimal charging pointing.
        """
        # Use solar panel optimal pointing if available
        if self.solar_panel is not None:
            target_ra, target_dec = self.solar_panel.optimal_charging_pointing(
                utime, self.ephem
            )
        else:
            # Fallback: point directly at Sun if no solar panel config and that
            # serves you right for not having solar panels!
            index = self.ephem.index(dtutcfromtimestamp(utime))
            target_ra = self.ephem.sun_ra_deg[index]
            target_dec = self.ephem.sun_dec_deg[index]

        # If actively slewing to safe mode position, use slew interpolation
        if (
            self.current_slew is not None
            and self.current_slew.obstype == "SAFE"
            and self.current_slew.is_slewing(utime)
        ):
            self.ra, self.dec = self.current_slew.ra_dec(utime)
        else:
            # After slew completes or for continuous tracking, maintain optimal pointing
            self.ra = target_ra
            self.dec = target_dec

    def _update_wheel_momentum(self, utime: float) -> None:
        """Update reaction wheel stored momentum during active slews.

        This method is called from `pointing()` before calculating pointing.
        It computes the wheel torque needed to realize the current slew
        (or to hold attitude when idle), applies that torque over the
        elapsed time since the last update, and clamps to wheel capabilities.
        """
        # Pull runtime toggle in case config is updated from a notebook.
        try:
            self._mtq_bleed_in_science = bool(
                getattr(self.acs_config, "mtq_bleed_in_science", False)
            )
        except Exception:
            self._mtq_bleed_in_science = False

        # track last update time
        if not hasattr(self, "_last_pointing_time") or self._last_pointing_time is None:
            self._last_pointing_time = utime
            if self.reaction_wheels:
                self._build_wheel_snapshot(utime, None, None)
            return

        dt = utime - self._last_pointing_time
        if dt <= 0:
            self._last_pointing_time = utime
            if self.reaction_wheels:
                self._build_wheel_snapshot(utime, None, None)
            return

        # Compute external disturbance torque (for both control and bookkeeping)
        disturbance_torque = self._compute_disturbance_torque(utime)

        # Apply magnetorquer desat bleed: during DESAT, or continuously when not in SCIENCE
        if self.magnetorquers:
            if self._desat_active and self._desat_use_mtq:
                self._apply_magnetorquer_desat(dt, utime)
            elif self.acsmode != ACSMode.SCIENCE or self._mtq_bleed_in_science:
                self._apply_magnetorquer_desat(dt, utime)
            else:
                self.mtq_power_w = 0.0
        # Check if a slew just completed (was slewing, now not)
        is_slewing = self.current_slew is not None and self.current_slew.is_slewing(
            utime
        )
        if self._was_slewing and not is_slewing:
            # Slew just completed - verify momentum consistency
            if self._last_slew_for_verification is not None:
                self._verify_slew_end_momentum(self._last_slew_for_verification, utime)
                self._last_slew_for_verification = None
        self._was_slewing = is_slewing

        # Only update when actively slewing and we have wheels
        if self.current_slew is None or not is_slewing:
            handled_pass = False
            # If in a ground pass, treat continuous tracking as a small slew for wheel accounting
            if (
                self.current_pass is not None
                and self.reaction_wheels
                and not self._desat_active
                and dt > 0
            ):
                handled_pass = self._apply_pass_wheel_update(dt, utime)

            # If holding attitude (no slew/pass), apply wheel torque to reject disturbances
            if self.reaction_wheels and not handled_pass:
                self._apply_hold_wheel_torque(disturbance_torque, dt, utime)

            # Proactively request desat if momentum is high while idle
            if self.reaction_wheels and not self._desat_active:
                max_frac = 0.0
                for w in self.reaction_wheels:
                    mm = float(getattr(w, "max_momentum", 0.0))
                    cm = float(getattr(w, "current_momentum", 0.0))
                    if mm > 0:
                        max_frac = max(max_frac, abs(cm) / mm)
                if max_frac >= 0.9:
                    self.request_desat(utime)
            self._last_pointing_time = utime
            return

        if not self.reaction_wheels:
            self._last_pointing_time = utime
            return

        # Determine current accel request (deg/s^2)
        accel_req = self._slew_accel_profile(self.current_slew, utime)
        if accel_req == 0.0:
            # During settle/hold portion of a slew, reject disturbances like a hold
            self._apply_hold_wheel_torque(disturbance_torque, dt, utime)
            self._last_pointing_time = utime
            return

        # Build inertia matrix from config
        moi_cfg = self.config.spacecraft_bus.attitude_control.spacecraft_moi
        I_mat = DisturbanceModel._build_inertia(moi_cfg)

        from math import pi

        # rotation axis (unit) provided by Slew.predict_slew; default to +Z
        axis = np.array(
            getattr(self.current_slew, "rotation_axis", (0.0, 0.0, 1.0)), dtype=float
        )
        a_nrm = np.linalg.norm(axis)
        if a_nrm <= 0:
            axis = np.array([0.0, 0.0, 1.0])
        else:
            axis = axis / a_nrm

        # Effective moment of inertia about the rotation axis: i_axis = axis^T * I * axis
        i_axis = float(axis.dot(I_mat.dot(axis)))

        # Compute requested scalar torque (N*m) about that axis
        requested_torque = (accel_req * (pi / 180.0)) * i_axis

        # Compute disturbance torque (external)
        disturbance = self._compute_disturbance_torque(utime)

        # Desired spacecraft torque vector (3D), including disturbance rejection
        T_desired = axis * requested_torque
        T_desired = T_desired - disturbance

        taus, taus_allowed, T_actual, clamped = self._allocate_wheel_torques(
            T_desired, dt, use_weights=False, bias_gain=0.1
        )
        if clamped:
            self._slew_clamped = True

        # Debug trace
        try:
            self._logger.debug(
                "_update_wheel_momentum: utime=%s dt=%s accel_req=%s requested_torque=%s T_desired=%s T_actual=%s",
                utime,
                dt,
                accel_req,
                requested_torque,
                T_desired.tolist() if hasattr(T_desired, "tolist") else T_desired,
                T_actual.tolist() if hasattr(T_actual, "tolist") else T_actual,
            )
        except Exception:
            pass

        # Apply external disturbance torque (changes total system momentum)
        self._apply_external_torque(disturbance, dt, source="disturbance")

        # Apply wheel torques with momentum conservation (internal exchange)
        self._apply_wheel_torques_conserving(taus_allowed, dt)

        # If we couldn't supply full torque, reduce accel_override magnitude accordingly
        achieved_scalar = float(np.dot(T_actual, axis))
        if abs(achieved_scalar) < abs(requested_torque) and i_axis > 0:
            # Only use a non-negative accel magnitude for slew timing.
            if achieved_scalar <= 0:
                new_accel = 0.0
            else:
                new_accel = (achieved_scalar / i_axis) * (180.0 / pi)
            self.current_slew._accel_override = new_accel
            if self.last_slew is self.current_slew:
                self.last_slew._accel_override = new_accel

        # Capture wheel snapshot for telemetry/resource tracking
        self._build_wheel_snapshot(utime, taus, taus_allowed)

        # Periodic conservation check (every ~10 updates to avoid overhead)
        if not hasattr(self, "_conservation_check_counter"):
            self._conservation_check_counter = 0
        self._conservation_check_counter += 1
        if self._conservation_check_counter >= 10:
            self._check_momentum_conservation(utime)
            self._conservation_check_counter = 0

        self._last_pointing_time = utime

    def _apply_magnetorquer_desat(self, dt: float, utime: float) -> None:
        """Bleed wheel momentum using magnetorquers during desat.

        Delegates the physics to WheelDynamics. This method computes the local
        magnetic field and passes it to WheelDynamics for momentum bleeding.
        """
        if not self.magnetorquers or dt <= 0:
            return

        # Get local magnetic field in body frame
        b_body, _ = self.disturbance_model.local_bfield_vector(utime, self.ra, self.dec)

        # Delegate to WheelDynamics for the physics
        self.mtq_power_w = self.wheel_dynamics.apply_magnetorquer_desat(b_body, dt)

        # Copy diagnostic values from WheelDynamics for telemetry
        self._last_mtq_proj_max = self.wheel_dynamics._last_mtq_proj_max
        self._last_mtq_torque_mag = self.wheel_dynamics._last_mtq_torque_mag

    def _slew_accel_profile(self, slew: Slew, utime: float) -> float:
        """Return signed slew acceleration (deg/s^2) at utime for bang-bang profile.

        Positive accel means speeding up along the slew axis, negative means decel.
        Returns 0 during cruise/settle or if inputs are invalid.
        """
        if slew is None:
            return 0.0
        t = float(utime - getattr(slew, "slewstart", 0.0))
        if t <= 0:
            return 0.0
        dist = float(getattr(slew, "slewdist", 0.0) or 0.0)
        if dist <= 0:
            return 0.0

        a = getattr(slew, "_accel_override", None)
        if a is None or a <= 0:
            a = float(self.acs_config.slew_acceleration)
        vmax = getattr(slew, "_vmax_override", None)
        if vmax is None or vmax <= 0:
            vmax = float(self.acs_config.max_slew_rate)
        if a <= 0 or vmax <= 0:
            return 0.0

        # Motion time excludes settle
        motion_time = float(self.acs_config.motion_time(dist, accel=a, vmax=vmax))
        if t >= motion_time:
            return 0.0

        t_accel = vmax / a
        d_accel = 0.5 * a * t_accel**2
        if 2 * d_accel >= dist:
            # Triangular
            t_peak = (dist / a) ** 0.5
            return a if t <= t_peak else -a

        # Trapezoidal
        d_cruise = dist - 2 * d_accel
        t_cruise = d_cruise / vmax if vmax > 0 else 0.0
        if t <= t_accel:
            return a
        if t <= t_accel + t_cruise:
            return 0.0
        return -a

    def _apply_hold_wheel_torque(
        self, disturbance_torque: np.ndarray, dt: float, utime: float
    ) -> None:
        """Apply wheel torque to reject disturbance torque while holding attitude."""
        if not self.reaction_wheels or dt <= 0:
            return

        # Apply external disturbance torque (changes total system momentum)
        self._apply_external_torque(disturbance_torque, dt, source="disturbance")

        # Wheels apply torque to counter disturbance (internal momentum exchange)
        T_desired = -disturbance_torque
        taus, taus_allowed, T_actual, _ = self._allocate_wheel_torques(
            T_desired, dt, use_weights=True
        )
        self._last_hold_torque_target_mag = float(np.linalg.norm(T_desired))
        self._last_hold_torque_actual_mag = float(np.linalg.norm(T_actual))

        # Apply wheel torques with momentum conservation
        self._apply_wheel_torques_conserving(taus_allowed, dt)

        self._build_wheel_snapshot(utime, taus, taus_allowed)

    def _allocate_wheel_torques(
        self,
        t_desired: np.ndarray,
        dt: float,
        use_weights: bool = False,
        bias_gain: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Solve wheel torques for a desired body torque and apply headroom clamps."""
        # Delegate to WheelDynamics
        taus, taus_allowed, t_actual, clamped = self.wheel_dynamics.allocate_torques(
            t_desired,
            dt,
            use_weights=use_weights,
            bias_gain=bias_gain,
        )
        try:
            self._last_t_actual_mag = float(np.linalg.norm(t_actual))
        except Exception:
            self._last_t_actual_mag = 0.0

        return taus, taus_allowed, t_actual, clamped

    def _build_wheel_snapshot(
        self, utime: float, taus_cmd: Any | None, taus_allowed: Any | None
    ) -> None:
        """Capture per-wheel torque/momentum state for resource tracking."""

        def _snap_float(val: Any) -> float:
            try:
                return float(val)
            except Exception:
                return 0.0

        # Clear cached diagnostics when not actively slewing (unless in a pass update)
        try:
            if self.current_slew is None or not self.current_slew.is_slewing(utime):
                if self.current_pass is None:
                    self._last_t_actual_mag = 0.0
                    self._last_pass_rate_deg_s = 0.0
                    self._last_pass_torque_target_mag = 0.0
                    self._last_pass_torque_actual_mag = 0.0
            # Only zero MTQ telemetry when SCIENCE bleed is disabled.
            if self.acsmode == ACSMode.SCIENCE and not self._mtq_bleed_in_science:
                self._last_mtq_proj_max = 0.0
                self._last_mtq_torque_mag = 0.0
                self.mtq_power_w = 0.0
        except Exception:
            pass

        wheels: list[WheelReading] = []
        max_m_frac = 0.0
        max_m_frac_raw = 0.0
        max_t_frac = 0.0
        saturated = False

        for i, w in enumerate(self.reaction_wheels):
            max_mom = _snap_float(getattr(w, "max_momentum", 0.0))
            max_torque = _snap_float(getattr(w, "max_torque", 0.0))
            momentum = _snap_float(getattr(w, "current_momentum", 0.0))
            m_frac = abs(momentum) / max_mom if max_mom > 0 else 0.0
            # Raw fraction is the actual abs(momentum)/capacity (clamped to 1.0 for reporting)
            m_frac_raw = min(1.0, m_frac)

            cmd = (
                _snap_float(taus_cmd[i])
                if taus_cmd is not None and len(taus_cmd) > i
                else 0.0
            )
            allowed = (
                _snap_float(taus_allowed[i])
                if taus_allowed is not None and len(taus_allowed) > i
                else cmd
            )
            t_frac = abs(allowed) / max_torque if max_torque > 0 else 0.0

            max_m_frac = max(max_m_frac, m_frac)
            max_m_frac_raw = max(max_m_frac_raw, m_frac_raw)
            max_t_frac = max(max_t_frac, t_frac)
            # Saturated only if momentum near capacity
            if max_mom > 0 and abs(momentum) >= 0.99 * max_mom:
                saturated = True

            wheels.append(
                WheelReading(
                    name=getattr(w, "name", f"wheel{i}"),
                    momentum=momentum,
                    max_momentum=max_mom,
                    momentum_fraction=m_frac,
                    momentum_fraction_raw=m_frac_raw,
                    torque_command=cmd,
                    torque_applied=allowed,
                    max_torque=max_torque,
                )
            )

        self._last_wheel_snapshot = WheelTelemetrySnapshot(
            utime=_snap_float(utime),
            wheels=wheels,
            max_momentum_fraction=_snap_float(max_m_frac),
            max_momentum_fraction_raw=_snap_float(max_m_frac_raw),
            max_torque_fraction=_snap_float(max_t_frac),
            saturated=saturated,
            t_actual_mag=_snap_float(getattr(self, "_last_t_actual_mag", 0.0)),
            hold_torque_target_mag=_snap_float(
                getattr(self, "_last_hold_torque_target_mag", 0.0)
            ),
            hold_torque_actual_mag=_snap_float(
                getattr(self, "_last_hold_torque_actual_mag", 0.0)
            ),
            pass_tracking_rate_deg_s=_snap_float(
                getattr(self, "_last_pass_rate_deg_s", 0.0)
            ),
            pass_torque_target_mag=_snap_float(
                getattr(self, "_last_pass_torque_target_mag", 0.0)
            ),
            pass_torque_actual_mag=_snap_float(
                getattr(self, "_last_pass_torque_actual_mag", 0.0)
            ),
            mtq_proj_max=_snap_float(getattr(self, "_last_mtq_proj_max", 0.0)),
            mtq_torque_mag=_snap_float(getattr(self, "_last_mtq_torque_mag", 0.0)),
            mtq_power_w=_snap_float(getattr(self, "mtq_power_w", 0.0)),
        )

    def wheel_snapshot(self) -> WheelTelemetrySnapshot:
        """Return the latest reaction wheel resource snapshot."""
        if self._last_wheel_snapshot is None:
            self._build_wheel_snapshot(
                getattr(self, "_last_pointing_time", 0.0), None, None
            )
        return self._last_wheel_snapshot or WheelTelemetrySnapshot(
            utime=getattr(self, "_last_pointing_time", 0.0),
            wheels=[],
            max_momentum_fraction=0.0,
            max_momentum_fraction_raw=0.0,
            max_torque_fraction=0.0,
            saturated=False,
            t_actual_mag=0.0,
            hold_torque_target_mag=0.0,
            hold_torque_actual_mag=0.0,
            pass_tracking_rate_deg_s=0.0,
            pass_torque_target_mag=0.0,
            pass_torque_actual_mag=0.0,
            mtq_proj_max=0.0,
            mtq_torque_mag=0.0,
            mtq_power_w=0.0,
        )

    @property
    def wheel_power_w(self) -> float:
        """Return total power draw from all reaction wheels in Watts."""
        if not self.reaction_wheels:
            return 0.0
        return sum(w.power_draw() for w in self.reaction_wheels)

    def _compute_disturbance_torque(self, utime: float) -> np.ndarray:
        """Compute aggregate disturbance torque in body frame."""
        torque, components = self.disturbance_model.compute(
            utime=utime,
            ra_deg=self.ra,
            dec_deg=self.dec,
            in_eclipse=bool(getattr(self, "in_eclipse", False)),
            moi_cfg=self.config.spacecraft_bus.attitude_control.spacecraft_moi,
        )
        self._last_disturbance_components = components
        return torque

    def _predict_wheel_peak_momentum(
        self,
        slew: Slew,
        axis: np.ndarray,
        acs_accel_deg: float,
        requested_vmax_deg: float,
        i_axis: float,
        margin: float = 0.9,
        utime: float = 0.0,
    ) -> bool:
        """Predict per-wheel peak momentum during the slew accel phase.

        Returns False if any wheel would exceed its max momentum (with margin) or torque.
        """
        if not self.reaction_wheels:
            return True
        try:
            axis = np.array(axis, dtype=float)
            nrm = np.linalg.norm(axis)
            if nrm <= 0:
                axis = np.array([0.0, 0.0, 1.0])
            else:
                axis = axis / nrm
        except Exception:
            axis = np.array([0.0, 0.0, 1.0])

        dist = float(getattr(slew, "slewdist", 0.0) or 0.0)
        if dist <= 0 or acs_accel_deg <= 0:
            return True

        accel = acs_accel_deg
        max_rate = requested_vmax_deg if requested_vmax_deg > 0 else float("inf")

        # Determine profile times (triangular vs trapezoidal) using deg units
        from math import sqrt

        triangular_vmax = sqrt(max(0.0, accel * dist))
        if triangular_vmax <= max_rate:
            t_acc = triangular_vmax / accel
        else:
            t_acc = max_rate / accel
        if t_acc <= 0:
            return True

        # Build wheel orientation matrix E
        cols = []
        max_torques = []
        max_moms = []
        curr_moms = []
        for w in self.reaction_wheels:
            try:
                v = np.array(w.orientation, dtype=float)
            except Exception:
                v = np.array([1.0, 0.0, 0.0])
            vn = np.linalg.norm(v)
            if vn > 0:
                v = v / vn
            cols.append(v)
            max_torques.append(float(getattr(w, "max_torque", 0.0)))
            max_moms.append(float(getattr(w, "max_momentum", 0.0)))
            curr_moms.append(float(getattr(w, "current_momentum", 0.0)))

        E = np.column_stack(cols) if cols else np.zeros((3, 0))
        if E.size == 0:
            return True

        # Desired torque vector during accel (countering disturbance)
        from math import pi

        requested_torque = (accel * (pi / 180.0)) * i_axis
        T_desired = axis * requested_torque
        T_desired = T_desired - self._compute_disturbance_torque(utime)

        try:
            taus, *_ = np.linalg.lstsq(E, T_desired, rcond=None)
        except Exception:
            return False

        for tau_i, mt, mm, cm in zip(taus, max_torques, max_moms, curr_moms):
            if mt > 0 and abs(tau_i) > mt + 1e-9:
                return False
            if mm <= 0:
                continue
            peak_mom = abs(cm) + abs(tau_i) * t_acc
            if peak_mom > mm * margin:
                return False

        return True

    def _apply_pass_wheel_update(self, dt: float, utime: float) -> bool:
        """Approximate wheel loading during continuous ground pass tracking."""
        if self.current_pass is None:
            self._last_pass_rate_deg_s = 0.0
            self._last_pass_torque_target_mag = 0.0
            self._last_pass_torque_actual_mag = 0.0
            self._build_wheel_snapshot(utime, None, None)
            return False

        try:
            target_ra, target_dec = self.current_pass.ra_dec(utime)
        except Exception:
            self._last_pass_rate_deg_s = 0.0
            self._last_pass_torque_target_mag = 0.0
            self._last_pass_torque_actual_mag = 0.0
            self._build_wheel_snapshot(utime, None, None)
            return False

        # Compute great-circle distance between current pointing and target (deg)
        try:
            ra0 = float(self.ra)
            dec0 = float(self.dec)
            ra1 = float(target_ra if target_ra is not None else 0.0)
            dec1 = float(target_dec if target_dec is not None else 0.0)
            r0 = np.deg2rad(ra0)
            d0 = np.deg2rad(dec0)
            r1 = np.deg2rad(ra1)
            d1 = np.deg2rad(dec1)
            cosc = np.sin(d0) * np.sin(d1) + np.cos(d0) * np.cos(d1) * np.cos(r1 - r0)
            cosc = np.clip(cosc, -1.0, 1.0)
            dist_rad = float(np.arccos(cosc))
            dist_deg = dist_rad * (180.0 / np.pi)
        except Exception:
            dist_deg = 0.0

        if dist_deg <= 0 or dt <= 0:
            self._last_pass_rate_deg_s = 0.0
            self._last_pass_torque_target_mag = 0.0
            self._last_pass_torque_actual_mag = 0.0
            self._build_wheel_snapshot(utime, None, None)
            return True

        # Continuous tracking: use rate change, not a rest-to-rest slew each step.
        omega_req = dist_deg / dt  # deg/s
        last_rate = float(getattr(self, "_last_pass_rate_deg_s", 0.0))
        accel_req = (omega_req - last_rate) / dt  # deg/s^2

        # Estimate rotation axis from endpoints
        try:
            v0 = np.array(
                [np.cos(d0) * np.cos(r0), np.cos(d0) * np.sin(r0), np.sin(d0)],
                dtype=float,
            )
            v1 = np.array(
                [np.cos(d1) * np.cos(r1), np.cos(d1) * np.sin(r1), np.sin(d1)],
                dtype=float,
            )
            axis = np.cross(v0, v1)
            nrm = np.linalg.norm(axis)
            if nrm <= 1e-12:
                axis = np.array([0.0, 0.0, 1.0])
            else:
                axis = axis / nrm
        except Exception:
            axis = np.array([0.0, 0.0, 1.0])

        # Use existing wheel update logic with this synthetic accel
        # Build inertia matrix
        moi_cfg = self.config.spacecraft_bus.attitude_control.spacecraft_moi
        I_mat = DisturbanceModel._build_inertia(moi_cfg)
        i_axis = float(axis.dot(I_mat.dot(axis)))
        if i_axis <= 0:
            self._build_wheel_snapshot(utime, None, None)
            return True

        from math import pi

        # Compute disturbance torque (external)
        disturbance = self._compute_disturbance_torque(utime)

        requested_torque = (accel_req * (pi / 180.0)) * i_axis
        T_desired = axis * requested_torque
        T_desired = T_desired - disturbance

        taus, taus_allowed, t_actual, _ = self._allocate_wheel_torques(
            T_desired, dt, use_weights=True
        )
        try:
            self._last_pass_rate_deg_s = float(omega_req)
            self._last_pass_torque_target_mag = float(np.linalg.norm(T_desired))
            self._last_pass_torque_actual_mag = float(np.linalg.norm(t_actual))
        except Exception:
            self._last_pass_rate_deg_s = 0.0
            self._last_pass_torque_target_mag = 0.0
            self._last_pass_torque_actual_mag = 0.0

        # Apply external disturbance torque (changes total system momentum)
        self._apply_external_torque(disturbance, dt, source="disturbance")

        # Apply wheel torques with momentum conservation
        self._apply_wheel_torques_conserving(taus_allowed, dt)

        self._build_wheel_snapshot(utime, taus, taus_allowed)
        return True

    def request_pass(self, gspass: Pass) -> None:
        """Request a groundstation pass."""
        # Check for overlap with existing passes
        for existing_pass in self.passrequests.passes:
            if self._passes_overlap(gspass, existing_pass):
                self._log_or_print(
                    gspass.begin, "ERROR", "ERROR: Pass overlap detected: %s" % gspass
                )
                return

        self.passrequests.passes.append(gspass)
        self._log_or_print(gspass.begin, "PASS", "Pass requested: %s" % gspass)

    def _passes_overlap(self, pass1: Pass, pass2: Pass) -> bool:
        """Check if two passes have overlapping time windows."""
        # Passes overlap if one starts before the other ends
        return not (pass1.end <= pass2.begin or pass1.begin >= pass2.end)

    def request_battery_charge(
        self, utime: float, ra: float, dec: float, obsid: int
    ) -> None:
        """Request emergency battery charging at specified pointing.

        Enqueues a START_BATTERY_CHARGE command with the given pointing parameters.
        The command will be executed at the specified time.
        """
        command = ACSCommand(
            command_type=ACSCommandType.START_BATTERY_CHARGE,
            execution_time=utime,
            ra=ra,
            dec=dec,
            obsid=obsid,
        )
        self.enqueue_command(command)
        self._log_or_print(
            utime,
            "CHARGING",
            f"Battery charge requested at RA={ra:.2f} Dec={dec:.2f} obsid={obsid}",
        )

    def request_end_battery_charge(self, utime: float) -> None:
        """Request termination of emergency battery charging.

        Enqueues an END_BATTERY_CHARGE command to be executed at the specified time.
        """
        command = ACSCommand(
            command_type=ACSCommandType.END_BATTERY_CHARGE,
            execution_time=utime,
        )
        self.enqueue_command(command)
        self._log_or_print(utime, "CHARGING", "End battery charge requested")

    def request_safe_mode(self, utime: float) -> None:
        """Request entry into safe mode.

        Enqueues an ENTER_SAFE_MODE command to be executed at the specified time.
        Once safe mode is entered, it cannot be exited. The spacecraft will point
        its solar panels at the Sun and obey bus-level constraints.

        Warning: This is an irreversible operation.
        """
        command = ACSCommand(
            command_type=ACSCommandType.ENTER_SAFE_MODE,
            execution_time=utime,
        )
        self.enqueue_command(command)
        self._log_or_print(
            utime,
            "SAFE",
            f"{unixtime2date(utime)}: Safe mode entry requested - this is irreversible",
        )

    def request_desat(self, utime: float, duration: float | None = None) -> None:
        """Request a reaction wheel desaturation period."""
        # If magnetorquers exist, we rely on continuous bleed instead of DESAT commands
        if getattr(self, "magnetorquers", None):
            return
        # If a desat is active or already queued, do nothing
        if self._desat_active:
            return
        if self._commands.has_pending_type(ACSCommandType.DESAT):
            return
        # Enforce cooldown between requests
        cooldown = 1800.0
        if utime < (self._last_desat_end + cooldown):
            return
        # Make desats long enough to meaningfully unload momentum
        dur = float(duration or 1200.0)
        # Start as soon as requested (even if slewing)
        start_time = utime

        command = ACSCommand(
            command_type=ACSCommandType.DESAT,
            execution_time=start_time,
            duration=dur,
        )
        self.enqueue_command(command)
        self.desat_requests += 1
        self._last_desat_request = utime
        self._last_desat_end = start_time + dur
        self._log_or_print(
            start_time,
            "DESAT",
            f"{unixtime2date(start_time)}: Desat requested for {dur:.0f}s",
        )

    def initiate_emergency_charging(
        self,
        utime: float,
        ephem: rust_ephem.Ephemeris,
        emergency_charging: EmergencyCharging,
        lastra: float,
        lastdec: float,
        current_ppt: "Pointing | None",
    ) -> tuple[float, float, Any]:
        """Initiate emergency charging by creating charging PPT and enqueuing charge command.

        Delegates to EmergencyCharging module to create the optimal charging pointing,
        then automatically enqueues the battery charge command via request_battery_charge().

        Returns:
            Tuple of (new_ra, new_dec, charging_ppt) where charging_ppt is the
            created charging pointing or None if charging could not be initiated.
        """
        charging_ppt = emergency_charging.initiate_emergency_charging(
            utime, ephem, lastra, lastdec, current_ppt
        )
        if charging_ppt is not None:
            self.request_battery_charge(
                utime, charging_ppt.ra, charging_ppt.dec, charging_ppt.obsid
            )
            return charging_ppt.ra, charging_ppt.dec, charging_ppt
        return lastra, lastdec, None

    def _start_battery_charge(self, command: ACSCommand, utime: float) -> None:
        """Handle START_BATTERY_CHARGE command execution.

        Initiates a slew to the optimal charging pointing.
        """
        if (
            command.ra is not None
            and command.dec is not None
            and command.obsid is not None
        ):
            self._log_or_print(
                utime,
                "CHARGING",
                f"Starting battery charge at RA={command.ra:.2f} Dec={command.dec:.2f} obsid={command.obsid}",
            )
            self._enqueue_slew(
                command.ra, command.dec, command.obsid, utime, obstype="CHARGE"
            )

    def _end_battery_charge(self, utime: float) -> None:
        """Handle END_BATTERY_CHARGE command execution.

        Terminates charging mode by returning to previous science pointing.
        """
        self._log_or_print(utime, "CHARGING", "Ending battery charge")

        # Clear the charging slew state immediately so _is_in_charging_mode returns False
        # This prevents staying in CHARGING mode while slewing back to science
        if self.last_slew is not None and self.last_slew.obstype == "CHARGE":
            self.last_slew = None

        # Return to the previous science PPT if one exists
        if self.last_ppt is not None:
            self._log_or_print(
                utime,
                "CHARGING",
                f"Returning to last PPT at RA={self.last_ppt.endra:.2f} Dec={self.last_ppt.enddec:.2f} obsid={self.last_ppt.obsid}",
            )
            self._enqueue_slew(
                self.last_ppt.endra,
                self.last_ppt.enddec,
                self.last_ppt.obsid,
                utime,
            )
