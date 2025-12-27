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
from ..config.constants import DTOR
from ..simulation.passes import PassTimes
from .acs_command import ACSCommand
from .disturbance import DisturbanceConfig, DisturbanceModel
from .emergency_charging import EmergencyCharging
from .passes import Pass
from .reaction_wheel import ReactionWheel
from .slew import Slew

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
    command_queue: list[ACSCommand]
    executed_commands: list[ACSCommand]
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
        self.command_queue = []
        self.executed_commands = []

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
        self._last_wheel_snapshot: dict[str, Any] | None = None
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
        wheels_iter = []
        if wheels_val:
            try:
                # Attempt to iterate; if it's a Mock or non-iterable, fall back to empty
                wheels_iter = list(wheels_val)
            except TypeError:
                wheels_iter = []

        for i, w in enumerate(wheels_iter):
            # expect dict with keys: orientation [x,y,z], max_torque, max_momentum
            orient = tuple(w.get("orientation", (1.0, 0.0, 0.0)))
            mt = float(w.get("max_torque", acs_cfg.wheel_max_torque or 0.0))
            mm = float(w.get("max_momentum", acs_cfg.wheel_max_momentum or 0.0))
            name = w.get("name", f"wheel{i}")
            self.reaction_wheels.append(
                ReactionWheel(
                    max_torque=mt, max_momentum=mm, orientation=orient, name=name
                )
            )

        # Magnetorquers (optional) for finite desat
        mtq_iter_raw = getattr(acs_cfg, "magnetorquers", None)
        mtq_iter = []
        if mtq_iter_raw:
            try:
                mtq_iter = list(mtq_iter_raw)
            except TypeError:
                mtq_iter = []
        for i, m in enumerate(mtq_iter):
            try:
                orient = tuple(m.get("orientation", (1.0, 0.0, 0.0)))
            except Exception:
                orient = (1.0, 0.0, 0.0)
            dipole = float(m.get("dipole_strength", 0.0))
            power = float(m.get("power_draw", 0.0))
            name = m.get("name", f"mtq{i}")
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
        try:
            self._magnetorquer_bfield = float(
                getattr(acs_cfg, "magnetorquer_bfield_T", 3e-5)
            )
        except Exception:
            self._magnetorquer_bfield = 3e-5
        try:
            self._cp_offset = np.array(
                getattr(acs_cfg, "cp_offset_body", (0.0, 0.0, 0.0)), dtype=float
            )
        except Exception:
            self._cp_offset = np.zeros(3, dtype=float)
        try:
            self._residual_moment = np.array(
                getattr(acs_cfg, "residual_magnetic_moment", (0.0, 0.0, 0.0)),
                dtype=float,
            )
        except Exception:
            self._residual_moment = np.zeros(3, dtype=float)
        try:
            self._drag_area = float(getattr(acs_cfg, "drag_area_m2", 0.0))
            self._drag_coeff = float(getattr(acs_cfg, "drag_coeff", 2.2))
        except Exception:
            self._drag_area = 0.0
            self._drag_coeff = 2.2
        try:
            self._solar_area = float(getattr(acs_cfg, "solar_area_m2", 0.0))
            self._solar_reflectivity = float(
                getattr(acs_cfg, "solar_reflectivity", 1.0)
            )
        except Exception:
            self._solar_area = 0.0
            self._solar_reflectivity = 1.0
        try:
            self._use_msis = bool(getattr(acs_cfg, "use_msis_density", False))
            self._msis_f107 = float(getattr(acs_cfg, "msis_f107", 200.0))
            self._msis_f107a = float(getattr(acs_cfg, "msis_f107a", 180.0))
            self._msis_ap = float(getattr(acs_cfg, "msis_ap", 12.0))
        except Exception:
            self._use_msis = False
            self._msis_f107 = 200.0
            self._msis_f107a = 180.0
            self._msis_ap = 12.0
        self.disturbance_model = DisturbanceModel(
            self.ephem,
            DisturbanceConfig(
                magnetorquer_bfield_T=self._magnetorquer_bfield,
                cp_offset_body=tuple(self._cp_offset.tolist()),
                residual_magnetic_moment=tuple(self._residual_moment.tolist()),
                drag_area_m2=self._drag_area,
                drag_coeff=self._drag_coeff,
                solar_area_m2=self._solar_area,
                solar_reflectivity=self._solar_reflectivity,
                use_msis_density=self._use_msis,
                msis_f107=self._msis_f107,
                msis_f107a=self._msis_f107a,
                msis_ap=self._msis_ap,
            ),
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

        self.command_queue.append(command)
        self.command_queue.sort(key=lambda cmd: cmd.execution_time)
        self._log_or_print(
            command.execution_time,
            "ACS",
            f"{unixtime2date(command.execution_time)}: Enqueued {command.command_type.name} command for execution  (queue size: {len(self.command_queue)})",
        )

    def _process_commands(self, utime: float) -> None:
        """Process all commands scheduled for execution at or before current time."""
        while self.command_queue and self.command_queue[0].execution_time <= utime:
            command = self.command_queue.pop(0)
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
            self.executed_commands.append(command)

    def _handle_slew_command(self, command: ACSCommand, utime: float) -> None:
        """Handle SLEW_TO_TARGET command."""
        if command.slew is not None:
            self._start_slew(command.slew, utime)

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
        self.command_queue.clear()
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
        self.command_queue = [
            c
            for c in self.command_queue
            if getattr(c, "command_type", None) != ACSCommandType.DESAT
        ]

    def _start_slew(self, slew: Slew, utime: float) -> None:
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
            achievable_accel = self._axis_accel_limit(axis, motion_time_est)
            if achievable_accel > 0:
                # Derive accel directly from wheel capability (not a fixed external parameter)
                accel_override = achievable_accel
            # Derive a max rate limit from wheel momentum capacity along axis
            vmax_limit = self._axis_rate_limit(axis)
            if vmax_limit > 0:
                vmax_override = vmax_limit
        # apply overrides to slew and calc time
        slew._accel_override = accel_override
        slew._vmax_override = vmax_override
        slew.calc_slewtime()

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

        # Update last_ppt if this is a science pointing
        if self._is_science_pointing(slew):
            self.last_ppt = slew

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
        if self._desat_active or any(
            getattr(cmd, "command_type", None) == ACSCommandType.DESAT
            for cmd in self.command_queue
        ):
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

    def _has_wheel_headroom(self, slew: Slew, utime: float | None = None) -> bool:
        """Check if reaction wheels have torque/momentum headroom for this slew."""
        if not self.reaction_wheels:
            return True

        # If already heavily loaded, request desat before any new slew
        max_frac = 0.0
        for w in self.reaction_wheels:
            mm = float(getattr(w, "max_momentum", 0.0))
            cm = float(getattr(w, "current_momentum", 0.0))
            if mm > 0:
                max_frac = max(max_frac, abs(cm) / mm)
        # Lower threshold to be more proactive
        if max_frac >= 0.6:
            if utime is None:
                utime = 0.0
            self._log_or_print(
                utime,
                "SLEW",
                f"{unixtime2date(utime)}: Slew rejected - wheels already at {max_frac:.2f} fraction, requesting desat",
            )
            self.request_desat(utime)
            return False

        dist = float(getattr(slew, "slewdist", 0.0) or 0.0)
        if dist <= 0:
            return True

        # Requested accel from ACS config
        try:
            acs_accel = float(getattr(self.acs_config, "slew_acceleration", 0.0))
        except Exception:
            acs_accel = 0.0

        if acs_accel <= 0:
            return True

        # Compute requested vmax for this slew (triangular vs trapezoidal)
        try:
            max_rate_req = float(
                getattr(self.acs_config, "max_slew_rate", float("inf"))
            )
        except Exception:
            max_rate_req = float("inf")
        from math import sqrt

        # Triangular profile peak rate if max_rate is high enough
        triangular_vmax = sqrt(max(0.0, acs_accel * dist))
        requested_vmax = min(
            max_rate_req, triangular_vmax if max_rate_req > 0 else triangular_vmax
        )

        # Estimate available accel along slew axis using all wheels
        axis = np.array(getattr(slew, "rotation_axis", (0.0, 0.0, 1.0)), dtype=float)
        try:
            motion_time = float(self.acs_config.motion_time(dist))
        except Exception:
            motion_time = 0.0
        achievable_accel = self._axis_accel_limit(axis, motion_time)
        if achievable_accel + 1e-6 < acs_accel:
            return False

        # Also ensure momentum headroom allows the required peak rate
        rate_limit = self._axis_rate_limit(axis)
        # Require generous margin so we desat well before hitting the absolute limit
        margin = 0.5
        if rate_limit * margin + 1e-6 < requested_vmax:
            return False

        # Direct momentum headroom check along slew axis (predict required peak H)
        # Compute inertia about axis
        moi_cfg = self.config.spacecraft_bus.attitude_control.spacecraft_moi
        try:
            if isinstance(moi_cfg, (list, tuple)):
                arr = np.array(moi_cfg, dtype=float)
                if arr.shape == (3, 3):
                    I_mat = arr
                elif arr.shape == (3,):
                    I_mat = np.diag(arr)
                else:
                    val = float(np.mean(arr))
                    I_mat = np.diag([val, val, val])
            else:
                val = float(moi_cfg)
                I_mat = np.diag([val, val, val])
        except Exception:
            I_mat = np.diag([1.0, 1.0, 1.0])

        i_axis = float(axis.dot(I_mat.dot(axis)))
        from math import pi

        required_axis_h = requested_vmax * (pi / 180.0) * i_axis

        total_axis_headroom = 0.0
        for w in self.reaction_wheels:
            try:
                v = np.array(w.orientation, dtype=float)
            except Exception:
                v = np.array([1.0, 0.0, 0.0])
            vn = np.linalg.norm(v)
            if vn > 0:
                v = v / vn
            proj = abs(np.dot(v, axis))
            max_mom = float(getattr(w, "max_momentum", 0.0))
            curr_mom = float(getattr(w, "current_momentum", 0.0))
            headroom = max(0.0, max_mom - abs(curr_mom))
            total_axis_headroom += proj * headroom

        # Use the smaller of the two (projection can over/under-estimate vs rate_limit-derived)
        headroom_axis_h = min(total_axis_headroom, rate_limit * i_axis * (pi / 180.0))

        if headroom_axis_h * margin + 1e-9 < required_axis_h:
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

        # Calculate roll angle
        # FIXME: Rolls should be pre-calculated, as this is computationally expensive
        if False:
            from ..simulation.roll import optimum_roll

            self.roll = optimum_roll(
                self.ra * DTOR,
                self.dec * DTOR,
                utime,
                self.ephem,
                self.solar_panel,
            )

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
            return (
                ACSMode.PASS if self.current_slew.obstype == "GSP" else ACSMode.SLEWING
            )

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
        elif self.last_slew is not None:
            self.ra, self.dec = self.last_slew.ra_dec(utime)
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
        # Only update when actively slewing and we have wheels
        if self.current_slew is None or not self.current_slew.is_slewing(utime):
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

        # Derive scalar moi approximation
        # Build inertia matrix from config. Support scalar, diagonal (Ixx,Iyy,Izz) or full 3x3.
        moi_cfg = self.config.spacecraft_bus.attitude_control.spacecraft_moi
        try:
            if isinstance(moi_cfg, (list, tuple)):
                # If nested list-like -> treat as full matrix
                if len(moi_cfg) == 3 and any(
                    isinstance(x, (list, tuple)) for x in moi_cfg
                ):
                    I_mat = np.array(moi_cfg, dtype=float)
                elif len(moi_cfg) == 3:
                    I_mat = np.diag([float(x) for x in moi_cfg])
                else:
                    # fallback to scalar conversion
                    val = float(sum(moi_cfg) / len(moi_cfg))
                    I_mat = np.diag([val, val, val])
            else:
                val = float(moi_cfg)
                I_mat = np.diag([val, val, val])
        except Exception:
            # if anything fails, fallback to scalar 1.0 inertia
            I_mat = np.diag([1.0, 1.0, 1.0])

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

        # Desired spacecraft torque vector (3D), including disturbance rejection
        T_desired = axis * requested_torque
        T_desired = T_desired - self._compute_disturbance_torque(utime)

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

        # Apply per-wheel torques (wheel stores scalar momentum along its axis)
        for i, w in enumerate(self.reaction_wheels):
            w.apply_torque(float(taus_allowed[i]), dt)

        # If we couldn't supply full torque, reduce accel_override magnitude accordingly
        achieved_scalar = float(np.dot(T_actual, axis))
        if abs(achieved_scalar) < abs(requested_torque) and i_axis > 0:
            new_accel = (achieved_scalar / i_axis) * (180.0 / pi)
            self.current_slew._accel_override = new_accel
            if self.last_slew is self.current_slew:
                self.last_slew._accel_override = new_accel

        # Capture wheel snapshot for telemetry/resource tracking
        self._build_wheel_snapshot(utime, taus, taus_allowed)

        self._last_pointing_time = utime

    def _apply_magnetorquer_desat(self, dt: float, utime: float) -> None:
        """Bleed wheel momentum using magnetorquers during desat."""
        if not self.magnetorquers or dt <= 0:
            return

        # Build total torque vector from magnetorquers using m x B in body frame
        B_body, Bmag = self.disturbance_model.local_bfield_vector(
            utime, self.ra, self.dec
        )
        T_mtq = np.zeros(3, dtype=float)
        total_power = 0.0
        for m in self.magnetorquers:
            try:
                v = np.array(m.get("orientation", (1.0, 0.0, 0.0)), dtype=float)
            except Exception:
                v = np.array([1.0, 0.0, 0.0])
            vn = np.linalg.norm(v)
            if vn <= 0:
                v = np.array([1.0, 0.0, 0.0])
            else:
                v = v / vn
            # Support both legacy "dipole" and the user-facing "dipole_strength"
            dipole = float(m.get("dipole_strength", m.get("dipole", 0.0)))
            m_vec = v * dipole  # A*m^2 in body frame
            T_mtq += np.cross(m_vec, B_body)
            total_power += float(m.get("power_draw", 0.0))

        if not self.reaction_wheels:
            return

        # Project MTQ torque onto each wheel axis and bleed momentum toward zero
        max_proj = 0.0
        for w in self.reaction_wheels:
            try:
                axis = np.array(w.orientation, dtype=float)
            except Exception:
                axis = np.array([1.0, 0.0, 0.0])
            an = np.linalg.norm(axis)
            if an <= 0:
                axis = np.array([1.0, 0.0, 0.0])
            else:
                axis = axis / an
            tau_w = float(np.dot(T_mtq, axis))
            max_proj = max(max_proj, abs(tau_w))
            if tau_w == 0:
                continue
            dm = tau_w * dt
            mom = float(getattr(w, "current_momentum", 0.0))
            if mom > 0:
                mom = max(0.0, mom - abs(dm))
            else:
                mom = min(0.0, mom + abs(dm))
            w.current_momentum = mom

        # Track instantaneous MTQ power draw (W)
        self.mtq_power_w = total_power
        try:
            self._last_mtq_proj_max = float(max_proj)
            self._last_mtq_torque_mag = float(np.linalg.norm(T_mtq))
        except Exception:
            self._last_mtq_proj_max = 0.0
            self._last_mtq_torque_mag = 0.0

        # Do not end desat early; run full duration

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

        T_desired = -disturbance_torque
        taus, taus_allowed, T_actual, _ = self._allocate_wheel_torques(
            T_desired, dt, use_weights=True
        )
        self._last_hold_torque_target_mag = float(np.linalg.norm(T_desired))
        self._last_hold_torque_actual_mag = float(np.linalg.norm(T_actual))

        for i, w in enumerate(self.reaction_wheels):
            w.apply_torque(float(taus_allowed[i]), dt)

        self._build_wheel_snapshot(utime, taus, taus_allowed)

    def _allocate_wheel_torques(
        self,
        t_desired: np.ndarray,
        dt: float,
        use_weights: bool = False,
        bias_gain: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
        """Solve wheel torques for a desired body torque and apply headroom clamps."""
        if not self.reaction_wheels or dt <= 0:
            return np.zeros(0), np.zeros(0), np.zeros(3), False

        # Build wheel orientation matrix E (3 x N), columns are wheel axis unit vectors
        E_cols: list[np.ndarray] = []
        curr_moms: list[float] = []
        max_moms: list[float] = []
        max_torques: list[float] = []
        for w in self.reaction_wheels:
            try:
                v = np.array(w.orientation, dtype=float)
            except Exception:
                v = np.array([1.0, 0.0, 0.0])
            vn = np.linalg.norm(v)
            if vn <= 0:
                v = np.array([1.0, 0.0, 0.0])
            else:
                v = v / vn
            E_cols.append(v)
            curr_moms.append(float(getattr(w, "current_momentum", 0.0)))
            max_moms.append(float(getattr(w, "max_momentum", 0.0)))
            max_torques.append(float(getattr(w, "max_torque", 0.0)))

        E = np.column_stack(E_cols) if E_cols else np.zeros((3, 0))
        if E.size == 0:
            return np.zeros(0), np.zeros(0), np.zeros(3), False

        # Solve least-squares for wheel torques (scalar per wheel) that produce t_desired.
        try:
            if use_weights:
                # Penalize wheels with higher stored momentum to spread load.
                weights = []
                for max_m, curr_m in zip(max_moms, curr_moms):
                    frac = abs(curr_m) / max_m if max_m > 0 else 0.0
                    weights.append(1.0 + 4.0 * frac * frac)
                W = np.diag(weights)
                lam = 1e-2
                cols = E.shape[1]
                reg = np.sqrt(lam) * W
                A = np.vstack([E, reg])
                b = np.concatenate([t_desired, np.zeros(cols, dtype=float)])
                taus, *_ = np.linalg.lstsq(A, b, rcond=None)
            else:
                taus, *_ = np.linalg.lstsq(E, t_desired, rcond=None)
        except Exception:
            # fallback: assign all torque to first wheel only (legacy behavior)
            taus = np.zeros((len(self.reaction_wheels),), dtype=float)
            if len(taus) > 0:
                taus[0] = float(np.linalg.norm(t_desired))

        # Null-space bias: drive wheel momentum toward zero without changing net torque.
        if bias_gain > 0.0:
            try:
                E_pinv = np.linalg.pinv(E)
                n = E.shape[1]
                null_proj = np.eye(n) - (E_pinv @ E)
                h_vec = np.array(curr_moms, dtype=float)
                taus = taus + null_proj.dot(-bias_gain * h_vec)
            except Exception:
                pass

        # Clamp per-wheel torques by peak torque and momentum headroom over dt
        taus_allowed = np.zeros_like(taus)
        clamped = False
        mom_margin = getattr(self, "_wheel_mom_margin", 0.99)
        for i, (mt, mm, cm) in enumerate(zip(max_torques, max_moms, curr_moms)):
            avail = max(0.0, (mm * mom_margin) - abs(cm))
            max_by_mom = avail / dt if dt > 0 else 0.0
            # If torque would reduce stored momentum magnitude, don't limit by momentum headroom
            if cm != 0 and (taus[i] * cm) < 0:
                limit = mt
            else:
                limit = min(mt, max_by_mom)
            allowed = np.sign(taus[i]) * min(abs(taus[i]), limit)
            taus_allowed[i] = allowed
            if abs(allowed) + 1e-9 < abs(taus[i]):
                clamped = True

        # Compute actual produced torque vector
        T_actual = E.dot(taus_allowed) if E.size else np.zeros(3)
        try:
            self._last_t_actual_mag = float(np.linalg.norm(T_actual))
        except Exception:
            self._last_t_actual_mag = 0.0

        return taus, taus_allowed, T_actual, clamped

    def _axis_accel_limit(self, axis: np.ndarray, motion_time: float) -> float:
        """Compute maximum achievable accel (deg/s^2) about a rotation axis using all wheels."""
        if not self.reaction_wheels:
            return float("inf")

        # Normalize axis
        try:
            axis = np.array(axis, dtype=float)
            nrm = np.linalg.norm(axis)
            if nrm <= 0:
                axis = np.array([0.0, 0.0, 1.0])
            else:
                axis = axis / nrm
        except Exception:
            axis = np.array([0.0, 0.0, 1.0])

        # Inertia about axis
        moi_cfg = self.config.spacecraft_bus.attitude_control.spacecraft_moi
        try:
            if isinstance(moi_cfg, (list, tuple)):
                arr = np.array(moi_cfg, dtype=float)
                if arr.shape == (3, 3):
                    I_mat = arr
                elif arr.shape == (3,):
                    I_mat = np.diag(arr)
                else:
                    val = float(np.mean(arr))
                    I_mat = np.diag([val, val, val])
            else:
                val = float(moi_cfg)
                I_mat = np.diag([val, val, val])
        except Exception:
            I_mat = np.diag([1.0, 1.0, 1.0])

        i_axis = float(axis.dot(I_mat.dot(axis)))
        if i_axis <= 0:
            return 0.0

        # Build wheel orientation matrix
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
            if vn <= 0:
                v = np.array([1.0, 0.0, 0.0])
            else:
                v = v / vn
            cols.append(v)
            max_torques.append(float(getattr(w, "max_torque", 0.0)))
            max_moms.append(float(getattr(w, "max_momentum", 0.0)))
            curr_moms.append(float(getattr(w, "current_momentum", 0.0)))

        E = np.column_stack(cols) if cols else np.zeros((3, 0))
        if E.size == 0:
            return 0.0

        # Desired unit torque along axis
        try:
            taus_unit, *_ = np.linalg.lstsq(E, axis, rcond=None)
        except Exception:
            return 0.0

        # Determine scaling limited by per-wheel torque and momentum over motion_time
        scales = []
        for tau_coeff, mt, mm, cm in zip(taus_unit, max_torques, max_moms, curr_moms):
            if abs(tau_coeff) < 1e-9:
                continue
            torque_cap = mt
            if motion_time and motion_time > 0 and mm > 0:
                avail = max(0.0, mm - abs(cm))
                torque_cap = min(torque_cap, avail / motion_time)
            if torque_cap <= 0:
                scales.append(0.0)
            else:
                scales.append(torque_cap / abs(tau_coeff))

        if not scales:
            return 0.0

        torque_along_axis = min(scales)
        # Convert to accel (deg/s^2)
        from math import pi

        return (torque_along_axis / i_axis) * (180.0 / pi)

    def _axis_rate_limit(self, axis: np.ndarray) -> float:
        """Estimate max angular rate (deg/s) about an axis based on wheel momentum capacity."""
        if not self.reaction_wheels:
            return float("inf")

        # Normalize axis
        try:
            axis = np.array(axis, dtype=float)
            nrm = np.linalg.norm(axis)
            if nrm <= 0:
                axis = np.array([0.0, 0.0, 1.0])
            else:
                axis = axis / nrm
        except Exception:
            axis = np.array([0.0, 0.0, 1.0])

        # Inertia about axis
        moi_cfg = self.config.spacecraft_bus.attitude_control.spacecraft_moi
        try:
            if isinstance(moi_cfg, (list, tuple)):
                arr = np.array(moi_cfg, dtype=float)
                if arr.shape == (3, 3):
                    I_mat = arr
                elif arr.shape == (3,):
                    I_mat = np.diag(arr)
                else:
                    val = float(np.mean(arr))
                    I_mat = np.diag([val, val, val])
            else:
                val = float(moi_cfg)
                I_mat = np.diag([val, val, val])
        except Exception:
            I_mat = np.diag([1.0, 1.0, 1.0])

        i_axis = float(axis.dot(I_mat.dot(axis)))
        if i_axis <= 0:
            return float("inf")

        # Project wheel momentum capacity onto axis
        total_axis_mom = 0.0
        for w in self.reaction_wheels:
            try:
                v = np.array(w.orientation, dtype=float)
            except Exception:
                v = np.array([1.0, 0.0, 0.0])
            vn = np.linalg.norm(v)
            if vn > 0:
                v = v / vn
            proj = abs(np.dot(v, axis))
            max_mom = float(getattr(w, "max_momentum", 0.0))
            curr_mom = float(getattr(w, "current_momentum", 0.0))
            headroom = max(0.0, max_mom - abs(curr_mom))
            total_axis_mom += proj * headroom

        if total_axis_mom <= 0:
            return float("inf")

        # omega = H / I; convert to deg/s
        from math import pi

        return (total_axis_mom / i_axis) * (180.0 / pi)

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
            # Only zero MTQ telemetry when SCIENCE bleed is disabled.
            if self.acsmode == ACSMode.SCIENCE and not self._mtq_bleed_in_science:
                self._last_mtq_proj_max = 0.0
                self._last_mtq_torque_mag = 0.0
                self.mtq_power_w = 0.0
        except Exception:
            pass

        wheels: list[dict[str, Any]] = []
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

            cmd = _snap_float(taus_cmd[i]) if taus_cmd is not None else 0.0
            allowed = _snap_float(taus_allowed[i]) if taus_allowed is not None else cmd
            t_frac = abs(allowed) / max_torque if max_torque > 0 else 0.0

            max_m_frac = max(max_m_frac, m_frac)
            max_m_frac_raw = max(max_m_frac_raw, m_frac_raw)
            max_t_frac = max(max_t_frac, t_frac)
            # Saturated only if momentum near capacity
            if max_mom > 0 and abs(momentum) >= 0.99 * max_mom:
                saturated = True

            wheels.append(
                {
                    "name": getattr(w, "name", f"wheel{i}"),
                    "momentum": momentum,
                    "max_momentum": max_mom,
                    "momentum_fraction": m_frac,
                    "momentum_fraction_raw": m_frac_raw,
                    "torque_command": cmd,
                    "torque_applied": allowed,
                    "max_torque": max_torque,
                }
            )

        self._last_wheel_snapshot = {
            "utime": _snap_float(utime),
            "wheels": wheels,
            "max_momentum_fraction": _snap_float(max_m_frac),
            "max_momentum_fraction_raw": _snap_float(max_m_frac_raw),
            "max_torque_fraction": _snap_float(max_t_frac),
            "saturated": saturated,
            "t_actual_mag": _snap_float(getattr(self, "_last_t_actual_mag", 0.0)),
            "hold_torque_target_mag": _snap_float(
                getattr(self, "_last_hold_torque_target_mag", 0.0)
            ),
            "hold_torque_actual_mag": _snap_float(
                getattr(self, "_last_hold_torque_actual_mag", 0.0)
            ),
            "mtq_proj_max": _snap_float(getattr(self, "_last_mtq_proj_max", 0.0)),
            "mtq_torque_mag": _snap_float(getattr(self, "_last_mtq_torque_mag", 0.0)),
            "mtq_power_w": _snap_float(getattr(self, "mtq_power_w", 0.0)),
        }

    def wheel_snapshot(self) -> dict[str, Any]:
        """Return the latest reaction wheel resource snapshot."""
        if self._last_wheel_snapshot is None:
            self._build_wheel_snapshot(
                getattr(self, "_last_pointing_time", 0.0), None, None
            )
        return self._last_wheel_snapshot or {
            "utime": getattr(self, "_last_pointing_time", 0.0),
            "wheels": [],
            "max_momentum_fraction": 0.0,
            "max_momentum_fraction_raw": 0.0,
            "max_torque_fraction": 0.0,
            "saturated": False,
            "hold_torque_target_mag": 0.0,
            "hold_torque_actual_mag": 0.0,
        }

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
            self._build_wheel_snapshot(utime, None, None)
            return False

        try:
            target_ra, target_dec = self.current_pass.ra_dec(utime)
        except Exception:
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
            self._build_wheel_snapshot(utime, None, None)
            return True

        # Approximate accel needed to traverse dist in dt with triangular profile: a = 2*s/t^2
        accel_req = (2.0 * dist_deg) / (dt * dt)

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
        try:
            if isinstance(moi_cfg, (list, tuple)):
                arr = np.array(moi_cfg, dtype=float)
                if arr.shape == (3, 3):
                    I_mat = arr
                elif arr.shape == (3,):
                    I_mat = np.diag(arr)
                else:
                    val = float(np.mean(arr))
                    I_mat = np.diag([val, val, val])
            else:
                val = float(moi_cfg)
                I_mat = np.diag([val, val, val])
        except Exception:
            I_mat = np.diag([1.0, 1.0, 1.0])

        i_axis = float(axis.dot(I_mat.dot(axis)))
        if i_axis <= 0:
            self._build_wheel_snapshot(utime, None, None)
            return True

        from math import pi

        requested_torque = (accel_req * (pi / 180.0)) * i_axis
        T_desired = axis * requested_torque
        T_desired = T_desired - self._compute_disturbance_torque(utime)

        taus, taus_allowed, _, _ = self._allocate_wheel_torques(
            T_desired, dt, use_weights=True
        )
        for i, w in enumerate(self.reaction_wheels):
            w.apply_torque(float(taus_allowed[i]), dt)

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
        if any(
            getattr(cmd, "command_type", None) == ACSCommandType.DESAT
            for cmd in self.command_queue
        ):
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
