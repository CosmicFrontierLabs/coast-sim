from typing import TYPE_CHECKING, cast

import rust_ephem

from ..common import (
    ACSCommandType,
    ACSMode,
    ObsType,
    dtutcfromtimestamp,
    unixtime2date,
)
from ..common.vector import sort_by_angular_separation
from ..config import AttitudeConstraintScope, FaultEvent, MissionConfig
from ..config.constraint import (
    attitude_constraint_names_for_scopes,
    attitude_constraint_scope_label,
    in_attitude_constraint_scopes,
)
from ..simulation.passes import PassTimes
from ..simulation.roll import optimum_roll
from .acs_command import ACSCommand
from .emergency_charging import EmergencyCharging
from .passes import Pass
from .slew import Slew

if TYPE_CHECKING:
    from ..ditl.ditl_log import DITLLog
    from ..targets import Pointing


IDLE_OBSID = 0
IDLE_SAFE_ATTITUDE_GRID_STEP_DEG = 15


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
    obstype: ObsType
    acsmode: ACSMode
    command_queue: list[ACSCommand]
    executed_commands: list[ACSCommand]
    current_slew: Slew | None
    last_ppt: Slew | None
    last_slew: Slew | None
    in_eclipse: bool
    star_tracker_hard_violations: int
    star_tracker_soft_violations: bool
    star_tracker_functional_count: int
    star_tracker_status: list[bool]
    radiator_hard_violations: int
    telescope_hard_violations: int
    radiator_sun_exposure: float
    radiator_earth_exposure: float
    radiator_heat_dissipation_w: float
    science_observation_active: bool

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
        self.last_slew = Slew(config=self.config)
        self.last_slew.endra = self.ra
        self.last_slew.enddec = self.dec

        # Current state
        self.roll = 0.0
        self.obstype = ObsType.PPT
        self.acsmode = ACSMode.IDLE
        self.science_observation_active = False
        self.in_eclipse = False  # Initialize eclipse state
        self.in_safe_mode = False  # Safe mode flag - once True, cannot be exited

        # Star tracker constraint state
        self.star_tracker_hard_violations = 0
        self.star_tracker_soft_violations = False
        self.star_tracker_functional_count = 0
        self.star_tracker_status: list[bool] = []
        self.radiator_hard_violations = 0
        self.telescope_hard_violations = 0
        self.radiator_sun_exposure = 0.0
        self.radiator_earth_exposure = 0.0
        self.radiator_heat_dissipation_w = 0.0

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
                obsid=self.last_slew.obsid if self.last_slew is not None else None,
                acs_mode=self.acsmode,
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
            and command.slew is not None
            and command.slew.obstype == ObsType.SAFE
        )

        # Prevent any commands from being enqueued in safe mode (except SAFE slews)
        if self.in_safe_mode and not is_safe_slew:
            self._log_or_print(
                command.execution_time,
                "ACS",
                f"{unixtime2date(command.execution_time)}: Command {command.command_type.name} rejected - spacecraft is in SAFE MODE",
            )
            return

        if command.command_type in (
            ACSCommandType.SLEW_TO_TARGET,
            ACSCommandType.START_PASS,
        ):
            self._cancel_pending_slew_commands(command)

        self.command_queue.append(command)
        self.command_queue.sort(key=lambda cmd: cmd.execution_time)
        self._log_or_print(
            command.execution_time,
            "ACS",
            f"{unixtime2date(command.execution_time)}: Enqueued {command.command_type.name} command for execution  (queue size: {len(self.command_queue)})",
        )

    def _cancel_pending_slew_commands(self, replacement: ACSCommand) -> None:
        """Drop queued target slews superseded by a newer attitude decision."""
        pending_slews = [
            command
            for command in self.command_queue
            if command.command_type == ACSCommandType.SLEW_TO_TARGET
        ]
        if not pending_slews:
            return

        self.command_queue = [
            command
            for command in self.command_queue
            if command.command_type != ACSCommandType.SLEW_TO_TARGET
        ]
        replacement_label = self._command_label(replacement)
        for canceled in pending_slews:
            canceled_obsid = self._command_obsid(canceled)
            self._log_or_print(
                replacement.execution_time,
                "ACS",
                (
                    f"{unixtime2date(replacement.execution_time)}: Canceled pending "
                    f"SLEW_TO_TARGET command scheduled for "
                    f"{unixtime2date(canceled.execution_time)} "
                    f"(obsid={canceled_obsid}) - superseded by "
                    f"{replacement_label}"
                ),
            )

    @staticmethod
    def _command_label(command: ACSCommand) -> str:
        obsid = ACS._command_obsid(command)
        if obsid is None:
            return command.command_type.name
        return f"{command.command_type.name} obsid={obsid}"

    @staticmethod
    def _command_obsid(command: ACSCommand) -> int | None:
        if command.slew is not None:
            return command.slew.obsid
        return command.obsid

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
            match command.command_type:
                case ACSCommandType.SLEW_TO_TARGET:
                    self._handle_slew_command(command, utime)
                case ACSCommandType.START_PASS:
                    self._start_pass(command, utime)
                case ACSCommandType.END_PASS:
                    self._end_pass(utime)
                case ACSCommandType.START_BATTERY_CHARGE:
                    self._start_battery_charge(command, utime)
                case ACSCommandType.END_BATTERY_CHARGE:
                    self._end_battery_charge(utime)
                case ACSCommandType.ENTER_SAFE_MODE:
                    self._handle_safe_mode_command(command, utime)

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
        self.acsmode = ACSMode.IDLE

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

        last_ppt_obsid = self.last_ppt.obsid if self.last_ppt is not None else "unknown"
        self._log_or_print(
            utime,
            "PASS",
            f"{unixtime2date(utime)}: Pass over - returning to last PPT {last_ppt_obsid}",
        )

    # Handle Safe Mode Command
    def _handle_safe_mode_command(self, command: "ACSCommand", utime: float) -> None:
        """Handle ENTER_SAFE_MODE command.

        Once safe mode is entered, it cannot be exited. The spacecraft will
        point solar panels at the Sun and obey bus-level constraints.
        """
        reason_str = f" - reason: {command.reason}" if command.reason else ""
        self._log_or_print(
            utime,
            "SAFE",
            f"{unixtime2date(utime)}: Entering SAFE MODE - irreversible{reason_str}",
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
        self._enqueue_slew(
            safe_ra, safe_dec, obsid=-999, utime=utime, obstype=ObsType.SAFE
        )

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
        slew.startroll = self.roll  # Start roll from current ACS roll
        slew.slewstart = utime
        slew.calc_slewtime()

        slewdist = slew.slewdist
        self._log_or_print(
            utime,
            "SLEW",
            f"{unixtime2date(utime)}: Starting slew from RA={self.ra:.2f} Dec={self.dec:.2f} "
            f"to RA={slew.endra:.2f} Dec={slew.enddec:.2f} "
            f"(duration: {slew.slewtime:.1f}s, distance: {slewdist:.1f} deg)",
        )

        self.current_slew = slew
        self.last_slew = slew

        # Update last_ppt if this is a science pointing
        if self._is_science_pointing(slew):
            self.science_observation_active = True
            self.last_ppt = slew
        else:
            self.science_observation_active = False

    def _is_science_pointing(self, slew: Slew) -> bool:
        """Check if slew represents a science pointing (not a pass)."""
        return slew.obstype == ObsType.PPT and isinstance(slew, Slew)

    def end_science_observation(self) -> None:
        """Mark the current held attitude as no longer collecting science."""
        self.science_observation_active = False

    def _enqueue_slew(
        self,
        ra: float,
        dec: float,
        obsid: int,
        utime: float,
        obstype: ObsType = ObsType.PPT,
        roll: float | None = None,
    ) -> bool:
        """Create and enqueue a slew command.

        This is a private helper method used internally by ACS for creating slew
        commands during battery charging operations.
        """
        # Create slew object
        slew = Slew(config=self.config)
        slew.ephem = self.ephem
        slew.slewrequest = utime
        slew.endra = ra
        slew.enddec = dec
        # If roll not provided, calculate optimal roll at target position
        if roll is None:
            slew.endroll = optimum_roll(
                ra, dec, utime, self.ephem, self.solar_panel, self.constraint
            )
        else:
            slew.endroll = roll
        slew.obstype = obstype
        slew.obsid = obsid

        # For SAFE mode, skip visibility checking (emergency situation)
        if obstype == ObsType.SAFE:
            # Initialize slew positions without target
            is_first_slew = self._initialize_slew_positions(slew)
            slew.at = None  # No visibility constraint in safe mode
            execution_time = utime  # Execute immediately
        else:
            # Set up target observation request and check visibility
            target_request = self._create_target_request(slew, roll)
            slew.at = target_request

            visstart = target_request.next_vis(utime)
            is_first_slew = self._initialize_slew_positions(slew)

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

    def _create_target_request(
        self, slew: Slew, roll: float | None = None
    ) -> "Pointing":
        """Create and configure a target observation request for visibility checking."""
        from ..targets import Pointing

        target = Pointing.from_config(
            config=self.config,
            ra=slew.endra,
            dec=slew.enddec,
            roll=roll if roll is not None else slew.endroll,
            obsid=slew.obsid,
        )
        target.isat = slew.obstype != ObsType.PPT

        target.visibility()
        return target

    def _initialize_slew_positions(self, slew: Slew) -> bool:
        """Initialize slew start positions from current ACS pointing.

        The ACS always drives the spacecraft from its current position
        (self.ra/dec/roll), regardless of whether a previous slew exists.
        Returns True if this is the first slew (used for accounting/logging).
        """
        slew.startra = self.ra
        slew.startdec = self.dec
        slew.startroll = self.roll
        return self.last_slew is None

    def _is_slew_valid(self, visstart: float, obstype: ObsType, utime: float) -> bool:
        """Check if the requested slew is valid (target is visible)."""
        if not visstart and obstype == ObsType.PPT:
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

        # Wait for current slew to finish if in progress
        if (
            not is_first_slew
            and self.last_slew is not None
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
        if visstart > execution_time and slew.obstype == ObsType.PPT:
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

    def pointing(self, utime: float) -> tuple[float, float, float, int]:
        """
        Calculate ACS pointing for the given time.

        This is the main state machine update method. It:
        1. Checks for upcoming passes and enqueues commands
        2. Processes any commands due for execution
        3. Updates the current ACS mode based on slew/pass state
        4. Calculates current RA/Dec pointing
        5. Calculates current roll angle to optimize solar panel illumination
        6. Checks current constraints using up-to-date pointing and roll
        """
        # Determine if the spacecraft is currently in eclipse
        self.in_eclipse = self.constraint.in_eclipse(ra=0, dec=0, time=utime)

        # Process any commands scheduled for execution at or before current time
        self._process_commands(utime)

        # Update ACS mode based on current state
        self._update_mode(utime)

        # Calculate current RA/Dec pointing
        self._calculate_pointing(utime)

        # Calculate roll angle (must run after _calculate_pointing so ra/dec are current).
        self.roll = self._compute_roll(utime)

        # Idle is an executed attitude, not a constraint-free gap. If a completed
        # observation is being held after science ends, move the hold to an
        # attitude that satisfies the configured IDLE scopes before recording.
        self._enforce_idle_constraint_safe_attitude(utime)

        # Check current constraints (must run after roll is updated)
        self._check_constraints(utime)

        # Return current pointing
        if self.current_pass is not None:
            return self.ra, self.dec, self.roll, self.current_pass.obsid
        if self.last_slew is not None:
            if self._is_science_pointing(self.last_slew):
                obsid = (
                    self.last_slew.obsid
                    if self.science_observation_active
                    else IDLE_OBSID
                )
                return self.ra, self.dec, self.roll, obsid
            return self.ra, self.dec, self.roll, self.last_slew.obsid
        else:
            return self.ra, self.dec, self.roll, IDLE_OBSID

    def get_mode(self, utime: float) -> ACSMode:
        """Determine current spacecraft mode based on ACS state and external factors.

        This is the authoritative source for determining spacecraft operational mode,
        considering slewing state, passes, SAA region, battery charging, and safe mode.
        """
        # Safe mode takes absolute priority - once entered, cannot be exited
        if self.in_safe_mode:
            return ACSMode.SAFE

        # Check if actively slewing
        if self._is_actively_slewing(utime):
            assert self.current_slew is not None, (
                "Current slew must be set when actively slewing"
            )
            return (
                ACSMode.PASS
                if self.current_slew.obstype == ObsType.GSP
                else ACSMode.SLEWING
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

        if self.last_slew is not None:
            if self._is_science_pointing(self.last_slew):
                return (
                    ACSMode.SCIENCE if self.science_observation_active else ACSMode.IDLE
                )
            return ACSMode.IDLE

        return ACSMode.SCIENCE if self.science_observation_active else ACSMode.IDLE

    def _is_actively_slewing(self, utime: float) -> bool:
        """Check if spacecraft is currently executing a slew."""
        return self.current_slew is not None and self.current_slew.is_slewing(utime)

    def _compute_roll(self, utime: float) -> float:
        """Return the roll angle for the current timestep.

        - During a slew: interpolate along the SLERP path.
        - Charging / safe mode: track solar-optimal roll continuously.
        - Settled at a science pointing: lock to the roll computed at scheduling
          time (slew.endroll) so constraints validated by the scheduler hold.
        - Initial boundary condition (no real slew yet): use optimal roll.
        """
        if self._is_actively_slewing(utime) and self.current_slew is not None:
            return self.current_slew.slew_roll(utime)
        if self._is_in_charging_mode(utime) or self.in_safe_mode:
            return optimum_roll(
                self.ra, self.dec, utime, self.ephem, self.solar_panel, self.constraint
            )
        if self.current_pass is not None and self.current_pass.in_pass(utime):
            return self.current_pass.roll_at(utime)
        if self.last_slew is not None and self.last_slew.slewstart > 0:
            return self.last_slew.endroll
        return optimum_roll(
            self.ra, self.dec, utime, self.ephem, self.solar_panel, self.constraint
        )

    def _enforce_idle_constraint_safe_attitude(self, utime: float) -> None:
        """Replace unsafe idle holds with an attitude that satisfies IDLE scopes."""
        if self.acsmode != ACSMode.IDLE:
            return
        if self.current_pass is not None or self.in_safe_mode:
            return

        scopes = self._idle_attitude_scopes()
        if not scopes:
            return

        if not self._idle_attitude_unsafe(self.ra, self.dec, self.roll, utime, scopes):
            return

        safe_attitude = self._find_constraint_safe_idle_attitude(utime, scopes)
        if safe_attitude is None:
            scope_label = attitude_constraint_scope_label(scopes)
            cause = (
                "No constraint-safe IDLE attitude found "
                f"(RA={self.ra:.2f} Dec={self.dec:.2f} scopes={scope_label}); "
                "requesting safe mode"
            )
            self._log_or_print(utime, "ERROR", f"{unixtime2date(utime)}: {cause}")
            self.config.fault_management.events.append(
                FaultEvent(
                    utime=utime,
                    event_type="safe_mode_trigger",
                    name="idle_attitude_constraint",
                    cause=cause,
                    metadata={
                        "ra": self.ra,
                        "dec": self.dec,
                        "scopes": scope_label,
                    },
                )
            )
            self.request_safe_mode(utime)
            return

        ra, dec, roll = safe_attitude
        self._hold_idle_attitude(ra, dec, roll, utime)
        self._log_or_print(
            utime,
            "ACS",
            f"{unixtime2date(utime)}: IDLE attitude constrained; holding safe attitude "
            f"RA={ra:.2f} Dec={dec:.2f} Roll={roll:.2f}",
        )

    def _find_constraint_safe_idle_attitude(
        self, utime: float, scopes: list[AttitudeConstraintScope]
    ) -> tuple[float, float, float] | None:
        """Find a deterministic nearby attitude that satisfies IDLE scopes."""
        candidates = self._idle_safe_attitude_candidates(utime)
        for candidate_ra, candidate_dec in candidates:
            optimal_roll = optimum_roll(
                candidate_ra,
                candidate_dec,
                utime,
                self.ephem,
                self.solar_panel,
                self.constraint,
            )
            for candidate_roll in self._idle_safe_roll_candidates(optimal_roll):
                if not self._idle_attitude_unsafe(
                    candidate_ra,
                    candidate_dec,
                    candidate_roll,
                    utime,
                    scopes,
                ):
                    return candidate_ra, candidate_dec, candidate_roll
        return None

    def _idle_attitude_scopes(self) -> list[AttitudeConstraintScope]:
        return self.config.attitude_constraint_scopes_for_mode(ACSMode.IDLE)

    def _idle_attitude_unsafe(
        self,
        ra: float,
        dec: float,
        roll: float,
        utime: float,
        scopes: list[AttitudeConstraintScope],
    ) -> bool:
        """Runtime instrument-safety check for idle holds.

        Distinct from post-simulation planning validation (queue_ditl): this asks
        "should the ACS repoint to protect hardware?" not "did the planner break a
        scheduling contract?"

        The configured IDLE scopes describe the attitude constraints that must
        not be sustained while idle. An empty list skips this check.
        """
        return in_attitude_constraint_scopes(
            self.constraint,
            scopes,
            ra,
            dec,
            utime,
            target_roll=roll,
            acs_mode=ACSMode.IDLE,
        )

    @staticmethod
    def _idle_safe_roll_candidates(optimal_roll: float) -> list[float]:
        """Try the solar-optimal roll first, then nearby grid-step alternatives."""
        rolls: list[float] = []

        def add(roll: float) -> None:
            normalized = roll % 360.0
            is_new = all(
                abs(((normalized - existing + 180.0) % 360.0) - 180.0) > 1e-6
                for existing in rolls
            )
            if is_new:
                rolls.append(normalized)

        add(optimal_roll)
        for offset in range(
            IDLE_SAFE_ATTITUDE_GRID_STEP_DEG, 360, IDLE_SAFE_ATTITUDE_GRID_STEP_DEG
        ):
            add(optimal_roll + offset)
            add(optimal_roll - offset)
        return rolls

    def _idle_safe_attitude_candidates(self, utime: float) -> list[tuple[float, float]]:
        """Generate deterministic IDLE hold candidates, nearest current attitude first."""
        raw_candidates: list[tuple[float, float]] = []

        def add(ra: float, dec: float) -> None:
            raw_candidates.append((ra % 360.0, max(-90.0, min(90.0, dec))))

        add(self.ra, self.dec)

        if self.solar_panel is not None:
            solar_ra, solar_dec = self.solar_panel.optimal_charging_pointing(
                utime, self.ephem
            )
            add(float(solar_ra), float(solar_dec))

        index = self.ephem.index(dtutcfromtimestamp(utime))
        add(
            (180.0 + float(self.ephem.earth_ra_deg[index])) % 360.0,
            -float(self.ephem.earth_dec_deg[index]),
        )
        add(
            (180.0 + float(self.ephem.sun_ra_deg[index])) % 360.0,
            -float(self.ephem.sun_dec_deg[index]),
        )

        grid: list[tuple[float, float]] = []
        for dec in range(-90, 91, IDLE_SAFE_ATTITUDE_GRID_STEP_DEG):
            if abs(dec) == 90:
                grid.append((0.0, float(dec)))
                continue
            for grid_ra in range(0, 360, IDLE_SAFE_ATTITUDE_GRID_STEP_DEG):
                grid.append((float(grid_ra), float(dec)))
        raw_candidates.extend(sort_by_angular_separation(grid, self.ra, self.dec))

        candidates: list[tuple[float, float]] = []
        seen: set[tuple[float, float]] = set()
        for candidate_ra, candidate_dec in raw_candidates:
            key = (round(candidate_ra, 6), round(candidate_dec, 6))
            if key in seen:
                continue
            seen.add(key)
            candidates.append((candidate_ra, candidate_dec))
        return candidates

    def _hold_idle_attitude(
        self, ra: float, dec: float, roll: float, utime: float
    ) -> None:
        """Install a zero-duration IDLE hold so future ticks keep the safe attitude."""
        hold = Slew.idle_hold(self.config, ra, dec, roll, utime)

        self.current_slew = None
        self.last_slew = hold
        self.science_observation_active = False
        self.ra = ra
        self.dec = dec
        self.roll = roll

    def _is_in_charging_mode(self, utime: float) -> bool:
        """Check if spacecraft is in charging mode (dwelling at charge pointing).

        Charging mode persists after slew completes until END_BATTERY_CHARGE command.
        Returns False during eclipse since charging is not useful without sunlight.
        """
        # Must have completed a CHARGE slew and not be actively slewing
        if not (
            self.last_slew is not None
            and self.last_slew.obstype == ObsType.CHARGE
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
            self.last_slew is not None
            and self.last_slew.at is not None
            and self.last_slew.obstype == ObsType.PPT
        ):
            scopes = self.config.attitude_constraint_scopes_for_mode(self.acsmode)
            constraint_names = attitude_constraint_names_for_scopes(
                self.constraint,
                scopes,
                self.last_slew.at.ra,
                self.last_slew.at.dec,
                utime,
                target_roll=self.roll,
                acs_mode=self.acsmode,
            )

            if constraint_names:
                self._log_or_print(
                    utime,
                    "CONSTRAINT",
                    "%s: CONSTRAINT: RA=%s Dec=%s obsid=%s %s (%s)"
                    % (
                        unixtime2date(utime),
                        self.last_slew.at.ra,
                        self.last_slew.at.dec,
                        self.last_slew.obsid,
                        " ".join(constraint_names),
                        attitude_constraint_scope_label(scopes),
                    ),
                )
        # Check star tracker constraints
        self._check_star_tracker_constraints(utime)
        self._check_radiator_constraints(utime)
        self._check_telescope_constraints(utime)

    def _check_star_tracker_constraints(self, utime: float) -> None:
        """Check and log star tracker constraint violations for current pointing.

        Checks both hard constraints (which make pointing invalid) and soft constraints
        (which indicate degraded performance) for all configured star trackers.
        """

        star_trackers = self.config.spacecraft_bus.star_trackers

        # Skip if no star trackers configured
        if star_trackers.num_trackers() == 0:
            return

        # Get current pointing (RA, Dec, roll)
        current_ra = self.ra
        current_dec = self.dec
        current_roll = self.roll

        # Check hard constraints
        hard_violations = star_trackers.trackers_violating_hard_constraints(
            current_ra,
            current_dec,
            utime,
            current_roll,
            mode=self.acsmode,
        )

        # Check soft constraints
        soft_violations = star_trackers.any_tracker_violating_soft_constraints(
            current_ra,
            current_dec,
            utime,
            current_roll,
            mode=self.acsmode,
        )
        soft_violation_count = star_trackers.trackers_violating_soft_constraints(
            current_ra,
            current_dec,
            utime,
            current_roll,
            mode=self.acsmode,
        )

        # Update ACS state for Housekeeping telemetry
        self.star_tracker_hard_violations = hard_violations
        self.star_tracker_soft_violations = soft_violations
        num_trackers = star_trackers.num_trackers()
        # Functional = not in soft constraint (i.e. tracking at full science quality).
        # Hard-constraint violations are always faulted separately; a tracker
        # being burned by the Sun is still "not soft-constrained" but the hard
        # constraint system handles that separately.
        self.star_tracker_functional_count = num_trackers - soft_violation_count
        # Per-tracker status: True = functional (not in soft constraint)
        self.star_tracker_status = [
            not st.in_soft_constraint(current_ra, current_dec, utime, current_roll)
            for st in star_trackers.star_trackers
        ]

        # Log hard constraint violations
        if hard_violations > 0:
            functional_trackers = star_trackers.num_trackers() - hard_violations
            min_required = star_trackers.min_functional_trackers

            self._log_or_print(
                utime,
                "STAR_TRACKER_HARD_CONSTRAINT",
                f"STAR_TRACKER: HARD_CONSTRAINT: RA={current_ra:.3f}° Dec={current_dec:.3f}° "
                f"roll={current_roll:.3f}° violations={hard_violations} "
                f"functional={functional_trackers} min_required={min_required}",
            )

        # Log soft constraint violations (degraded performance)
        if soft_violations:
            self._log_or_print(
                utime,
                "STAR_TRACKER_SOFT_CONSTRAINT",
                f"STAR_TRACKER: SOFT_CONSTRAINT: RA={current_ra:.3f}° Dec={current_dec:.3f}° "
                f"roll={current_roll:.3f}° - Degraded star tracker performance",
            )

    def _check_radiator_constraints(self, utime: float) -> None:
        """Check radiator constraints and compute Sun/Earth exposure metrics."""
        radiators = self.config.spacecraft_bus.radiators

        if radiators.num_radiators() == 0:
            return

        current_ra = self.ra
        current_dec = self.dec
        current_roll = self.roll

        # Build solar panel geometry lookup so radiators can compute shadow fractions.
        solar_panel_geometries = None
        if self.config.solar_panel is not None:
            geom_map = {
                p.name: p.geometry
                for p in self.config.solar_panel.panels
                if p.geometry is not None
            }
            if geom_map:
                solar_panel_geometries = geom_map

        metrics = radiators.exposure_metrics(
            ra_deg=current_ra,
            dec_deg=current_dec,
            utime=utime,
            ephem=self.ephem,
            roll_deg=current_roll,
            solar_panel_geometries=solar_panel_geometries,
        )

        per_radiator = cast(
            list[dict[str, float | str | bool]], metrics["per_radiator"]
        )
        self.radiator_hard_violations = sum(
            1 for r in per_radiator if bool(r.get("hard_violation"))
        )
        self.radiator_sun_exposure = cast(float, metrics["sun_exposure"])
        self.radiator_earth_exposure = cast(float, metrics["earth_exposure"])
        self.radiator_heat_dissipation_w = cast(float, metrics["heat_dissipation_w"])

        if self.radiator_hard_violations > 0:
            self._log_or_print(
                utime,
                "RADIATOR_HARD_CONSTRAINT",
                f"RADIATOR: HARD_CONSTRAINT: RA={current_ra:.3f}° Dec={current_dec:.3f}° "
                f"roll={current_roll:.3f}° violations={self.radiator_hard_violations}",
            )

    def _check_telescope_constraints(self, utime: float) -> None:
        """Check telescope hard constraint for current pointing."""
        if self.constraint.telescope_hard_constraint is None:
            self.telescope_hard_violations = 0
            return
        self.telescope_hard_violations = (
            1
            if self.constraint.in_telescope_hard(
                self.ra, self.dec, utime, target_roll=self.roll
            )
            else 0
        )
        if self.telescope_hard_violations > 0:
            self._log_or_print(
                utime,
                "TELESCOPE_HARD_CONSTRAINT",
                f"TELESCOPE: HARD_CONSTRAINT: RA={self.ra:.3f}° Dec={self.dec:.3f}° "
                f"roll={self.roll:.3f}°",
            )

    def _calculate_pointing(self, utime: float) -> None:
        """Calculate current RA/Dec based on slew state or safe mode."""
        # Safe mode overrides all other pointing
        if self.in_safe_mode:
            self._calculate_safe_mode_pointing(utime)
        # If we are in a groundstations pass
        elif self.current_pass is not None:
            pass_ra, pass_dec = self.current_pass.ra_dec(utime)
            if pass_ra is not None and pass_dec is not None:
                self.ra, self.dec = pass_ra, pass_dec
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
            and self.current_slew.obstype == ObsType.SAFE
            and self.current_slew.is_slewing(utime)
        ):
            self.ra, self.dec = self.current_slew.ra_dec(utime)
            self.roll = self.current_slew.roll(utime)
        else:
            # After slew completes or for continuous tracking, maintain optimal pointing
            self.ra = target_ra
            self.dec = target_dec

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
        self, utime: float, ra: float, dec: float, roll: float, obsid: int
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
            roll=roll,
            obsid=obsid,
        )
        self.enqueue_command(command)
        self._log_or_print(
            utime,
            "CHARGING",
            f"Battery charge requested at RA={ra:.2f} Dec={dec:.2f} Roll={roll:.2f} obsid={obsid}",
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

    def initiate_emergency_charging(
        self,
        utime: float,
        ephem: rust_ephem.Ephemeris,
        emergency_charging: EmergencyCharging,
        lastra: float,
        lastdec: float,
        current_ppt: "Pointing | None",
    ) -> "tuple[float, float, Pointing | None]":
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
                utime,
                charging_ppt.ra,
                charging_ppt.dec,
                charging_ppt.roll,
                charging_ppt.obsid,
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
                command.ra,
                command.dec,
                command.obsid,
                utime,
                obstype=ObsType.CHARGE,
                roll=command.roll,
            )

    def cancel_pending_battery_charge(self, utime: float) -> bool:
        """Drop a not-yet-executed START_BATTERY_CHARGE command from the queue.

        Emergency charging can be initiated and then immediately terminated within
        the same planning step (e.g. the charging pointing is itself constrained).
        In that case the START_BATTERY_CHARGE enqueued at initiation has not run
        yet — letting it execute on the next step would start a charge slew (and
        cancel any science slew queued in the meantime) for a charge session that
        was already abandoned.  Returns True if such a pending command was removed.

        QueueDITL invariant: _initiate_charging (the only enqueue site) is gated
        on charging_ppt is None, and charging_ppt is not cleared until
        _terminate_charging_ppt runs.  Therefore at most one START_BATTERY_CHARGE
        can be in the queue at any time; the assert below enforces this.
        """
        pending = [
            command
            for command in self.command_queue
            if command.command_type == ACSCommandType.START_BATTERY_CHARGE
        ]
        assert len(pending) <= 1, (
            f"Invariant violated: {len(pending)} START_BATTERY_CHARGE commands "
            "in queue; expected at most 1"
        )
        if not pending:
            return False
        self.command_queue = [
            command
            for command in self.command_queue
            if command.command_type != ACSCommandType.START_BATTERY_CHARGE
        ]
        self._log_or_print(
            utime,
            "CHARGING",
            f"{unixtime2date(utime)}: Canceled pending START_BATTERY_CHARGE - "
            "charge session abandoned before it started",
        )
        return True

    def _end_battery_charge(self, utime: float) -> None:
        """Handle END_BATTERY_CHARGE command execution."""
        self._log_or_print(utime, "CHARGING", "Ending battery charge")

        # Clear the charging slew state immediately so _is_in_charging_mode returns False
        # This prevents staying in CHARGING mode while slewing back to science
        if self.last_slew is not None and self.last_slew.obstype == ObsType.CHARGE:
            self.last_slew = None
        self.science_observation_active = False

        # Queue-driven scheduling owns the next science target. Returning to the
        # previous PPT here can resurrect a closed observation after charging.
        if self.last_ppt is not None:
            self._log_or_print(
                utime,
                "CHARGING",
                "Charge complete; awaiting next target command",
            )
