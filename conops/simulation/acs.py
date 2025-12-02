from typing import TYPE_CHECKING, Any

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
from ..targets import Pointing
from .acs_command import ACSCommand
from .passes import Pass
from .slew import Slew

if TYPE_CHECKING:
    from ..ditl.ditl_log import DITLLog


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
        self.ra = (180 + self.ephem.earth[0].ra.deg) % 360
        self.dec = -self.ephem.earth[0].dec.deg
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
            handlers = {
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
            safe_ra = self.ephem.sun[index].ra.deg
            safe_dec = self.ephem.sun[index].dec.deg

        self._log_or_print(
            utime,
            "SAFE",
            f"{unixtime2date(utime)}: Initiating slew to safe mode pointing at RA={safe_ra:.2f} Dec={safe_dec:.2f}",
        )
        # Enqueue slew to safe pointing with a special obsid
        self._enqueue_slew(safe_ra, safe_dec, obsid=-999, utime=utime, obstype="SAFE")

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
        slew.calc_slewtime()

        self._log_or_print(
            utime,
            "SLEW",
            f"{unixtime2date(utime)}: Starting slew from RA={self.ra:.2f} Dec={self.dec:.2f} "
            f"to RA={slew.endra:.2f} Dec={slew.enddec:.2f} (duration: {slew.slewtime:.1f}s)",
        )

        self.current_slew = slew
        self.last_slew = slew

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
            execution_time = utime  # Execute immediately
        else:
            # Set up target observation request and check visibility
            target_request = self._create_target_request(slew, utime)
            slew.at = target_request

            visstart = target_request.next_vis(utime)
            is_first_slew = self._initialize_slew_positions(slew, utime)

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

    def _create_target_request(self, slew: Slew, utime: float) -> Pointing:
        """Create and configure a target observation request for visibility checking."""
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
        self.in_eclipse = self.constraint.in_eclipse(ra=0, dec=0, time=utime)  # type: ignore[assignment]

        # Process any commands scheduled for execution at or before current time
        self._process_commands(utime)

        # Update ACS mode based on current state
        self._update_mode(utime)

        # Check current constraints
        self._check_constraints(utime)

        # Calculate current RA/Dec pointing
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
            self.ra, self.dec = self.last_slew.ra_dec(utime)  # type: ignore[assignment]
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
            target_ra = self.ephem.sun[index].ra.deg
            target_dec = self.ephem.sun[index].dec.deg

        # If actively slewing to safe mode position, use slew interpolation
        if (
            self.current_slew is not None
            and self.current_slew.obstype == "SAFE"
            and self.current_slew.is_slewing(utime)
        ):
            self.ra, self.dec = self.current_slew.ra_dec(utime)  # type: ignore[assignment]
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

    def initiate_emergency_charging(
        self,
        utime: float,
        ephem,
        emergency_charging,
        lastra: float,
        lastdec: float,
        current_ppt,
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
