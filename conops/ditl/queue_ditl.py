from datetime import datetime, timezone
from typing import Any

import numpy as np
import rust_ephem
from pydantic import BaseModel

from ..common import ACSMode, angular_separation, unixtime2date
from ..common.enums import ACSCommandType
from ..config import MissionConfig
from ..simulation.acs_command import ACSCommand
from ..simulation.emergency_charging import EmergencyCharging
from ..simulation.slew import Slew
from ..targets import Plan, Pointing, Queue
from .ditl_log import DITLLog
from .ditl_mixin import DITLMixin
from .ditl_stats import DITLStats


class TOORequest(BaseModel):
    """A Target of Opportunity (TOO) request waiting to be executed.

    Attributes:
        obsid: Unique observation identifier for this TOO
        ra: Right ascension in degrees
        dec: Declination in degrees
        merit: Priority merit value (higher = more urgent)
        exptime: Requested exposure time in seconds
        name: Human-readable name for the TOO target
        submit_time: Unix timestamp when the TOO becomes active. If 0.0,
            the TOO is active immediately from simulation start.
        executed: Whether this TOO has been executed
    """

    obsid: int
    ra: float
    dec: float
    merit: float
    exptime: int
    name: str
    submit_time: float = 0.0
    executed: bool = False


class QueueDITL(DITLMixin, DITLStats):
    """
    Class to run a Day In The Life (DITL) simulation based on a target
    Queue. Rather than creating a observing plan and then running it, this
    dynamically pulls a target off the Queue when the current target is done.
    Therefore this is simulating a queue scheduled telescope. However, this
    makes for a very simple DITL simulator as we don't have to create a
    separate Plan first.
    """

    # QueueDITL-specific type definitions (types defined in DITLMixin are inherited)
    ppt: Pointing | None  # Override to use Pointing instead of PlanEntry
    charging_ppt: Pointing | None
    emergency_charging: EmergencyCharging
    utime: list[float]  # Override to specify float instead of generic list
    ephem: rust_ephem.Ephemeris  # Override to make non-optional
    queue: Queue
    too_register: list[TOORequest]  # Register of pending TOO requests

    def __init__(
        self,
        config: MissionConfig,
        ephem: rust_ephem.Ephemeris | None = None,
        begin: datetime | None = None,
        end: datetime | None = None,
        queue: Queue | None = None,
    ) -> None:
        # Initialize mixin
        DITLMixin.__init__(self, config=config, ephem=ephem, begin=begin, end=end)

        # Current target (already set in mixin but repeated for clarity)
        self.ppt = None

        # Pointing history
        self.ra = list()
        self.dec = list()
        self.roll = list()
        self.mode = list()
        self.obsid = list()
        self.plan = Plan()

        # Power and battery history
        self.panel = list()
        self.batterylevel = list()
        self.charge_state = list()
        self.power = list()
        self.panel_power = list()
        # Subsystem power tracking
        self.power_bus = list()
        self.power_payload = list()
        self.mtq_power: list[float] = []
        # Data recorder tracking
        self.recorder_volume_gb = list()
        self.recorder_fill_fraction = list()
        self.recorder_alert = list()
        self.data_generated_gb = list()
        self.data_downlinked_gb = list()

        # Event log
        self.log = DITLLog()

        # TOO (Target of Opportunity) register - holds pending TOOs
        self.too_register: list[TOORequest] = []

        # Target Queue (use provided queue or create default)
        if queue is not None:
            self.queue = queue
            if self.queue.log is None:
                self.queue.log = self.log
        else:
            self.queue = Queue(
                config=self.config,
                log=self.log,
                ephem=self.ephem,
            )

        # Wire log into ACS so it can log events (if ACS exists)
        if hasattr(self, "acs"):
            self.acs.log = self.log

        # Initialize emergency charging manager (will be fully set up after ACS is available)
        self.charging_ppt = None
        self.emergency_charging = EmergencyCharging(
            config=self.config,
            starting_obsid=999000,
            log=self.log,
        )

    def get_acs_queue_status(self) -> dict[str, Any]:
        """
        Get the current status of the ACS command queue.

        Returns a dictionary with queue diagnostics useful for debugging
        the queue-driven state machine.
        """
        return {
            "queue_size": len(self.acs.command_queue),
            "pending_commands": [
                {
                    "type": cmd.command_type.name,
                    "execution_time": cmd.execution_time,
                    "time_formatted": unixtime2date(cmd.execution_time),
                }
                for cmd in self.acs.command_queue
            ],
            "current_slew": type(self.acs.current_slew).__name__
            if self.acs.current_slew
            else None,
            "acs_mode": self.acs.acsmode.name,
        }

    def submit_too(
        self,
        obsid: int,
        ra: float,
        dec: float,
        merit: float,
        exptime: int,
        name: str,
        submit_time: float | datetime | None = None,
    ) -> TOORequest:
        """Submit a Target of Opportunity (TOO) request.

        The TOO is added to a special register and will be checked during
        the simulation. When the TOO's merit is higher than the current
        observation's merit and the TOO target is visible, the current
        observation will be abandoned and the TOO will be observed immediately.

        The TOO can be scheduled before the DITL starts running by setting
        `submit_time` to a future time. The TOO will not become active until
        that time is reached during the simulation.

        Args:
            obsid: Unique observation identifier for this TOO
            ra: Right ascension in degrees
            dec: Declination in degrees
            merit: Priority merit value (higher = more urgent). Should be higher
                   than normal queue targets to ensure immediate observation.
            exptime: Requested exposure time in seconds
            name: Human-readable name for the TOO target
            submit_time: When the TOO becomes active. Can be:
                - None: TOO is active immediately from simulation start
                - float: Unix timestamp when TOO becomes active
                - datetime: Datetime when TOO becomes active (will be converted to Unix timestamp)

        Returns:
            The created TOORequest object

        Example:
            >>> # TOO active immediately
            >>> ditl.submit_too(
            ...     obsid=1000001,
            ...     ra=180.0,
            ...     dec=45.0,
            ...     merit=10000.0,
            ...     exptime=3600,
            ...     name="GRB 250101A",
            ... )

            >>> # TOO scheduled for 1 hour into the simulation (using Unix timestamp)
            >>> ditl.submit_too(
            ...     obsid=1000002,
            ...     ra=90.0,
            ...     dec=-30.0,
            ...     merit=10000.0,
            ...     exptime=1800,
            ...     name="GRB 250101B",
            ...     submit_time=ditl.ustart + 3600,
            ... )

            >>> # TOO scheduled for a specific datetime
            >>> from datetime import datetime
            >>> ditl.submit_too(
            ...     obsid=1000003,
            ...     ra=270.0,
            ...     dec=60.0,
            ...     merit=10000.0,
            ...     exptime=2400,
            ...     name="GRB 250101C",
            ...     submit_time=datetime(2025, 11, 1, 12, 0, 0),
            ... )
        """
        # Convert datetime to Unix timestamp if needed
        if isinstance(submit_time, datetime):
            # Ensure timezone-aware datetime
            if submit_time.tzinfo is None:
                submit_time = submit_time.replace(tzinfo=timezone.utc)
            effective_submit_time = submit_time.timestamp()
        elif submit_time is not None:
            effective_submit_time = submit_time
        else:
            # None means active immediately (use 0.0 which is always <= any simulation time)
            effective_submit_time = 0.0

        too = TOORequest(
            obsid=obsid,
            ra=ra,
            dec=dec,
            merit=merit,
            exptime=exptime,
            name=name,
            submit_time=effective_submit_time,
            executed=False,
        )
        self.too_register.append(too)
        return too

    def _check_too_interrupt(self, utime: float, ra: float, dec: float) -> bool:
        """Check if a pending TOO should interrupt the current observation.

        A TOO will interrupt the current observation if:
        1. The TOO has been submitted (submit_time <= utime)
        2. The TOO has not yet been executed
        3. The TOO target is currently visible
        4. The TOO's merit is higher than the current PPT's merit

        Args:
            utime: Current simulation time
            ra: Current spacecraft RA
            dec: Current spacecraft Dec

        Returns:
            True if a TOO interrupt occurred, False otherwise
        """
        # Get pending TOOs that are ready and not executed
        pending_toos = [
            too
            for too in self.too_register
            if too.submit_time <= utime and not too.executed
        ]

        if not pending_toos:
            return False

        # Get current PPT merit (if any)
        current_merit = self.ppt.merit if self.ppt is not None else -float("inf")

        # Check each pending TOO
        for too in pending_toos:
            # Skip if TOO merit is not higher than current
            if too.merit <= current_merit:
                continue

            # Create a temporary Pointing to check visibility
            too_pointing = Pointing(
                config=self.config,
                ra=too.ra,
                dec=too.dec,
                obsid=too.obsid,
                name=too.name,
                merit=too.merit,
                exptime=too.exptime,
            )
            too_pointing.visibility()

            # Check if TOO is currently visible
            if not too_pointing.visible(utime, utime):
                continue

            # TOO should interrupt! Log the event
            self.log.log_event(
                utime=utime,
                event_type="TOO",
                description=f"TOO interrupt: {too.name} (obsid={too.obsid}, merit={too.merit}) "
                f"preempting current observation (merit={current_merit})",
                obsid=too.obsid,
                acs_mode=self.acs.acsmode,
            )

            # Terminate current observation if any
            if self.ppt is not None:
                self._terminate_ppt(
                    utime,
                    reason=f"Preempted by TOO {too.name} (obsid={too.obsid})",
                    mark_done=False,  # Don't mark as done, it was interrupted
                )

            # Add TOO to queue with boosted merit to ensure immediate observation
            # Use merit + 100000 to guarantee it's selected next
            boosted_merit = too.merit + 100000.0
            self.queue.add(
                ra=too.ra,
                dec=too.dec,
                obsid=too.obsid,
                name=too.name,
                merit=boosted_merit,
                exptime=too.exptime,
            )

            self.log.log_event(
                utime=utime,
                event_type="TOO",
                description=f"Added TOO {too.name} to queue with boosted merit {boosted_merit}",
                obsid=too.obsid,
                acs_mode=self.acs.acsmode,
            )

            # Mark TOO as executed
            too.executed = True

            # Fetch the TOO as the new PPT
            self._fetch_new_ppt(utime, ra, dec)

            return True

        return False

    def calc(self) -> bool:
        """
        Run the DITL (Day In The Life) simulation.

        This simulation uses a queue-driven ACS (Attitude Control System) where
        spacecraft state transitions (slews, passes, etc.) are managed through
        a command queue, providing explicit, traceable control flow.
        """
        # If begin/end datetimes are naive, assume UTC by making them timezone-aware
        if self.begin.tzinfo is None:
            self.begin = self.begin.replace(tzinfo=timezone.utc)
        if self.end.tzinfo is None:
            self.end = self.end.replace(tzinfo=timezone.utc)

        # Check that ephemeris is set
        assert self.ephem is not None, "Ephemeris must be set before running DITL"

        # Set step_size from ephem
        self.step_size = self.ephem.step_size

        # Set ACS ephemeris if not already set
        if self.acs.ephem is None:
            self.acs.ephem = self.ephem

        # If safe mode already requested, enqueue immediately and exit
        if (
            getattr(self.config.fault_management, "safe_mode_requested", False)
            and not self.acs.in_safe_mode
        ):
            safe_cmd = ACSCommand(
                command_type=ACSCommandType.ENTER_SAFE_MODE,
                execution_time=self.begin.timestamp(),
            )
            self.acs.enqueue_command(safe_cmd)
            self.config.fault_management.safe_mode_requested = False
            return True

        # Set up timing and schedule passes
        if not self._setup_simulation_timing():
            return False

        # Schedule groundstation passes (these will be queued in ACS)
        self._schedule_groundstation_passes()

        # Set up simulation length from begin/end datetimes
        simlen = int((self.end - self.begin).total_seconds() / self.step_size)

        # DITL loop
        for i in range(simlen):
            utime = self.ustart + i * self.step_size

            # Track PPT in timeline
            self._track_ppt_in_timeline()

            # Get current pointing and mode from ACS
            ra, dec, roll, obsid = self.acs.pointing(utime)
            mode = self.acs.get_mode(utime)

            # If safe mode requested, prioritize it and skip other ops this step
            if (
                getattr(self.config.fault_management, "safe_mode_requested", False)
                and not self.acs.in_safe_mode
            ):
                safe_cmd = ACSCommand(
                    command_type=ACSCommandType.ENTER_SAFE_MODE,
                    execution_time=utime,
                )
                self.acs.enqueue_command(safe_cmd)
                # Prevent duplicate requests
                self.config.fault_management.safe_mode_requested = False
                continue

            # Check pass timing and manage passes
            self._check_and_manage_passes(utime, ra, dec)

            # Handle spacecraft operations based on current mode
            self._handle_mode_operations(mode, utime, ra, dec)

            # Close PPT timeline segment if no active observation
            self._close_ppt_timeline_if_needed(utime)

            # Re-query mode after operations to capture any newly enqueued commands
            # This prevents 1-step mode gaps when commands are enqueued during this step
            mode = self.acs.get_mode(utime)

            # Record pointing and mode
            self._record_pointing_data(ra, dec, roll, obsid, mode)

            # Calculate and record power data
            self._record_power_data(
                i, utime, ra, dec, mode, in_eclipse=self.acs.in_eclipse
            )

            # Record reaction wheel resource usage
            self._record_wheel_resource(utime)
            self._maybe_request_desat(utime)
            # Track desat active time
            if getattr(self.acs, "_desat_active", False):
                self.desat_time_steps += 1
            # Record disturbance torques (magnitudes) if available
            if hasattr(self.acs, "_last_disturbance_components"):
                comps = self.acs._last_disturbance_components or {}

                def _safe(val: Any) -> float:
                    try:
                        return float(val)
                    except Exception:
                        return 0.0

                self.disturbance_total.append(_safe(comps.get("total", 0.0)))
                self.disturbance_gg.append(_safe(comps.get("gg", 0.0)))
                self.disturbance_drag.append(_safe(comps.get("drag", 0.0)))
                self.disturbance_srp.append(_safe(comps.get("srp", 0.0)))
                self.disturbance_mag.append(_safe(comps.get("mag", 0.0)))
                vec = comps.get("vector", (0.0, 0.0, 0.0))
                if isinstance(vec, (list, tuple)) and len(vec) == 3:
                    try:
                        self.disturbance_vec.append(
                            (float(vec[0]), float(vec[1]), float(vec[2]))
                        )
                    except Exception:
                        self.disturbance_vec.append((0.0, 0.0, 0.0))
                else:
                    self.disturbance_vec.append((0.0, 0.0, 0.0))
            else:
                self.disturbance_total.append(0.0)
                self.disturbance_gg.append(0.0)
                self.disturbance_drag.append(0.0)
                self.disturbance_srp.append(0.0)
                self.disturbance_mag.append(0.0)
                self.disturbance_vec.append((0.0, 0.0, 0.0))

            # Handle data generation and downlink
            self._handle_data_management(utime, mode)

            # Fault management checks (e.g., battery level thresholds)
            self._handle_fault_management(utime)

        # Make sure the last PPT of the day ends (if any)
        if self.plan:
            self.plan[-1].end = utime

        # Capture ACS momentum management counters
        if hasattr(self.acs, "desat_events"):
            self.desat_event_count = self.acs.desat_events
        if hasattr(self.acs, "desat_requests"):
            self.desat_request_count = self.acs.desat_requests
        if hasattr(self.acs, "headroom_rejects"):
            self.headroom_rejects = self.acs.headroom_rejects

        return True

    def _handle_data_management(self, utime: float, mode: ACSMode) -> None:
        """Handle data generation during observations and downlink during passes."""
        # Use the mixin method to process data generation and downlink
        data_generated, data_downlinked = self._process_data_management(
            utime, mode, self.step_size
        )

        # Record data telemetry (cumulative values)
        prev_generated = self.data_generated_gb[-1] if self.data_generated_gb else 0.0
        prev_downlinked = (
            self.data_downlinked_gb[-1] if self.data_downlinked_gb else 0.0
        )

        self.recorder_volume_gb.append(self.recorder.current_volume_gb)
        self.recorder_fill_fraction.append(self.recorder.get_fill_fraction())
        self.recorder_alert.append(self.recorder.get_alert_level())
        self.data_generated_gb.append(prev_generated + data_generated)
        self.data_downlinked_gb.append(prev_downlinked + data_downlinked)

    def _handle_fault_management(self, utime: float) -> None:
        """Handle fault management checks and safe mode requests."""
        if self.config.fault_management is not None:
            self.config.fault_management.check(
                values={
                    "battery_level": self.battery.battery_level,
                    "recorder_fill_fraction": self.recorder.get_fill_fraction(),
                },
                utime=utime,
                step_size=self.step_size,
                acs=self.acs,
                ephem=self.ephem,
                ra=self.ra[-1] if self.ra else None,
                dec=self.dec[-1] if self.dec else None,
            )
            # Check if safe mode has been requested by fault management
            if (
                self.config.fault_management.safe_mode_requested
                and not self.acs.in_safe_mode
            ):
                command = ACSCommand(
                    command_type=ACSCommandType.ENTER_SAFE_MODE,
                    execution_time=utime,
                )
                self.acs.enqueue_command(command)

    def _track_ppt_in_timeline(self) -> None:
        """Track the start of a new PPT in the plan timeline."""
        if self.ppt is not None and (
            len(self.plan) == 0 or self.ppt.begin != self.plan[-1].begin
        ):
            # Before adding new PPT, close the previous one if it has placeholder end time
            if len(self.plan) > 0:
                last_entry = self.plan[-1]
                # Check if end time looks like a placeholder (>= 86400 seconds from begin)
                # Charging PPTs use exactly 86400, science obs use larger values
                if last_entry.end >= last_entry.begin + 86400:
                    # Set end to the begin time of new PPT (no gap between entries)
                    self.plan[-1].end = self.ppt.begin

            self.plan.append(self.ppt.copy())

    def _close_ppt_timeline_if_needed(self, utime: float) -> None:
        """Close the last PPT segment in timeline if no active observation.

        This is a safety net to ensure plan timeline is closed if ppt becomes None
        and the end time hasn't been set yet (e.g., has placeholder value).
        """
        if self.ppt is None and len(self.plan) > 0:
            last_entry = self.plan[-1]
            # Check if end time looks like a placeholder (>= 86400 seconds from begin)
            # Charging PPTs use exactly 86400, science obs use larger values
            if last_entry.end >= last_entry.begin + 86400:
                self.plan[-1].end = utime

    def _handle_mode_operations(
        self, mode: ACSMode, utime: float, ra: float, dec: float
    ) -> None:
        """Handle spacecraft operations based on current mode."""
        if mode == ACSMode.PASS:
            self._handle_pass_mode(utime)
        elif mode == ACSMode.CHARGING:
            self._handle_charging_mode(utime)
        else:
            # Science or SAA modes: handle observations and battery management
            self._handle_science_mode(utime, ra, dec, mode)

    def _handle_science_mode(
        self, utime: float, ra: float, dec: float, mode: ACSMode
    ) -> None:
        """Handle science mode operations: charging, observations, and target acquisition."""
        # Check for battery alert and initiate emergency charging if needed
        if self._should_initiate_charging(utime):
            self._initiate_charging(utime, ra, dec)

        # Check for TOO interrupts (before managing PPT lifecycle)
        # This allows TOOs to preempt ongoing observations
        if self._check_too_interrupt(utime, ra, dec):
            return  # TOO took over, skip normal PPT handling

        # Manage current science PPT lifecycle
        self._manage_ppt_lifecycle(utime, mode)

        # Fetch new PPT if none is active
        if self.ppt is None:
            self._fetch_new_ppt(utime, ra, dec)

    def _should_initiate_charging(self, utime: float) -> bool:
        """Check if emergency charging should be initiated."""
        return (
            self.charging_ppt is None
            and self.emergency_charging.should_initiate_charging(
                utime, self.ephem, self.battery.battery_alert
            )
        )

    def _initiate_charging(self, utime: float, ra: float, dec: float) -> None:
        """Initiate emergency charging by creating charging PPT and sending command to ACS."""
        self.charging_ppt = self.emergency_charging.initiate_emergency_charging(
            utime, self.ephem, ra, dec, self.ppt
        )

        # If charging PPT created successfully, send command to ACS and replace current PPT
        if self.charging_ppt is not None:
            command = ACSCommand(
                command_type=ACSCommandType.START_BATTERY_CHARGE,
                execution_time=utime,
                ra=self.charging_ppt.ra,
                dec=self.charging_ppt.dec,
                obsid=self.charging_ppt.obsid,
            )
            self.acs.enqueue_command(command)
            self.ppt = self.charging_ppt

    def _setup_simulation_timing(self) -> bool:
        """Set up timing aspect of simulation."""
        self.ustart = self.begin.timestamp()
        self.uend = self.end.timestamp()
        # Check that the start/end times fall within the ephemeris
        # Ephemeris uses timestamp attribute which is a list of datetime objects
        if (
            self.begin not in self.ephem.timestamp
            or self.end not in self.ephem.timestamp
        ):
            raise ValueError("ERROR: Ephemeris does not cover simulation date range")

        self.utime = (
            np.arange(self.ustart, self.uend, self.step_size).astype(float).tolist()
        )
        return True

    def _schedule_groundstation_passes(self) -> None:
        """Populate groundstation passes for the simulation window."""
        if (
            self.acs.passrequests.passes is None
            or len(self.acs.passrequests.passes) == 0
        ):
            self.log.log_event(
                utime=self.ustart,
                event_type="INFO",
                description="Scheduling groundstation passes...",
            )
            # Extract year and day-of-year from begin datetime
            year = self.begin.year
            day = self.begin.timetuple().tm_yday
            # Calculate length in days from begin/end
            # Use at least 1 day to avoid empty pass windows on short sims
            length = max(1, int((self.end - self.begin).total_seconds() / 86400))
            self.acs.passrequests.get(year, day, length)
            if self.acs.passrequests.passes:
                for p in self.acs.passrequests.passes:
                    self.log.log_event(
                        utime=self.ustart,
                        event_type="PASS",
                        description=f"Scheduled pass: {p}",
                    )
            else:
                self.log.log_event(
                    utime=self.ustart,
                    event_type="INFO",
                    description="No groundstation passes scheduled.",
                )

    def _check_and_manage_passes(self, utime: float, ra: float, dec: float) -> None:
        """Check pass timing and send appropriate commands to ACS."""

        # Check if we're in a pass, if yes, command ACS to start the pass
        current_pass = self.acs.passrequests.current_pass(utime)
        if current_pass is not None and self.acs.acsmode != ACSMode.PASS:
            self.log.log_event(
                utime=utime,
                event_type="PASS",
                description="In pass, commanding ACS to start pass",
                acs_mode=self.acs.acsmode,
            )
            command = ACSCommand(
                command_type=ACSCommandType.START_PASS,
                execution_time=utime,
            )
            self.acs.enqueue_command(command)
            return

        # Check if a pass just ended, if yes, command ACS to end the pass.
        # FIXME: This works but isn't super clean.
        if (
            self.acs.passrequests.current_pass(utime - self.ephem.step_size)
            and current_pass is None
        ):
            self.log.log_event(
                utime=utime,
                event_type="PASS",
                description="Pass ended, commanding ACS to end pass",
                acs_mode=self.acs.acsmode,
            )
            command = ACSCommand(
                command_type=ACSCommandType.END_PASS,
                execution_time=utime,
            )
            self.acs.enqueue_command(command)
            return

        # Check to see if it's time to slew to the next pass
        # Skip if already in PASS or SLEWING mode - can't interrupt a slew mid-motion.
        # Pass-aware scheduling in _fetch_new_ppt prevents conflicts.
        if self.acs.acsmode in (ACSMode.PASS, ACSMode.SLEWING):
            return

        next_pass = self.acs.passrequests.next_pass(utime)

        # If there's no next pass, nothing to do
        if next_pass is None:
            return

        # Check if it's time to start slewing for the next pass
        if next_pass.time_to_slew(utime=utime, ra=ra, dec=dec):
            # If it's time to slew, enqueue the slew command
            self.log.log_event(
                utime=utime,
                event_type="SLEW",
                description=f"Slewing for pass to {next_pass.station}",
                acs_mode=self.acs.acsmode,
            )

            # Create slew object for the pass
            slew = Slew(
                config=self.config,
            )

            slew.startra = ra
            slew.startdec = dec
            slew.endra = next_pass.gsstartra
            slew.enddec = next_pass.gsstartdec
            slew.obstype = "GSP"  # Ground Station Pass slew
            command = ACSCommand(
                command_type=ACSCommandType.SLEW_TO_TARGET,
                execution_time=utime,
                slew=slew,
            )
            self.acs.enqueue_command(command)
            return

    def _handle_pass_mode(self, utime: float) -> None:
        """Handle spacecraft behavior during ground station passes."""
        # Terminate any active observations during passes
        self._terminate_science_ppt_for_pass(utime)
        if self.charging_ppt is not None:
            self._terminate_charging_ppt(utime)

    def _handle_charging_mode(self, utime: float) -> None:
        """Monitor battery and constraints during emergency charging."""
        # Sync state for legacy test compatibility
        self._sync_charging_state()

        # Check if charging should terminate
        termination_reason = self.emergency_charging.check_termination(
            utime, self.battery, self.ephem
        )
        if termination_reason is not None:
            self._terminate_emergency_charging(termination_reason, utime)
            # Immediately fetch a new PPT to avoid 1-step SCIENCE gap
            # Get current pointing from ACS
            ra, dec = self.acs.ra, self.acs.dec
            self._fetch_new_ppt(utime, ra, dec)

    def _sync_charging_state(self) -> None:
        """Synchronize emergency_charging module state with queue state."""
        if (
            self.charging_ppt is not None
            and self.emergency_charging.current_charging_ppt is None
        ):
            self.emergency_charging.current_charging_ppt = self.charging_ppt

    def _manage_ppt_lifecycle(self, utime: float, mode: ACSMode) -> None:
        """Manage the lifecycle of the current pointing (PPT)."""
        if self.ppt is None:
            return

        # Handle charging PPT constraint checks (regardless of mode)
        if self.ppt == self.charging_ppt:
            # Check constraints for charging PPT even if mode hasn't transitioned yet
            if self.constraint.in_constraint(self.ppt.ra, self.ppt.dec, utime):
                constraint_name = self._get_constraint_name(
                    self.ppt.ra, self.ppt.dec, utime
                )
                self.log.log_event(
                    utime=utime,
                    event_type="CHARGING",
                    description=f"Charging PPT {constraint_name} constrained, terminating",
                    obsid=self.ppt.obsid,
                    acs_mode=self.acs.acsmode,
                )
                self._terminate_emergency_charging("constraint", utime)
            return

        # Decrement exposure time when actively observing
        if mode == ACSMode.SCIENCE:
            self._decrement_exposure_time()

        # Check termination conditions
        self._check_ppt_termination(utime)

    def _decrement_exposure_time(self) -> None:
        """Decrement PPT exposure time by one timestep."""
        assert self.ppt is not None
        assert self.ppt.exptime is not None, "Exposure time should not be None here"
        self.ppt.exptime -= self.step_size

    def _check_ppt_termination(self, utime: float) -> None:
        """Check if PPT should terminate due to constraints, completion, or timeout."""
        assert self.ppt is not None

        # Enforce minimum snapshot dwell unless constrained
        try:
            begin = float(getattr(self.ppt, "begin", utime))
        except (TypeError, ValueError):
            begin = utime
        elapsed = utime - begin
        try:
            min_dwell = float(getattr(self.ppt, "ss_min", 0.0) or 0.0)
        except (TypeError, ValueError):
            min_dwell = 0.0

        if self.constraint.in_constraint(self.ppt.ra, self.ppt.dec, utime):
            constraint_name = self._get_constraint_name(
                self.ppt.ra, self.ppt.dec, utime
            )
            self._terminate_ppt(
                utime,
                reason=f"Target {constraint_name} constrained, ending observation",
            )
        elif self.ppt.exptime is None or self.ppt.exptime <= 0:
            if elapsed >= min_dwell:
                self._terminate_ppt(
                    utime,
                    reason="Exposure complete, ending observation",
                    mark_done=True,
                )
        elif utime >= self.ppt.end:
            if elapsed >= min_dwell:
                self._terminate_ppt(
                    utime, reason="Time window elapsed, ending observation"
                )

    def _terminate_ppt(
        self, utime: float, reason: str, mark_done: bool = False
    ) -> None:
        """Terminate the current PPT.

        Parameters
        ----------
        utime : float
            Current time
        reason : str
            Reason for termination (for logging)
        mark_done : bool
            Whether to mark the PPT as done
        """
        assert self.ppt is not None
        self.log.log_event(
            utime=utime,
            event_type="OBSERVATION",
            description=reason,
            obsid=self.ppt.obsid,
            acs_mode=self.acs.acsmode,
        )

        # Update plan timeline with actual end time
        if len(self.plan) > 0:
            self.plan[-1].end = utime

        if mark_done:
            self.ppt.done = True

        self.ppt = None
        self.acs.last_slew = None

    def _get_constraint_name(self, ra: float, dec: float, utime: float) -> str:
        """Determine which constraint is violated."""
        if self.constraint.in_earth(ra, dec, utime):
            return "Earth Limb"
        elif self.constraint.in_moon(ra, dec, utime):
            return "Moon"
        elif self.constraint.in_sun(ra, dec, utime):
            return "Sun"
        elif self.constraint.in_panel(ra, dec, utime):
            return "Panel"
        return "Unknown"

    def _fetch_new_ppt(self, utime: float, ra: float, dec: float) -> None:
        """Fetch a new pointing target from the queue and enqueue slew command."""
        # Don't issue science slews during an active pass - this prevents the
        # teleportation bug where a slew is issued with start position from the
        # pass tracking, but the spacecraft continues following the pass instead.
        # When the pass ends, the stale slew would report incorrect positions.
        current_pass = self.acs.passrequests.current_pass(utime)
        if current_pass is not None:
            self.log.log_event(
                utime=utime,
                event_type="QUEUE",
                description="Deferring PPT fetch - pass in progress",
                acs_mode=self.acs.acsmode,
            )
            return

        self.log.log_event(
            utime=utime,
            event_type="QUEUE",
            description=f"Fetching new PPT from Queue (last RA/Dec {ra:.2f}/{dec:.2f})",
            acs_mode=self.acs.acsmode,
        )

        self.ppt = self.queue.get(ra, dec, utime)

        if self.ppt is not None:
            self.log.log_event(
                utime=utime,
                event_type="QUEUE",
                description=f"Fetched PPT: {self.ppt}",
                obsid=self.ppt.obsid,
                acs_mode=self.acs.acsmode,
            )

            # Create and configure a Slew object
            slew = Slew(
                config=self.config,
            )
            slew.ephem = self.acs.ephem
            slew.slewrequest = utime
            slew.endra = self.ppt.ra
            slew.enddec = self.ppt.dec
            slew.obstype = "PPT"
            slew.obsid = self.ppt.obsid

            # Use the PPT from the queue which already has visibility calculated
            slew.at = self.ppt

            # Check if target is visible
            visstart = self.ppt.next_vis(utime)
            if not visstart and slew.obstype == "PPT":
                self.log.log_event(
                    utime=utime,
                    event_type="SLEW",
                    description="Slew rejected - target not visible",
                    obsid=self.ppt.obsid,
                    acs_mode=self.acs.acsmode,
                )
                return

            # Initialize slew start positions from current ACS pointing
            slew.startra = self.acs.ra
            slew.startdec = self.acs.dec

            # Calculate slew timing
            execution_time = utime

            # Wait for current slew to finish if in progress
            if (
                self.acs.last_slew is not None
                and isinstance(self.acs.last_slew, Slew)
                and self.acs.last_slew.is_slewing(utime)
            ):
                execution_time = (
                    self.acs.last_slew.slewstart + self.acs.last_slew.slewtime
                )
                self.log.log_event(
                    utime=utime,
                    event_type="SLEW",
                    description=f"Slewing - delaying next slew until {unixtime2date(execution_time)}",
                    obsid=self.ppt.obsid,
                    acs_mode=self.acs.acsmode,
                )

            # Wait for target visibility if constrained
            if visstart and visstart > execution_time and slew.obstype == "PPT":
                self.log.log_event(
                    utime=utime,
                    event_type="SLEW",
                    description=f"Slew delayed by {visstart - execution_time:.1f}s",
                    obsid=self.ppt.obsid,
                    acs_mode=self.acs.acsmode,
                )
                execution_time = visstart

            slew.slewstart = execution_time
            slew.calc_slewtime()

            # Verify slew won't overlap with a pass - check both start and end
            slew_end = execution_time + slew.slewtime
            if self.acs.passrequests.current_pass(execution_time) is not None:
                self.log.log_event(
                    utime=utime,
                    event_type="SLEW",
                    description="Slew rejected - execution time falls during a pass",
                    obsid=self.ppt.obsid,
                    acs_mode=self.acs.acsmode,
                )
                self.ppt = None
                return
            if self.acs.passrequests.current_pass(slew_end) is not None:
                self.log.log_event(
                    utime=utime,
                    event_type="SLEW",
                    description="Slew rejected - would complete during a pass",
                    obsid=self.ppt.obsid,
                    acs_mode=self.acs.acsmode,
                )
                self.ppt = None
                return

            self.acs.slew_dists.append(slew.slewdist)

            # Check if there's enough observation time before the next pass
            slew_end_time = slew.slewstart + slew.slewtime
            next_pass = self.acs.passrequests.next_pass(slew_end_time)
            if next_pass is not None:
                # Calculate when we need to start slewing to the pass
                time_to_pass_slew = next_pass.begin - slew_end_time
                # Estimate slew time to pass start position using existing utilities
                pass_slew_dist = angular_separation(
                    self.ppt.ra, self.ppt.dec, next_pass.gsstartra, next_pass.gsstartdec
                )
                acs_cfg = self.config.spacecraft_bus.attitude_control
                pass_slew_time = acs_cfg.slew_time(pass_slew_dist)

                # Available observation time = time until pass - slew time to pass
                available_obs_time = time_to_pass_slew - pass_slew_time
                if available_obs_time < self.ppt.ss_min:
                    self.log.log_event(
                        utime=utime,
                        event_type="QUEUE",
                        description=f"Target {self.ppt.obsid} rejected - only {available_obs_time:.0f}s "
                        f"available before pass (need {self.ppt.ss_min}s)",
                        obsid=self.ppt.obsid,
                        acs_mode=self.acs.acsmode,
                    )
                    self.ppt = None
                    return

            # Update PPT timing based on slew
            self.ppt.begin = int(execution_time)
            # Update PPT end time to ensure it has enough time for slew + max observation
            self.ppt.end = int(execution_time + slew.slewtime + self.ppt.ss_max)

            # Enqueue the slew command
            command = ACSCommand(
                command_type=ACSCommandType.SLEW_TO_TARGET,
                execution_time=slew.slewstart,
                slew=slew,
            )
            self.acs.enqueue_command(command)

            # Return the new target coordinates
            return
        else:
            self.log.log_event(
                utime=utime,
                event_type="QUEUE",
                description="No targets available from Queue",
                acs_mode=self.acs.acsmode,
            )
            return

    def _record_pointing_data(
        self, ra: float, dec: float, roll: float, obsid: int, mode: ACSMode
    ) -> None:
        """Record spacecraft pointing and mode data."""
        self.mode.append(mode)
        self.ra.append(ra)
        self.dec.append(dec)
        self.roll.append(roll)
        self.obsid.append(obsid)

    def _record_power_data(
        self,
        i: int,
        utime: float,
        ra: float,
        dec: float,
        mode: ACSMode,
        in_eclipse: bool,
    ) -> None:
        """Calculate and record power generation, consumption, and battery state."""
        # Calculate solar panel power
        panel_illumination, panel_power = self._calculate_panel_power(i, utime, ra, dec)
        self.panel.append(panel_illumination)
        self.panel_power.append(panel_power)

        # Calculate power consumption by subsystem
        bus_power, payload_power, total_power, mtq_power, wheel_power = (
            self._calculate_power_consumption(mode=mode, in_eclipse=in_eclipse)
        )
        self.power_bus.append(bus_power)
        self.power_payload.append(payload_power)
        self.mtq_power.append(mtq_power)
        self.wheel_power.append(wheel_power)
        self.power.append(total_power)

        # Update battery state
        self._update_battery_state(total_power, panel_power)

    def _calculate_panel_power(
        self, i: int, utime: float, ra: float, dec: float
    ) -> tuple[float, float]:
        """Calculate solar panel illumination and power generation."""
        panel_illumination, panel_power = (
            self.config.solar_panel.illumination_and_power(
                time=self.utime[i], ra=ra, dec=dec, ephem=self.ephem
            )
        )
        assert isinstance(panel_illumination, float)
        assert isinstance(panel_power, float)
        return panel_illumination, panel_power

    def _calculate_power_consumption(
        self, mode: ACSMode, in_eclipse: bool
    ) -> tuple[float, float, float, float, float]:
        """Calculate total spacecraft power consumption broken down by subsystem.

        Returns:
            Tuple of (bus_power, payload_power, total_power, mtq_power, wheel_power)
            in watts. MTQ power reflects ACS telemetry and may be non-zero in SCIENCE
            when mtq_bleed_in_science is enabled. Wheel power includes idle and
            torque-dependent consumption from all reaction wheels.
        """
        bus_power = self.spacecraft_bus.power(mode=mode, in_eclipse=in_eclipse)
        payload_power = self.payload.power(mode=mode, in_eclipse=in_eclipse)
        try:
            mtq_power = float(getattr(self.acs, "mtq_power_w", 0.0))
        except (TypeError, ValueError):
            mtq_power = 0.0
        try:
            wheel_power = float(getattr(self.acs, "wheel_power_w", 0.0))
        except (TypeError, ValueError):
            wheel_power = 0.0
        total_power = bus_power + payload_power + mtq_power + wheel_power
        return bus_power, payload_power, total_power, mtq_power, wheel_power

    def _update_battery_state(
        self, consumed_power: float, generated_power: float
    ) -> None:
        """Update battery level based on power consumption and generation."""
        self.battery.drain(consumed_power, self.step_size)
        self.battery.charge(generated_power, self.step_size)
        self.batterylevel.append(self.battery.battery_level)
        self.charge_state.append(self.battery.charge_state)

    def _record_wheel_resource(self, utime: float) -> None:
        """Record aggregated reaction wheel torque/momentum usage."""
        if not hasattr(self.acs, "wheel_snapshot"):
            self.wheel_momentum_fraction.append(0.0)
            self.wheel_momentum_fraction_raw.append(0.0)
            self.wheel_torque_fraction.append(0.0)
            self.wheel_saturation.append(0)
            self.wheel_torque_actual_mag.append(0.0)
            self.hold_torque_target_mag.append(0.0)
            self.hold_torque_actual_mag.append(0.0)
            self.pass_tracking_rate_deg_s.append(0.0)
            self.pass_torque_target_mag.append(0.0)
            self.pass_torque_actual_mag.append(0.0)
            self.mtq_proj_max.append(0.0)
            self.mtq_torque_mag.append(0.0)
            return

        snapshot = self.acs.wheel_snapshot()
        # Handle both real WheelTelemetrySnapshot and Mock objects
        self.wheel_momentum_fraction.append(
            getattr(snapshot, "max_momentum_fraction", 0.0)
        )
        self.wheel_momentum_fraction_raw.append(
            getattr(snapshot, "max_momentum_fraction_raw", 0.0)
        )
        self.wheel_torque_fraction.append(getattr(snapshot, "max_torque_fraction", 0.0))
        self.wheel_saturation.append(1 if getattr(snapshot, "saturated", False) else 0)
        self.wheel_torque_actual_mag.append(getattr(snapshot, "t_actual_mag", 0.0))
        self.hold_torque_target_mag.append(
            getattr(snapshot, "hold_torque_target_mag", 0.0)
        )
        self.hold_torque_actual_mag.append(
            getattr(snapshot, "hold_torque_actual_mag", 0.0)
        )
        self.pass_tracking_rate_deg_s.append(
            getattr(snapshot, "pass_tracking_rate_deg_s", 0.0)
        )
        self.pass_torque_target_mag.append(
            getattr(snapshot, "pass_torque_target_mag", 0.0)
        )
        self.pass_torque_actual_mag.append(
            getattr(snapshot, "pass_torque_actual_mag", 0.0)
        )
        self.mtq_proj_max.append(getattr(snapshot, "mtq_proj_max", 0.0))
        self.mtq_torque_mag.append(getattr(snapshot, "mtq_torque_mag", 0.0))
        # Track per-wheel raw maxima across the run
        wheels = getattr(snapshot, "wheels", None)
        if wheels is None or not hasattr(wheels, "__iter__"):
            return
        try:
            for w in wheels:
                name = getattr(w, "name", None)
                if not name:
                    continue
                # Store signed momentum time series
                hist = self.wheel_momentum_history.setdefault(name, [])
                hist.append(getattr(w, "momentum", 0.0))
                # Store applied torque time series
                thist = self.wheel_torque_history.setdefault(name, [])
                thist.append(getattr(w, "torque_applied", 0.0))
                frac = getattr(w, "momentum_fraction_raw", 0.0)
                prev = self.wheel_per_wheel_max_raw.get(name, 0.0)
                if frac > prev:
                    self.wheel_per_wheel_max_raw[name] = frac
        except TypeError:
            # wheels is not iterable (e.g., Mock object)
            pass

    @staticmethod
    def _safe_float(val: float | int | None) -> float:
        if val is None:
            return 0.0
        try:
            return float(val)
        except Exception:
            return 0.0

    def _maybe_request_desat(self, utime: float) -> None:
        """Trigger a desaturation when wheels have been saturated for several steps."""
        if len(self.wheel_saturation) < 3:
            return

        # If MTQ bleed is enabled in SCIENCE, skip command-based desat
        if getattr(self.acs, "magnetorquers", None) and getattr(
            self.acs, "_mtq_bleed_in_science", False
        ):
            return

        if self.acs._desat_active:
            return

        # Require sustained saturation and a cooldown between desats
        if all(s == 1 for s in self.wheel_saturation[-5:]):
            if utime - getattr(self, "_last_desat_request", 0.0) < 1800:
                return
            self.log.log_event(
                utime=utime,
                event_type="DESAT",
                description="Reaction wheels saturated, requesting desat",
                acs_mode=self.acs.acsmode,
            )
            self.acs.request_desat(utime)
            self._last_desat_request = utime

    def _terminate_science_ppt_for_pass(self, utime: float) -> None:
        """Terminate the current science PPT during ground station pass."""
        if self.ppt is not None and self.ppt != self.charging_ppt:
            # Update plan timeline with actual end time
            if len(self.plan) > 0:
                self.plan[-1].end = utime
            self.ppt.end = utime
            self.ppt.done = True
            self.ppt = None

    def _terminate_charging_ppt(self, utime: float) -> None:
        """Terminate the current charging PPT if active."""
        if self.charging_ppt is not None:
            # Update plan timeline with actual end time
            if len(self.plan) > 0:
                self.plan[-1].end = utime
            self.charging_ppt.end = utime
            self.charging_ppt.done = True
            self.charging_ppt = None
            self.acs.last_slew = None

    def _terminate_emergency_charging(self, reason: str, utime: float) -> None:
        """Terminate emergency charging and log the reason."""
        # Log why we're terminating
        termination_messages = {
            "battery_recharged": f"Battery recharged to {self.battery.battery_level:.2%}, ending emergency charging",
            "constraint": "Charging pointing constrained, terminating",
            "eclipse": "Entered eclipse, terminating emergency charging and suppressing restarts until sunlight",
        }
        message = termination_messages.get(reason, f"Unknown reason: {reason}")
        self.log.log_event(
            utime=utime,
            event_type="CHARGING",
            description=message,
            obsid=self.charging_ppt.obsid if self.charging_ppt else None,
            acs_mode=self.acs.acsmode,
        )

        # Clean up charging state - send END_BATTERY_CHARGE command to ACS
        command = ACSCommand(
            command_type=ACSCommandType.END_BATTERY_CHARGE,
            execution_time=utime,
        )
        self.acs.enqueue_command(command)
        self._terminate_charging_ppt(utime)
        self.emergency_charging.terminate_current_charging(utime)
