import copy
from datetime import timezone
from typing import Any

import numpy as np
import rust_ephem

from ..common import ACSMode, unixtime2date
from ..common.enums import ACSCommandType
from ..config import Config
from ..simulation.acs_command import ACSCommand
from ..simulation.emergency_charging import EmergencyCharging
from ..simulation.slew import Slew
from ..targets import Plan, Pointing, Queue
from .ditl_mixin import DITLMixin


class QueueDITL(DITLMixin):
    """
    Class to run a Day In The Life (DITL) simulation based on a target
    Queue. Rather than creating a observing plan and then running it, this
    dynamically pulls a target off the Queue when the current target is done.
    Therefore this is simulating a queue scheduled telescope. However, this
    makes for a very simple DITL simulator as we don't have to create a
    separate Plan first.
    """

    ppt: Pointing | None
    charging_ppt: Pointing | None
    emergency_charging: EmergencyCharging
    ra: list[float]
    dec: list[float]
    roll: list[float]
    mode: list[int]
    panel: list[float]
    power: list[float]
    panel_power: list[float]
    batterylevel: list[float]
    obsid: list[int]
    ppst: Plan
    utime: list[float]
    ephem: rust_ephem.TLEEphemeris

    def __init__(self, config: Config) -> None:
        DITLMixin.__init__(self, config=config)
        # Initialize subsystems from config
        self.constraint = self.config.constraint
        self.battery = self.config.battery
        self.spacecraft_bus = self.config.spacecraft_bus
        self.payload = self.config.payload

        # Current target
        self.ppt = None

        # Pointing history
        self.ra = list()
        self.dec = list()
        self.roll = list()
        self.mode = list()
        self.obsid = list()
        self.ppst = Plan()

        # Power and battery history
        self.panel = list()
        self.batterylevel = list()
        self.charge_state = list()
        self.power = list()
        self.panel_power = list()
        # Subsystem power tracking
        self.power_bus = list()
        self.power_payload = list()
        # Target Queue
        self.queue = Queue()

        # Initialize emergency charging manager (will be fully set up after ACS is available)
        self.charging_ppt = None
        self.emergency_charging = EmergencyCharging(
            constraint=self.constraint,
            solar_panel=self.config.solar_panel,
            acs_config=self.config.spacecraft_bus.attitude_control,
            starting_obsid=999000,
        )

    def timeindex(self, utime: float) -> int:
        return np.where(self.utime <= utime)[0][-1]  # type: ignore[operator]

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

        # Set initial last pointing
        lastra = 0.0
        lastdec = 0.0

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
            print(f"{unixtime2date(utime)}, RA: {ra}, Dec: {dec}, Mode: {mode.name}")

            # Check pass timing and manage passes
            self._check_and_manage_passes(utime, ra, dec)

            # Handle spacecraft operations based on current mode
            lastra, lastdec = self._handle_mode_operations(mode, utime, lastra, lastdec)

            # Close PPT timeline segment if no active observation
            self._close_ppt_timeline_if_needed(utime)

            # Record pointing and mode
            self._record_pointing_data(ra, dec, roll, obsid, mode)

            # Calculate and record power data
            self._record_power_data(
                i, utime, ra, dec, mode, in_eclipse=self.acs.in_eclipse
            )

            # Fault management checks (e.g., battery level thresholds)
            self._handle_fault_management(utime)

        # Make sure the last PPT of the day ends (if any)
        if self.ppst:
            self.ppst[-1].end = utime

        return True

    def _handle_fault_management(self, utime: float) -> None:
        """Handle fault management checks and safe mode requests."""
        if self.config.fault_management is not None:
            self.config.fault_management.check(
                values={"battery_level": self.battery.battery_level},
                utime=utime,
                step_size=self.step_size,
                acs=self.acs,
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
            len(self.ppst) == 0 or self.ppt.begin != self.ppst[-1].begin
        ):
            # Before adding new PPT, close the previous one if it has placeholder end time
            if len(self.ppst) > 0:
                last_entry = self.ppst[-1]
                # Check if end time looks like a placeholder (>= 86400 seconds from begin)
                # Charging PPTs use exactly 86400, science obs use larger values
                if last_entry.end >= last_entry.begin + 86400:
                    # Set end to the begin time of new PPT (no gap between entries)
                    self.ppst[-1].end = self.ppt.begin

            self.ppst.append(self.ppt.copy())

    def _close_ppt_timeline_if_needed(self, utime: float) -> None:
        """Close the last PPT segment in timeline if no active observation.

        This is a safety net to ensure ppst timeline is closed if ppt becomes None
        and the end time hasn't been set yet (e.g., has placeholder value).
        """
        if self.ppt is None and len(self.ppst) > 0:
            last_entry = self.ppst[-1]
            # Check if end time looks like a placeholder (>= 86400 seconds from begin)
            # Charging PPTs use exactly 86400, science obs use larger values
            if last_entry.end >= last_entry.begin + 86400:
                self.ppst[-1].end = utime

    def _handle_mode_operations(
        self, mode: ACSMode, utime: float, lastra: float, lastdec: float
    ) -> tuple[float, float]:
        """Handle spacecraft operations based on current mode.

        Returns:
            Updated (lastra, lastdec) tuple
        """
        if mode == ACSMode.PASS:
            self._handle_pass_mode(utime)
        elif mode == ACSMode.CHARGING:
            self._handle_charging_mode(utime)
        else:
            # Science or SAA modes: handle observations and battery management
            lastra, lastdec = self._handle_science_mode(utime, lastra, lastdec, mode)

        return lastra, lastdec

    def _handle_science_mode(
        self, utime: float, lastra: float, lastdec: float, mode: ACSMode
    ) -> tuple[float, float]:
        """Handle science mode operations: charging, observations, and target acquisition."""
        # Check if we're at a charging pointing during eclipse (mode=SCIENCE but ppt=charging_ppt)
        # This happens when eclipse causes charging mode to revert to science mode
        if self.ppt == self.charging_ppt and self.charging_ppt is not None:
            # We're pointed at a charging location but in SCIENCE mode (likely eclipse)
            # Clear charging PPT and fetch a new science target
            self.charging_ppt = None
            self.ppt = None

        # Check for battery alert and initiate emergency charging if needed
        if self._should_initiate_charging(utime):
            lastra, lastdec = self._initiate_charging(utime, lastra, lastdec)

        # Manage current science PPT lifecycle
        self._manage_ppt_lifecycle(utime, mode)

        # Fetch new PPT if none is active
        if self.ppt is None:
            lastra, lastdec = self._fetch_new_ppt(utime, lastra, lastdec)

        return lastra, lastdec

    def _should_initiate_charging(self, utime: float) -> bool:
        """Check if emergency charging should be initiated."""
        return (
            self.charging_ppt is None
            and self.emergency_charging.should_initiate_charging(
                utime, self.ephem, self.battery.battery_alert
            )
        )

    def _initiate_charging(
        self, utime: float, lastra: float, lastdec: float
    ) -> tuple[float, float]:
        """Initiate emergency charging by creating charging PPT and sending command to ACS."""
        self.charging_ppt = self.emergency_charging.initiate_emergency_charging(
            utime, self.ephem, lastra, lastdec, self.ppt
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
            lastra = self.charging_ppt.ra
            lastdec = self.charging_ppt.dec
            self.ppt = self.charging_ppt

        return lastra, lastdec

    def _setup_simulation_timing(self) -> bool:
        """Set up timing aspect of simulation."""
        self.ustart = self.begin.timestamp()
        self.uend = self.end.timestamp()
        # Check that the start/end times fall within the ephemeris
        # TLEEphemeris uses timestamp attribute which is a list of datetime objects
        if (
            self.begin not in self.ephem.timestamp
            or self.end not in self.ephem.timestamp
        ):
            print("ERROR: Ephemeris not valid for date range")
            return False

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
            print("Scheduling groundstation passes...")
            # Extract year and day-of-year from begin datetime
            year = self.begin.year
            day = self.begin.timetuple().tm_yday
            # Calculate length in days from begin/end
            length = int((self.end - self.begin).total_seconds() / 86400)
            self.acs.passrequests.get(year, day, length)
            if self.acs.passrequests.passes:
                for p in self.acs.passrequests.passes:
                    print(f"Scheduled pass: {p}")
            else:
                print("No groundstation passes scheduled.")

    def _check_and_manage_passes(self, utime: float, ra: float, dec: float) -> None:
        """Check pass timing and send appropriate commands to ACS."""

        # Check what actions are needed for passes
        pass_actions = self.acs.passrequests.check_pass_timing(
            utime, ra, dec, self.step_size
        )

        # Handle pass end
        if pass_actions["end_pass"]:
            command = ACSCommand(
                command_type=ACSCommandType.END_PASS,
                execution_time=utime,
            )
            self.acs.enqueue_command(command)

        # Handle pass start
        if pass_actions["start_pass"] is not None:
            pass_obj = pass_actions["start_pass"]
            # Set the obsid from last science pointing
            pass_obj.obsid = getattr(self.acs.last_ppt, "obsid", 0xFFFF)

            # Only start pass if we're in science or slewing mode
            if self.acs.acsmode in (ACSMode.SCIENCE, ACSMode.SLEWING):
                print(f"{unixtime2date(utime)} Pass start: {pass_obj.station}")
                command = ACSCommand(
                    command_type=ACSCommandType.START_PASS,
                    execution_time=pass_obj.slewrequired,
                    slew=copy.copy(pass_obj),
                )
                self.acs.enqueue_command(command)

    def _handle_pass_mode(self, utime: float) -> None:
        """Handle spacecraft behavior during ground station passes."""
        # Terminate any active observations during passes
        self._terminate_ppt(utime)
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

    def _sync_charging_state(self) -> None:
        """Synchronize emergency_charging module state with queue state."""
        if (
            self.charging_ppt is not None
            and self.emergency_charging.current_charging_ppt is None
        ):
            self.emergency_charging.current_charging_ppt = self.charging_ppt

    def _manage_ppt_lifecycle(self, utime: float, mode: ACSMode) -> None:
        """Manage the lifecycle of the current pointing (PPT)."""
        if self.ppt is None or self.ppt == self.charging_ppt:
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

        if self.constraint.inoccult(self.ppt.ra, self.ppt.dec, utime):
            self._terminate_ppt_due_to_constraint(utime)
        elif self.ppt.exptime is None or self.ppt.exptime <= 0:
            self._terminate_ppt_exposure_complete(utime)
        elif utime >= self.ppt.end:
            self._terminate_ppt_timeout(utime)

    def _terminate_ppt_due_to_constraint(self, utime: float) -> None:
        """Terminate PPT because target is constrained."""
        assert self.ppt is not None
        constraint_name = self._get_constraint_name(self.ppt.ra, self.ppt.dec, utime)
        print(
            f"{unixtime2date(utime)} Target {constraint_name} constrained, ending observation"
        )
        # Update ppst timeline with actual end time
        if len(self.ppst) > 0:
            self.ppst[-1].end = utime
        self.ppt = None

    def _terminate_ppt_exposure_complete(self, utime: float) -> None:
        """Terminate PPT because exposure is complete."""
        assert self.ppt is not None
        print(f"{unixtime2date(utime)} Exposure complete, ending observation")
        # Update ppst timeline with actual end time
        if len(self.ppst) > 0:
            self.ppst[-1].end = utime
        self.ppt.done = True
        self.ppt = None

    def _terminate_ppt_timeout(self, utime: float) -> None:
        """Terminate PPT because time window elapsed."""
        print(f"{unixtime2date(utime)} Time window elapsed, ending observation")
        # Update ppst timeline with actual end time
        if len(self.ppst) > 0:
            self.ppst[-1].end = utime
        self.ppt = None

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

    def _fetch_new_ppt(
        self, utime: float, lastra: float, lastdec: float
    ) -> tuple[float, float]:
        """Fetch a new pointing target from the queue and enqueue slew command."""
        print(
            f"{unixtime2date(utime)} Fetching new PPT from Queue (last RA/Dec {lastra:.2f}/{lastdec:.2f})"
        )

        self.ppt = self.queue.get(lastra, lastdec, utime)

        if self.ppt is not None:
            print(f"{unixtime2date(utime)} Fetched PPT: {self.ppt}")

            # Create and configure a Slew object
            slew = Slew(
                constraint=self.constraint,
                acs_config=self.config.spacecraft_bus.attitude_control,
            )
            slew.ephem = self.acs.ephem
            slew.slewrequest = utime
            slew.endra = self.ppt.ra
            slew.enddec = self.ppt.dec
            slew.obstype = "PPT"
            slew.obsid = self.ppt.obsid

            # Set up target observation request and check visibility
            target_request = Pointing(
                constraint=self.constraint,
                acs_config=self.config.spacecraft_bus.attitude_control,
            )
            target_request.ra = slew.endra
            target_request.dec = slew.enddec
            target_request.obsid = slew.obsid
            target_request.isat = slew.obstype != "PPT"

            target_request.visibility()
            slew.at = target_request

            # Check if target is visible
            visstart = target_request.next_vis(utime)
            if not visstart and slew.obstype == "PPT":
                print(f"{unixtime2date(utime)} Slew rejected - target not visible")
                return lastra, lastdec

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
                print(
                    f"{unixtime2date(utime)} Slewing - delaying next slew until {unixtime2date(execution_time)}"
                )

            # Wait for target visibility if constrained
            if visstart and visstart > execution_time and slew.obstype == "PPT":
                print(
                    f"{unixtime2date(utime)} Slew delayed by {visstart - execution_time:.1f}s"
                )
                execution_time = visstart

            slew.slewstart = execution_time
            slew.calc_slewtime()
            self.acs.slew_dists.append(slew.slewdist)

            # Enqueue the slew command
            command = ACSCommand(
                command_type=ACSCommandType.SLEW_TO_TARGET,
                execution_time=slew.slewstart,
                slew=slew,
            )
            self.acs.enqueue_command(command)

            # Return the new target coordinates
            return self.ppt.ra, self.ppt.dec
        else:
            print(f"{unixtime2date(utime)} No targets available from Queue")
            return lastra, lastdec

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
        bus_power, payload_power, total_power = self._calculate_power_consumption(
            mode=mode, in_eclipse=in_eclipse
        )
        self.power_bus.append(bus_power)
        self.power_payload.append(payload_power)
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
    ) -> tuple[float, float, float]:
        """Calculate total spacecraft power consumption broken down by subsystem.

        Returns:
            Tuple of (bus_power, payload_power, total_power) in watts
        """
        bus_power = self.spacecraft_bus.power(mode=mode, in_eclipse=in_eclipse)
        payload_power = self.payload.power(mode=mode, in_eclipse=in_eclipse)
        total_power = bus_power + payload_power
        return bus_power, payload_power, total_power

    def _update_battery_state(
        self, consumed_power: float, generated_power: float
    ) -> None:
        """Update battery level based on power consumption and generation."""
        self.battery.drain(consumed_power, self.step_size)
        self.battery.charge(generated_power, self.step_size)
        self.batterylevel.append(self.battery.battery_level)
        self.charge_state.append(self.battery.charge_state)

    def _terminate_ppt(self, utime: float) -> None:
        """Terminate the current science PPT if active."""
        if self.ppt is not None and self.ppt != self.charging_ppt:
            # Update ppst timeline with actual end time
            if len(self.ppst) > 0:
                self.ppst[-1].end = utime
            self.ppt.end = utime
            self.ppt.done = True
            self.ppt = None

    def _terminate_charging_ppt(self, utime: float) -> None:
        """Terminate the current charging PPT if active."""
        if self.charging_ppt is not None:
            # Update ppst timeline with actual end time
            if len(self.ppst) > 0:
                self.ppst[-1].end = utime
            self.charging_ppt.end = utime
            self.charging_ppt.done = True
            self.charging_ppt = None

    def _terminate_emergency_charging(self, reason: str, utime: float) -> None:
        """Terminate emergency charging and log the reason."""
        # Log why we're terminating
        termination_messages = {
            "battery_recharged": f"Battery recharged to {self.battery.battery_level:.2%}, ending emergency charging",
            "constraint": "Charging pointing constrained, terminating",
            "eclipse": "Entered eclipse, terminating emergency charging and suppressing restarts until sunlight",
        }
        message = termination_messages.get(reason, f"Unknown reason: {reason}")
        print(f"{unixtime2date(utime)} {message}")

        # Clean up charging state - send END_BATTERY_CHARGE command to ACS
        command = ACSCommand(
            command_type=ACSCommandType.END_BATTERY_CHARGE,
            execution_time=utime,
        )
        self.acs.enqueue_command(command)
        self._terminate_charging_ppt(utime)
        self.emergency_charging.terminate_current_charging(utime)
