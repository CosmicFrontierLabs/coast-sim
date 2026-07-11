from bisect import bisect_left
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import numpy as np
import rust_ephem
from pydantic import BaseModel

from ..common import (
    ACSMode,
    ObsType,
    angular_separation,
    dtutcfromtimestamp,
    radec2vec,
    scbodyvector,
    unixtime2date,
)
from ..common.enums import ACSCommandType
from ..common.vector import attitude_to_quat, quaternion_attitude_distance
from ..config import DAY_SECONDS, MissionConfig
from ..config.constraint import (
    all_attitude_constraint_name,
    attitude_constraint_name_for_scopes,
    attitude_constraint_scope_label,
)
from ..simulation.acs_command import ACSCommand
from ..simulation.emergency_charging import EmergencyCharging
from ..simulation.passes import Pass, pass_slew_trigger_buffer
from ..simulation.roll import optimum_roll
from ..simulation.slew import Slew
from ..targets import Plan, PlanEntry, Pointing, Queue, TargetSlewEstimate
from .ditl_log import DITLLog
from .ditl_mixin import DITLMixin
from .ditl_stats import DITLStats
from .telemetry import Housekeeping, PayloadData


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


class PlanExecutionMismatchError(RuntimeError):
    """Raised when an exported plan entry does not match executed ACS telemetry."""


@dataclass(frozen=True)
class PlanExecutionMismatch:
    """A single mismatch between an exported plan entry and ACS telemetry."""

    utime: float
    message: str
    obsid: int | None = None

    def __str__(self) -> str:
        """Return the mismatch message as the string representation."""
        return self.message


@dataclass(frozen=True)
class _ScienceDeadlineInputs:
    """Target-independent deadline components for one scheduler fetch."""

    simulation_end: float
    charge_deadline: float | None


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
        calculate_field_of_regard: bool = False,
    ) -> None:
        """Initialize a queue-driven DITL simulation."""
        # Initialize mixin
        DITLMixin.__init__(
            self,
            config=config,
            ephem=ephem,
            begin=begin,
            end=end,
            calculate_field_of_regard=calculate_field_of_regard,
        )

        # Current target (already set in mixin but repeated for clarity)
        self.ppt = None

        # Pointing history
        self.plan = Plan()

        # Power and battery history
        self.panel = list()
        self.batterylevel = list()
        self.charge_state = list()
        self.power = list()
        self.panel_power = list()
        # Eclipse tracking
        self.in_eclipse = list()
        self._ephem_utime_cache: list[float] | None = None
        self._ephem_utime_cache_source: Any | None = None
        self._ppt_optimum_roll_cache: dict[
            tuple[float, float, float, int, int, int], float
        ] = {}
        # Subsystem power tracking
        self.power_bus = list()
        self.power_payload = list()
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

        # Wire log into ACS so it can log events
        self.acs.log = self.log

        # Initialize emergency charging manager (will be fully set up after ACS is available)
        self.charging_ppt = None
        self.emergency_charging = EmergencyCharging(
            config=self.config,
            starting_obsid=999000,
            log=self.log,
        )
        self._temporary_rejected_ppts: list[tuple[Any, bool]] | None = None
        self._retry_ppt_fetch_requested = False

        # Track whether the last PPT fetch attempt was unsuccessful (no target dispatched).
        # Set True when the queue is empty or all candidates are rejected; False on
        # successful dispatch; None during a pass (spacecraft is occupied).
        self._ppt_unavailable: bool | None = None
        self._planned_gsp_keys: set[tuple[str, float, float]] = set()
        self._gsp_slew_plan_entries: dict[Slew, PlanEntry] = {}
        self._synced_executed_slew_count = 0
        self._dropped_science_windows: list[tuple[float, float, int]] = []
        self._attitude_constraint_violations: list[tuple[str, str] | None] = []
        self._active_gsp_end_time: float | None = None

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
        # Reset per-run state so re-runs on the same instance start clean
        self._attitude_constraint_violations = []
        self._active_gsp_end_time = None

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

        # Set up timing and schedule passes
        if not self._setup_simulation_timing():
            return False

        # Schedule groundstation passes (these will be queued in ACS)
        self._schedule_groundstation_passes()

        # Set up simulation length from begin/end datetimes
        simlen = int((self.end - self.begin).total_seconds() / self.step_size)
        self._dropped_science_windows.clear()

        # DITL loop
        for i in range(simlen):
            utime = self.ustart + i * self.step_size

            # Track PPT in timeline
            self._track_ppt_in_timeline()

            # Get current pointing and mode from ACS
            ra, dec, roll, obsid = self.acs.pointing(utime)
            self._sync_acs_slew_metadata()
            mode = self.acs.get_mode(utime)

            # Check pass timing and manage passes
            pass_command_due = self._check_and_manage_passes(utime, ra, dec, roll)
            if pass_command_due:
                ra, dec, roll, obsid = self.acs.pointing(utime)
                self._sync_acs_slew_metadata()
                mode = self.acs.get_mode(utime)

            # Handle spacecraft operations based on current mode
            self._handle_mode_operations(mode, utime, ra, dec)

            same_tick_pointing = self._process_due_acs_commands(utime)
            if same_tick_pointing is not None:
                ra, dec, roll, obsid = same_tick_pointing

            # Close PPT timeline segment if no active observation
            self._close_ppt_timeline_if_needed(utime)

            # Re-query attitude after operations when they changed mode or queued
            # an immediate command. This keeps IDLE telemetry from inheriting the
            # stale science attitude after an observation terminates.
            ra, dec, roll, obsid, mode = self._refresh_pointing_after_operations(
                mode, utime, ra, dec, roll, obsid
            )

            # Record pointing and mode
            self._record_pointing_data(ra, dec, roll, obsid, mode)

            # Calculate and record power data
            self._record_power_data(
                i, utime, ra, dec, roll, mode, in_eclipse=self.acs.in_eclipse
            )

            # Handle data generation and downlink
            self._handle_data_management(utime, mode)

            # Create housekeeping snapshot from current timestep state
            hk = self._create_housekeeping_record(utime, ra, dec, roll, mode)
            self.telemetry.housekeeping.append(hk)

            # Fault management checks (e.g., battery level thresholds)
            self._handle_fault_management(utime, hk)

        # Make sure an active PPT ends at the simulation boundary.
        if self.plan and self.ppt is not None:
            self._close_last_plan_entry(self.uend)

        self._assert_plan_matches_execution()
        self._attach_execution_timeseries_to_plan()

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

        # Create payload data record if data was generated
        if data_generated > 0:
            pd = PayloadData(
                timestamp=datetime.fromtimestamp(utime, tz=timezone.utc),
                data_size_gb=data_generated,
            )
            self.telemetry.data.append(pd)

    def _handle_fault_management(
        self, utime: float, hk: Housekeeping | None = None
    ) -> None:
        """Handle fault management checks and safe mode requests."""
        # Get the most recent housekeeping record
        if hk is None:
            if not self.telemetry.housekeeping:
                return
            hk = self.telemetry.housekeeping[-1]

        self.config.fault_management.check(
            housekeeping=hk,
            acs=self.acs,
        )
        # Check if safe mode has been requested by fault management
        if (
            self.config.fault_management.safe_mode_requested
            and not self.acs.in_safe_mode
        ):
            reason = None
            trigger_event = next(
                (
                    e
                    for e in reversed(self.config.fault_management.events)
                    if e.event_type == "safe_mode_trigger"
                ),
                None,
            )
            if trigger_event is not None:
                reason = trigger_event.cause
            command = ACSCommand(
                command_type=ACSCommandType.ENTER_SAFE_MODE,
                execution_time=utime,
                reason=reason,
            )
            self.acs.enqueue_command(command)

    def _science_collection_time(
        self, entry: PlanEntry, end_time: float | None = None
    ) -> float | None:
        """Calculate the effective science collection time for a given plan
        entry."""
        if entry.obstype != ObsType.AT:
            return None
        end = end_time if end_time is not None else float(entry.end)
        return (
            end
            - entry.begin
            - max(0.0, float(entry.slewtime))
            - max(0.0, float(entry.insaa))
        )

    def _is_short_science_entry(self, entry: PlanEntry) -> bool:
        """Determine if a science observation entry failed to meet the minimum
        snapshot requirement."""
        collection_time = self._science_collection_time(entry)
        if collection_time is None:
            return False
        return collection_time < max(0.0, float(entry.ss_min))

    def _entry_obstype(self, entry: PlanEntry) -> ObsType | None:
        """Return the entry's obstype as an ObsType, or None if it cannot be coerced."""
        obstype = getattr(entry, "obstype", None)
        if isinstance(obstype, ObsType):
            return obstype
        try:
            return ObsType(obstype)
        except (TypeError, ValueError):
            return None

    def _mode_at_index(self, index: int) -> ACSMode | None:
        """Return the telemetry mode at index as an ACSMode, or None if it cannot be coerced."""
        mode = self.mode[index]
        if isinstance(mode, ACSMode):
            return mode
        try:
            return ACSMode(mode)
        except (TypeError, ValueError):
            return None

    def _plan_execution_tolerance_deg(self) -> float:
        """Return the pointing error tolerance, in degrees, used for plan/execution checks."""
        tolerance = float(self.config.spacecraft_bus.attitude_control.slew_accuracy)
        return tolerance if tolerance > 0 else 0.01

    def _telemetry_length_mismatch(self) -> PlanExecutionMismatch | None:
        """Return a mismatch if the per-step telemetry lists are not all the same length."""
        lengths = {
            "utime": len(self.utime),
            "ra": len(self.ra),
            "dec": len(self.dec),
            "roll": len(self.roll),
            "obsid": len(self.obsid),
            "mode": len(self.mode),
        }
        if len(set(lengths.values())) == 1:
            return None
        return PlanExecutionMismatch(
            utime=self.uend or self.ustart,
            message=f"telemetry_length_mismatch: {lengths}",
        )

    def _science_start_time(self, entry: PlanEntry) -> float:
        """Return the time science collection begins for an entry, after its slew."""
        return float(entry.begin) + max(0.0, float(entry.slewtime))

    def _window_indices(self, start: float, end: float) -> range:
        """Return the range of telemetry indices whose utime falls within [start, end)."""
        lo = max(0, bisect_left(self.utime, start))
        hi = min(len(self.utime), bisect_left(self.utime, end))
        return range(lo, hi)

    def _execution_mismatch(
        self,
        utime: float,
        interval: str,
        mismatch_type: str,
        detail: str,
        obsid: int | None = None,
    ) -> PlanExecutionMismatch:
        """Build a PlanExecutionMismatch with a formatted, timestamped message."""
        return PlanExecutionMismatch(
            utime=utime,
            message=f"{unixtime2date(utime)} {interval} {mismatch_type}: {detail}",
            obsid=obsid,
        )

    def _mode_mismatch(
        self,
        entry: PlanEntry,
        utime: float,
        actual_mode: ACSMode | None,
        expected_mode: ACSMode,
        interval: str,
    ) -> PlanExecutionMismatch:
        """Build a mismatch for telemetry executing in an unexpected ACS mode."""
        return self._execution_mismatch(
            utime,
            interval,
            "mode_mismatch",
            f"obsid {entry.obsid} expected {expected_mode} got {actual_mode}",
            obsid=int(entry.obsid),
        )

    def _obsid_mismatch(
        self, entry: PlanEntry, utime: float, actual_obsid: int, interval: str
    ) -> PlanExecutionMismatch:
        """Build a mismatch for telemetry reporting an unexpected obsid."""
        return self._execution_mismatch(
            utime,
            interval,
            "obsid_mismatch",
            f"expected {int(entry.obsid)} got {int(actual_obsid)}",
            obsid=int(entry.obsid),
        )

    def _pointing_mismatch(
        self,
        entry: PlanEntry,
        utime: float,
        error_deg: float,
        interval: str,
    ) -> PlanExecutionMismatch:
        """Build a mismatch for pointing error exceeding tolerance."""
        return self._execution_mismatch(
            utime,
            interval,
            "pointing_mismatch",
            f"obsid {int(entry.obsid)} error {error_deg:.3f} deg",
            obsid=int(entry.obsid),
        )

    def _attitude_constraint_mismatch(
        self, index: int, constraint_name: str, scope_label: str
    ) -> PlanExecutionMismatch:
        """Build a mismatch for an attitude constraint violation at a telemetry sample."""
        mode = self._mode_at_index(index)
        obsid = int(self.obsid[index])
        return self._execution_mismatch(
            self.utime[index],
            "attitude",
            "constraint_violation",
            (
                f"mode {mode.name if mode is not None else self.mode[index]} "
                f"obsid {obsid} violates {constraint_name} ({scope_label}); "
                f"ra={float(self.ra[index]):.3f} "
                f"dec={float(self.dec[index]):.3f} "
                f"roll={float(self.roll[index]):.3f}"
            ),
            obsid=obsid,
        )

    def _unknown_mode_mismatch(self, index: int) -> PlanExecutionMismatch:
        """Build a mismatch for a telemetry sample whose mode could not be resolved."""
        return self._execution_mismatch(
            self.utime[index],
            "attitude",
            "unknown_mode",
            f"cannot apply constraint scopes for mode {self.mode[index]}",
            obsid=int(self.obsid[index]),
        )

    def _validate_plan_entry_structure(self) -> list[PlanExecutionMismatch]:
        """Check exported plan entries for invalid intervals and non-monotonic ordering."""
        mismatches: list[PlanExecutionMismatch] = []
        previous_begin: float | None = None
        for index, entry in enumerate(self.plan):
            obstype = self._entry_obstype(entry)
            if obstype is None or obstype == ObsType.PPT:
                continue
            begin = float(entry.begin)
            end = float(entry.end)
            obsid = int(entry.obsid) if entry.obsid is not None else None
            if end <= begin:
                mismatches.append(
                    self._execution_mismatch(
                        begin,
                        "plan",
                        "invalid_interval",
                        (
                            f"entry {index} obsid {obsid} ends at or before it begins "
                            f"({end:.0f} <= {begin:.0f})"
                        ),
                        obsid=obsid,
                    )
                )
            if previous_begin is not None and begin < previous_begin:
                mismatches.append(
                    self._execution_mismatch(
                        begin,
                        "plan",
                        "non_monotonic_begin",
                        (
                            f"entry {index} obsid {obsid} begins before previous "
                            f"entry ({begin:.0f} < {previous_begin:.0f})"
                        ),
                        obsid=obsid,
                    )
                )
            previous_begin = begin
        return mismatches

    def _validate_science_entry_execution(
        self, entry: PlanEntry, tolerance_deg: float
    ) -> list[PlanExecutionMismatch]:
        """Check telemetry over a science entry's window against expected mode, obsid, and pointing."""
        start = self._science_start_time(entry)
        end = float(entry.end)
        mismatches: list[PlanExecutionMismatch] = []
        if end <= start:
            return mismatches

        samples = 0
        science_samples = 0
        for i in self._window_indices(start, end):
            utime = self.utime[i]
            samples += 1
            mode = self._mode_at_index(i)
            if mode == ACSMode.SAA:
                continue
            if mode != ACSMode.SCIENCE:
                mismatches.append(
                    self._mode_mismatch(entry, utime, mode, ACSMode.SCIENCE, "science")
                )
                continue

            science_samples += 1
            if int(self.obsid[i]) != int(entry.obsid):
                mismatches.append(
                    self._obsid_mismatch(entry, utime, self.obsid[i], "science")
                )

            error_deg = angular_separation(
                float(self.ra[i]), float(self.dec[i]), float(entry.ra), float(entry.dec)
            )
            if error_deg > tolerance_deg:
                mismatches.append(
                    self._pointing_mismatch(entry, utime, error_deg, "science")
                )

        if samples > 0 and science_samples == 0:
            mismatches.append(
                self._execution_mismatch(
                    start,
                    "science",
                    "no_science_execution",
                    f"obsid {int(entry.obsid)}",
                    obsid=int(entry.obsid),
                )
            )
        return mismatches

    def _validate_execution_is_planned(self) -> list[PlanExecutionMismatch]:
        """Check that every executed science/pass telemetry sample is covered by an exported plan entry."""
        mismatches: list[PlanExecutionMismatch] = []

        # Build obsid → entries lookups once to avoid O(N×M) linear plan scans.
        science_by_obsid: dict[int, list[PlanEntry]] = {}
        gsp_by_obsid: dict[int, list[PlanEntry]] = {}
        for entry in self.plan:
            obstype = self._entry_obstype(entry)
            if obstype in (ObsType.AT, ObsType.TOO):
                science_by_obsid.setdefault(int(entry.obsid), []).append(entry)
            elif obstype == ObsType.GSP:
                gsp_by_obsid.setdefault(int(entry.obsid), []).append(entry)

        dropped_by_obsid: dict[int, list[tuple[float, float]]] = {}
        for start, end, dropped_obsid in self._dropped_science_windows:
            dropped_by_obsid.setdefault(dropped_obsid, []).append((start, end))

        for i, utime in enumerate(self.utime):
            mode = self._mode_at_index(i)
            if mode == ACSMode.SCIENCE:
                obsid = int(self.obsid[i])
                entries = science_by_obsid.get(obsid, [])
                covered = any(float(e.begin) <= utime < float(e.end) for e in entries)
                if not covered:
                    if any(s <= utime < e for s, e in dropped_by_obsid.get(obsid, [])):
                        continue
                    mismatches.append(
                        self._execution_mismatch(
                            utime,
                            "execution",
                            "unplanned_science",
                            f"obsid {obsid} has no matching exported science entry",
                            obsid=obsid,
                        )
                    )
            elif mode == ACSMode.PASS:
                obsid = int(self.obsid[i])
                entries = gsp_by_obsid.get(obsid, [])
                if not any(float(e.begin) <= utime <= float(e.end) for e in entries):
                    mismatches.append(
                        self._execution_mismatch(
                            utime,
                            "execution",
                            "unplanned_contact",
                            f"obsid {obsid} has no matching exported GSP entry",
                            obsid=obsid,
                        )
                    )
        return mismatches

    def _attitude_constraint_name_for_attitude(
        self,
        ra: float,
        dec: float,
        roll: float,
        utime: float,
        mode: ACSMode,
    ) -> tuple[str, str] | None:
        """Return the violated constraint name and scope label for one attitude."""
        scopes = self.config.attitude_constraint_scopes_for_mode(mode)
        name = attitude_constraint_name_for_scopes(
            self.constraint,
            scopes,
            ra,
            dec,
            utime,
            target_roll=roll,
            acs_mode=mode,
        )
        if name is not None:
            return name, attitude_constraint_scope_label(scopes)
        return None

    def _attitude_constraint_name_for_sample(
        self, index: int, mode: ACSMode
    ) -> tuple[str, str] | None:
        """Return the violated constraint name and scope label for one sample."""
        return self._attitude_constraint_name_for_attitude(
            float(self.ra[index]),
            float(self.dec[index]),
            float(self.roll[index]),
            self.utime[index],
            mode,
        )

    def validate_attitude_constraints(self) -> list[PlanExecutionMismatch]:
        """Check post-simulation attitude constraint compliance.

        Reports telemetry samples where the commanded attitude violated a constraint
        the planner was expected to enforce. The check is gated by each mode's
        configured attitude constraint scopes. An empty scope list disables
        validation for that mode.
        """
        mismatches: list[PlanExecutionMismatch] = []
        use_cached = len(self._attitude_constraint_violations) == len(self.utime)
        for i in range(len(self.utime)):
            mode = self._mode_at_index(i)
            if mode is None:
                mismatches.append(self._unknown_mode_mismatch(i))
                continue

            violation = (
                self._attitude_constraint_violations[i]
                if use_cached
                else self._attitude_constraint_name_for_sample(i, mode)
            )
            if violation is None:
                continue

            constraint_name, scope_label = violation
            mismatches.append(
                self._attitude_constraint_mismatch(i, constraint_name, scope_label)
            )
        return mismatches

    def _matching_pass_for_entry(self, entry: PlanEntry) -> Pass | None:
        """Find the ACS-scheduled pass matching a GSP plan entry's station and contact window."""
        station = entry.station
        contact_begin = entry.contact_begin
        contact_end = entry.contact_end
        for gspass in self.acs.passrequests.passes:
            if station is not None and gspass.station != station:
                continue
            begin_matches = (
                contact_begin is None
                or abs(float(gspass.begin) - float(contact_begin)) <= 1e-6
            )
            end_matches = (
                contact_end is None
                or abs(float(gspass.end) - float(contact_end)) <= 1e-6
            )
            if begin_matches and end_matches:
                return gspass
        return None

    def _validate_gsp_entry_execution(
        self, entry: PlanEntry, tolerance_deg: float
    ) -> list[PlanExecutionMismatch]:
        """Check telemetry over a GSP entry's contact window against expected mode, obsid, and antenna pointing."""
        contact_begin = entry.contact_begin
        contact_end = entry.contact_end
        start = (
            float(contact_begin)
            if contact_begin is not None
            else self._science_start_time(entry)
        )
        end = float(contact_end) if contact_end is not None else float(entry.end)
        mismatches: list[PlanExecutionMismatch] = []
        if end <= start:
            return mismatches

        gspass = self._matching_pass_for_entry(entry)
        missing_profile = gspass is None or not gspass.ra or not gspass.dec
        if missing_profile:
            mismatches.append(
                self._execution_mismatch(
                    start,
                    "contact",
                    "pass_profile_missing",
                    f"obsid {int(entry.obsid)} station {entry.station}",
                    obsid=int(entry.obsid),
                )
            )

        for i in self._window_indices(start, end):
            utime = self.utime[i]
            mode = self._mode_at_index(i)
            if mode != ACSMode.PASS:
                mismatches.append(
                    self._mode_mismatch(entry, utime, mode, ACSMode.PASS, "contact")
                )

            if int(self.obsid[i]) != int(entry.obsid):
                mismatches.append(
                    self._obsid_mismatch(entry, utime, self.obsid[i], "contact")
                )

            if missing_profile:
                continue
            assert gspass is not None
            expected_ra, expected_dec, expected_roll = gspass.attitude_at(utime)
            if expected_ra is None or expected_dec is None:
                continue

            actual_quat = attitude_to_quat(
                float(self.ra[i]), float(self.dec[i]), float(self.roll[i])
            )
            expected_quat = attitude_to_quat(
                float(expected_ra), float(expected_dec), float(expected_roll)
            )
            dot = min(1.0, abs(float(np.dot(actual_quat, expected_quat))))
            error_deg = float(np.rad2deg(2.0 * np.arccos(dot)))
            antenna_error = gspass.antenna_pointing_error(
                float(self.ra[i]), float(self.dec[i]), float(self.roll[i]), utime
            )
            if antenna_error is not None:
                error_deg = max(error_deg, antenna_error)
            if error_deg > tolerance_deg:
                mismatches.append(
                    self._pointing_mismatch(entry, utime, error_deg, "contact")
                )
        return mismatches

    def validate_plan_matches_execution(self) -> list[PlanExecutionMismatch]:
        """Return plan/ACS mismatches over exported science and contact intervals."""
        mismatch = self._telemetry_length_mismatch()
        if mismatch is not None:
            return [mismatch]

        tolerance_deg = self._plan_execution_tolerance_deg()
        mismatches = self._validate_plan_entry_structure()
        if mismatches:
            return mismatches

        for entry in self.plan:
            obstype = self._entry_obstype(entry)
            if obstype in (ObsType.AT, ObsType.TOO):
                mismatches.extend(
                    self._validate_science_entry_execution(entry, tolerance_deg)
                )
            elif obstype == ObsType.GSP:
                mismatches.extend(
                    self._validate_gsp_entry_execution(entry, tolerance_deg)
                )
        mismatches.extend(self.validate_attitude_constraints())
        mismatches.extend(self._validate_execution_is_planned())
        return mismatches

    def _assert_plan_matches_execution(self) -> None:
        """Raise if the exported plan doesn't match executed telemetry, logging the first mismatch."""
        mismatches = self.validate_plan_matches_execution()
        if not mismatches:
            return

        examples = "; ".join(str(mismatch) for mismatch in mismatches[:5])
        message = (
            f"Plan execution validation failed with {len(mismatches)} mismatch(es): "
            f"{examples}"
        )
        first = mismatches[0]
        self.log.log_event(
            utime=first.utime,
            event_type="ERROR",
            description=message,
            obsid=first.obsid,
            acs_mode=self.acs.acsmode,
        )
        raise PlanExecutionMismatchError(message)

    def _close_last_plan_entry(self, end_time: float) -> None:
        """Close the last plan entry in the timeline by setting its end time.
        If it's a science observation that didn't meet the minimum snapshot
        requirement, log it and remove it from the plan."""
        if len(self.plan) == 0:
            return
        entry = self.plan[-1]
        entry_begin = float(entry.begin)
        entry_end = float(entry.end)
        # An entry's end is finalized the moment its activity actually stops.
        # Later close calls — the end-of-simulation sweep, an emergency-charge
        # teardown, etc. — must never push that end *later* over the idle/slew
        # gap that followed, or the plan would claim science/activity that never
        # executed.  Closing a still-open entry (placeholder end ~ begin + a day)
        # or trimming a finalized one earlier (e.g. to free room for a ground
        # station pass) is still allowed.
        is_open = entry_end >= entry_begin + DAY_SECONDS
        if not is_open and end_time >= entry_end:
            return
        self.plan[-1].end = end_time
        if self._is_short_science_entry(self.plan[-1]):
            entry = self.plan[-1]
            self._dropped_science_windows.append(
                (float(entry.begin), float(entry.end), int(entry.obsid))
            )
            collection_time = self._science_collection_time(entry)
            collected = max(0.0, collection_time or 0.0)
            self.log.log_event(
                utime=end_time,
                event_type="QUEUE",
                description=(
                    f"Dropping under-collected science entry {entry.obsid} - "
                    f"collected {collected:.0f}s of required {float(entry.ss_min):.0f}s"
                ),
                obsid=entry.obsid,
                acs_mode=self.acs.acsmode,
            )
            self.plan.pop()

    def _create_housekeeping_record(
        self, utime: float, ra: float, dec: float, roll: float, mode: ACSMode
    ) -> Housekeeping:
        """Create a housekeeping telemetry record from current timestep state."""
        panel_illumination = self.panel[-1] if self.panel else None
        total_power = self.power[-1] if self.power else None
        bus_power = self.power_bus[-1] if self.power_bus else None
        payload_power = self.power_payload[-1] if self.power_payload else None

        violated = self.constraint.in_constraint(
            ra, dec, utime, target_roll=roll, acs_mode=mode
        )
        in_constraint_name = (
            self._get_constraint_name(ra, dec, utime, roll=roll, mode=mode)
            if violated
            else None
        )

        # Pre-compute scope-scoped attitude constraint violations for
        # post-simulation validation.
        scopes = self.config.attitude_constraint_scopes_for_mode(mode)
        scope_constraint_name = attitude_constraint_name_for_scopes(
            self.constraint,
            scopes,
            ra,
            dec,
            utime,
            target_roll=roll,
            acs_mode=mode,
        )
        scope_label = attitude_constraint_scope_label(scopes)
        _constraint_violation = (
            (scope_constraint_name, scope_label)
            if scope_constraint_name is not None
            else None
        )
        self._attitude_constraint_violations.append(_constraint_violation)

        ei = self.ephem.index(dtutcfromtimestamp(utime))
        _sun_bv = scbodyvector(
            np.radians(ra),
            np.radians(dec),
            np.radians(roll),
            radec2vec(
                np.radians(float(self.ephem.sun_ra_deg[ei])),
                np.radians(float(self.ephem.sun_dec_deg[ei])),
            ),
        )
        sun_body_vector: list[float] = [
            float(_sun_bv[0]),
            float(_sun_bv[1]),
            float(_sun_bv[2]),
        ]
        _pos = np.asarray(self.ephem.gcrs_pv.position[ei], dtype=np.float64)
        earth_body_vector: list[float] = list(-_pos / np.linalg.norm(_pos))

        nominal_roll = optimum_roll(ra, dec, utime, self.ephem, self.config.solar_panel)
        roll_offset_deg = (roll - nominal_roll + 180.0) % 360.0 - 180.0

        _q = attitude_to_quat(ra, dec, roll)
        return Housekeeping(
            timestamp=datetime.fromtimestamp(utime, tz=timezone.utc),
            ra=ra,
            dec=dec,
            roll=roll,
            roll_offset_deg=roll_offset_deg,
            acs_mode=mode,
            panel_illumination=panel_illumination,
            power_usage=total_power,
            power_bus=bus_power,
            power_payload=payload_power,
            battery_level=self.battery.battery_level,
            charge_state=int(self.battery.charge_state),
            battery_alert=self.battery.battery_alert,
            obsid=self.obsid[-1] if self.obsid else None,
            recorder_volume_gb=self.recorder.current_volume_gb,
            recorder_fill_fraction=self.recorder.get_fill_fraction(),
            recorder_alert=self.recorder.get_alert_level(),
            sun_angle_deg=self._compute_sun_angle(utime, ra, dec, ephem_index=ei),
            earth_angle_deg=self._compute_earth_angle(utime, ra, dec, ephem_index=ei),
            moon_angle_deg=self._compute_moon_angle(utime, ra, dec, ephem_index=ei),
            for_solid_angle_sr=(
                self.constraint.instantaneous_field_of_regard(utime=utime)
                if self.calculate_field_of_regard
                else None
            ),
            in_eclipse=self.acs.in_eclipse,
            star_tracker_hard_violations=self.acs.star_tracker_hard_violations,
            star_tracker_soft_violations=self.acs.star_tracker_soft_violations,
            star_tracker_functional_count=self.acs.star_tracker_functional_count,
            star_tracker_status=self.acs.star_tracker_status,
            in_constraint=in_constraint_name,
            attitude_constraint=scope_constraint_name,
            attitude_constraint_scope=scope_label
            if scope_constraint_name is not None
            else None,
            ppt_unavailable=self._ppt_unavailable,
            radiator_hard_violations=self.acs.radiator_hard_violations,
            telescope_hard_violations=self.acs.telescope_hard_violations,
            radiator_sun_exposure=self.acs.radiator_sun_exposure,
            radiator_earth_exposure=self.acs.radiator_earth_exposure,
            radiator_heat_dissipation_w=self.acs.radiator_heat_dissipation_w,
            sun_body_vector=sun_body_vector,
            earth_body_vector=earth_body_vector,
            quat_w=float(_q[0]),
            quat_x=float(_q[1]),
            quat_y=float(_q[2]),
            quat_z=float(_q[3]),
        )

    def _track_ppt_in_timeline(self) -> None:
        """Track the start of a new PPT in the plan timeline."""
        if self.ppt is not None and (
            len(self.plan) == 0 or self.ppt.begin != self.plan[-1].begin
        ):
            # Before adding new PPT, close the previous one if it has placeholder end time
            if len(self.plan) > 0:
                last_entry = self.plan[-1]
                # Check if end time looks like a placeholder (>= one day from begin).
                # Charging PPTs use exactly one day, science obs use larger values.
                if last_entry.end >= last_entry.begin + DAY_SECONDS:
                    # Set end to the begin time of new PPT (no gap between entries)
                    self._close_last_plan_entry(self.ppt.begin)

            self.plan.append(self.ppt.copy())

    def _process_due_acs_commands(
        self, utime: float
    ) -> tuple[float, float, float, int] | None:
        """Process ACS commands queued for the current tick.

        QueueDITL may enqueue a command after the tick's first ACS update.  If
        that command is due now, re-enter ACS at the same timestamp so exported
        plan entries start when the simulator actually commanded the slew.
        """
        if not self.acs.command_queue:
            return None

        if self.acs.command_queue[0].execution_time > utime:
            return None

        pointing = self.acs.pointing(utime)
        self._sync_acs_slew_metadata()
        self._clear_canceled_ppt(utime)
        self._track_ppt_in_timeline()
        return pointing

    @staticmethod
    def _slew_matches_obsid(slew: Slew | None, obsid: int) -> bool:
        """Return whether a slew exists and targets the given obsid."""
        return slew is not None and int(slew.obsid) == obsid

    def _ppt_has_pending_or_active_slew(self, obsid: int) -> bool:
        """Return whether a slew for this obsid is active, just completed, or still queued."""
        if self._slew_matches_obsid(self.acs.current_slew, obsid):
            return True
        if self._slew_matches_obsid(self.acs.last_slew, obsid):
            return True
        return any(
            command.command_type == ACSCommandType.SLEW_TO_TARGET
            and self._slew_matches_obsid(command.slew, obsid)
            for command in self.acs.command_queue
        )

    def _clear_canceled_ppt(self, utime: float) -> None:
        """Clear a just-selected science PPT whose ACS slew was canceled."""
        if self.ppt is None or self.ppt is self.charging_ppt:
            return
        obstype = self._entry_obstype(self.ppt)
        if obstype not in (ObsType.AT, ObsType.TOO):
            return

        obsid = int(self.ppt.obsid)
        if self._ppt_has_pending_or_active_slew(obsid):
            return

        self.log.log_event(
            utime=utime,
            event_type="QUEUE",
            description=f"Clearing PPT {obsid} - ACS slew was canceled before execution",
            obsid=obsid,
            acs_mode=self.acs.acsmode,
        )
        self.ppt = None

    def _close_ppt_timeline_if_needed(self, utime: float) -> None:
        """Close the last PPT segment in timeline if no active observation.

        This is a safety net to ensure plan timeline is closed if ppt becomes None
        and the end time hasn't been set yet (e.g., has placeholder value).
        """
        if self.ppt is None and len(self.plan) > 0:
            last_entry = self.plan[-1]
            # Check if end time looks like a placeholder (>= one day from begin).
            # Charging PPTs use exactly one day, science obs use larger values.
            if last_entry.end >= last_entry.begin + DAY_SECONDS:
                self._close_last_plan_entry(utime)

    @staticmethod
    def _apply_slew_metadata(
        entry: PlanEntry, slew: Slew, *, update_end: bool = False
    ) -> None:
        """Copy executed slew timing, distance, and roll onto a plan entry."""
        entry.begin = int(slew.slewstart)
        entry.slewtime = int(round(slew.slewtime))
        entry.slewdist = float(slew.slewdist)
        entry.slewpath = slew.slewpath
        entry.roll = float(slew.endroll)
        if update_end:
            entry.end = int(slew.slewstart + slew.slewtime + entry.ss_max)

    @staticmethod
    def _entry_matches_science_slew(entry: PlanEntry, slew: Slew) -> bool:
        """Return whether a plan entry corresponds to the given executed science slew."""
        if entry.obstype != ObsType.AT or entry.obsid != slew.obsid:
            return False

        entry_begin = float(entry.begin)
        entry_end = float(entry.end)
        slew_start = float(slew.slewstart)
        has_placeholder_end = entry_end >= entry_begin + DAY_SECONDS
        if has_placeholder_end:
            return True

        # The executed slew is recomputed when it actually runs: `_start_slew`
        # resets `slewstart` to the current step and recalculates `slewtime` from
        # the real attitude.  That actual start almost never equals the predicted
        # start recorded on the entry at enqueue time (the command is processed at
        # least one step later, and may be delayed further for visibility/chaining).
        # Re-sync whenever the slew actually begins within the window the entry
        # already occupies, so stale predicted timing is corrected.
        #
        # Safety: a retry of the same obsid cannot start inside the closed window.
        # An entry is closed at the simulation step that interrupts/ends it
        # (entry.end = t_close).  Any subsequent slew for the same obsid is
        # scheduled at utime >= t_close, so slew_start >= entry_end, which makes
        # the condition below false.  Simulation time is strictly monotonic and
        # _fetch_new_ppt only schedules slews at utime or later, so this boundary
        # holds by construction.
        if entry_begin <= slew_start < entry_end:
            return True

        return abs(entry_begin - slew_start) <= 1e-6

    def _sync_gsp_slew_metadata(self, slew: Slew) -> None:
        """Update the GSP plan entry's timing/roll from its associated executed slew."""
        entry = self._gsp_slew_plan_entries.get(slew)
        if entry is not None:
            entry.begin = int(slew.slewstart)
            entry.slewtime = int(round(slew.slewtime))
            entry.slewdist = float(slew.slewdist)
            entry.roll = float(slew.endroll)

    def _sync_slew_metadata(self, slew: Slew) -> None:
        """Sync slew metadata onto the GSP or PPT plan entry the slew belongs to."""
        if slew.obstype == ObsType.GSP:
            self._sync_gsp_slew_metadata(slew)
            return

        if slew.obstype != ObsType.PPT:
            return

        if self.ppt is not None and self._entry_matches_science_slew(self.ppt, slew):
            self._apply_slew_metadata(self.ppt, slew, update_end=True)

        for entry in reversed(list(self.plan)):
            if self._entry_matches_science_slew(entry, slew):
                update_end = entry.end >= entry.begin + DAY_SECONDS
                self._apply_slew_metadata(entry, slew, update_end=update_end)
                return

    def _sync_acs_slew_metadata(self) -> None:
        """Copy ACS science slew metadata onto exported plan entries."""
        new_commands = self.acs.executed_commands[self._synced_executed_slew_count :]
        for command in new_commands:
            if command.command_type == ACSCommandType.SLEW_TO_TARGET:
                slew = command.slew
                if slew is not None:
                    self._sync_slew_metadata(slew)
        self._synced_executed_slew_count = len(self.acs.executed_commands)

        if self.acs.current_slew is not None:
            self._sync_slew_metadata(self.acs.current_slew)

    def _expected_slew_start_attitude(
        self, utime: float, execution_time: float
    ) -> tuple[float, float, float]:
        """Return the attitude a new slew should start from, accounting for an in-progress slew."""
        active_slew = self.acs.last_slew
        if active_slew is not None and active_slew.is_slewing(utime):
            slewend = active_slew.slewstart + active_slew.slewtime
            if execution_time >= slewend:
                return active_slew.endra, active_slew.enddec, active_slew.endroll
        return self.acs.ra, self.acs.dec, self.acs.roll

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

    def _refresh_pointing_after_operations(
        self,
        previous_mode: ACSMode,
        utime: float,
        ra: float,
        dec: float,
        roll: float,
        obsid: int,
    ) -> tuple[float, float, float, int, ACSMode]:
        """Re-query ACS pointing and mode if the mode changed or a command is now due."""
        mode = self.acs.get_mode(utime)
        if mode == previous_mode and not self._has_due_acs_command(utime):
            return ra, dec, roll, obsid, mode

        ra, dec, roll, obsid = self.acs.pointing(utime)
        self._sync_acs_slew_metadata()
        return ra, dec, roll, obsid, self.acs.get_mode(utime)

    def _has_due_acs_command(self, utime: float) -> bool:
        """Return whether the ACS command queue has a command due at or before utime."""
        return (
            bool(self.acs.command_queue)
            and self.acs.command_queue[0].execution_time <= utime
        )

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
            if self._gsp_activity_in_progress(utime):
                self._ppt_unavailable = None
                return
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
        charging_ppt = self.emergency_charging.create_charging_pointing(
            utime, self.ephem, ra, dec
        )
        if charging_ppt is None:
            return

        slew = Slew(config=self.config)
        slew.ephem = self.acs.ephem
        slew.slewrequest = utime
        slew.slewstart = utime
        slew.startra = ra
        slew.startdec = dec
        slew.startroll = self.acs.roll
        slew.endra = charging_ppt.ra
        slew.enddec = charging_ppt.dec
        slew.endroll = charging_ppt.roll
        slew.obstype = ObsType.CHARGE
        slew.obsid = charging_ppt.obsid
        slew.at = charging_ppt
        slew.calc_slewtime()
        violation = self._slew_attitude_constraint_violation(slew, ACSMode.SLEWING)
        if violation is not None:
            violation_time, constraint_name, scope_label = violation
            self.log.log_event(
                utime=utime,
                event_type="CHARGING",
                description=(
                    f"Skipping emergency charge slew - path violates "
                    f"{constraint_name} ({scope_label}) at "
                    f"{unixtime2date(violation_time)}"
                ),
                obsid=charging_ppt.obsid,
                acs_mode=self.acs.acsmode,
            )
            self.emergency_charging.current_charging_ppt = None
            return

        interrupted_ppt = self.ppt
        self.charging_ppt = charging_ppt
        if interrupted_ppt is not None and not interrupted_ppt.done:
            self.log.log_event(
                utime=utime,
                event_type="CHARGING",
                description=(
                    "Battery below recharge threshold; interrupting science observation "
                    "for charging"
                ),
                obsid=interrupted_ppt.obsid,
                acs_mode=self.acs.acsmode,
            )
            interrupted_ppt.end = utime
            interrupted_ppt.done = True

        if interrupted_ppt is not None:
            if (
                len(self.plan) > 0
                and self._entry_obstype(self.plan[-1]) == ObsType.AT
                and int(self.plan[-1].obsid) == int(interrupted_ppt.obsid)
            ):
                self._close_last_plan_entry(utime)
            if self._is_short_science_entry(interrupted_ppt):
                interrupted_ppt.done = False

        command = ACSCommand(
            command_type=ACSCommandType.START_BATTERY_CHARGE,
            execution_time=utime,
            ra=self.charging_ppt.ra,
            dec=self.charging_ppt.dec,
            roll=self.charging_ppt.roll,
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
            length = int((self.end - self.begin).total_seconds() / 86400)
            self.acs.passrequests.get(year, day, length)
            for p in self.acs.passrequests.passes:
                self.log.log_event(
                    utime=self.ustart,
                    event_type="PASS",
                    description=f"Scheduled pass: {p}",
                )

            for dropped, selected in self.acs.passrequests.dropped_overlapping_passes:
                self.log.log_event(
                    utime=self.ustart,
                    event_type="PASS",
                    description=(
                        f"Skipped overlapping pass opportunity: {dropped} "
                        f"overlaps selected pass {selected}"
                    ),
                )

            for dropped in self.acs.passrequests.dropped_constraint_passes:
                self.log.log_event(
                    utime=self.ustart,
                    event_type="PASS",
                    description=f"Skipped constraint-unsafe pass opportunity: {dropped}",
                )

            if not self.acs.passrequests.passes:
                self.log.log_event(
                    utime=self.ustart,
                    event_type="INFO",
                    description="No groundstation passes scheduled.",
                )

    def _ground_pass_key(self, gspass: Pass) -> tuple[str, float, float]:
        """Return the (station, begin, end) identity key for a pass."""
        return gspass.station, gspass.begin, gspass.end

    def _overlaps_planned_gsp(self, gspass: Pass) -> bool:
        """Return whether a pass overlaps an already-planned GSP entry."""
        return any(
            entry.obstype == ObsType.GSP
            and gspass.begin < entry.end
            and gspass.end > entry.begin
            for entry in self.plan
        )

    def _gsp_activity_in_progress(self, utime: float) -> bool:
        """Return whether a ground station contact is still active at utime."""
        return (
            self._active_gsp_end_time is not None and utime < self._active_gsp_end_time
        )

    def _terminate_active_ppt_for_gsp(self, utime: float) -> None:
        """Terminate whichever PPT (charging or science) is active ahead of a ground station pass."""
        if self.ppt is None:
            return
        if self.ppt == self.charging_ppt:
            self._terminate_charging_ppt(utime)
        else:
            self._terminate_science_ppt_for_pass(utime)

    def _gsp_plan_entry_allowed(self, gspass: Pass, reserved_begin: float) -> bool:
        """Check whether this pass should be represented in the exported plan."""
        key = self._ground_pass_key(gspass)
        if key in self._planned_gsp_keys:
            return True

        if self._overlaps_planned_gsp(gspass):
            self.log.log_event(
                utime=reserved_begin,
                event_type="PASS",
                description=f"Skipping overlapping pass opportunity for {gspass.station}",
                obsid=gspass.obsid,
                acs_mode=self.acs.acsmode,
            )
            return False

        return True

    def _record_gsp_plan_entry(
        self, gspass: Pass, reserved_begin: float, slew: Slew | None = None
    ) -> bool:
        """Record an accepted ground-station pass command in the exported plan."""
        key = self._ground_pass_key(gspass)
        if key in self._planned_gsp_keys:
            return True

        station, pass_begin, pass_end = key

        self._terminate_active_ppt_for_gsp(reserved_begin)

        if len(self.plan) > 0:
            last_entry = self.plan[-1]
            if last_entry.obstype != ObsType.GSP and last_entry.end > reserved_begin:
                self._close_last_plan_entry(reserved_begin)

        entry = PlanEntry(config=self.config)
        entry.name = f"{station}_PASS"
        entry.ra = gspass.gsstartra
        entry.dec = gspass.gsstartdec
        entry.roll = gspass.gsstartroll
        entry.begin = reserved_begin
        entry.end = pass_end
        entry.slewtime = (
            int(round(slew.slewtime))
            if slew is not None
            else max(0, int(round(pass_begin - reserved_begin)))
        )
        if slew is not None:
            entry.slewdist = float(slew.slewdist)
        entry.obsid = gspass.obsid
        entry.obstype = ObsType.GSP
        entry.ss_min = 0
        contact_duration = max(0, int(round(pass_end - max(pass_begin, entry.begin))))
        entry.ss_max = contact_duration
        entry.exptime = contact_duration
        entry.station = station
        station_location = self._gsp_station_location(station)
        if station_location is not None:
            entry.station_lat_deg, entry.station_lon_deg, entry.station_alt_m = (
                station_location
            )
        entry.contact_begin = pass_begin
        entry.contact_end = pass_end
        entry.track_start_ra = gspass.gsstartra
        entry.track_start_dec = gspass.gsstartdec
        entry.track_start_roll = gspass.gsstartroll
        entry.track_end_ra = gspass.gsendra
        entry.track_end_dec = gspass.gsenddec
        entry.track_end_roll = gspass.gsendroll

        self.plan.append(entry)
        self._active_gsp_end_time = pass_end
        if slew is not None:
            self._gsp_slew_plan_entries[slew] = entry
        self._planned_gsp_keys.add(key)
        self.log.log_event(
            utime=reserved_begin,
            event_type="PASS",
            description=(
                f"Added GSP plan entry for {station} pass from "
                f"{unixtime2date(reserved_begin)} to {unixtime2date(pass_end)}"
            ),
            obsid=entry.obsid,
            acs_mode=self.acs.acsmode,
        )
        return True

    def _gsp_station_location(self, station: str) -> tuple[float, float, float] | None:
        """Look up a ground station's lat/lon/elevation for plan export, if available."""
        try:
            ground_station = self.config.ground_stations.get(station)
        except (AttributeError, KeyError):
            return None

        try:
            return (
                float(ground_station.latitude_deg),
                float(ground_station.longitude_deg),
                float(ground_station.elevation_m),
            )
        except (AttributeError, TypeError, ValueError):
            return None

    def _planned_gsp_entry(self, gspass: Pass) -> PlanEntry | None:
        """Find the exported plan entry matching a scheduled pass."""
        station, pass_begin, pass_end = self._ground_pass_key(gspass)
        for entry in reversed(self.plan):
            if self._entry_obstype(entry) != ObsType.GSP:
                continue
            contact_begin = entry.contact_begin
            contact_end = entry.contact_end
            if (
                entry.station == station
                and contact_begin is not None
                and contact_end is not None
                and abs(float(contact_begin) - pass_begin) <= 1e-6
                and abs(float(contact_end) - pass_end) <= 1e-6
            ):
                return entry
        return None

    def _close_gsp_plan_entry(self, gspass: Pass, executed_end: float) -> None:
        """Extend a GSP plan entry's end time to match when the pass actually ended."""
        entry = self._planned_gsp_entry(gspass)
        if entry is None:
            return
        if entry.end < executed_end:
            entry.end = executed_end

    def _check_and_manage_passes(
        self, utime: float, ra: float, dec: float, roll: float = 0.0
    ) -> bool:
        """Check pass timing and send appropriate commands to ACS."""

        if self.acs.in_safe_mode or self.acs.acsmode == ACSMode.SAFE:
            return False

        commanded_pass = self.acs.current_pass
        if (
            self.acs.acsmode == ACSMode.PASS
            and commanded_pass is not None
            and not commanded_pass.in_pass(utime)
        ):
            self.log.log_event(
                utime=utime,
                event_type="PASS",
                description="Commanded pass ended, commanding ACS to end pass",
                acs_mode=self.acs.acsmode,
            )
            self._close_gsp_plan_entry(commanded_pass, utime)
            self._active_gsp_end_time = None
            command = ACSCommand(
                command_type=ACSCommandType.END_PASS,
                execution_time=utime,
            )
            self.acs.enqueue_command(command)
            return True

        # Check if we're in a pass, if yes, command ACS to start the pass
        current_pass = self.acs.passrequests.current_pass(utime)
        if current_pass is not None and self.acs.acsmode != ACSMode.PASS:
            if not self._gsp_plan_entry_allowed(current_pass, reserved_begin=utime):
                return False
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
            self._record_gsp_plan_entry(current_pass, reserved_begin=utime)
            return True

        # Check if a pass just ended, if yes, command ACS to end the pass.
        # FIXME: This works but isn't super clean.
        previous_pass = self.acs.passrequests.current_pass(utime - self.ephem.step_size)
        if previous_pass and current_pass is None:
            self.log.log_event(
                utime=utime,
                event_type="PASS",
                description="Pass ended, commanding ACS to end pass",
                acs_mode=self.acs.acsmode,
            )
            self._close_gsp_plan_entry(previous_pass, utime)
            self._active_gsp_end_time = None
            command = ACSCommand(
                command_type=ACSCommandType.END_PASS,
                execution_time=utime,
            )
            self.acs.enqueue_command(command)
            return True

        # Check to see if it's time to slew to the next pass
        # Skip if already in PASS or SLEWING mode - can't interrupt a slew mid-motion.
        # Pass-aware scheduling in _fetch_new_ppt prevents conflicts.
        if self.acs.acsmode in (ACSMode.PASS, ACSMode.SLEWING):
            return False

        next_pass = self.acs.passrequests.next_pass(utime)

        # If there's no next pass, nothing to do
        if next_pass is None:
            return False

        # Check if it's time to start slewing for the next pass
        if next_pass.time_to_slew(utime=utime, ra=ra, dec=dec, roll=roll):
            if not self._gsp_plan_entry_allowed(next_pass, reserved_begin=utime):
                return False
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
            slew.startroll = roll
            slew.slewstart = utime
            slew.endra = next_pass.gsstartra
            slew.enddec = next_pass.gsstartdec
            slew.endroll = next_pass.gsstartroll
            slew.obstype = ObsType.GSP  # Ground Station Pass slew
            slew.obsid = next_pass.obsid
            slew.calc_slewtime()
            violation = self._slew_attitude_constraint_violation(slew, ACSMode.PASS)
            if violation is not None:
                violation_time, constraint_name, scope_label = violation
                self.log.log_event(
                    utime=utime,
                    event_type="PASS",
                    description=(
                        f"Skipping pass slew to {next_pass.station} - path violates "
                        f"{constraint_name} ({scope_label}) at "
                        f"{unixtime2date(violation_time)}"
                    ),
                    obsid=next_pass.obsid,
                    acs_mode=self.acs.acsmode,
                )
                return False
            command = ACSCommand(
                command_type=ACSCommandType.SLEW_TO_TARGET,
                execution_time=utime,
                slew=slew,
            )
            self.acs.enqueue_command(command)
            self._record_gsp_plan_entry(next_pass, reserved_begin=utime, slew=slew)
            return True

        return False

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
            violation = self._attitude_constraint_name_for_attitude(
                self.ppt.ra,
                self.ppt.dec,
                self.acs.roll,
                utime,
                ACSMode.CHARGING,
            )
            if violation is not None:
                constraint_name, scope_label = violation
                constraint_text = f"{constraint_name} ({scope_label})"
                self.log.log_event(
                    utime=utime,
                    event_type="CHARGING",
                    description=f"Charging PPT {constraint_text} constrained, terminating",
                    obsid=self.ppt.obsid,
                    acs_mode=self.acs.acsmode,
                )
                self._terminate_emergency_charging("constraint", utime)
            return

        if mode == ACSMode.SLEWING:
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

        violation = self._attitude_constraint_name_for_attitude(
            self.ppt.ra,
            self.ppt.dec,
            self.acs.roll,
            utime,
            ACSMode.SCIENCE,
        )
        if violation is not None:
            constraint_name, scope_label = violation
            constraint_text = f"{constraint_name} ({scope_label})"
            self._terminate_ppt(
                utime,
                reason=f"Target {constraint_text} constrained, ending observation",
            )
        elif self.ppt.exptime is None or self.ppt.exptime <= 0:
            self._terminate_ppt(
                utime, reason="Exposure complete, ending observation", mark_done=True
            )
        elif utime >= self.ppt.end:
            self._terminate_ppt(utime, reason="Time window elapsed, ending observation")

    def _constraint_name_for_science_attitude(
        self,
        ra: float,
        dec: float,
        roll: float,
        utime: float,
    ) -> tuple[str, str] | None:
        """Return the violated constraint name/scope for an attitude in SCIENCE mode."""
        return self._attitude_constraint_name_for_attitude(
            ra, dec, roll, utime, ACSMode.SCIENCE
        )

    def _check_locked_roll_window(
        self,
        obs_start_time: float,
        obs_val_end: float,
        target_roll: float,
    ) -> tuple[float, str, str] | None:
        """Check whether a locked roll stays constraint-free for the minimum observation window."""
        assert self.ppt is not None
        ephem = self.acs.ephem
        begin_idx = max(0, ephem.index(dtutcfromtimestamp(obs_start_time)))
        end_idx = min(
            len(ephem.timestamp), ephem.index(dtutcfromtimestamp(obs_val_end)) + 1
        )
        for t in ephem.timestamp[begin_idx:end_idx]:
            t_unix = self._ephem_timestamp_to_utime(t)
            violation = self._constraint_name_for_science_attitude(
                self.ppt.ra,
                self.ppt.dec,
                target_roll,
                t_unix,
            )
            if violation is not None:
                constraint_name, scope_label = violation
                return t_unix, constraint_name, scope_label
        return None

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
            self._close_last_plan_entry(utime)

        if mark_done:
            self.ppt.done = True

        self.ppt = None
        self.acs.end_science_observation()
        # Do NOT clear last_slew here. The spacecraft is physically still pointing
        # at the science target; clearing last_slew would cause pointing() to call
        # optimum_roll() and jump the roll on the next tick. Roll stays locked to
        # last_slew.endroll until the next executed slew replaces last_slew.

    def _get_constraint_name(
        self,
        ra: float,
        dec: float,
        utime: float,
        roll: float | None = None,
        mode: int | None = None,
    ) -> str:
        """Determine which constraint is violated.

        Check order matches Constraint.in_constraint() so that when multiple
        constraints are simultaneously active the reported name is consistent
        with the one that actually triggered termination.
        """
        return (
            all_attitude_constraint_name(
                self.constraint,
                ra,
                dec,
                utime,
                target_roll=roll,
                acs_mode=mode,
            )
            or "Unknown"
        )

    def _simulation_end_deadline(self) -> float:
        """Return the Unix timestamp for the end of the simulation."""
        return self.uend if self.uend > 0.0 else self.end.timestamp()

    def _current_ppt_visibility_deadline(
        self, slew_end: float, target: Pointing | None = None
    ) -> float | None:
        """Calculate the end of the current PPT visibility window, if any, after the slew ends."""
        ppt = target or self.ppt
        assert ppt is not None
        if not ppt.windows:
            return None
        for window in ppt.windows:
            if window[0] <= slew_end <= window[1]:
                return window[1]
        return slew_end

    def _slew_attitude_constraint_violation(
        self, slew: Slew, mode: ACSMode
    ) -> tuple[float, str, str] | None:
        """Return first scoped violation along a planned slew path, if any."""
        if slew.slewtime <= 0:
            return None

        for sample_utime in self._ephem_utimes():
            if sample_utime < slew.slewstart:
                continue
            if sample_utime >= slew.slewend:
                break

            sample_ra, sample_dec = slew.ra_dec(sample_utime)
            sample_roll = slew.slew_roll(sample_utime)
            violation = self._attitude_constraint_name_for_attitude(
                float(sample_ra),
                float(sample_dec),
                float(sample_roll),
                sample_utime,
                mode,
            )
            if violation is not None:
                constraint_name, scope_label = violation
                return sample_utime, constraint_name, scope_label

        return None

    def _next_pass_science_deadline(
        self, slew_end: float, target: Pointing | None = None
    ) -> float | None:
        """Calculate the next pass deadline after the slew ends, accounting for
        slew time to the pass start."""
        ppt = target or self.ppt
        assert ppt is not None
        next_pass = self.acs.passrequests.next_pass(slew_end)
        if next_pass is None:
            return None

        pass_slew_dist = angular_separation(
            ppt.ra, ppt.dec, next_pass.gsstartra, next_pass.gsstartdec
        )
        acs_cfg = self.config.spacecraft_bus.attitude_control
        pass_slew_time = float(acs_cfg.slew_time(pass_slew_dist))

        return next_pass.begin - pass_slew_time - self._pass_slew_trigger_buffer()

    def _pass_slew_trigger_buffer(self) -> float:
        """Return the lead time to trigger a pass slew, based on the ephemeris step size."""
        return pass_slew_trigger_buffer(self.ephem.step_size)

    @staticmethod
    def _ephem_timestamp_to_utime(timestamp: Any) -> float:
        """Convert an ephemeris timestamp entry to a Unix timestamp."""
        if hasattr(timestamp, "timestamp"):
            return float(timestamp.timestamp())
        return float(timestamp)

    def _ephem_utimes(self) -> list[float]:
        """Return the ephemeris timestamps converted to Unix time, caching the result."""
        timestamps = self.ephem.timestamp
        if (
            self._ephem_utime_cache is None
            or self._ephem_utime_cache_source is not timestamps
            or len(self._ephem_utime_cache) != len(timestamps)
        ):
            self._ephem_utime_cache = [
                self._ephem_timestamp_to_utime(ts) for ts in timestamps
            ]
            self._ephem_utime_cache_source = timestamps
        return self._ephem_utime_cache

    def _next_charge_science_deadline(self, current_time: float) -> float | None:
        """Return when pending battery recharge can next preempt science."""
        if not self.battery.battery_alert or self.charging_ppt is not None:
            return None

        timestamps = self._ephem_utimes()
        index = bisect_left(timestamps, current_time)
        if index >= len(timestamps):
            index = len(timestamps) - 1

        constraint_time = timestamps[index]
        if not self.constraint.in_eclipse(ra=0, dec=0, time=constraint_time):
            return constraint_time

        for next_index in range(index + 1, len(timestamps)):
            utime = timestamps[next_index]
            if not self.constraint.in_eclipse(ra=0, dec=0, time=utime):
                return utime
        return None

    def _science_deadline_inputs(self, current_time: float) -> _ScienceDeadlineInputs:
        """Calculate deadline inputs that are stable during one queue fetch."""
        return _ScienceDeadlineInputs(
            simulation_end=self._simulation_end_deadline(),
            charge_deadline=self._next_charge_science_deadline(current_time),
        )

    def _next_science_deadline(
        self,
        slew_end: float,
        current_time: float,
        target: Pointing | None = None,
        deadline_inputs: _ScienceDeadlineInputs | None = None,
    ) -> tuple[float, str]:
        """Determine the next science deadline after the slew ends, which could
        be the end of the current visibility window, the next pass, or the end
        of the simulation."""
        if deadline_inputs is None:
            deadline_inputs = self._science_deadline_inputs(current_time)

        deadlines: list[tuple[float, str]] = [
            (deadline_inputs.simulation_end, "simulation end")
        ]

        visibility_end = self._current_ppt_visibility_deadline(slew_end, target=target)
        if visibility_end is not None:
            deadlines.append((visibility_end, "visibility window"))

        pass_deadline = self._next_pass_science_deadline(slew_end, target=target)
        if pass_deadline is not None:
            deadlines.append((pass_deadline, "pass"))

        if deadline_inputs.charge_deadline is not None:
            deadlines.append((deadline_inputs.charge_deadline, "charge opportunity"))

        return min(deadlines, key=lambda item: item[0])

    def _ppt_slew_execution_time(self, utime: float) -> float:
        """Return when a new slew can execute, waiting for an in-progress slew to finish."""
        if self.acs.last_slew is not None and self.acs.last_slew.is_slewing(utime):
            return self.acs.last_slew.slewstart + self.acs.last_slew.slewtime
        return utime

    def _new_ppt_slew(self, target: Pointing, utime: float) -> Slew:
        """Build a new, not-yet-timed Slew object targeting a Pointing."""
        slew = Slew(config=self.config)
        slew.ephem = self.acs.ephem
        slew.slewrequest = utime
        slew.endra = target.ra
        slew.enddec = target.dec
        slew.obstype = ObsType.PPT
        slew.obsid = target.obsid
        slew.at = target
        return slew

    def _complete_ppt_slew(
        self,
        slew: Slew,
        target: Pointing,
        utime: float,
        execution_time: float,
    ) -> None:
        """Fill in a PPT slew's start attitude, end roll, and computed slew time."""
        slew.slewstart = execution_time
        slew.startra, slew.startdec, slew.startroll = (
            self._expected_slew_start_attitude(utime, execution_time)
        )
        slew.endroll = self._ppt_optimum_roll(target, execution_time)
        slew.calc_slewtime()

    def _ppt_optimum_roll(self, target: Pointing, execution_time: float) -> float:
        """Return the optimum roll for a target at the given time, caching the result."""
        key = (
            float(target.ra),
            float(target.dec),
            float(execution_time),
            id(self.acs.ephem),
            id(self.config.solar_panel),
            id(self.config.constraint),
        )
        cached = self._ppt_optimum_roll_cache.get(key)
        if cached is not None:
            return cached

        roll = optimum_roll(
            target.ra,
            target.dec,
            execution_time,
            self.acs.ephem,
            self.config.solar_panel,
            self.config.constraint,
        )
        self._ppt_optimum_roll_cache[key] = roll
        return roll

    def _estimate_ppt_slew(self, target: Pointing, utime: float) -> TargetSlewEstimate:
        """Estimate slew distance and time to a candidate target, for queue scoring."""
        execution_time = self._ppt_slew_execution_time(utime)
        startra, startdec, startroll = self._expected_slew_start_attitude(
            utime, execution_time
        )
        endroll = self._ppt_optimum_roll(target, execution_time)
        slewdist = quaternion_attitude_distance(
            startra,
            startdec,
            startroll,
            target.ra,
            target.dec,
            endroll,
        )
        slewtime = round(
            self.config.spacecraft_bus.attitude_control.slew_time(slewdist)
        )
        return TargetSlewEstimate(slewtime=float(slewtime), slewdist=slewdist)

    def _can_retry_without_current_ppt(self) -> bool:
        """Sanity check to see if we can retry fetching a new PPT without just
        returning the same one"""
        return len(self.queue.targets) > 1

    def _retry_fetch_without_current_ppt(
        self, utime: float, ra: float, dec: float
    ) -> None:
        """
        Retry fetching a new PPT without the current one, if possible. If
        the current PPT is the only one in the queue, mark PPT as unavailable.
        """
        rejected_ppt = self.ppt
        if rejected_ppt is None or not self._can_retry_without_current_ppt():
            self.ppt = None
            self._ppt_unavailable = True
            return

        was_done = rejected_ppt.done
        rejected_ppt.done = True
        self.ppt = None

        if self._temporary_rejected_ppts is not None:
            self._temporary_rejected_ppts.append((rejected_ppt, was_done))
            self._retry_ppt_fetch_requested = True
            return

        try:
            self._fetch_new_ppt(utime, ra, dec)
        finally:
            rejected_ppt.done = was_done

    def _reject_current_ppt_if_insufficient_collect_time(
        self, slew_end: float, utime: float, ra: float, dec: float
    ) -> bool:
        """Check if the current PPT has enough time to collect before the next
        science deadline."""
        assert self.ppt is not None
        if self.ppt.ss_min <= 0:
            return False

        # Check if there's enough time to collect before the next science
        # deadline (visibility window end, next pass, or simulation end)
        deadline, reason = self._next_science_deadline(slew_end, current_time=utime)
        available_collect_time = deadline - slew_end
        if available_collect_time >= self.ppt.ss_min:
            return False

        # If not enough time to collect, reject this target and try fetching
        # another one
        ss_min = self.ppt.ss_min
        obsid = self.ppt.obsid
        self.log.log_event(
            utime=utime,
            event_type="QUEUE",
            description=(
                f"Target {obsid} rejected - only "
                f"{available_collect_time:.0f}s available before {reason} "
                f"(need {ss_min:.0f}s)"
            ),
            obsid=obsid,
            acs_mode=self.acs.acsmode,
        )
        self._retry_fetch_without_current_ppt(utime, ra, dec)
        return True

    def _fetch_new_ppt(self, utime: float, ra: float, dec: float) -> None:
        """Fetch a new pointing target from the queue and enqueue slew command."""
        self._temporary_rejected_ppts = []
        self._retry_ppt_fetch_requested = False
        try:
            while True:
                self._retry_ppt_fetch_requested = False
                self._fetch_new_ppt_inner(utime, ra, dec)
                if self._retry_ppt_fetch_requested:
                    continue
                return
        finally:
            temporary_rejections = self._temporary_rejected_ppts or []
            for rejected_ppt, was_done in temporary_rejections:
                rejected_ppt.done = was_done
            self._temporary_rejected_ppts = None
            self._retry_ppt_fetch_requested = False

    def _fetch_new_ppt_inner(self, utime: float, ra: float, dec: float) -> None:
        """Fetch one candidate PPT from the queue, validate it, and enqueue its slew.

        May request a retry (via `_retry_fetch_without_current_ppt`) if the
        candidate is rejected; `_fetch_new_ppt` loops this until a target is
        accepted or none remain.
        """
        if self.battery.below_minimum_charge_level:
            self.log.log_event(
                utime=utime,
                event_type="QUEUE",
                description="Deferring PPT fetch - battery below minimum charge level",
                acs_mode=self.acs.acsmode,
            )
            self._ppt_unavailable = None
            return

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
            self._ppt_unavailable = None
            return

        self.log.log_event(
            utime=utime,
            event_type="QUEUE",
            description=f"Fetching new PPT from Queue (last RA/Dec {ra:.2f}/{dec:.2f})",
            acs_mode=self.acs.acsmode,
        )

        # Fetch the next PPT from the queue based on current pointing and time.
        # These deadline inputs are target-independent for one queue fetch, but
        # build them lazily so unscored queue modes do not pay for them.
        deadline_inputs: _ScienceDeadlineInputs | None = None

        def collection_deadline(target: Pointing, slew_end: float) -> float:
            """Return the next science deadline for a candidate target, for queue scoring."""
            nonlocal deadline_inputs
            if deadline_inputs is None:
                deadline_inputs = self._science_deadline_inputs(utime)
            # The cached inputs cover only fetch-wide pieces; each callback
            # still evaluates target-specific visibility and pass deadlines.
            return self._next_science_deadline(
                slew_end,
                current_time=utime,
                target=target,
                deadline_inputs=deadline_inputs,
            )[0]

        self.ppt = self.queue.get(
            ra,
            dec,
            utime,
            collection_deadline=collection_deadline,
            slew_estimator=lambda target: self._estimate_ppt_slew(target, utime),
        )

        if self.ppt is not None:
            slew = self._new_ppt_slew(self.ppt, utime)

            # Check if target is visible
            visstart = self.ppt.next_vis(utime)
            if not visstart and slew.obstype == ObsType.PPT:
                self.log.log_event(
                    utime=utime,
                    event_type="SLEW",
                    description="Slew rejected - target not visible",
                    obsid=self.ppt.obsid,
                    acs_mode=self.acs.acsmode,
                )
                self._ppt_unavailable = True
                return

            # Calculate slew timing
            execution_time = self._ppt_slew_execution_time(utime)

            # Wait for current slew to finish if in progress
            if execution_time > utime:
                self.log.log_event(
                    utime=utime,
                    event_type="SLEW",
                    description=f"Slewing - delaying next slew until {unixtime2date(execution_time)}",
                    obsid=self.ppt.obsid,
                    acs_mode=self.acs.acsmode,
                )

            # Wait for target visibility if constrained
            if visstart and visstart > execution_time and slew.obstype == ObsType.PPT:
                self.log.log_event(
                    utime=utime,
                    event_type="SLEW",
                    description=f"Slew delayed by {visstart - execution_time:.1f}s",
                    obsid=self.ppt.obsid,
                    acs_mode=self.acs.acsmode,
                )
                execution_time = visstart

            # When ignore_roll=True, verify that a valid roll exists before slewing.
            # optimum_roll() falls back to the unconstrained solar roll when
            # roll_range() is empty (no roll satisfies all constraints), which
            # would put star trackers into a constraint zone.  Skip the target
            # instead so a better one can be selected.
            if self.config.constraint.ignore_roll:
                _constraint_obj = self.config.constraint.constraint
                if _constraint_obj is not None:
                    # Snap to the nearest ephemeris timestamp — roll_range() requires
                    # an exact match and execution_time may be between grid points.
                    _snapped_dt = self.acs.ephem.timestamp[
                        self.acs.ephem.index(dtutcfromtimestamp(execution_time))
                    ]
                    _valid_ranges = _constraint_obj.roll_range(
                        time=_snapped_dt,
                        ephemeris=self.acs.ephem,
                        target_ra=self.ppt.ra,
                        target_dec=self.ppt.dec,
                    )
                    if not _valid_ranges:
                        self.log.log_event(
                            utime=utime,
                            event_type="QUEUE",
                            description=f"Target {self.ppt.obsid} skipped — no valid roll satisfies all constraints at this time",
                            obsid=self.ppt.obsid,
                            acs_mode=self.acs.acsmode,
                        )
                        self._retry_fetch_without_current_ppt(utime, ra, dec)
                        return

            self._complete_ppt_slew(slew, self.ppt, utime, execution_time)
            violation = self._slew_attitude_constraint_violation(slew, ACSMode.SLEWING)
            if violation is not None:
                violation_time, constraint_name, scope_label = violation
                self.log.log_event(
                    utime=utime,
                    event_type="QUEUE",
                    description=(
                        f"Target {self.ppt.obsid} skipped - slew path violates "
                        f"{constraint_name} ({scope_label}) at "
                        f"{unixtime2date(violation_time)}"
                    ),
                    obsid=self.ppt.obsid,
                    acs_mode=self.acs.acsmode,
                )
                self._retry_fetch_without_current_ppt(utime, ra, dec)
                return

            # Validate that the locked roll satisfies constraints for at least
            # ss_min seconds after slew completion.  The ACS holds roll constant
            # during observations, so a roll that is immediately constrained
            # would cause the observation to be terminated before any science
            # is collected.  Skip the target now rather than wasting a slew.
            obs_start_time = execution_time + slew.slewtime
            ephem = self.acs.ephem
            ephem_end = self._ephem_timestamp_to_utime(ephem.timestamp[-1])
            obs_val_end = min(obs_start_time + self.ppt.ss_min, ephem_end)
            violation = self._check_locked_roll_window(
                obs_start_time,
                obs_val_end,
                slew.endroll,
            )
            if violation is not None:
                t_unix, constraint_name, scope_label = violation
                self.log.log_event(
                    utime=utime,
                    event_type="QUEUE",
                    description=(
                        f"Target {self.ppt.obsid} skipped — locked roll "
                        f"{slew.endroll:.1f}° violates {constraint_name} "
                        f"({scope_label}) within minimum observation window "
                        f"(at t+{t_unix - obs_start_time:.0f}s)"
                    ),
                    obsid=self.ppt.obsid,
                    acs_mode=self.acs.acsmode,
                )
                self._retry_fetch_without_current_ppt(utime, ra, dec)
                return

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
                self._ppt_unavailable = True
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
                self._ppt_unavailable = True
                return

            if self._reject_current_ppt_if_insufficient_collect_time(
                slew_end, utime, ra, dec
            ):
                return

            self._apply_slew_metadata(self.ppt, slew, update_end=True)
            self.acs.slew_dists.append(slew.slewdist)

            self.log.log_event(
                utime=utime,
                event_type="QUEUE",
                description=f"Fetched PPT: {self.ppt}",
                obsid=self.ppt.obsid,
                acs_mode=self.acs.acsmode,
            )

            # Enqueue the slew command
            command = ACSCommand(
                command_type=ACSCommandType.SLEW_TO_TARGET,
                execution_time=slew.slewstart,
                slew=slew,
            )
            self.acs.enqueue_command(command)
            self._ppt_unavailable = False

            # Return the new target coordinates
            return
        else:
            self.log.log_event(
                utime=utime,
                event_type="QUEUE",
                description="No targets available from Queue",
                acs_mode=self.acs.acsmode,
            )
            self._ppt_unavailable = True
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
        self.in_eclipse.append(self.acs.in_eclipse)

    def _record_power_data(
        self,
        i: int,
        utime: float,
        ra: float,
        dec: float,
        roll: float,
        mode: ACSMode,
        in_eclipse: bool,
    ) -> None:
        """Calculate and record power generation, consumption, and battery state."""
        # Calculate solar panel power
        panel_illumination, panel_power = self._calculate_panel_power(
            i, utime, ra, dec, roll
        )
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
        self, i: int, utime: float, ra: float, dec: float, roll: float
    ) -> tuple[float, float]:
        """Calculate solar panel illumination and power generation."""
        panel_illumination, panel_power = (
            self.config.solar_panel.illumination_and_power(
                time=self.utime[i], ra=ra, dec=dec, ephem=self.ephem, roll=roll
            )
        )
        assert isinstance(panel_illumination, float)
        assert isinstance(panel_power, float)
        return panel_illumination, panel_power

    def _compute_sun_angle(
        self, utime: float, ra: float, dec: float, ephem_index: int | None = None
    ) -> float | None:
        """Compute angular distance from pointing to the Sun in degrees."""
        if self.ephem is None:
            return None

        try:
            idx = (
                ephem_index
                if ephem_index is not None
                else self.ephem.index(dtutcfromtimestamp(utime))
            )
            sun_ra = self.ephem.sun_ra_deg[idx]
            sun_dec = self.ephem.sun_dec_deg[idx]
        except Exception:
            return None

        return angular_separation(sun_ra, sun_dec, ra, dec)

    def _compute_earth_angle(
        self, utime: float, ra: float, dec: float, ephem_index: int | None = None
    ) -> float | None:
        """Compute angular distance from pointing to the Earth in degrees."""
        if self.ephem is None:
            return None

        try:
            idx = (
                ephem_index
                if ephem_index is not None
                else self.ephem.index(dtutcfromtimestamp(utime))
            )
            earth_ra = self.ephem.earth_ra_deg[idx]
            earth_dec = self.ephem.earth_dec_deg[idx]
        except Exception:
            return None

        return angular_separation(earth_ra, earth_dec, ra, dec)

    def _compute_moon_angle(
        self, utime: float, ra: float, dec: float, ephem_index: int | None = None
    ) -> float | None:
        """Compute angular distance from pointing to the Moon in degrees."""
        if self.ephem is None:
            return None

        try:
            idx = (
                ephem_index
                if ephem_index is not None
                else self.ephem.index(dtutcfromtimestamp(utime))
            )
            moon_ra = self.ephem.moon_ra_deg[idx]
            moon_dec = self.ephem.moon_dec_deg[idx]
        except Exception:
            return None

        return angular_separation(moon_ra, moon_dec, ra, dec)

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

    def _terminate_science_ppt_for_pass(self, utime: float) -> None:
        """Terminate the current science PPT during ground station pass."""
        if self.ppt is not None and self.ppt != self.charging_ppt:
            # Update plan timeline with actual end time
            if len(self.plan) > 0:
                self._close_last_plan_entry(utime)
            self.ppt.end = utime
            self.ppt.done = True
            self.ppt = None

    def _terminate_charging_ppt(self, utime: float) -> None:
        """Terminate the current charging PPT if active."""
        if self.charging_ppt is not None:
            # Update plan timeline with actual end time, but only when the last
            # plan entry actually belongs to this charging PPT.  A charging PPT
            # that is immediately constrained is terminated before it is ever
            # tracked into the plan, so `plan[-1]` is still the previous (already
            # closed) science entry; closing it here would wrongly extend that
            # observation's end past when it really stopped.  Match on obsid, the
            # same guard `_initiate_charging` uses for the interrupted science PPT.
            if (
                len(self.plan) > 0
                and self.plan[-1].obsid is not None
                and int(self.plan[-1].obsid) == int(self.charging_ppt.obsid)
            ):
                self._close_last_plan_entry(utime)
            self.charging_ppt.end = utime
            self.charging_ppt.done = True
            if self.ppt is self.charging_ppt:
                self.ppt = None
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

        # Clean up charging state.  If the START_BATTERY_CHARGE for this session
        # has not executed yet (charging was abandoned in the same step it was
        # initiated), cancel it outright: there is no charge to end, and letting
        # it fire would start a charge slew that cancels the next science slew.
        # Otherwise send END_BATTERY_CHARGE to stop the active charge.
        if not self.acs.cancel_pending_battery_charge(utime):
            command = ACSCommand(
                command_type=ACSCommandType.END_BATTERY_CHARGE,
                execution_time=utime,
            )
            self.acs.enqueue_command(command)
        self._terminate_charging_ppt(utime)
        self.emergency_charging.terminate_current_charging(utime)
