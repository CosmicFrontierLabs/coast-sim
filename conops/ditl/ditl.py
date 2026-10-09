from datetime import datetime, timezone

import numpy as np
import rust_ephem

from conops.targets.plan import Plan

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
from ..common.ephemeris import position_vectors
from ..common.vector import attitude_to_quat
from ..config import AttitudeConstraintScope, MissionConfig
from ..config.constraint import (
    all_attitude_constraint_name,
    attitude_constraint_name_for_scopes,
    attitude_constraint_scope_label,
)
from ..simulation.acs_command import ACSCommand
from ..simulation.passes import Pass
from ..simulation.roll import optimum_body_roll, optimum_instrument_roll
from ..simulation.slew import Slew
from ..targets import PlanEntry
from .ditl_log import DITLLog
from .ditl_mixin import DITLMixin
from .ditl_stats import DITLStats
from .plan_validator import (
    PLAN_SCIENCE_OBSTYPES,
    PlanExecutionMismatch,
    PlanExecutionValidator,
    entry_obstype,
    matching_pass_for_entry,
)
from .telemetry import Housekeeping, PayloadData


class DITL(DITLMixin, DITLStats):
    """Day In The Life (DITL) simulation class.

    Simulates spacecraft operations by executing a pre-planned observing
    schedule and tracking spacecraft state including power usage, battery
    levels, pointing angles, and data management.

    Every plan entry is commanded when its ``begin`` is reached:

    - Science entries (PPT, AT, TOO) slew to the target and collect until
      ``end``.
    - Ground-station entries (GSP) slew to the pass tracking profile, start
      the contact at ``contact_begin`` and end it at ``end``. Passes are
      predicted from the configured ground stations and matched to the entry
      by station and contact window.
    - Charging entries (CHARGE) slew to the charging attitude and charge
      until ``end``.

    The plan is authoritative: no science, contact or charging activity
    outside it is started. Use
    :meth:`validate_plan_matches_execution` after :meth:`calc` to compare the
    executed telemetry with the plan.

    Inherits from DITLMixin which provides shared initialization and plotting
    functionality for DITL simulations.

    Attributes:
        constraint (Constraint): Spacecraft constraint model (sun, earth, moon avoidance).
        battery (Battery): Battery model for power tracking and management.
        spacecraft_bus (SpacecraftBus): Spacecraft bus configuration and power draw.
        payload (Payload): Instrument configuration and power draw.
        solar_panel (SolarPanelSet): Solar panel configuration and power generation.
        recorder (OnboardRecorder): Onboard data storage device.
        ephem (Ephemeris): Ephemeris data for position and illumination calculations.
        plan (Plan): Pre-planned pointing schedule to execute.
        acs (ACS): Attitude Control System for pointing and slew calculations.
        begin (datetime): Start time for simulation (default: None).
        end (datetime): End time for simulation (default: None).
        step_size (int): Time step in seconds (default: 60).

    Telemetry Arrays (populated during `calc()`):
        ra (np.ndarray): Right ascension at each timestep.
        dec (np.ndarray): Declination at each timestep.
        mode (np.ndarray): ACS mode at each timestep.
        panel (np.ndarray): Solar panel illumination fraction at each timestep.
        power (np.ndarray): Power usage at each timestep.
        batterylevel (np.ndarray): Battery state of charge at each timestep.
        batteryalert (np.ndarray): Battery alert status at each timestep.
        obsid (np.ndarray): Observation ID at each timestep.
        recorder_volume_gb (np.ndarray): Recorder data volume in Gb at each timestep.
        recorder_fill_fraction (np.ndarray): Recorder fill fraction (0-1) at each timestep.
        recorder_alert (np.ndarray): Recorder alert level (0/1/2) at each timestep.
        data_generated_gb (np.ndarray): Data generated in Gb at each timestep.
        data_downlinked_gb (np.ndarray): Data downlinked in Gb at each timestep.

    Attributes:
        telemetry (Telemetry): Structured telemetry data container.
    """

    def __init__(
        self,
        config: MissionConfig,
        ephem: rust_ephem.Ephemeris | None = None,
        plan: Plan | None = None,
        begin: datetime | None = None,
        end: datetime | None = None,
        calculate_field_of_regard: bool = False,
    ) -> None:
        """Initialize DITL with spacecraft configuration.

        Args:
            config (MissionConfig): Spacecraft configuration containing all subsystems
                (spacecraft_bus, payload, solar_panel, battery, constraint,
                ground_stations). Must not be None.
            ephem (Ephemeris, optional): Ephemeris data for position and illumination calculations.
            plan (Plan, optional): Pre-planned pointing schedule to execute.
            begin (datetime, optional): Start time for simulation (timezone-aware).
            end (datetime, optional): End time for simulation (timezone-aware).
            calculate_field_of_regard (bool, optional): Whether to compute
                instantaneous field-of-regard telemetry (for_solid_angle_sr).
                Defaults to False.

        Raises:
            AssertionError: If config is None. MissionConfig must be provided as it contains
                all necessary spacecraft subsystems and constraints.

        Note:
            DITLMixin.__init__ is called to set up base simulation parameters.
            All subsystems are extracted from the provided config for direct access.
        """
        DITLMixin.__init__(
            self,
            config=config,
            ephem=ephem,
            begin=begin,
            end=end,
            plan=plan,
            calculate_field_of_regard=calculate_field_of_regard,
        )
        # DITL also needs solar_panel
        self.solar_panel = self.config.solar_panel

        # Event log
        self.log = DITLLog()
        # Wire log into ACS so it can log events
        self.acs.log = self.log
        # Per-step attitude constraint violation, recorded by calc()
        self._attitude_constraint_violations: list[tuple[str, str] | None] = []

    def calc(self) -> bool:
        """Execute Day In The Life simulation.

        Runs the complete DITL simulation by:
        1. Validating that ephemeris and plan are loaded
        2. Setting up timing for the simulation period
        3. Initializing telemetry arrays
        4. Executing the main simulation loop for each timestep
        5. Recording spacecraft state, power calculations, and battery changes

        The simulation loop:
        - Gets current pointing from ACS
        - Determines spacecraft mode (SCIENCE, SLEWING, PASS, SAA)
        - Calculates power usage based on mode and configuration
        - Calculates solar panel power generation
        - Updates battery (drain for usage, charge from panels)
        - Records all telemetry

        Returns:
            bool: True if simulation completed successfully, False if errors occurred
                (missing ephemeris, missing plan, or invalid ephemeris date range).

        Raises:
            ValueError: If the ephemeris or plan is missing, or the ephemeris does
                not cover the simulation date range.
            AttitudeRateContinuityError: If adjacent executed attitude samples
                exceed the configured maximum slew rate.

        Note:
            The simulation respects the class attributes:
            - begin: Start datetime (timezone-aware)
            - end: End datetime (timezone-aware)
            - step_size: Time step in seconds
            - ephem: Must be loaded before calling calc()
            - plan: Must be loaded before calling calc()
        """
        # A few sanity checks before we start
        if self.ephem is None:
            raise ValueError("ERROR: No ephemeris loaded")

        if self.plan is None:
            raise ValueError("ERROR: No plan loaded")

        self._begin_fault_run()
        self.acs.solar_array_drive_state = self.solar_panel.initial_drive_state()
        # Plans intentionally exclude runtime objects from their serialized form.
        # Rebind here as well as during construction so assigning Plan.load(...)
        # after DITL initialization remains safe.
        if isinstance(self.plan, Plan):
            self.plan.bind_runtime(self.config, self.ephem)

        self._reset_stored_momentum_tracker()

        # Set up ACS ephemeris if not already set
        if self.acs.ephem is None:
            self.acs.ephem = self.ephem

        # Set up timing aspect of simulation
        self.ustart = self.begin.timestamp()
        self.uend = self.end.timestamp()
        ephem_utime = [dt.timestamp() for dt in self.ephem.timestamp]
        if self.ustart not in ephem_utime or self.uend not in ephem_utime:
            raise ValueError("ERROR: Ephemeris does not cover simulation date range")

        self.utime = np.arange(self.ustart, self.uend, self.step_size).tolist()

        # Initialize telemetry arrays for backward compatibility
        simlen = len(self.utime)
        self.ra = np.zeros(simlen).tolist()
        self.dec = np.zeros(simlen).tolist()
        self.mode = np.zeros(simlen).astype(int).tolist()
        self.panel = np.zeros(simlen).tolist()
        self.obsid = np.zeros(simlen).astype(int).tolist()
        self.batterylevel = np.zeros(simlen).tolist()
        self.charge_state = np.zeros(simlen).astype(int).tolist()
        self.batteryalert = np.zeros(simlen).tolist()
        self.power = np.zeros(simlen).tolist()
        # Subsystem power tracking
        self.power_bus = np.zeros(simlen).tolist()
        self.power_payload = np.zeros(simlen).tolist()
        # Data recorder tracking
        self.recorder_volume_gb = np.zeros(simlen).tolist()
        self.recorder_fill_fraction = np.zeros(simlen).tolist()
        self.recorder_alert = np.zeros(simlen).astype(int).tolist()
        self.data_generated_gb = np.zeros(simlen).tolist()
        self.data_downlinked_gb = np.zeros(simlen).tolist()

        self.roll = np.zeros(simlen).tolist()
        self._attitude_constraint_violations = []

        self._start_plan_execution()

        ##
        ## DITL LOOP
        ##
        for i in range(simlen):
            # Advance the current plan entry. Plan entries are not guaranteed to
            # be time-ordered, so only fall back to a full lookup when the
            # cached entry no longer covers this timestep.
            if self.ppt is None or not (self.ppt.begin <= self.utime[i] < self.ppt.end):
                self.ppt = self.plan.which_ppt(self.utime[i])

            # End an expired plan entry before ACS processes this step's
            # commands, so a slew it deferred to this step cannot start.
            self._finish_expired_entry(self.utime[i])

            # Obtain the current pointing information
            ra, dec, roll, obsid = self.acs.pointing(self.utime[i])

            # Command the plan activities due now from the attitude ACS holds at
            # this step, then re-enter ACS so they take effect at this step.
            if self._execute_plan(self.utime[i]):
                ra, dec, roll, obsid = self.acs.pointing(self.utime[i])

            # Get current mode from ACS (it now determines mode internally)
            mode = self.acs.get_mode(self.utime[i])

            if (
                self.ppt is not None
                and self.ppt.collection_begin is None
                and mode in (ACSMode.SCIENCE, ACSMode.SLEWING)
            ):
                self.ppt.set_collection_window(self.config.payload.observation_timing)
            collection_seconds = self._collection_seconds_for_step(
                self.utime[i], mode, obsid
            )

            # Determine the power usage in Watts based on mode from config
            bus_power = self.spacecraft_bus.power(mode, in_eclipse=self.acs.in_eclipse)
            payload_power = self.payload.power(mode, in_eclipse=self.acs.in_eclipse)
            power_usage = bus_power + payload_power

            # Calculate solar panel illumination and power (more efficient than separate calls)
            panel_illumination, panel_power, drive_state = (
                self.solar_panel.evaluate_executed_attitude(
                    time=self.utime[i],
                    ra=ra,
                    dec=dec,
                    ephem=self.ephem,
                    drive_state=self.acs.solar_array_drive_state,
                    roll=roll,
                    acs_mode=mode,
                )
            )
            self.acs.solar_array_drive_state = drive_state
            assert isinstance(panel_illumination, float)
            assert isinstance(panel_power, float)

            # Record all the useful DITL values
            self.batteryalert[i] = self.battery.battery_alert
            self.ra[i] = ra
            self.dec[i] = dec
            self.roll[i] = roll
            self.mode[i] = mode
            self.panel[i] = panel_illumination
            self.power[i] = power_usage
            self.power_bus[i] = bus_power
            self.power_payload[i] = payload_power
            # Drain the battery based on power usage
            self.battery.drain(power_usage, self.step_size)
            # Charge the battery based on solar panel power
            self.battery.charge(panel_power, self.step_size)
            # Record battery level and charge state
            self.batterylevel[i] = self.battery.battery_level
            self.charge_state[i] = self.battery.charge_state
            self.obsid[i] = obsid

            # Create housekeeping telemetry record for fault checking
            science_target = (
                self.ppt
                if self.ppt is not None
                and issubclass(type(self.ppt), PlanEntry)
                and mode == ACSMode.SCIENCE
                else None
            )
            mounted_target = (
                science_target
                if science_target is not None and science_target.uses_mounted_attitude()
                else None
            )
            instrument_roll = (
                mounted_target.roll if mounted_target is not None else roll
            )
            if (
                mounted_target is not None
                and self.acs.last_slew is not None
                and self.acs.last_slew.instrument_roll is not None
            ):
                instrument_roll = self.acs.last_slew.instrument_roll
            if instrument_roll == -1.0:
                instrument_roll = 0.0
            if mounted_target is not None:
                telescope = mounted_target.science_telescope()
                assert telescope is not None
                nominal_roll = optimum_instrument_roll(
                    mounted_target.ra,
                    mounted_target.dec,
                    self.utime[i],
                    self.ephem,
                    telescope,
                    self.solar_panel,
                    self.constraint,
                    drive_state=self.acs.solar_array_drive_state,
                )
            else:
                nominal_roll = optimum_body_roll(
                    ra,
                    dec,
                    self.utime[i],
                    self.ephem,
                    self.solar_panel,
                    drive_state=self.acs.solar_array_drive_state,
                )
            roll_offset_deg = (instrument_roll - nominal_roll + 180.0) % 360.0 - 180.0
            sun_angle_deg = self._compute_sun_angle(self.utime[i], ra, dec)
            _sun_bv = scbodyvector(
                np.radians(ra),
                np.radians(dec),
                np.radians(roll),
                radec2vec(
                    np.radians(float(self.ephem.sun_ra_deg[i])),
                    np.radians(float(self.ephem.sun_dec_deg[i])),
                ),
            )
            sun_body_vector: list[float] = [
                float(_sun_bv[0]),
                float(_sun_bv[1]),
                float(_sun_bv[2]),
            ]
            _pos = np.asarray(position_vectors(self.ephem, "gcrs")[i], dtype=np.float64)
            earth_body_vector: list[float] = list(-_pos / np.linalg.norm(_pos))
            for_solid_angle_sr = (
                self.constraint.instantaneous_field_of_regard(utime=self.utime[i])
                if self.calculate_field_of_regard
                else None
            )
            if mounted_target is not None:
                all_names = mounted_target.attitude_constraint_names(
                    list(AttitudeConstraintScope),
                    (ra, dec, roll),
                    self.utime[i],
                    mode,
                )
                in_constraint_name = all_names[0] if all_names else None
            else:
                violated = self.constraint.in_constraint(
                    ra, dec, self.utime[i], target_roll=roll, acs_mode=mode
                )
                in_constraint_name = None
                if violated:
                    in_constraint_name = (
                        all_attitude_constraint_name(
                            self.constraint,
                            ra,
                            dec,
                            self.utime[i],
                            target_roll=roll,
                            acs_mode=mode,
                        )
                        or "Unknown"
                    )
            scopes = self.config.attitude_constraint_scopes_for_mode(mode)
            scoped_names = (
                mounted_target.attitude_constraint_names(
                    scopes,
                    (ra, dec, roll),
                    self.utime[i],
                    mode,
                )
                if mounted_target is not None
                else []
            )
            if mounted_target is not None:
                scope_constraint_name = scoped_names[0] if scoped_names else None
            else:
                scope_constraint_name = attitude_constraint_name_for_scopes(
                    self.constraint,
                    scopes,
                    ra,
                    dec,
                    self.utime[i],
                    target_roll=roll,
                    acs_mode=mode,
                )
            scope_label = attitude_constraint_scope_label(scopes)
            self._attitude_constraint_violations.append(
                (scope_constraint_name, scope_label)
                if scope_constraint_name is not None
                else None
            )
            _q = attitude_to_quat(ra, dec, roll)
            momentum_sample = self._update_stored_momentum(self.utime[i], _pos, _q)
            drive_angles = self._solar_array_drive_telemetry()
            hk = Housekeeping(
                timestamp=datetime.fromtimestamp(self.utime[i], tz=timezone.utc),
                ra=ra,
                dec=dec,
                roll=roll,
                roll_offset_deg=roll_offset_deg,
                acs_mode=mode,
                collection_seconds=collection_seconds,
                panel_illumination=panel_illumination,
                solar_array_drive_angles=drive_angles,
                power_usage=power_usage,
                power_bus=bus_power,
                power_payload=payload_power,
                battery_level=self.battery.battery_level,
                charge_state=int(self.battery.charge_state),
                battery_alert=self.battery.battery_alert,
                obsid=obsid,
                recorder_volume_gb=self.recorder.current_volume_gb,
                recorder_fill_fraction=self.recorder.get_fill_fraction(),
                recorder_alert=self.recorder.get_alert_level(),
                sun_angle_deg=sun_angle_deg,
                for_solid_angle_sr=for_solid_angle_sr,
                in_eclipse=self.acs.in_eclipse,
                star_tracker_hard_violations=self.acs.star_tracker_hard_violations,
                star_tracker_soft_violations=self.acs.star_tracker_soft_violations,
                star_tracker_functional_count=self.acs.star_tracker_functional_count,
                star_tracker_status=self.acs.star_tracker_status,
                radiator_hard_violations=self.acs.radiator_hard_violations,
                telescope_hard_violations=self.acs.telescope_hard_violations,
                in_constraint=in_constraint_name,
                attitude_constraint=scope_constraint_name,
                attitude_constraint_scope=scope_label
                if scope_constraint_name is not None
                else None,
                radiator_sun_exposure=self.acs.radiator_sun_exposure,
                radiator_earth_exposure=self.acs.radiator_earth_exposure,
                radiator_heat_dissipation_w=self.acs.radiator_heat_dissipation_w,
                sun_body_vector=sun_body_vector,
                earth_body_vector=earth_body_vector,
                quat_w=float(_q[0]),
                quat_x=float(_q[1]),
                quat_y=float(_q[2]),
                quat_z=float(_q[3]),
                gravity_gradient_torque_body_n_m=(
                    list(momentum_sample.gravity_gradient_torque_body_n_m)
                    if momentum_sample is not None
                    else None
                ),
                stored_momentum_body_n_m_s=(
                    list(momentum_sample.stored_momentum_body_n_m_s)
                    if momentum_sample is not None
                    else None
                ),
                stored_momentum_norm_n_m_s=(
                    momentum_sample.stored_momentum_norm_n_m_s
                    if momentum_sample is not None
                    else None
                ),
            )

            # Check fault management thresholds and red limit constraints
            self.fault_management.check(
                housekeeping=hk,
                acs=self.acs,
            )

            # Check if safe mode was requested by fault management
            if self.fault_management.safe_mode_requested and not self.acs.in_safe_mode:
                self.acs.request_safe_mode(self.utime[i])
                self.fault_management.safe_mode_requested = False  # Reset flag

            # Store housekeeping telemetry
            self.telemetry.housekeeping.append(hk)

            # Data management: generate and downlink data
            data_generated, data_downlinked = self._process_data_management(
                self.utime[i],
                mode,
                self.step_size,
                collection_seconds=collection_seconds,
            )

            # Record data telemetry (cumulative values)
            prev_generated = self.data_generated_gb[i - 1] if i > 0 else 0.0
            prev_downlinked = self.data_downlinked_gb[i - 1] if i > 0 else 0.0

            self.recorder_volume_gb[i] = self.recorder.current_volume_gb
            self.recorder_fill_fraction[i] = self.recorder.get_fill_fraction()
            self.recorder_alert[i] = self.recorder.get_alert_level()
            self.data_generated_gb[i] = prev_generated + data_generated
            self.data_downlinked_gb[i] = prev_downlinked + data_downlinked

            # Create payload data record if data was generated
            if data_generated > 0:
                pd = PayloadData(
                    timestamp=datetime.fromtimestamp(self.utime[i], tz=timezone.utc),
                    data_size_gb=data_generated,
                )
                self.telemetry.data.append(pd)

        self._assert_attitude_rate_continuity()
        self._attach_execution_timeseries_to_plan()
        return not any(
            event.name
            in ("attitude_execution", "attitude_braking", "attitude_recovery")
            and event.event_type == "operational_fault"
            for event in self.fault_management.events
        )

    def validate_plan_matches_execution(self) -> list[PlanExecutionMismatch]:
        """Compare the executed telemetry from :meth:`calc` with the plan.

        Returns:
            list[PlanExecutionMismatch]: Science, contact, attitude-constraint
                and attitude-rate mismatches. Empty when the plan executed as
                written.
        """
        return PlanExecutionValidator(
            config=self.config,
            plan=self.plan,
            utime=self.utime,
            ra=self.ra,
            dec=self.dec,
            roll=self.roll,
            obsid=self.obsid,
            mode=self.mode,
            end_time=self.uend or self.ustart,
            passes=self.acs.passrequests.passes,
            attitude_violation_at=lambda index, _mode: (
                self._attitude_constraint_violations[index]
            ),
            rate_mismatches=[
                PlanExecutionMismatch(
                    utime=violation.utime,
                    message=str(violation),
                    obsid=violation.obsid,
                )
                for violation in self._attitude_rate_violations()
            ],
            science_obstypes=PLAN_SCIENCE_OBSTYPES,
        ).validate()

    def _start_plan_execution(self) -> None:
        """Order the plan entries for commanding and predict the passes they use."""
        self._entries_to_command = sorted(
            (entry for entry in self.plan if float(entry.end) > self.ustart),
            key=lambda entry: float(entry.begin),
        )
        self._next_entry_index = 0
        self._active_entry: PlanEntry | None = None
        self._active_pass: Pass | None = None
        self._entry_passes: dict[int, Pass] = {}
        if any(
            entry_obstype(entry) == ObsType.GSP for entry in self._entries_to_command
        ):
            self._resolve_planned_passes()

    def _resolve_planned_passes(self) -> None:
        """Match each GSP entry to a predicted pass and drop unplanned passes.

        The ACS starts whichever pass is current when a START_PASS command
        executes, so only the passes the plan schedules are kept.
        """
        passrequests = self.acs.passrequests
        if not passrequests.passes:
            length = max(1, int(np.ceil((self.uend - self.ustart) / 86400)))
            passrequests.get(self.begin.year, self.begin.timetuple().tm_yday, length)
        planned: list[Pass] = []
        for entry in self._entries_to_command:
            if entry_obstype(entry) != ObsType.GSP:
                continue
            gspass = matching_pass_for_entry(entry, passrequests.passes)
            if gspass is None:
                self.log.log_event(
                    utime=float(entry.begin),
                    event_type="ERROR",
                    description=(
                        f"No predicted pass matches planned contact at station "
                        f"{entry.station} from {unixtime2date(entry.contact_begin or 0)}"
                        f" to {unixtime2date(entry.contact_end or 0)}"
                    ),
                    obsid=entry.obsid,
                )
                continue
            profile = self._planned_tracking_profile(entry, gspass)
            if profile is None and entry.track_start_ra is not None:
                self.log.log_event(
                    utime=float(entry.begin),
                    event_type="ERROR",
                    description="No pass tracking profile matches the planned attitude",
                    obsid=entry.obsid,
                )
                continue
            if profile:
                gspass.select_tracking_profile(profile)
            self._entry_passes[id(entry)] = gspass
            planned.append(gspass)
        passrequests.passes = planned

    def _execute_plan(self, utime: float) -> bool:
        """Issue the ACS commands that the plan makes due at ``utime``.

        Returns:
            bool: True if the ACS state changed and must be re-evaluated.
        """
        if self.acs.in_safe_mode:
            return False
        changed = False
        if (
            self._active_pass is not None
            and self.acs.current_pass is None
            and self._active_pass.in_pass(utime)
            and self._active_pass.at_selected_tracking_attitude(
                utime, self.acs.ra, self.acs.dec, self.acs.roll
            )
        ):
            self.acs.enqueue_command(
                ACSCommand(
                    command_type=ACSCommandType.START_PASS,
                    execution_time=utime,
                )
            )
            changed = True
        while self._next_entry_index < len(self._entries_to_command):
            entry = self._entries_to_command[self._next_entry_index]
            if float(entry.begin) > utime:
                break
            self._next_entry_index += 1
            if float(entry.end) <= utime:
                continue
            if self._active_entry is not None:
                self._finish_active_entry(utime)
                changed = True
            changed = self._command_entry(entry, utime) or changed
        return changed

    def _finish_expired_entry(self, utime: float) -> None:
        """End the active plan entry if it has expired by ``utime``."""
        if self.acs.in_safe_mode:
            return
        if self._active_entry is not None and utime >= float(self._active_entry.end):
            self._finish_active_entry(utime)

    def _finish_active_entry(self, utime: float) -> None:
        """End the activity commanded for the active plan entry."""
        entry = self._active_entry
        assert entry is not None
        obstype = entry_obstype(entry)
        if obstype == ObsType.GSP:
            if self.acs.current_pass is not None:
                self.acs.enqueue_command(
                    ACSCommand(
                        command_type=ACSCommandType.END_PASS,
                        execution_time=utime,
                    )
                )
            self._active_pass = None
        elif obstype == ObsType.CHARGE:
            self.acs.request_end_battery_charge(utime)
        else:
            self.acs.cancel_pending_slews(entry.obsid)
            self.acs.end_science_observation()
        self._active_entry = None

    def _command_entry(self, entry: PlanEntry, utime: float) -> bool:
        """Command the activity a plan entry describes, starting at ``utime``.

        Returns:
            bool: True if the entry was commanded.
        """
        obstype = entry_obstype(entry)
        if obstype in PLAN_SCIENCE_OBSTYPES:
            self._command_science_entry(entry, utime)
        elif obstype == ObsType.GSP:
            if not self._command_pass_entry(entry, utime):
                return False
        elif obstype == ObsType.CHARGE:
            self.acs.request_battery_charge(
                utime, entry.ra, entry.dec, entry.roll, entry.obsid
            )
        else:
            self.log.log_event(
                utime=utime,
                event_type="ERROR",
                description=(
                    f"Plan entry {entry.obsid} has obstype {entry.obstype!r}, "
                    "which DITL cannot execute; skipping it"
                ),
                obsid=entry.obsid,
            )
            return False
        self._active_entry = entry
        return True

    def _command_science_entry(self, entry: PlanEntry, utime: float) -> None:
        """Slew to a science entry's target attitude."""
        instrument_roll = entry.roll
        mounted = entry.uses_mounted_attitude()
        if instrument_roll == -1.0:
            telescope = entry.science_telescope()
            instrument_roll = (
                optimum_instrument_roll(
                    entry.ra,
                    entry.dec,
                    utime,
                    self.ephem,
                    telescope,
                    self.solar_panel,
                    self.constraint,
                    drive_state=self.acs.solar_array_drive_state,
                )
                if telescope is not None
                else optimum_body_roll(
                    entry.ra,
                    entry.dec,
                    utime,
                    self.ephem,
                    self.solar_panel,
                    self.constraint,
                    drive_state=self.acs.solar_array_drive_state,
                )
            )
            entry.roll = instrument_roll
        body_ra, body_dec, body_roll = entry.target_body_attitude(instrument_roll)
        entry.spacecraft_attitude = (body_ra, body_dec, body_roll) if mounted else None
        self.acs._enqueue_slew(
            body_ra,
            body_dec,
            entry.obsid,
            utime,
            obstype=ObsType.PPT,
            roll=body_roll,
            target_request=entry,
            instrument_roll=instrument_roll,
        )

    def _command_pass_entry(self, entry: PlanEntry, utime: float) -> bool:
        """Slew onto the tracking profile of a GSP entry's pass.

        Returns:
            bool: False if no predicted pass matches the entry.
        """
        gspass = self._entry_passes.get(id(entry))
        if gspass is None:
            return False
        # Join an already-running contact on the profile, as QueueDITL does.
        end_ra, end_dec, end_roll = (
            gspass.gsstartra,
            gspass.gsstartdec,
            gspass.gsstartroll,
        )
        if utime >= gspass.begin:
            ra, dec, roll = gspass.attitude_at(utime)
            if ra is not None and dec is not None:
                end_ra, end_dec, end_roll = ra, dec, roll
        slew = Slew(config=self.config)
        slew.startra, slew.startdec, slew.startroll = (
            self.acs.ra,
            self.acs.dec,
            self.acs.roll,
        )
        slew.slewstart = utime
        slew.endra, slew.enddec, slew.endroll = end_ra, end_dec, end_roll
        slew.obstype = ObsType.GSP
        slew.obsid = gspass.obsid
        slew.calc_slewtime()
        self.acs.enqueue_command(
            ACSCommand(
                command_type=ACSCommandType.SLEW_TO_TARGET,
                execution_time=utime,
                slew=slew,
            )
        )
        self._active_pass = gspass
        return True

    @staticmethod
    def _planned_tracking_profile(
        entry: PlanEntry, gspass: Pass
    ) -> list[tuple[float, float, float]] | None:
        """Return the pass tracking profile that starts at the entry's planned attitude."""
        profiles = gspass.available_tracking_profiles()
        if (
            entry.track_start_ra is None
            or entry.track_start_dec is None
            or entry.track_start_roll is None
        ):
            return profiles[0] if profiles else None
        for profile in profiles:
            ra, dec, roll = profile[0]
            if (
                angular_separation(ra, dec, entry.track_start_ra, entry.track_start_dec)
                <= 1e-6
                and abs((roll - entry.track_start_roll + 180.0) % 360.0 - 180.0) <= 1e-6
            ):
                return profile
        return None

    def _compute_sun_angle(self, utime: float, ra: float, dec: float) -> float | None:
        """Compute angular distance from pointing to the Sun in degrees."""
        if self.ephem is None:
            return None

        try:
            idx = self.ephem.index(dtutcfromtimestamp(utime))
            sun_ra = self.ephem.sun_ra_deg[idx]
            sun_dec = self.ephem.sun_dec_deg[idx]
        except Exception:
            return None

        return angular_separation(sun_ra, sun_dec, ra, dec)


class DITLs:
    """Container for analyzing results of multiple DITL simulations.

    Stores and provides analysis methods for a collection of DITL objects,
    typically from Monte Carlo simulations where the same scenario is run
    multiple times with varying inputs or random effects.

    Attributes:
        ditls (list[DITL]): List of DITL simulation results.
        total (int): Total count (used for statistics).
        suncons (int): Sun constraint violations count (used for statistics).

    Example:
        >>> ditls = DITLs()
        >>> for config in configs:
        ...     ditl = DITL(config=config)
        ...     ditl.calc()
        ...     ditls.append(ditl)
        >>> num_simulations = len(ditls)
        >>> passes_per_sim = ditls.number_of_passes
    """

    def __init__(self) -> None:
        """Initialize empty DITLs collection.

        Creates an empty list to store DITL simulation results and initializes
        statistics counters.
        """
        self.ditls: list[DITL] = list()
        self.total = 0
        self.suncons = 0

    def __getitem__(self, number: int) -> "DITL":
        """Get DITL simulation result by index.

        Args:
            number (int): Index of the DITL to retrieve.

        Returns:
            DITL: The DITL simulation result at the given index.

        Raises:
            IndexError: If index is out of range.
        """
        return self.ditls[number]

    def __len__(self) -> int:
        """Get number of DITL simulations in collection.

        Returns:
            int: Number of DITL results stored.
        """
        return len(self.ditls)

    def append(self, ditl: "DITL") -> None:
        """Add a DITL simulation result to the collection.

        Args:
            ditl (DITL): The DITL simulation result to add.
        """
        self.ditls.append(ditl)

    @property
    def number_of_passes(self) -> list[int]:
        """Get number of executed passes for each DITL simulation.

        Returns:
            list[int]: List where each element is the count of executed passes
                for the corresponding DITL simulation.
        """
        return [len(d.executed_passes) for d in self.ditls]
