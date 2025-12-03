from datetime import datetime

import numpy as np
import rust_ephem

from conops.targets.plan import Plan

from ..config import MissionConfig
from .ditl_log import DITLLog
from .ditl_mixin import DITLMixin
from .ditl_stats import DITLStats


class DITL(DITLMixin, DITLStats):
    """Day In The Life (DITL) simulation class.

    Simulates a single day of spacecraft operations by executing a pre-planned
    observing schedule  and tracking spacecraft state including power usage,
    battery levels, pointing angles, and data management.

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
    """

    def __init__(
        self,
        config: MissionConfig,
        ephem: rust_ephem.Ephemeris | None = None,
        plan: Plan = Plan(),
        begin: datetime | None = None,
        end: datetime | None = None,
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
            step_size (int, optional): Time step in seconds (default: 60).

        Raises:
            AssertionError: If config is None. MissionConfig must be provided as it contains
                all necessary spacecraft subsystems and constraints.

        Note:
            DITLMixin.__init__ is called to set up base simulation parameters.
            All subsystems are extracted from the provided config for direct access.
        """
        DITLMixin.__init__(
            self, config=config, ephem=ephem, begin=begin, end=end, plan=plan
        )
        # DITL also needs solar_panel
        self.solar_panel = self.config.solar_panel

        # Event log
        self.log = DITLLog()
        # Wire log into ACS so it can log events (if ACS exists)
        if hasattr(self, "acs"):
            self.acs.log = self.log

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
            No exceptions raised; errors are logged to stdout and return False.

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

        # Set up simulation telemetry arrays
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

        # Set up initial target in ACS
        self.ppt = self.plan.which_ppt(self.utime[0])
        if self.ppt is not None:
            self.acs._enqueue_slew(
                self.ppt.ra,
                self.ppt.dec,
                self.ppt.obsid,
                self.utime[0],
                obstype=self.ppt.obstype,
            )

        ##
        ## DITL LOOP
        ##
        for i in range(simlen):
            # Obtain the current pointing information
            ra, dec, roll, obsid = self.acs.pointing(self.utime[i])

            # Get current mode from ACS (it now determines mode internally)
            mode = self.acs.get_mode(self.utime[i])

            # Determine the power usage in Watts based on mode from config
            bus_power = self.spacecraft_bus.power(mode, in_eclipse=self.acs.in_eclipse)
            payload_power = self.payload.power(mode, in_eclipse=self.acs.in_eclipse)
            power_usage = bus_power + payload_power

            # Calculate solar panel illumination and power (more efficient than separate calls)
            panel_illumination, panel_power = self.solar_panel.illumination_and_power(
                time=self.utime[i], ra=ra, dec=dec, ephem=self.ephem
            )
            assert isinstance(panel_illumination, float)
            assert isinstance(panel_power, float)

            # Record all the useful DITL values
            self.batteryalert[i] = self.battery.battery_alert
            self.ra[i] = ra
            self.dec[i] = dec
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

            # Data management: generate and downlink data
            data_generated, data_downlinked = self._process_data_management(
                self.utime[i], mode, self.step_size
            )

            # Record data telemetry (cumulative values)
            prev_generated = self.data_generated_gb[i - 1] if i > 0 else 0.0
            prev_downlinked = self.data_downlinked_gb[i - 1] if i > 0 else 0.0

            self.recorder_volume_gb[i] = self.recorder.current_volume_gb
            self.recorder_fill_fraction[i] = self.recorder.get_fill_fraction()
            self.recorder_alert[i] = self.recorder.get_alert_level()
            self.data_generated_gb[i] = prev_generated + data_generated
            self.data_downlinked_gb[i] = prev_downlinked + data_downlinked

        return True


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
