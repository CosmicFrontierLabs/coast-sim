import numpy as np

from .config import Config  # type: ignore[attr-defined]
from .ditl_mixin import DITLMixin


class DITL(DITLMixin):
    """Day In The Life (DITL) simulation class.

    Simulates a single day of spacecraft operations by executing a pre-planned
    observing schedule (PPST - Pointing Plan and Scheduling Tool) and tracking
    spacecraft state including power usage, battery levels, and pointing angles.

    Inherits from DITLMixin which provides shared initialization and plotting
    functionality for DITL simulations.

    Attributes:
        constraint (Constraint): Spacecraft constraint model (sun, earth, moon avoidance).
        battery (Battery): Battery model for power tracking and management.
        spacecraft_bus (SpacecraftBus): Spacecraft bus configuration and power draw.
        payload (Payload): Instrument configuration and power draw.
        solar_panel (SolarPanelSet): Solar panel configuration and power generation.
        ephem (Ephemeris): Ephemeris data for position and illumination calculations.
        ppst (Plan): Pre-planned pointing schedule to execute.
        acs (ACS): Attitude Control System for pointing and slew calculations.
        begin (datetime): Start time for simulation (default: Nov 27, 2018 00:00:00 UTC).
        end (datetime): End time for simulation (default: Nov 28, 2018 00:00:00 UTC).
        step_size (int): Time step in seconds (default: 60).

    Telemetry Arrays (populated during calc()):
        ra (np.ndarray): Right ascension at each timestep.
        dec (np.ndarray): Declination at each timestep.
        mode (np.ndarray): ACS mode at each timestep.
        panel (np.ndarray): Solar panel illumination fraction at each timestep.
        power (np.ndarray): Power usage at each timestep.
        batterylevel (np.ndarray): Battery state of charge at each timestep.
        batteryalert (np.ndarray): Battery alert status at each timestep.
        obsid (np.ndarray): Observation ID at each timestep.
    """

    def __init__(self, config: Config) -> None:
        """Initialize DITL with spacecraft configuration.

        Args:
            config (Config): Spacecraft configuration containing all subsystems
                (spacecraft_bus, payload, solar_panel, battery, constraint,
                ground_stations). Must not be None.

        Raises:
            AssertionError: If config is None. Config must be provided as it contains
                all necessary spacecraft subsystems and constraints.

        Note:
            DITLMixin.__init__ is called to set up base simulation parameters.
            All subsystems are extracted from the provided config for direct access.
        """
        DITLMixin.__init__(self, config=config)
        # Initialize subsystems from config
        self.constraint = self.config.constraint
        self.battery = self.config.battery
        self.spacecraft_bus = self.config.spacecraft_bus
        self.payload = self.config.payload
        self.solar_panel = self.config.solar_panel

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
            - ppst: Must be loaded before calling calc()
        """
        # A few sanity checks before we start
        if self.ephem is None:
            print("ERROR: No ephemeris loaded")
            return False
        if self.ppst is None:
            print("ERROR: No Plan loaded")
            return False

        # Set up ACS ephemeris if not already set
        if self.acs.ephem is None:
            self.acs.ephem = self.ephem

        # Set up timing aspect of simulation
        self.ustart = self.begin.timestamp()
        self.uend = self.end.timestamp()
        ephem_utime = [dt.timestamp() for dt in self.ephem.timestamp]
        if self.ustart not in ephem_utime or self.uend not in ephem_utime:
            print("ERROR: Ephemeris not valid for date range")
            return False
        self.utime = np.arange(self.ustart, self.uend, self.step_size).tolist()

        # Set up simulation telemetry arrays
        simlen = len(self.utime)
        self.ra = np.zeros(simlen).tolist()
        self.dec = np.zeros(simlen).tolist()
        self.mode = np.zeros(simlen).astype(int).tolist()
        self.panel = np.zeros(simlen).tolist()
        self.obsid = np.zeros(simlen).astype(int).tolist()
        self.batterylevel = np.zeros(simlen).tolist()
        self.batteryalert = np.zeros(simlen).tolist()
        self.power = np.zeros(simlen).tolist()
        # Subsystem power tracking
        self.power_bus = np.zeros(simlen).tolist()
        self.power_payload = np.zeros(simlen).tolist()

        # Set up initial target in ACS
        self.ppt = self.ppst.which_ppt(self.utime[0])
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
            # Record battery level
            self.batterylevel[i] = self.battery.battery_level
            self.obsid[i] = obsid

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

    def __init__(self):
        """Initialize empty DITLs collection.

        Creates an empty list to store DITL simulation results and initializes
        statistics counters.
        """
        self.ditls = list()
        self.reset_stats()
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
