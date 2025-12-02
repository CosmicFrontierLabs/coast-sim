import matplotlib.pyplot as plt
import rust_ephem

from ..config import MissionConfig
from ..simulation.acs import ACS
from ..simulation.passes import PassTimes
from ..targets import Plan, PlanEntry


class DITLMixin:
    ppt: PlanEntry | None
    ra: list[float]
    dec: list[float]
    roll: list[float]
    mode: list[int]
    panel: list[float]
    power: list[float]
    panel_power: list[float]
    batterylevel: list[float]
    charge_state: list[int]
    obsid: list[int]
    plan: Plan
    utime: list
    ephem: rust_ephem.Ephemeris | None
    # Subsystem power tracking
    power_bus: list[float]
    power_payload: list[float]
    # Data recorder tracking
    recorder_volume_gb: list[float]
    recorder_fill_fraction: list[float]
    recorder_alert: list[int]
    data_generated_gb: list[float]
    data_downlinked_gb: list[float]

    def __init__(
        self, config: MissionConfig, ephem: rust_ephem.Ephemeris | None = None
    ) -> None:
        # Initialize mixin
        self.config = config

        # Set ephemeris if provided
        if ephem is not None:
            self.ephem = ephem
            self.config.constraint.ephem = ephem
        else:
            assert config.constraint.ephem is not None, (
                "Ephemeris must be set in Config Constraint"
            )
            self.ephem = config.constraint.ephem

        self.ra = []
        self.dec = []
        self.utime = []
        self.mode = []
        self.obsid = []
        self.ephem = ephem
        # Defining when the model is run
        self.step_size = 60  # seconds
        self.ustart = 0.0  # Calculate these
        self.uend = 0.0  # later
        self.plan = Plan()
        self.saa = None
        self.passes = PassTimes(config=config)
        self.executed_passes = PassTimes(config=config)

        # Set up event based ACS
        assert self.config.constraint.ephem is not None, (
            "Ephemeris must be set in Config Constraint"
        )
        # Note: log will be set by subclass (DITL/QueueDITL) before use
        # For now, create ACS without log (will be set later)
        self.acs = ACS(config=self.config, log=None)

        # Current target
        self.ppt = None

        # Initialize common subsystems (can be overridden by subclasses)
        self._init_subsystems()

    def _init_subsystems(self) -> None:
        """Initialize subsystems from config. Can be overridden by subclasses."""
        self.constraint = self.config.constraint
        self.battery = self.config.battery
        self.spacecraft_bus = self.config.spacecraft_bus
        self.payload = self.config.payload
        self.recorder = self.config.recorder

    def plot(self) -> None:
        """Plot DITL timeline.

        .. deprecated::
            Use :func:`conops.visualization.plot_ditl_telemetry` instead.
            This method is maintained for backward compatibility.
        """
        from ..visualization import plot_ditl_telemetry

        plot_ditl_telemetry(self, config=getattr(self.config, "visualization", None))
        plt.show()

    def _find_current_pass(self, utime: float):
        """Find the current pass at the given time.

        Args:
            utime: Unix timestamp to check.

        Returns:
            Pass object if currently in a pass, None otherwise.
        """
        # Check in ACS passrequests (scheduled passes)
        if hasattr(self, "acs") and hasattr(self.acs, "passrequests"):
            if self.acs.passrequests.passes:
                for pass_obj in self.acs.passrequests.passes:
                    if pass_obj.in_pass(utime):
                        return pass_obj

        # Fallback to executed_passes for backwards compatibility
        if hasattr(self, "executed_passes") and self.executed_passes is not None:
            if self.executed_passes.passes:
                for pass_obj in self.executed_passes.passes:
                    if pass_obj.in_pass(utime):
                        return pass_obj

        return None

    def _process_data_management(
        self, utime: float, mode, step_size: int
    ) -> tuple[float, float]:
        """Process data generation and downlink for a single timestep.

        Args:
            utime: Unix timestamp for current timestep.
            mode: Current ACS mode.
            step_size: Time step in seconds.

        Returns:
            Tuple of (data_generated, data_downlinked) in Gb for this timestep.
        """
        from ..common.enums import ACSMode

        data_generated = 0.0
        data_downlinked = 0.0

        # Generate data during SCIENCE mode
        if mode == ACSMode.SCIENCE:
            data_generated = self.payload.data_generated(step_size)
            self.recorder.add_data(data_generated)

        # Downlink data during PASS mode
        if mode == ACSMode.PASS:
            current_pass = self._find_current_pass(utime)
            if current_pass is not None:
                station = self.config.ground_stations.get(current_pass.station)

                # Determine actual data rate based on both ground station and spacecraft capabilities
                effective_rate_mbps = self._get_effective_data_rate(
                    station, current_pass
                )

                if effective_rate_mbps is not None and effective_rate_mbps > 0:
                    # Convert Mbps to Gb per step: Mbps * seconds / 1000 / 8 = Gb
                    megabits_per_step = effective_rate_mbps * step_size
                    data_to_downlink = megabits_per_step / 1000.0 / 8.0  # Convert to Gb
                    data_downlinked = self.recorder.remove_data(data_to_downlink)

        return data_generated, data_downlinked

    def _get_effective_data_rate(self, station, current_pass) -> float | None:
        """Calculate effective downlink data rate based on ground station and spacecraft capabilities.

        The effective rate is, per band, min(GS downlink rate, SC downlink rate);
        we take the maximum of this across all common bands.

        Args:
            station: GroundStation object with antenna capabilities
            current_pass: Pass object with communications configuration

        Returns:
            Effective data rate in Mbps, or None if no compatible bands/rates
        """
        # If pass has no comms config, use GS overall maximum across bands
        if current_pass.config.spacecraft_bus.communications is None:
            return station.get_overall_max_downlink()

        # If GS has no per-band capabilities, no defined rate
        gs_bands = set(station.supported_bands()) if station.bands else set()
        if not gs_bands:
            # No bands defined on ground station
            return None

        # Compute effective rate per common band
        best_effective = 0.0
        for band in gs_bands:
            gs_rate = station.get_downlink_rate(band) or 0.0
            sc_rate = (
                current_pass.config.spacecraft_bus.communications.get_downlink_rate(
                    band
                )
                or 0.0
            )
            if gs_rate > 0.0 and sc_rate > 0.0:
                effective = min(gs_rate, sc_rate)
                if effective > best_effective:
                    best_effective = effective

        return best_effective if best_effective > 0.0 else None
