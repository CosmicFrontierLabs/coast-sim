from __future__ import annotations

from pydantic import BaseModel

from .battery import Battery
from .constraint import Constraint
from .fault_management import FaultManagement
from .groundstation import GroundStationRegistry
from .instrument import Payload
from .solar_panel import SolarPanelSet
from .spacecraft_bus import SpacecraftBus


class Config(BaseModel):
    """
    Configuration class for the spacecraft and its subsystems.
    """

    name: str = "Default Config"
    spacecraft_bus: SpacecraftBus
    solar_panel: SolarPanelSet
    payload: Payload
    battery: Battery
    constraint: Constraint
    ground_stations: GroundStationRegistry
    fault_management: FaultManagement | None = None

    def init_fault_management_defaults(self) -> None:
        """Initialize default fault thresholds if none provided.

        Currently sets up a battery_level threshold using the battery
        max_depth_of_discharge as the minimum allowed charge level for YELLOW
        and (max_depth_of_discharge + 0.1) as RED for demonstration.
        Users can override via config serialization.
        """
        if self.fault_management is None:
            return
        # Only add battery threshold if not already present
        if "battery_level" not in self.fault_management.thresholds:
            # Yellow alert at minimum allowed charge level (1.0 - max_depth_of_discharge)
            yellow = 1.0 - self.battery.max_depth_of_discharge
            # Red alert at 10% below the minimum allowed charge level
            red = max(yellow - 0.1, 0.0)  # Ensure non-negative
            self.fault_management.add_threshold(
                name="battery_level", yellow=yellow, red=red, direction="below"
            )

    @classmethod
    def from_json_file(cls, filepath: str) -> Config:
        """Load configuration from a JSON file."""
        with open(filepath) as f:
            return cls.model_validate_json(f.read())

    def to_json_file(self, filepath: str) -> None:
        """Save configuration to a JSON file."""
        with open(filepath, "w") as f:
            f.write(self.model_dump_json(indent=4))
