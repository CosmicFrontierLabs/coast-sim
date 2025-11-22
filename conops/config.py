from pydantic import BaseModel

from .battery import Battery
from .constraint import Constraint
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
    instruments: Payload
    battery: Battery
    constraint: Constraint
    ground_stations: GroundStationRegistry
