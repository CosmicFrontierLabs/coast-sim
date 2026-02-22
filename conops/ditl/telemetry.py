"""Telemetry data storage for DITL simulations."""

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..common import ACSMode


class Housekeeping(BaseModel):
    """
    Housekeeping telemetry record from ACS.

    Records spacecraft state and power data at a single timestep.

    Attributes:
        timestamp: UTC timestamp of the recording
        ra: Right ascension in degrees
        dec: Declination in degrees
        roll: Roll angle in degrees
        acs_mode: Current ACS mode (SCIENCE, SLEWING, SAFE, SAA, etc.)
        panel_illumination: Solar panel illumination fraction (0-1)
        power_usage: Total power usage in W
        power_bus: Spacecraft bus power usage in W
        power_payload: Payload power usage in W
        battery_level: Battery state of charge (0-1)
        charge_state: Battery charging state (0=discharging, 1=charging)
        battery_alert: Battery alert level (0/1/2)
        obsid: Current observation ID
        recorder_volume_gb: Recorder data volume in Gb
        recorder_fill_fraction: Recorder fill fraction (0-1)
        recorder_alert: Recorder alert level (0/1/2)
        sun_angle_deg: Angular distance from pointing to the Sun in degrees
        in_eclipse: Whether spacecraft is in eclipse
    """

    timestamp: datetime = Field(description="UTC timestamp")
    ra: float | None = Field(default=None, description="Right ascension in degrees")
    dec: float | None = Field(default=None, description="Declination in degrees")
    roll: float | None = Field(default=0.0, description="Roll angle in degrees")
    acs_mode: ACSMode | int | None = Field(default=None, description="ACS mode")
    panel_illumination: float | None = Field(
        default=None, description="Solar panel illumination fraction (0-1)"
    )
    power_usage: float | None = Field(
        default=None, description="Total power usage in W"
    )
    power_bus: float | None = Field(
        default=None, description="Spacecraft bus power usage in W"
    )
    power_payload: float | None = Field(
        default=None, description="Payload power usage in W"
    )
    battery_level: float | None = Field(
        default=None, description="Battery state of charge (0-1)"
    )
    charge_state: int | None = Field(default=None, description="Battery charging state")
    battery_alert: int | None = Field(
        default=None, description="Battery alert level (0/1/2)"
    )
    obsid: int | None = Field(default=None, description="Current observation ID")
    recorder_volume_gb: float | None = Field(
        default=None, description="Recorder data volume in Gb"
    )
    recorder_fill_fraction: float | None = Field(
        default=None, description="Recorder fill fraction (0-1)"
    )
    recorder_alert: int | None = Field(
        default=None, description="Recorder alert level (0/1/2)"
    )
    sun_angle_deg: float | None = Field(
        default=None, description="Angular distance to Sun in degrees"
    )
    in_eclipse: bool | None = Field(
        default=None, description="Whether spacecraft is in eclipse"
    )
    star_tracker_hard_violations: int | None = Field(
        default=None, description="Number of star trackers violating hard constraints"
    )
    star_tracker_soft_violations: bool | None = Field(
        default=None, description="Whether any star tracker violates soft constraints"
    )
    star_tracker_functional_count: int | None = Field(
        default=None, description="Number of functional star trackers"
    )

    @classmethod
    def extract_field(cls, records: list["Housekeeping"], field_name: str) -> list[Any]:
        """
        Extract a single field from a list of Housekeeping records.

        Parameters
        ----------
        records : list[Housekeeping]
            List of housekeeping records
        field_name : str
            Name of the field to extract (must be a valid Housekeeping attribute)

        Returns
        -------
        list[Any]
            List of field values from all records

        Raises
        ------
        AttributeError
            If field_name is not a valid Housekeeping attribute
        """
        if not records:
            return []
        return [getattr(record, field_name) for record in records]

    @classmethod
    def extract_fields(
        cls, records: list["Housekeeping"], field_names: list[str]
    ) -> dict[str, list[Any]]:
        """
        Extract multiple fields from a list of Housekeeping records.

        Parameters
        ----------
        records : list[Housekeeping]
            List of housekeeping records
        field_names : list[str]
            Names of fields to extract (must be valid Housekeeping attributes)

        Returns
        -------
        dict[str, list]
            Dictionary mapping field names to lists of values

        Raises
        ------
        AttributeError
            If any field_name is not a valid Housekeeping attribute
        """
        if not records:
            return {name: [] for name in field_names}
        return {name: cls.extract_field(records, name) for name in field_names}


class PayloadData(BaseModel):
    """
    Payload data record.

    Records a data generation event with timestamp and data size.

    Attributes:
        timestamp: UTC timestamp of the data record
        data_size_gb: Size of data generated in Gb
    """

    timestamp: datetime = Field(description="UTC timestamp")
    data_size_gb: float = Field(description="Size of data generated in Gb")


class HousekeepingList(list[Housekeeping]):
    """List of Housekeeping records with convenient property access to fields."""

    @property
    def timestamp(self) -> list[datetime | None]:
        """Get timestamp values from all housekeeping records."""
        return [hk.timestamp for hk in self]

    @property
    def ra(self) -> list[float | None]:
        """Get RA values from all housekeeping records."""
        return [hk.ra for hk in self]

    @property
    def dec(self) -> list[float | None]:
        """Get Dec values from all housekeeping records."""
        return [hk.dec for hk in self]

    @property
    def roll(self) -> list[float | None]:
        """Get roll values from all housekeeping records."""
        return [hk.roll for hk in self]

    @property
    def acs_mode(self) -> list[ACSMode | int | None]:
        """Get ACS mode values from all housekeeping records."""
        return [hk.acs_mode for hk in self]

    @property
    def panel_illumination(self) -> list[float | None]:
        """Get panel illumination values from all housekeeping records."""
        return [hk.panel_illumination for hk in self]

    @property
    def power_usage(self) -> list[float | None]:
        """Get power usage values from all housekeeping records."""
        return [hk.power_usage for hk in self]

    @property
    def power_bus(self) -> list[float | None]:
        """Get power bus values from all housekeeping records."""
        return [hk.power_bus for hk in self]

    @property
    def power_payload(self) -> list[float | None]:
        """Get power payload values from all housekeeping records."""
        return [hk.power_payload for hk in self]

    @property
    def battery_level(self) -> list[float | None]:
        """Get battery level values from all housekeeping records."""
        return [hk.battery_level for hk in self]

    @property
    def charge_state(self) -> list[int | None]:
        """Get charge state values from all housekeeping records."""
        return [hk.charge_state for hk in self]

    @property
    def battery_alert(self) -> list[int | None]:
        """Get battery alert values from all housekeeping records."""
        return [hk.battery_alert for hk in self]

    @property
    def obsid(self) -> list[int | None]:
        """Get obsid values from all housekeeping records."""
        return [hk.obsid for hk in self]

    @property
    def recorder_volume_gb(self) -> list[float | None]:
        """Get recorder volume values from all housekeeping records."""
        return [hk.recorder_volume_gb for hk in self]

    @property
    def recorder_fill_fraction(self) -> list[float | None]:
        """Get recorder fill fraction values from all housekeeping records."""
        return [hk.recorder_fill_fraction for hk in self]

    @property
    def recorder_alert(self) -> list[int | None]:
        """Get recorder alert values from all housekeeping records."""
        return [hk.recorder_alert for hk in self]

    @property
    def sun_angle_deg(self) -> list[float | None]:
        """Get sun angle values from all housekeeping records."""
        return [hk.sun_angle_deg for hk in self]

    @property
    def in_eclipse(self) -> list[bool | None]:
        """Get eclipse state values from all housekeeping records."""
        return [hk.in_eclipse for hk in self]

    @property
    def star_tracker_hard_violations(self) -> list[int | None]:
        """Get star tracker hard violation counts from all housekeeping records."""
        return [hk.star_tracker_hard_violations for hk in self]

    @property
    def star_tracker_soft_violations(self) -> list[bool | None]:
        """Get star tracker soft violation states from all housekeeping records."""
        return [hk.star_tracker_soft_violations for hk in self]

    @property
    def star_tracker_functional_count(self) -> list[int | None]:
        """Get star tracker functional counts from all housekeeping records."""
        return [hk.star_tracker_functional_count for hk in self]


class Telemetry(BaseModel):
    """
    Container for telemetry data from a DITL simulation.

    Stores housekeeping records (ACS state at each timestep) and payload data records
    (data generation events), organized as lists of typed records rather than parallel arrays.

    Attributes:
        housekeeping: List of Housekeeping records from the simulation
        data: List of PayloadData records generated during the simulation
    """

    housekeeping: HousekeepingList = Field(
        default_factory=HousekeepingList, description="List of housekeeping records"
    )
    data: list[PayloadData] = Field(
        default_factory=list, description="List of payload data records"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)
