"""Telemetry data storage for DITL simulations."""

from collections.abc import Iterable
from typing import Any, SupportsIndex, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from ..common import ACSMode


class Housekeeping(BaseModel):
    """
    Housekeeping telemetry record from ACS.

    Records spacecraft state and power data at a single timestep.

    Attributes:
        utime: Unix timestamp of the recording
        ra: Right ascension in degrees
        dec: Declination in degrees
        roll: Roll angle in degrees
        mode: Current ACS mode (SCIENCE, SLEWING, SAFE, SAA, etc.)
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
    """

    utime: float | None = Field(default=None, description="Unix timestamp")
    ra: float | None = Field(default=None, description="Right ascension in degrees")
    dec: float | None = Field(default=None, description="Declination in degrees")
    roll: float | None = Field(default=0.0, description="Roll angle in degrees")
    mode: ACSMode | int | None = Field(default=None, description="ACS mode")
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
        utime: Unix timestamp of the data record
        data_size_gb: Size of data generated in Gb
    """

    utime: float = Field(description="Unix timestamp")
    data_size_gb: float = Field(description="Size of data generated in Gb")


T = TypeVar("T")


class _CacheInvalidatingList(list[T]):
    """List wrapper that invalidates cache when modified."""

    def __init__(
        self, *args: Any, telemetry: "Telemetry | None" = None, **kwargs: Any
    ) -> None:
        """Initialize with reference to parent Telemetry object."""
        super().__init__(*args, **kwargs)
        self._telemetry = telemetry

    def append(self, item: T) -> None:
        """Append item and clear cache."""
        super().append(item)
        if self._telemetry and hasattr(self._telemetry, "clear_cache"):
            self._telemetry.clear_cache()

    def extend(self, items: Iterable[T]) -> None:
        """Extend with items and clear cache."""
        super().extend(items)
        if self._telemetry and hasattr(self._telemetry, "clear_cache"):
            self._telemetry.clear_cache()

    def __setitem__(self, key: SupportsIndex | slice, value: T | Iterable[T]) -> None:
        """Set item and clear cache."""
        super().__setitem__(key, value)  # type: ignore
        if self._telemetry and hasattr(self._telemetry, "clear_cache"):
            self._telemetry.clear_cache()

    def __delitem__(self, key: SupportsIndex | slice) -> None:
        """Delete item and clear cache."""
        super().__delitem__(key)
        if self._telemetry and hasattr(self._telemetry, "clear_cache"):
            self._telemetry.clear_cache()

    def insert(self, index: SupportsIndex, item: T) -> None:
        """Insert item and clear cache."""
        super().insert(index, item)
        if self._telemetry and hasattr(self._telemetry, "clear_cache"):
            self._telemetry.clear_cache()

    def remove(self, item: T) -> None:
        """Remove item and clear cache."""
        super().remove(item)
        if self._telemetry and hasattr(self._telemetry, "clear_cache"):
            self._telemetry.clear_cache()

    def pop(self, index: SupportsIndex = -1) -> T:
        """Pop item and clear cache."""
        result = super().pop(index)
        if self._telemetry and hasattr(self._telemetry, "clear_cache"):
            self._telemetry.clear_cache()
        return result

    def clear(self) -> None:
        """Clear list and invalidate cache."""
        super().clear()
        if self._telemetry and hasattr(self._telemetry, "clear_cache"):
            self._telemetry.clear_cache()

    def __getattr__(self, name: str) -> list[Any]:
        """Allow attribute access to extract field values from all items."""
        # Check if this is a valid attribute of the items in the list
        if self and hasattr(self[0], name):
            return [getattr(item, name) for item in self]
        # Fall back to normal attribute access
        raise AttributeError(
            f"'{type(self).__name__}' object has no attribute '{name}'"
        )


class Telemetry(BaseModel):
    """
    Container for telemetry data from a DITL simulation.

    Stores housekeeping records (ACS state at each timestep) and payload data records
    (data generation events), organized as lists of typed records rather than parallel arrays.

    Attributes:
        housekeeping: List of Housekeeping records from the simulation
        data: List of PayloadData records generated during the simulation
    """

    housekeeping: list[Housekeeping] = Field(
        default_factory=list, description="List of housekeeping records"
    )
    data: list[PayloadData] = Field(
        default_factory=list, description="List of payload data records"
    )

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self, **data: Any) -> None:
        """Initialize Telemetry and set up field cache."""
        super().__init__(**data)
        # Cache for extracted fields to avoid recomputation
        self._field_cache: dict[str, list[Any]] = {}
        self._fields_cache: dict[tuple[str, ...], dict[str, list[Any]]] = {}
        # Convert lists to cache-invalidating versions
        self._convert_to_cache_aware_lists()

    def _convert_to_cache_aware_lists(self) -> None:
        """Convert housekeeping and data lists to cache-invalidating versions."""
        if not isinstance(self.housekeeping, _CacheInvalidatingList):
            housekeeping: _CacheInvalidatingList[Housekeeping] = _CacheInvalidatingList(
                self.housekeeping, telemetry=self
            )
            object.__setattr__(self, "housekeeping", housekeeping)
        if not isinstance(self.data, _CacheInvalidatingList):
            data: _CacheInvalidatingList[PayloadData] = _CacheInvalidatingList(
                self.data, telemetry=self
            )
            object.__setattr__(self, "data", data)

    def __setattr__(self, name: str, value: Any) -> None:
        """Override to clear cache and wrap lists for housekeeping or data fields."""
        # Wrap lists and clear cache if housekeeping or data fields are being reassigned
        if name in ("housekeeping", "data") and hasattr(self, "_field_cache"):
            self._field_cache.clear()
            self._fields_cache.clear()
            # Wrap the value if it's not already wrapped
            if not isinstance(value, _CacheInvalidatingList):
                if name == "housekeeping":
                    value = _CacheInvalidatingList[Housekeeping](value, telemetry=self)
                else:  # name == "data"
                    value = _CacheInvalidatingList[PayloadData](value, telemetry=self)
        super().__setattr__(name, value)

    def get_housekeeping_field(self, field_name: str) -> list[Any]:
        """
        Get a cached array of a single housekeeping field.

        Uses internal caching to avoid recomputing the same field multiple times.

        Parameters
        ----------
        field_name : str
            Name of the field to extract (must be a valid Housekeeping attribute)

        Returns
        -------
        list
            List of field values from all housekeeping records

        Raises
        ------
        AttributeError
            If field_name is not a valid Housekeeping attribute
        """
        if field_name not in self._field_cache:
            self._field_cache[field_name] = Housekeeping.extract_field(
                self.housekeeping, field_name
            )
        return self._field_cache[field_name]

    def get_housekeeping_fields(self, field_names: list[str]) -> dict[str, list[Any]]:
        """
        Get cached arrays of multiple housekeeping fields.

        Uses internal caching to avoid recomputing the same fields multiple times.

        Parameters
        ----------
        field_names : list[str]
            Names of fields to extract (must be valid Housekeeping attributes)

        Returns
        -------
        dict[str, list[Any]]
            Dictionary mapping field names to lists of values

        Raises
        ------
        AttributeError
            If any field_name is not a valid Housekeeping attribute
        """
        field_tuple = tuple(sorted(field_names))
        if field_tuple not in self._fields_cache:
            self._fields_cache[field_tuple] = Housekeeping.extract_fields(
                self.housekeeping, field_names
            )
        return self._fields_cache[field_tuple]

    def clear_cache(self) -> None:
        """Clear all cached field extractions."""
        self._field_cache.clear()
        self._fields_cache.clear()
