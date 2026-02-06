from typing import Any

from pydantic import BaseModel, Field, field_validator


class OnboardRecorder(BaseModel):
    """
    A model representing the spacecraft's onboard data storage device.

    This class simulates an onboard solid-state recorder (SSR) or data storage unit
    that buffers science data collected by instruments before downlink to ground stations.
    It tracks data volume in Gigabits (Gb) and provides alerts when storage is filling up.

    Attributes:
        name (str): The name of the recorder. Defaults to "Default Recorder".
        capacity_gb (float): Maximum storage capacity in Gigabits (Gb). Defaults to 32 Gb.
        current_volume_gb (float): Current data volume stored in Gigabits (Gb). Defaults to 0.
        yellow_threshold (float): Fraction of capacity that triggers yellow alert (0-1).
            Defaults to 0.7 (70% full).
        red_threshold (float): Fraction of capacity that triggers red alert (0-1).
            Defaults to 0.9 (90% full).

    Methods:
        add_data(data_gb): Add data to the recorder, returns amount actually stored.
        remove_data(data_gb): Remove data from the recorder during downlink.
        get_fill_fraction(): Get the current fill level as a fraction (0-1).
        get_alert_level(): Get current alert level (0=none, 1=yellow, 2=red).
        is_full(): Check if recorder is at or above capacity.
        available_capacity(): Get remaining available capacity in Gb.

    Example:
        >>> recorder = OnboardRecorder(name="SSR", capacity_gb=64)
        >>> recorder.add_data(10.5)  # Add 10.5 Gb
        10.5
        >>> recorder.get_fill_fraction()
        0.1640625
        >>> recorder.get_alert_level()
        0
    """

    name: str = Field(
        default="Default Recorder", description="Recorder name/identifier"
    )
    capacity_gb: float = Field(
        default=32.0, gt=0, description="Maximum storage capacity in Gigabits"
    )
    current_volume_gb: float = Field(
        default=0.0, ge=0, description="Current data volume stored in Gigabits"
    )
    yellow_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Fraction of capacity for yellow alert",
    )
    red_threshold: float = Field(
        default=0.9, ge=0.0, le=1.0, description="Fraction of capacity for red alert"
    )

    @field_validator("current_volume_gb")
    @classmethod
    def validate_current_volume(cls, v: float, info: Any) -> float:
        """Ensure current volume doesn't exceed capacity."""
        # Access capacity from ValidationInfo if available
        capacity: float = info.data.get("capacity_gb", 32.0)
        if v > capacity:
            return capacity
        return v

    @field_validator("red_threshold")
    @classmethod
    def validate_thresholds(cls, v: float, info: Any) -> float:
        """Ensure red threshold is greater than or equal to yellow threshold."""
        yellow = info.data.get("yellow_threshold", 0.7)
        if v < yellow:
            raise ValueError("red_threshold must be >= yellow_threshold")
        return v

    def add_data(self, data_gb: float) -> float:
        """Add data to the recorder.

        Args:
            data_gb: Amount of data to add in Gigabits.

        Returns:
            float: Amount of data actually stored (may be less if recorder fills up).

        Note:
            If adding data would exceed capacity, only the available space is filled.
            The recorder will be at capacity after this operation if requested amount
            exceeded available space.
        """
        if data_gb < 0:
            return 0.0

        available = self.available_capacity()
        stored = min(data_gb, available)
        self.current_volume_gb += stored
        return stored

    def remove_data(self, data_gb: float) -> float:
        """Remove data from the recorder (e.g., during downlink).

        Args:
            data_gb: Amount of data to remove in Gigabits.

        Returns:
            float: Amount of data actually removed (may be less if insufficient data).

        Note:
            If removing more data than available, all remaining data is removed.
            The recorder will be empty after this operation if requested amount
            exceeded current volume.
        """
        if data_gb < 0:
            return 0.0

        removed = min(data_gb, self.current_volume_gb)
        self.current_volume_gb -= removed
        return removed

    def get_fill_fraction(self) -> float:
        """Get the current fill level as a fraction.

        Returns:
            float: Fill fraction between 0.0 (empty) and 1.0 (full).
        """
        return self.current_volume_gb / self.capacity_gb

    def get_alert_level(self) -> int:
        """Get the current alert level based on fill fraction.

        Returns:
            int: Alert level:
                0 = No alert (below yellow threshold)
                1 = Yellow alert (at or above yellow, below red)
                2 = Red alert (at or above red threshold)
        """
        fill = self.get_fill_fraction()
        if fill >= self.red_threshold:
            return 2
        elif fill >= self.yellow_threshold:
            return 1
        else:
            return 0

    def is_full(self) -> bool:
        """Check if recorder is at or above capacity.

        Returns:
            bool: True if current volume is at or above capacity.
        """
        return self.current_volume_gb >= self.capacity_gb

    def available_capacity(self) -> float:
        """Get remaining available capacity.

        Returns:
            float: Available capacity in Gigabits.
        """
        return max(0.0, self.capacity_gb - self.current_volume_gb)

    def reset(self) -> None:
        """Reset the recorder to empty state."""
        self.current_volume_gb = 0.0
