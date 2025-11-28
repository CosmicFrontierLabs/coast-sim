"""Communications system configuration for spacecraft downlink and uplink."""

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class AntennaType(str, Enum):
    """Antenna mounting and pointing configuration."""

    OMNI = "omni"  # Omnidirectional antenna
    FIXED = "fixed"  # Fixed pointing antenna
    GIMBALED = "gimbaled"  # Gimbaled (steerable) antenna


class Polarization(str, Enum):
    """Antenna polarization type."""

    LINEAR_HORIZONTAL = "linear_horizontal"
    LINEAR_VERTICAL = "linear_vertical"
    CIRCULAR_RIGHT = "circular_right"  # RHCP
    CIRCULAR_LEFT = "circular_left"  # LHCP
    DUAL = "dual"  # Supports multiple polarizations


class BandCapability(BaseModel):
    """Defines a communication band's data rate capabilities."""

    band: Literal["S", "X", "Ka", "Ku", "L", "C", "K"]  # Common spacecraft bands
    uplink_rate_mbps: float = Field(
        default=0.0, description="Uplink data rate in Mbps", ge=0.0
    )
    downlink_rate_mbps: float = Field(
        default=0.0, description="Downlink data rate in Mbps", ge=0.0
    )


class AntennaPointing(BaseModel):
    """Defines antenna pointing configuration based on antenna type."""

    antenna_type: AntennaType = Field(
        default=AntennaType.FIXED,
        description="Type of antenna: omni, fixed, or gimbaled",
    )

    # Fixed antenna pointing (body-fixed direction)
    fixed_azimuth_deg: float = Field(
        default=0.0,
        description="Azimuth angle in spacecraft body frame (deg). 0=nadir, 180=zenith",
        ge=0.0,
        le=360.0,
    )
    fixed_elevation_deg: float = Field(
        default=0.0,
        description="Elevation angle in spacecraft body frame (deg)",
        ge=-90.0,
        le=90.0,
    )

    # Gimbaled antenna range (angular range of motion from boresight)
    gimbal_range_deg: float = Field(
        default=0.0,
        description="Angular range for gimbaled antenna (deg) from boresight",
        ge=0.0,
        le=180.0,
    )

    def is_nadir_pointing(self) -> bool:
        """Check if antenna is nadir-pointing (opposite of telescope)."""
        return (
            self.antenna_type == AntennaType.FIXED
            and self.fixed_azimuth_deg == 0.0
            and self.fixed_elevation_deg == 0.0
        )


class CommunicationsSystem(BaseModel):
    """Onboard communications system configuration.

    Defines band capabilities, data rates, antenna type, pointing,
    and signal requirements for ground station passes.
    """

    name: str = Field(default="Default Comms", description="Communications system name")

    # Band capabilities (multiple bands supported)
    band_capabilities: list[BandCapability] = Field(
        default_factory=lambda: [
            BandCapability(band="S", uplink_rate_mbps=2.0, downlink_rate_mbps=10.0)
        ],
        description="List of supported frequency bands with data rates",
    )

    # Antenna configuration
    antenna_pointing: AntennaPointing = Field(
        default_factory=lambda: AntennaPointing(antenna_type=AntennaType.FIXED),
        description="Antenna pointing configuration",
    )

    # Signal quality requirement
    pointing_accuracy_deg: float = Field(
        default=5.0,
        description="Maximum pointing error (deg) to maintain good signal",
        ge=0.0,
        le=180.0,
    )

    # Antenna polarization
    polarization: Polarization = Field(
        default=Polarization.CIRCULAR_RIGHT,
        description="Antenna polarization type",
    )

    def get_band(self, band: str) -> BandCapability | None:
        """Get capability for a specific band.

        Args:
            band: Band identifier (e.g., "S", "X", "Ka")

        Returns:
            BandCapability if found, None otherwise
        """
        for capability in self.band_capabilities:
            if capability.band == band:
                return capability
        return None

    def get_downlink_rate(self, band: str) -> float:
        """Get downlink rate for a specific band in Mbps.

        Args:
            band: Band identifier (e.g., "S", "X", "Ka")

        Returns:
            Downlink rate in Mbps, or 0.0 if band not supported
        """
        capability = self.get_band(band)
        return capability.downlink_rate_mbps if capability else 0.0

    def get_uplink_rate(self, band: str) -> float:
        """Get uplink rate for a specific band in Mbps.

        Args:
            band: Band identifier (e.g., "S", "X", "Ka")

        Returns:
            Uplink rate in Mbps, or 0.0 if band not supported
        """
        capability = self.get_band(band)
        return capability.uplink_rate_mbps if capability else 0.0

    def can_communicate(self, pointing_error_deg: float) -> bool:
        """Check if communication is possible given pointing error.

        Args:
            pointing_error_deg: Pointing error in degrees

        Returns:
            True if pointing error is within acceptable range
        """
        # Omni antenna can always communicate
        if self.antenna_pointing.antenna_type == AntennaType.OMNI:
            return True

        # For fixed/gimbaled, check against pointing accuracy requirement
        return pointing_error_deg <= self.pointing_accuracy_deg
