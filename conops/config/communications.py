"""Communications system configuration for spacecraft downlink and uplink."""

import warnings
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ..common.enums import AntennaType, Polarization
from ._base import ConfigModel


class BandCapability(ConfigModel):
    """Defines a communication band's data rate capabilities."""

    band: Literal["S", "X", "Ka", "Ku", "L", "C", "K"]  # Common spacecraft bands
    uplink_rate_mbps: float = Field(
        default=0.0, description="Uplink data rate in Mbps", ge=0.0
    )
    downlink_rate_mbps: float = Field(
        default=0.0, description="Downlink data rate in Mbps", ge=0.0
    )

    @model_validator(mode="before")
    @classmethod
    def _apply_standard_band_defaults(cls, data: dict[str, Any]) -> dict[str, Any]:
        """Fill in standard per-band defaults when rates are not provided.

        Only applies when `uplink_rate_mbps` and/or `downlink_rate_mbps` are
        not passed in the input data. Custom-provided values are preserved.
        """
        if isinstance(data, dict):
            band: str | None = data.get("band")
            defaults = {
                "S": {"uplink": 2.0, "downlink": 10.0},
                "X": {"uplink": 10.0, "downlink": 150.0},
                "Ka": {"uplink": 20.0, "downlink": 300.0},
                "Ku": {"uplink": 5.0, "downlink": 50.0},
                "L": {"uplink": 0.5, "downlink": 1.0},
                "C": {"uplink": 2.0, "downlink": 20.0},
                "K": {"uplink": 15.0, "downlink": 200.0},
            }
            if band in defaults:
                if (
                    "uplink_rate_mbps" not in data
                    or data.get("uplink_rate_mbps") is None
                ):
                    data["uplink_rate_mbps"] = defaults[band]["uplink"]
                if (
                    "downlink_rate_mbps" not in data
                    or data.get("downlink_rate_mbps") is None
                ):
                    data["downlink_rate_mbps"] = defaults[band]["downlink"]
        return data


class AntennaPointing(ConfigModel):
    """Defines antenna pointing configuration based on antenna type."""

    antenna_type: AntennaType = Field(
        default=AntennaType.FIXED,
        description="Type of antenna: omni, fixed, or gimbaled",
    )

    # Fixed antenna pointing (body-fixed direction)
    fixed_boresight_body: tuple[float, float, float] = Field(
        default=(-1.0, 0.0, 0.0),
        description=(
            "Fixed antenna boresight as a unit vector in spacecraft body frame. "
            "Default -X preserves the legacy ground-station-pass convention."
        ),
    )
    fixed_azimuth_deg: float = Field(
        default=0.0,
        description="Legacy fixed antenna azimuth metadata (deg)",
        ge=0.0,
        le=360.0,
    )
    fixed_elevation_deg: float = Field(
        default=0.0,
        description="Legacy fixed antenna elevation metadata (deg)",
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

    @field_validator("fixed_boresight_body")
    @classmethod
    def validate_fixed_boresight_body(
        cls, v: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        magnitude = sum(component * component for component in v) ** 0.5
        if magnitude < 0.99 or magnitude > 1.01:
            raise ValueError(
                f"Fixed antenna boresight must be a unit vector. Got magnitude {magnitude}"
            )
        return v

    @model_validator(mode="after")
    def warn_legacy_azimuth_elevation(self) -> "AntennaPointing":
        if self.fixed_azimuth_deg != 0.0 or self.fixed_elevation_deg != 0.0:
            warnings.warn(
                "AntennaPointing.fixed_azimuth_deg and fixed_elevation_deg are "
                "legacy metadata fields and no longer affect GSP attitude "
                "generation. Set fixed_boresight_body explicitly to the desired "
                "body-frame unit vector.",
                DeprecationWarning,
                stacklevel=2,
            )
        return self


class CommunicationsSystem(ConfigModel):
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
