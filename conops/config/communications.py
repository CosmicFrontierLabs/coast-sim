"""Communications system configuration for spacecraft downlink and uplink."""

import warnings
from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from ..common.enums import AntennaType, Polarization
from ..common.numerics import UNIT_MAGNITUDE_TOLERANCE
from ._base import ConfigModel


def validate_range_table(
    table: list[tuple[float, float]] | None, name: str, *, positive: bool = False
) -> list[tuple[float, float]] | None:
    """Check a table of ``(range_km, value)`` rows: at least one row, ranges
    increasing from zero or more, values not negative (``positive``: above
    zero). Returns the rows as floats."""
    if table is None:
        return None
    if not table:
        raise ValueError(f"{name} must have at least one row")
    for range_km, value in table:
        if range_km < 0.0 or value < 0.0 or (positive and value <= 0.0):
            raise ValueError(
                f"{name} ranges must be >= 0 and values "
                + ("> 0" if positive else ">= 0")
            )
    ranges = [row[0] for row in table]
    if any(b <= a for a, b in zip(ranges, ranges[1:])):
        raise ValueError(f"{name} ranges must increase")
    return [(float(r), float(value)) for r, value in table]


def interpolate_range_table(table: list[tuple[float, float]], range_km: float) -> float:
    """The table's value at a range: linear between rows, held at the end rows."""
    if range_km <= table[0][0]:
        return table[0][1]
    if range_km >= table[-1][0]:
        return table[-1][1]
    for (r0, v0), (r1, v1) in zip(table, table[1:]):
        if r0 <= range_km <= r1:
            return v0 + (v1 - v0) * (range_km - r0) / (r1 - r0)
    return table[-1][1]


class BandCapability(ConfigModel):
    """Defines a communication band's data rate capabilities."""

    band: Literal["UHF", "S", "X", "Ka", "Ku", "L", "C", "K"]  # Common bands
    uplink_rate_mbps: float = Field(
        default=0.0, description="Uplink data rate in Mbps", ge=0.0
    )
    downlink_rate_mbps: float = Field(
        default=0.0, description="Downlink data rate in Mbps", ge=0.0
    )
    downlink_rate_vs_range_km: list[tuple[float, float]] | None = Field(
        default=None,
        description=(
            "Downlink rate against spacecraft-to-station range, as (range_km, "
            "rate_mbps) rows in increasing range, interpolated linearly between "
            "rows and held at the end rows beyond them. Used where the range is "
            "known by the link-coverage model. Without it the "
            "rate is downlink_rate_mbps at every range."
        ),
    )

    @field_validator("downlink_rate_vs_range_km")
    @classmethod
    def _validate_rate_table(
        cls, table: list[tuple[float, float]] | None
    ) -> list[tuple[float, float]] | None:
        return validate_range_table(table, "downlink_rate_vs_range_km")

    def downlink_rate_at(self, range_km: float | None) -> float:
        """The downlink rate in Mbps at a range, or the nominal rate without one.

        The table, where there is one, is interpolated linearly between its
        rows and held at its end rows beyond them.
        """
        table = self.downlink_rate_vs_range_km
        if range_km is None or table is None:
            return self.downlink_rate_mbps
        return interpolate_range_table(table, range_km)

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

    repoint_threshold_deg: float | None = Field(
        default=None,
        description=(
            "For a gimballed antenna: the antenna is re-pointed at the tracked "
            "station when the station's direction has moved this far since "
            "the last repoint, as the spacecraft moves along its orbit. "
            "None: no tracking repoints."
        ),
        gt=0.0,
        le=180.0,
    )
    repoint_interval_vs_range_km: list[tuple[float, float]] | None = Field(
        default=None,
        description=(
            "Instead of the threshold: the interval between repoints in "
            "seconds against the spacecraft's distance from the Earth's centre "
            "in km, as (range_km, seconds) rows in increasing range, "
            "interpolated at the range at each repoint."
        ),
    )
    repoint_seconds: float = Field(
        default=0.0,
        description=(
            "Seconds a repoint of a gimballed antenna takes, during which "
            "nothing is downlinked and an observation of a category that "
            "forbids repoints cannot collect"
        ),
        ge=0.0,
    )

    @field_validator("repoint_interval_vs_range_km")
    @classmethod
    def _validate_repoint_table(
        cls, table: list[tuple[float, float]] | None
    ) -> list[tuple[float, float]] | None:
        return validate_range_table(
            table, "repoint_interval_vs_range_km", positive=True
        )

    @model_validator(mode="after")
    def _repoints_need_a_gimbal(self) -> "AntennaPointing":
        """Only a gimballed antenna is re-pointed: an omni antenna has no
        direction and a body-fixed one moves with the spacecraft."""
        if self.antenna_type != AntennaType.GIMBALED and (
            self.repoint_threshold_deg is not None
            or self.repoint_interval_vs_range_km is not None
            or self.repoint_seconds > 0.0
        ):
            raise ValueError(
                "repoint_threshold_deg, repoint_interval_vs_range_km and "
                "repoint_seconds apply to a gimballed antenna only"
            )
        return self

    @field_validator("fixed_boresight_body")
    @classmethod
    def validate_fixed_boresight_body(
        cls, v: tuple[float, float, float]
    ) -> tuple[float, float, float]:
        magnitude = sum(component * component for component in v) ** 0.5
        if abs(magnitude - 1.0) > UNIT_MAGNITUDE_TOLERANCE:
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

    acquisition_seconds: float = Field(
        default=0.0,
        description=(
            "Seconds to acquire the link during passes without spacecraft "
            "tracking, during which nothing is downlinked"
        ),
        ge=0.0,
    )

    handover_seconds: float = Field(
        default=0.0,
        description=(
            "Seconds to switch the link straight from one ground station to "
            "another during passes without spacecraft tracking, on top of the "
            "acquisition, during which nothing is downlinked"
        ),
        ge=0.0,
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

    def get_downlink_rate(self, band: str, range_km: float | None = None) -> float:
        """Get downlink rate for a specific band in Mbps.

        Args:
            band: Band identifier (e.g., "S", "X", "Ka")
            range_km: The range to the station, where known: the rate is then
                read from the band's ``downlink_rate_vs_range_km`` table if it
                has one (see :meth:`BandCapability.downlink_rate_at`).

        Returns:
            Downlink rate in Mbps, or 0.0 if band not supported
        """
        capability = self.get_band(band)
        return capability.downlink_rate_at(range_km) if capability else 0.0

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
