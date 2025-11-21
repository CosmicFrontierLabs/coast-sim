from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class Antenna(BaseModel):
    """Represents antenna capabilities for a ground station."""

    bands: list[str] = Field(
        default_factory=list, description="Supported frequency bands, e.g., S, X"
    )
    gain_db: float | None = Field(default=None, description="Antenna gain in dB")
    max_data_rate_mbps: float | None = Field(
        default=None, description="Maximum downlink/uplink data rate in Mbps"
    )
    simultaneous_links: int = Field(
        default=1, ge=1, description="Number of simultaneous spacecraft links supported"
    )


class GroundStation(BaseModel):
    """Pydantic model describing a ground station location and capabilities."""

    code: str = Field(description="Short code identifier (e.g., GHA, RWA)")
    name: str = Field(description="Human readable name")
    latitude_deg: float = Field(description="Latitude in degrees")
    longitude_deg: float = Field(description="Longitude in degrees")
    elevation_m: float = Field(
        default=0.0, description="Elevation above sea level in meters"
    )
    min_elevation_deg: float = Field(
        default=10.0, ge=0.0, le=90.0, description="Minimum elevation for a valid pass"
    )
    schedule_probability: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Probability of scheduling a pass when available",
    )
    antenna: Antenna = Field(default_factory=Antenna)

    @field_validator("code")
    @classmethod
    def code_upper(cls, v: str) -> str:  # noqa: D401
        """Ensure code is upper-case without whitespace."""
        return v.strip().upper()


class GroundStationRegistry(BaseModel):
    """Container holding all defined ground stations (list-backed)."""

    stations: list[GroundStation] = Field(default_factory=list)

    def add(self, station: GroundStation) -> None:
        """Add or replace a ground station by code.

        If a station with the same code already exists it is replaced to keep
        code uniqueness invariant.
        """
        for i, existing in enumerate(self.stations):
            if existing.code == station.code:
                self.stations[i] = station
                return
        self.stations.append(station)

    def get(self, code: str) -> GroundStation:
        """Return station matching code or raise KeyError."""
        for station in self.stations:
            if station.code == code:
                return station
        raise KeyError(code)

    def codes(self) -> list[str]:
        return [s.code for s in self.stations]

    def __iter__(self):  # noqa: D401
        """Iterate over stored stations."""
        return iter(self.stations)

    def __contains__(self, code: str) -> bool:  # noqa: D401
        """Return True if a station code exists in registry."""
        return any(s.code == code for s in self.stations)

    @classmethod
    def default(cls) -> GroundStationRegistry:
        """Return registry pre-populated with baseline stations."""
        reg = cls()
        reg.add(
            GroundStation(
                code="MAL",
                name="Malindi",
                latitude_deg=-3.22,
                longitude_deg=40.12,
                elevation_m=0.0,
                min_elevation_deg=10.0,
                schedule_probability=1.0,
                antenna=Antenna(bands=["S"], gain_db=None, max_data_rate_mbps=None),
            )
        )
        reg.add(
            GroundStation(
                code="HI1",
                name="Hawaii (Kokee Park)",
                latitude_deg=22.126,
                longitude_deg=-159.671,
                elevation_m=0.0,
                min_elevation_deg=10.0,
                schedule_probability=1.0,
                antenna=Antenna(bands=["S"], gain_db=None, max_data_rate_mbps=None),
            )
        )
        reg.add(
            GroundStation(
                code="SGS",
                name="Svalbard",
                latitude_deg=78.229,
                longitude_deg=15.407,
                elevation_m=0.0,
                min_elevation_deg=10.0,
                schedule_probability=1.0,
                antenna=Antenna(bands=["S"], gain_db=None, max_data_rate_mbps=None),
            )
        )
        return reg

    def min_elevation(self, code: str) -> float:
        return self.get(code).min_elevation_deg

    def schedule_probability_for(self, code: str) -> float:
        return self.get(code).schedule_probability
