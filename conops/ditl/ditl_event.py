"""DITL Event logging data structure."""

from datetime import datetime, timezone

from pydantic import BaseModel, Field

from ..common import ACSMode


class DITLEvent(BaseModel):
    """
    A single event logged during a DITL simulation.

    Attributes:
        time: Unix timestamp of the event
        timestamp: ISO 8601 formatted timestamp string
        event_type: Category of event (e.g., 'PASS', 'SLEW', 'OBSERVATION', 'ERROR', 'INFO')
        description: Human-readable description of the event
        obsid: Observation ID associated with the event (if applicable)
        acs_mode: Current ACS mode at the time of the event (if applicable)
    """

    time: float = Field(..., description="Unix timestamp of the event")
    timestamp: str = Field(..., description="ISO 8601 formatted timestamp string")
    event_type: str = Field(
        ..., description="Category of event (e.g., 'PASS', 'SLEW', 'OBSERVATION')"
    )
    description: str = Field(..., description="Human-readable description of the event")
    obsid: int | None = Field(None, description="Observation ID (if applicable)")
    acs_mode: ACSMode | None = Field(
        None, description="Current ACS mode (if applicable)"
    )

    @classmethod
    def from_utime(
        cls,
        utime: float,
        event_type: str,
        description: str,
        obsid: int | None = None,
        acs_mode: ACSMode | None = None,
    ) -> "DITLEvent":
        """
        Create a DITLEvent from a unix timestamp.

        Args:
            utime: Unix timestamp
            event_type: Category of event
            description: Human-readable description
            obsid: Optional observation ID
            acs_mode: Optional ACS mode

        Returns:
            A new DITLEvent instance
        """
        dt = datetime.fromtimestamp(utime, tz=timezone.utc)
        timestamp = dt.isoformat(timespec="seconds")
        return cls(
            time=utime,
            timestamp=timestamp,
            event_type=event_type,
            description=description,
            obsid=obsid,
            acs_mode=acs_mode,
        )

    def __str__(self) -> str:
        """Format the event for display."""
        return f"[{self.timestamp}] {self.description}"
