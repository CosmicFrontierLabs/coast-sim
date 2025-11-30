"""DITL event logging system."""

from pydantic import BaseModel, Field

from ..common import ACSMode
from .ditl_event import DITLEvent


class DITLLog(BaseModel):
    """
    Container for DITL simulation events with logging and printing methods.

    Attributes:
        events: List of DITLEvent objects logged during simulation
    """

    events: list[DITLEvent] = Field(
        default_factory=list, description="List of logged events"
    )

    def log_event(
        self,
        utime: float,
        event_type: str,
        description: str,
        obsid: int | None = None,
        acs_mode: ACSMode | None = None,
    ) -> None:
        """
        Log a DITL event.

        Parameters
        ----------
        utime : float
            Unix timestamp of the event
        event_type : str
            Category of event (e.g., 'PASS', 'SLEW', 'OBSERVATION', 'ERROR', 'INFO')
        description : str
            Human-readable description of the event
        obsid : int | None
            Optional observation ID associated with the event
        acs_mode : ACSMode | None
            Optional ACS mode at the time of the event
        """
        event = DITLEvent.from_utime(
            utime=utime,
            event_type=event_type,
            description=description,
            obsid=obsid,
            acs_mode=acs_mode,
        )
        self.events.append(event)

    def print_log(self) -> None:
        """Print the DITL event log to stdout."""
        for event in self.events:
            print(str(event))

    def clear(self) -> None:
        """Clear all logged events."""
        self.events.clear()

    def __len__(self) -> int:
        """Return the number of logged events."""
        return len(self.events)

    def __iter__(self):
        """Iterate over logged events."""
        return iter(self.events)

    def __getitem__(self, index: int) -> DITLEvent:
        """Get event by index."""
        return self.events[index]
