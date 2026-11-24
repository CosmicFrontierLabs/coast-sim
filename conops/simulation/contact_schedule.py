"""Ground-station booking rules shared by all antenna types."""

from collections.abc import Callable, Sequence
from numbers import Integral
from typing import Protocol, TypeVar

import numpy as np

from ..common.enums import AntennaType
from ..config import CommunicationsSystem, GroundStation, MissionConfig


class RandomSource(Protocol):
    def random(self) -> float: ...


class ContactWindow(Protocol):
    @property
    def begin(self) -> float: ...

    @property
    def end(self) -> float: ...


_Contact = TypeVar("_Contact", bound=ContactWindow)


def deconflict_contacts(
    contacts: Sequence[_Contact],
    key: Callable[[_Contact], tuple[float, float, float]],
) -> tuple[list[_Contact], list[tuple[_Contact, _Contact]]]:
    """Select whole, non-overlapping passes; report (dropped, selected) pairs.

    Keep the existing fixed-antenna policy: compare overlapping passes by
    downlink volume, then duration, then earlier start.
    """
    selected: list[_Contact] = []
    dropped: list[tuple[_Contact, _Contact]] = []
    for contact in sorted(contacts, key=lambda c: c.begin):
        if not selected or selected[-1].end <= contact.begin:
            selected.append(contact)
            continue
        incumbent = selected[-1]
        if key(contact) > key(incumbent):
            selected[-1] = contact
            dropped.append((incumbent, contact))
        else:
            dropped.append((contact, incumbent))
    return selected, dropped


def _min_pass_seconds(communications: CommunicationsSystem | None) -> float:
    """The configured minimum, or the default where there is no communications system."""
    if communications is None:
        return float(CommunicationsSystem.model_fields["min_pass_seconds"].default)
    return communications.min_pass_seconds


class ContactScheduler:
    """Configure station booking before antenna-specific execution is considered.

    Probability is drawn once per eligible station visibility window, in
    station/time order. A configured seed reproduces the bookings when the
    same ephemeris is used by planning and execution.
    """

    def __init__(self, config: MissionConfig, rng: RandomSource | None = None) -> None:
        seed = config.random_seed
        valid_seed = (
            int(seed)
            if isinstance(seed, Integral) and not isinstance(seed, bool)
            else None
        )
        self.rng = rng if rng is not None else np.random.default_rng(valid_seed)
        communications = config.spacecraft_bus.communications
        self.continuous_contacts = (
            communications is not None
            and communications.antenna_pointing.antenna_type != AntennaType.FIXED
            and communications.contact_policy == "continuous"
        )
        self._spacecraft_bus = config.spacecraft_bus
        self._minlen: float | None = None
        self.schedule_chance = 1.0

    @property
    def minlen(self) -> float:
        """Shortest window booked, in seconds: the configuration's, unless set here.

        Short visibility windows and trimmed handovers are valid network
        contacts; the minimum belongs only to whole-pass booking.
        """
        if self._minlen is not None:
            return self._minlen
        if self.continuous_contacts:
            return 0.0
        return _min_pass_seconds(self._spacecraft_bus.communications)

    @minlen.setter
    def minlen(self, seconds: float) -> None:
        self._minlen = seconds

    def _random(self) -> float:
        return float(self.rng.random())

    def accepts(
        self,
        station: GroundStation,
        length: float,
        *,
        apply_scheduling: bool = True,
    ) -> bool:
        """Apply the booking rules: the minimum duration and the probability.

        Without ``apply_scheduling`` the contacts are already chosen, and a
        window of any duration may be one of them.
        """
        if length <= 0.0:
            return False
        if not apply_scheduling:
            return True
        if length < self.minlen:
            return False
        probability = self.schedule_chance * station.schedule_probability
        return probability >= 1.0 or (
            probability > 0.0 and self._random() <= probability
        )
