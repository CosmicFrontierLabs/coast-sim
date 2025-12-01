from typing import Any

import numpy as np
import rust_ephem

from ..common import unixtime2date
from ..ditl.ditl_log import DITLLog
from . import Pointing


class Queue:
    """Target Queue class, contains a list of targets for Spacecraft to observe."""

    targets: list[Pointing]
    ephem: rust_ephem.Ephemeris | None
    utime: float | None
    gs: Any
    log: DITLLog | None

    def __init__(
        self, ephem: rust_ephem.Ephemeris | None = None, log: DITLLog | None = None
    ):
        self.targets = []
        self.ephem = ephem
        self.utime = None
        self.gs = None
        self.log = log

    def __getitem__(self, number: int) -> Pointing:
        return self.targets[number]

    def __len__(self) -> int:
        return len(self.targets)

    def append(self, target: Pointing) -> None:
        self.targets.append(target)

    def meritsort(self, ra: float, dec: float) -> None:
        """Sort target queue by merit based on visibility, type, and trigger recency."""

        for target in self.targets:
            # Initialize merit using any pre-configured merit on the target.
            # Previously this used a `fom` attribute; prefer setting `merit`
            # directly on targets now.
            if getattr(target, "fom", None) is None:
                target.merit = 100
            else:
                target.merit = target.fom

            # Penalize constrained targets
            if target.visible(self.utime, self.utime) is False:
                target.merit = -900
                continue

        # Add randomness to break ties
        for target in self.targets:
            target.merit += np.random.random()

        # Sort by merit (highest first)
        self.targets.sort(key=lambda x: x.merit, reverse=True)

    def get(self, ra: float, dec: float, utime: float) -> Pointing | None:
        """Get the next best target to observe from the queue.

        Given current position (ra, dec) and time, returns the next highest-merit
        target that is visible for the minimum exposure time.

        Args:
            ra: Current right ascension in degrees.
            dec: Current declination in degrees.
            utime: Current time in Unix seconds.

        Returns:
            Next target to observe, or None if no suitable target found.
        """
        assert self.ephem is not None, (
            "Ephemeris must be set in TargetQueue before get()"
        )
        self.utime = utime
        self.meritsort(ra, dec)

        # Select targets from queue
        targets = [t for t in self.targets if t.merit > 0 and not t.done]

        msg = (
            f"{unixtime2date(self.utime)} Searching {len(targets)} targets in queue..."
        )
        if self.log is not None:
            self.log.log_event(
                utime=utime,
                event_type="QUEUE",
                description=msg,
                obsid=None,
                acs_mode=None,
            )
        else:
            print(msg)
        # Check each candidate target
        for target in targets:
            target.slewtime = target.calc_slewtime(ra, dec)

            # Calculate observation window
            endtime = utime + target.slewtime + target.ss_min

            # Use timestamp for the end-of-ephemeris bound
            last_unix = self.ephem.timestamp[-1].timestamp()

            # If the end time exceeds ephemeris, clamp it
            if endtime > last_unix:
                endtime = last_unix

            # Check if target is visible for full observation
            if target.visible(utime, endtime):
                target.begin = int(utime)
                target.end = int(utime + target.slewtime + target.ss_max)
                return target

        return None

    def reset(self) -> None:
        """Reset queue by resetting target status.

        Resets done flags on remaining targets for reuse in subsequent
        scheduling cycles.
        """
        for target in self.targets:
            target.reset()


TargetQueue = Queue
