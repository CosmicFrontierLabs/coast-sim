from typing import Any

import numpy as np
from pydantic import BaseModel, Field, field_validator

from ..common import unixtime2date
from ..config import AttitudeControlSystem, Constraint, MissionConfig
from ..ditl.ditl_log import DITLLog
from . import Pointing


class TargetQueue(BaseModel):
    """TargetQueue class, contains a list of targets for Spacecraft to observe."""

    model_config = {"arbitrary_types_allowed": True}

    # Pydantic fields
    targets: list[Pointing] = Field(default_factory=list)
    ephem: Any = None  # Allow any type for ephem to support mocks in tests
    utime: float = 0.0
    gs: Any = None
    log: DITLLog | None = None
    constraint: Constraint | None = (
        None  # Allow any type for constraint to support mocks in tests
    )
    acs_config: AttitudeControlSystem | None = (
        None  # Allow any type for acs_config to support mocks in tests
    )
    config: MissionConfig = Field(default_factory=MissionConfig, exclude=True)
    slew_distance_weight: float = 0.0

    @field_validator("config")
    @classmethod
    def validate_config(cls, v: Any) -> Any:
        """Validate config parameter, allowing None or extracting values from MissionConfig-like objects."""
        if v is None:
            raise ValueError("Config must be provided to TargetQueue")
        return v

    def __getitem__(self, number: int) -> Pointing:
        return self.targets[number]

    def __len__(self) -> int:
        return len(self.targets)

    def append(self, target: Pointing) -> None:
        self.targets.append(target)

    def add(
        self,
        ra: float = 0.0,
        dec: float = 0.0,
        obsid: int = 0,
        name: str = "FakeTarget",
        merit: float = 100.0,
        exptime: int = 1000,
        ss_min: int = 300,
        ss_max: int = 86400,
    ) -> None:
        """Add a pointing target to the queue.

        Creates a new Pointing object with the specified parameters and adds it to the queue.

        Args:
            ra: Right ascension in degrees
            dec: Declination in degrees
            obsid: Observation ID
            name: Target name
            merit: Merit value for scheduling priority
            exptime: Exposure time in seconds
            ss_min: Minimum snapshot size in seconds
            ss_max: Maximum snapshot size in seconds
        """

        pointing = Pointing(
            config=self.config,
            ra=ra,
            dec=dec,
            obsid=obsid,
            name=name,
            merit=merit,
            exptime=exptime,
            ss_min=ss_min,
            ss_max=ss_max,
        )
        pointing.visibility()
        self.targets.append(pointing)

    def meritsort(self) -> None:
        """Sort target queue by merit based on visibility, type, and trigger recency."""

        for target in self.targets:
            # Initialize merit using any pre-configured merit on the target.
            # directly on targets now.
            if target.merit is None:
                target.merit = 100

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
        self.meritsort()

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
        best_target = None
        best_score = -np.inf

        # Check each candidate target
        for target in targets:
            # Skip targets with remaining exposure less than minimum snapshot
            if target.exptime is not None and target.exptime < target.ss_min:
                continue

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
                # If no slew weighting, return first visible target (fast path)
                if self.slew_distance_weight == 0.0:
                    return target
                # Otherwise, score all visible targets and pick best
                slewdist = getattr(target, "slewdist", 0.0)
                score = target.merit - self.slew_distance_weight * slewdist
                if score > best_score:
                    best_score = score
                    best_target = target

        return best_target

    def reset(self) -> None:
        """Reset queue by resetting target status.

        Resets done flags on remaining targets for reuse in subsequent
        scheduling cycles.
        """
        for target in self.targets:
            target.reset()


Queue = TargetQueue
