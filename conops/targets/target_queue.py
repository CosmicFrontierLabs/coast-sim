import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import rust_ephem

from ..common import unixtime2date
from ..config import AttitudeControlSystem, Constraint, MissionConfig
from ..ditl.ditl_log import DITLLog
from . import Pointing


@dataclass(frozen=True)
class TargetSlewEstimate:
    """Estimated slew cost used for target selection."""

    slewtime: float
    slewdist: float
    slewpath: tuple[list[float], list[float]] | None = None


class TargetQueue:
    """TargetQueue class, contains a list of targets for Spacecraft to observe."""

    targets: list[Pointing]
    ephem: rust_ephem.Ephemeris | None
    utime: float
    gs: Any
    log: DITLLog | None
    constraint: Constraint | None
    acs_config: AttitudeControlSystem | None
    config: MissionConfig | None
    random_seed: int

    def __init__(
        self,
        config: MissionConfig | None = None,
        ephem: rust_ephem.Ephemeris | None = None,
        log: DITLLog | None = None,
    ):
        # Extract config parameters from Config object
        if config is None:
            raise ValueError("Config must be provided to TargetQueue")
        self.config = config
        self.constraint = config.constraint
        self.acs_config = config.spacecraft_bus.attitude_control

        self.targets = []
        self.ephem = ephem
        self.utime = 0.0
        self.gs = None
        self.log = log
        # Optional weight to penalize long slews when selecting next target
        self.slew_distance_weight = config.targets.slew_distance_weight
        self.slew_time_weight = config.targets.slew_time_weight
        self.collection_time_weight = config.targets.collection_time_weight
        self.radiator_sun_exposure_weight = config.targets.radiator_sun_exposure_weight
        self.radiator_earth_exposure_weight = (
            config.targets.radiator_earth_exposure_weight
        )
        self.random_seed = config.random_seed if config.random_seed is not None else 0

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

        self.targets.sort(
            key=lambda target: (target.merit, self._target_tie_breaker(target)),
            reverse=True,
        )

    def _target_tie_breaker(self, target: Pointing) -> int:
        """Return a deterministic tie-break value for equal-merit targets."""
        identifier = getattr(target, "obsid", None)
        if identifier is None:
            identifier = getattr(target, "targetid", None)
        if identifier is None:
            identifier = getattr(target, "name", "")
        payload = f"{self.random_seed}:{identifier}".encode()
        digest = hashlib.blake2b(payload, digest_size=8).digest()
        return int.from_bytes(digest, "big")

    def _candidate_collection_seconds(
        self,
        target: Pointing,
        visibility_window: list[float],
        utime: float,
        deadline: float | None = None,
        slewtime: float | None = None,
    ) -> float:
        """Estimate useful science collection available after the slew."""
        collection_start = utime + float(
            target.slewtime if slewtime is None else slewtime
        )
        collection_end = float(visibility_window[1])
        if deadline is not None:
            collection_end = min(collection_end, float(deadline))
        window_seconds = max(0.0, collection_end - collection_start)
        max_snapshot = float(target.ss_max)
        remaining_exposure = target.exptime
        if remaining_exposure is None:
            return min(window_seconds, max_snapshot)
        return min(window_seconds, float(remaining_exposure), max_snapshot)

    def _can_fit_min_snapshot_with_zero_slew(
        self,
        target: Pointing,
        utime: float,
        last_unix: float,
        collection_deadline: Callable[[Pointing, float], float | None] | None = None,
    ) -> bool:
        """Check if a candidate can fit under the current rules with no slew.

        This is only an impossibility filter. If zero slew cannot fit the
        immediate visibility/deadline window, any nonnegative slew also cannot.
        """
        zero_slew_endtime = utime + float(target.ss_min)
        if zero_slew_endtime > last_unix:
            zero_slew_endtime = last_unix

        visibility_window = target.visible(utime, zero_slew_endtime)
        if not visibility_window:
            return False

        if collection_deadline is None:
            return True

        deadline = collection_deadline(target, utime)
        collection_seconds = self._candidate_collection_seconds(
            target=target,
            visibility_window=visibility_window,
            utime=utime,
            deadline=deadline,
            slewtime=0.0,
        )
        return collection_seconds >= float(target.ss_min)

    def _estimate_slew(
        self,
        target: Pointing,
        ra: float,
        dec: float,
        slew_estimator: Callable[[Pointing], TargetSlewEstimate] | None,
    ) -> None:
        if slew_estimator is None:
            target.slewtime = target.calc_slewtime(ra, dec)
            return

        estimate = slew_estimator(target)
        target.slewtime = round(estimate.slewtime)
        target.slewdist = estimate.slewdist
        if estimate.slewpath is not None:
            target.slewpath = estimate.slewpath

    def get(
        self,
        ra: float,
        dec: float,
        utime: float,
        collection_deadline: Callable[[Pointing, float], float | None] | None = None,
        slew_estimator: Callable[[Pointing], TargetSlewEstimate] | None = None,
    ) -> Pointing | None:
        """Get the next best target to observe from the queue.

        Given current position (ra, dec), roll, and time, returns the next
        highest-merit target that is visible for the minimum exposure time.

        Args:
            ra: Current right ascension in degrees.
            dec: Current declination in degrees.
            utime: Current time in Unix seconds.
            collection_deadline: Optional callback returning the latest time
                science may be collected for a candidate target after its slew.
            slew_estimator: Optional callback returning attitude-aware slew
                cost for a candidate target.

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
        score_candidates = (
            self.slew_distance_weight != 0.0
            or self.slew_time_weight != 0.0
            or self.collection_time_weight != 0.0
            or self.radiator_sun_exposure_weight != 0.0
            or self.radiator_earth_exposure_weight != 0.0
        )

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
        last_unix = self.ephem.timestamp[-1].timestamp()

        # Check each candidate target
        for target in targets:
            # Skip targets with remaining exposure less than minimum snapshot
            if target.exptime is not None and target.exptime < target.ss_min:
                continue

            if not self._can_fit_min_snapshot_with_zero_slew(
                target=target,
                utime=utime,
                last_unix=last_unix,
                collection_deadline=collection_deadline if score_candidates else None,
            ):
                continue

            self._estimate_slew(
                target,
                ra,
                dec,
                slew_estimator if score_candidates else None,
            )

            # Calculate observation window
            endtime = utime + target.slewtime + target.ss_min

            # If the end time exceeds ephemeris, clamp it
            if endtime > last_unix:
                endtime = last_unix

            # Check if target is visible for full observation
            visibility_window = target.visible(utime, endtime)
            if visibility_window:
                target.begin = int(utime)
                target.end = int(utime + target.slewtime + target.ss_max)
                # If no slew weighting, return first visible target (fast path)
                if not score_candidates:
                    return target
                # Otherwise, score all visible targets and pick best
                slewdist = getattr(target, "slewdist", 0.0)
                slew_minutes = float(target.slewtime) / 60.0
                score = (
                    target.merit
                    - self.slew_distance_weight * slewdist
                    - self.slew_time_weight * slew_minutes
                )
                slew_end = utime + float(target.slewtime)
                deadline = (
                    collection_deadline(target, slew_end)
                    if collection_deadline is not None
                    else None
                )
                collection_seconds = self._candidate_collection_seconds(
                    target=target,
                    visibility_window=visibility_window,
                    utime=utime,
                    deadline=deadline,
                )
                if collection_deadline is not None and collection_seconds < float(
                    target.ss_min
                ):
                    continue
                collection_minutes = collection_seconds / 60.0
                score += self.collection_time_weight * collection_minutes

                if (
                    self.radiator_sun_exposure_weight > 0.0
                    or self.radiator_earth_exposure_weight > 0.0
                ):
                    assert self.config is not None
                    radiators = self.config.spacecraft_bus.radiators
                    if radiators.num_radiators() > 0 and self.ephem is not None:
                        metrics = radiators.exposure_metrics(
                            ra_deg=target.ra,
                            dec_deg=target.dec,
                            utime=utime,
                            ephem=self.ephem,
                            roll_deg=getattr(target, "roll", 0.0),
                        )
                        sun_exposure = cast(float, metrics.get("sun_exposure", 0.0))
                        earth_exposure = cast(float, metrics.get("earth_exposure", 0.0))
                        score -= (
                            self.radiator_sun_exposure_weight * sun_exposure
                            + self.radiator_earth_exposure_weight * earth_exposure
                        )

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
