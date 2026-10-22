import hashlib
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, cast

import numpy as np
import rust_ephem
from pydantic import BaseModel, ConfigDict

from ..common import unixtime2date
from ..config import (
    AttitudeControlSystem,
    Constraint,
    MissionConfig,
    ObservationTiming,
    ScienceData,
)
from ..config.acs import scheduled_slew_time
from ..config.observation_categories import ObservationCategories
from ..ditl.ditl_log import DITLLog
from . import Pointing
from .merit import MeritBreakdown, MeritModel
from .plan_entry import extra_data_gb, instrument_modes_of, snapshot_min_seconds
from .sequence import ObservationSequence
from .visits import VisitWindow

if TYPE_CHECKING:
    from ..simulation.coverage import StationCoverage


class TargetSlewEstimate(BaseModel):
    """Estimated slew cost used for target selection."""

    model_config = ConfigDict(frozen=True)

    slewtime: float
    slewdist: float
    slewpath: tuple[list[float], list[float]] | None = None
    instrument_roll: float | None = None
    spacecraft_attitude: tuple[float, float, float] | None = None


class TargetQueue:
    """TargetQueue class, contains a list of targets for Spacecraft to observe."""

    targets: list[Pointing]
    ephem: rust_ephem.Ephemeris | None
    utime: float
    log: DITLLog | None
    constraint: Constraint | None
    acs_config: AttitudeControlSystem | None
    config: MissionConfig | None
    random_seed: int
    prefer: Callable[[int], bool] | None
    """Returns, for an obsid, whether to favour it, such as a target a
    long-range allocation assigns to now. None (the default) treats all
    targets alike."""
    allocation_bonus: float
    """Fraction of its value a preferred target gains."""

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
        self.observation_timing = config.payload.observation_timing
        self.coverage: "StationCoverage | None" = None
        """The run's station coverage, where its antenna needs no attitude:
        a candidate whose category forbids antenna repoints during an
        observation is kept between them."""

        self.targets = []
        self.ephem = ephem
        self.utime = 0.0
        self.log = log
        # Optional weight to penalize long slews when selecting next target
        self.slew_distance_weight = config.targets.slew_distance_weight
        self.slew_time_weight = config.targets.slew_time_weight
        self.collection_time_weight = config.targets.collection_time_weight
        self.data_volume_weight = config.targets.data_volume_weight
        self.radiator_sun_exposure_weight = config.targets.radiator_sun_exposure_weight
        self.radiator_earth_exposure_weight = (
            config.targets.radiator_earth_exposure_weight
        )
        self.random_seed = config.random_seed if config.random_seed is not None else 0
        self.merit_model = MeritModel(config)
        self.prefer = None
        self.allocation_bonus = 0.5

    def _bonus(self, target: Pointing) -> float:
        """Fraction of its value a target gains for being preferred."""
        if self.prefer is None or not self.prefer(int(target.obsid)):
            return 0.0
        return self.allocation_bonus

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
        instrument_name: str | None = None,
        deadline: float | None = None,
        earliest_start: float | None = None,
        science_data: Mapping[str, ScienceData] | None = None,
        sequence: ObservationSequence | None = None,
        visits: int = 1,
        visit_windows: Sequence[VisitWindow] | None = None,
        instrument_modes: Mapping[str, str] | None = None,
    ) -> Pointing:
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
            instrument_name: Science instrument to observe with
            deadline: Latest time (Unix seconds) science collection may begin
            earliest_start: Earliest time (Unix seconds) science collection
                may begin
            science_data: Science data observing the target generates, keyed
                by instrument name; other instruments generate their own
            sequence: The steps a visit of the target observes, if any
            visits: How many visits of the sequence are wanted
            visit_windows: When each visit may begin collecting, one visit
                per window; the target is selected only while an unvisited
                window is open
            instrument_modes: The instruments the observation uses and the
                mode each is in, by name; instruments not named idle

        Returns:
            The added target.
        """

        assert self.config is not None
        telescope = self.config.payload.telescope_for_target(instrument_name)
        pointing = Pointing(
            config=self.config,
            ra=ra,
            dec=dec,
            obsid=obsid,
            name=name,
            merit=merit,
            ss_min=ss_min,
            ss_max=ss_max,
            instrument_name=telescope.name if telescope is not None else None,
            deadline=deadline,
            earliest_start=earliest_start,
            science_data=dict(science_data or {}),
            sequence=sequence,
            visits=visits,
            visit_windows=list(visit_windows or []),
            instrument_modes=dict(instrument_modes or {}),
        )
        if pointing.visit_windows:
            pointing.exptime = pointing.visits * pointing.per_visit_seconds
        else:
            pointing.exptime = (
                visits * sequence.science_seconds if sequence is not None else exptime
            )
        # A request naming no modes uses every instrument with modes.
        pointing.check_instrument_modes()
        pointing.visibility()
        self.targets.append(pointing)
        return pointing

    def _timing(self, target: Pointing) -> ObservationTiming:
        """The timing budgets of ``target``: its category's where that sets
        them, else the queue's, which is the payload's unless changed."""
        categories = getattr(self.config, "observation_categories", None)
        obsid = getattr(target, "obsid", None)
        # A test double's target has no number for an obsid.
        if isinstance(categories, ObservationCategories) and isinstance(
            obsid, (int, float)
        ):
            timing = categories.get_category(int(obsid)).observation_timing
            if timing is not None:
                return timing
        return self.observation_timing

    def _visit_overhead(self, target: Pointing) -> float:
        """Seconds a visit of a sequence target takes beyond its science and
        the payload's budgets; 0 for a plain pointing or a test double."""
        overhead = getattr(target, "sequence_overhead_seconds", None)
        if getattr(target, "sequence", None) is None or not callable(overhead):
            return 0.0
        seconds = overhead(self.acs_config, self._timing(target))
        return float(seconds) if isinstance(seconds, (int, float)) else 0.0

    def _repoints_forbidden(self, target: Pointing) -> bool:
        """Whether an antenna repoint must not fall in an observation of ``target``."""
        if self.coverage is None or not self.coverage.repoints:
            return False
        assert self.config is not None
        categories = self.config.observation_categories
        return categories.antenna_repoints(int(target.obsid)) == "forbidden"

    def meritsort(self) -> None:
        """Sort target queue by merit, ranking targets not visible now below
        every visible target."""
        self.targets.sort(
            key=lambda target: (
                target.merit * (1.0 + self._bonus(target))
                if self._visible_now(target)
                else -900,
                self._target_tie_breaker(target),
            ),
            reverse=True,
        )

    def _visible_now(self, target: Pointing) -> bool:
        """Whether ``target`` is visible at the queue's current time."""
        # Cached per utime: the same utime is checked many times per fetch.
        utime = self.utime
        if getattr(target, "_vis_cache_utime", None) != utime:
            target._vis_cache_utime = utime  # type: ignore[attr-defined]
            target._vis_cache_result = target.visible(utime, utime)  # type: ignore[attr-defined]
        return getattr(target, "_vis_cache_result", None) is not False

    def _target_tie_breaker(self, target: Pointing) -> int:
        """Return a deterministic tie-break value for equal-merit targets."""
        cached: int | None = getattr(target, "_tie_breaker_cache", None)
        if cached is not None:
            return cached
        identifier = getattr(target, "obsid", None)
        if identifier is None:
            identifier = getattr(target, "targetid", None)
        if identifier is None:
            identifier = getattr(target, "name", "")
        payload = f"{self.random_seed}:{identifier}".encode()
        value = int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")
        target._tie_breaker_cache = value  # type: ignore[attr-defined]
        return value

    def _candidate_collection_seconds(
        self,
        target: Pointing,
        visibility_window: list[float],
        utime: float,
        deadline: float | None = None,
        slewtime: float | None = None,
    ) -> float:
        """Estimate useful science collection available after the slew."""
        end = float(visibility_window[1])
        if deadline is not None:
            end = min(end, float(deadline))
        collection_start, collection_end = self._timing(target).collection_window(
            utime, float(target.slewtime if slewtime is None else slewtime), end
        )
        window_seconds = max(0.0, collection_end - collection_start)
        max_snapshot = float(target.ss_max)
        overhead = self._visit_overhead(target)
        if overhead > 0.0:
            # A sequence visit runs whole or not at all: its science, with
            # the calibration and slews between its steps around it.
            if window_seconds < max_snapshot + overhead:
                return 0.0
            return max_snapshot
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
        zero_slew_endtime = (
            utime
            + float(snapshot_min_seconds(target))
            + self._timing(target).total_seconds
            + self._visit_overhead(target)
        )
        if zero_slew_endtime > last_unix:
            zero_slew_endtime = last_unix

        visibility_window = target.visible(utime, zero_slew_endtime)
        if not visibility_window:
            return False

        # A deadline can only shorten collection. Reject insufficient exposure
        # or visibility before asking the caller to optimize the attitude.
        if self._candidate_collection_seconds(
            target, visibility_window, utime, slewtime=0.0
        ) < float(snapshot_min_seconds(target)):
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
        return collection_seconds >= float(snapshot_min_seconds(target))

    def _score_upper_bound_pruning_is_safe(self) -> bool:
        """Return True when configured weights make the score bound conservative."""
        return (
            self.slew_distance_weight >= 0.0
            and self.slew_time_weight >= 0.0
            and self.collection_time_weight >= 0.0
            and self.radiator_sun_exposure_weight >= 0.0
            and self.radiator_earth_exposure_weight >= 0.0
        )

    def _candidate_score_upper_bound(
        self,
        target: Pointing,
        utime: float,
        last_unix: float,
    ) -> float:
        """Calculate a conservative best possible score before slew estimation.

        This ignores all nonnegative penalties and assumes zero slew. If that
        optimistic score cannot beat the current best score, the real score
        cannot either.
        """
        zero_slew_endtime = (
            utime
            + float(snapshot_min_seconds(target))
            + self._timing(target).total_seconds
            + self._visit_overhead(target)
        )
        if zero_slew_endtime > last_unix:
            zero_slew_endtime = last_unix

        visibility_window = target.visible(utime, zero_slew_endtime)
        if not visibility_window:
            return -np.inf

        upper_bound = (float(target.merit) + self.merit_model.max_dynamic_value) * (
            1.0 + self._bonus(target)
        )
        if self.collection_time_weight > 0.0:
            collection_seconds = self._candidate_collection_seconds(
                target=target,
                visibility_window=visibility_window,
                utime=utime,
                slewtime=0.0,
            )
            upper_bound += self.collection_time_weight * (collection_seconds / 60.0)
        return upper_bound

    def _estimate_slew(
        self,
        target: Pointing,
        ra: float,
        dec: float,
        roll: float | None,
        slew_estimator: Callable[[Pointing], TargetSlewEstimate] | None,
    ) -> None:
        if slew_estimator is None:
            assert self.acs_config is not None
            if self.acs_config.direction_dependent_slew:
                target.slewtime = target.calc_slewtime(ra, dec, roll)
            else:
                target.slewtime = target.calc_slewtime(ra, dec)
            if issubclass(type(target), Pointing) and target.uses_mounted_attitude():
                target.spacecraft_attitude = target.target_body_attitude()
            return

        estimate = slew_estimator(target)
        target.slewtime = scheduled_slew_time(estimate.slewtime)
        target.slewdist = estimate.slewdist
        if estimate.instrument_roll is not None:
            target.roll = estimate.instrument_roll
        target.spacecraft_attitude = estimate.spacecraft_attitude
        if estimate.slewpath is not None:
            target.slewpath = estimate.slewpath

    @staticmethod
    def _bounds(target: Pointing, utime: float) -> tuple[float | None, float | None]:
        """When ``target`` may begin collecting if selected at ``utime``: its
        next visit window's bounds, or its own earliest start and deadline."""
        windows = getattr(target, "visit_windows", None)
        # A test double's target has no windows.
        if isinstance(windows, list) and windows:
            return target.window_bounds(utime)
        start = getattr(target, "earliest_start", None)
        deadline = getattr(target, "deadline", None)
        return (
            start if isinstance(start, (int, float)) else None,
            deadline if isinstance(deadline, (int, float)) else None,
        )

    def _open_at(self, target: Pointing, utime: float) -> bool:
        """Whether ``target`` may be selected at ``utime``: at or after its
        earliest start and by its deadline, or within an unvisited visit
        window; a target whose windows are all past or visited is not."""
        start, deadline = self._bounds(target, utime)
        windows = getattr(target, "visit_windows", None)
        if isinstance(windows, list) and windows and deadline is None:
            return False
        if deadline is not None and utime > deadline:
            return False
        # Selected now, it collects after now.
        return start is None or utime >= start

    def request_openings(self, utime: float) -> list[tuple[float, float, int]]:
        """The requests whose windows open after ``utime``, as
        ``(opening, value, obsid)`` in opening order: a request with an
        ``earliest_start`` still to come opens at its first visibility from
        then, unless that is past its deadline."""
        openings: list[tuple[float, float, int]] = []
        for other in self.targets:
            start, _ = self._bounds(other, utime)
            if start is None or start <= utime:
                continue
            if other.done or other.merit <= 0:
                continue
            opening = other.next_vis(float(start))
            if not isinstance(opening, (int, float)) or isinstance(opening, bool):
                continue
            deadline = getattr(other, "deadline", None)
            if isinstance(deadline, (int, float)) and opening > deadline:
                continue
            value = float(other.merit) * (1.0 + self._bonus(other))
            openings.append((float(opening), value, int(other.obsid)))
        openings.sort()
        return openings

    def request_cut(
        self,
        target: Pointing,
        openings: list[tuple[float, float, int]],
        end: float,
    ) -> tuple[float, int] | None:
        """When, before ``end``, the window of a request ranked above
        ``target`` opens, and that request's obsid; None when none does."""
        merit = getattr(target, "merit", None)
        # A test double's target has no number for a merit.
        if not openings or not isinstance(merit, (int, float)):
            return None
        value = float(merit) * (1.0 + self._bonus(target))
        obsid = int(target.obsid)
        for opening, other_value, other in openings:
            if opening >= end:
                return None
            if other_value > value and other != obsid:
                return opening, other
        return None

    def get(
        self,
        ra: float,
        dec: float,
        utime: float,
        collection_deadline: Callable[[Pointing, float], float | None] | None = None,
        slew_estimator: Callable[[Pointing], TargetSlewEstimate] | None = None,
        roll: float | None = None,
        recorder_room_gb: float | None = None,
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
            roll: Current roll in degrees. Required when body-axis slew limits
                are configured and no attitude-aware slew estimator is supplied.
            recorder_room_gb: Gb of science the recorder can take before
                its next contact, under its planning limit. A candidate
                collects no more than the room holds at its own rates, and
                one that cannot collect its minimum snapshot within it is
                passed over, as the schedulers pass over what the recorder
                cannot hold. None checks nothing.

        A candidate's snapshot ends when the window of a request ranked
        above it opens, if that is sooner, so a long observation does not
        block a window that opens inside it: a request with an
        ``earliest_start`` still to come opens at its first visibility from
        then. A candidate whose minimum snapshot does not fit before the
        opening is passed over, for a shorter one or for nothing.

        Returns:
            Next target to observe, or None if no suitable target found.
        """
        assert self.ephem is not None, (
            "Ephemeris must be set in TargetQueue before get()"
        )
        self.utime = utime
        self.meritsort()
        merit_model = self.merit_model

        # Select targets from queue
        targets = [
            t
            for t in self.targets
            if t.merit > 0
            and self._visible_now(t)
            and not t.done
            and self._open_at(t, utime)
        ]
        score_candidates = (
            self.slew_distance_weight != 0.0
            or self.slew_time_weight != 0.0
            or self.collection_time_weight != 0.0
            or self.data_volume_weight != 0.0
            or self.radiator_sun_exposure_weight != 0.0
            or self.radiator_earth_exposure_weight != 0.0
            or merit_model.is_dynamic
        )
        overfill = 0
        """Candidates passed over because the recorder cannot hold their data."""
        openings = self.request_openings(utime)
        blocked: list[tuple[float, int]] = []
        """The openings that candidates were passed over for."""
        cuts: dict[int, tuple[float, int]] = {}
        """The opening each shortened candidate ends at, by obsid."""
        delivered_shares = (
            merit_model.delivered_shares(self.targets)
            if merit_model.completion_deficit_weight > 0.0
            else None
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
        best_breakdown: MeritBreakdown | None = None
        best_score = -np.inf
        last_unix = self.ephem.timestamp[-1].timestamp()
        prune_by_score_bound = (
            score_candidates and self._score_upper_bound_pruning_is_safe()
        )

        # Check each candidate target
        for target in targets:
            # Skip targets with remaining exposure less than minimum snapshot
            if target.exptime is not None and target.exptime < snapshot_min_seconds(
                target
            ):
                continue

            # Reject score-bound losers before the deadline callback can compute
            # their roll and ground-pass ingress geometry.
            if prune_by_score_bound:
                upper_bound = self._candidate_score_upper_bound(
                    target=target,
                    utime=utime,
                    last_unix=last_unix,
                )
                if upper_bound <= best_score:
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
                roll,
                slew_estimator
                if score_candidates or target.uses_mounted_attitude() is True
                else None,
            )

            # A target must start collecting by its deadline.
            _, deadline = self._bounds(target, utime)
            if (
                deadline is not None
                and utime
                + float(target.slewtime)
                + self.observation_timing.setup_seconds
                > deadline
            ):
                continue

            # Calculate observation window
            timing = self._timing(target)
            endtime = (
                utime
                + target.slewtime
                + snapshot_min_seconds(target)
                + timing.total_seconds
                + self._visit_overhead(target)
            )

            # If the end time exceeds ephemeris, clamp it
            if endtime > last_unix:
                endtime = last_unix

            # Check if target is visible for full observation
            visibility_window = target.visible(utime, endtime)
            if visibility_window and self._repoints_forbidden(target):
                # Not during a repoint, and the minimum snapshot before the
                # next one; the window then ends where the repoint begins.
                assert self.coverage is not None
                cut = self.coverage.next_repoint_cut(utime)
                if cut is not None and (cut[0] <= utime or cut[0] < endtime):
                    continue
                if cut is not None:
                    visibility_window = [
                        float(visibility_window[0]),
                        min(float(visibility_window[1]), cut[0]),
                    ]
            cut = (
                self.request_cut(target, openings, float(visibility_window[1]))
                if visibility_window and openings
                else None
            )
            if visibility_window and cut is not None:
                # The snapshot ends where a request ranked above this
                # candidate opens.
                visibility_window = [float(visibility_window[0]), cut[0]]
                cuts[int(target.obsid)] = cut
            if visibility_window:
                target.begin = int(utime)
                windows = getattr(target, "visit_windows", None)
                if isinstance(windows, list) and windows:
                    # The visit this selection would be.
                    target.visit = target.next_visit(utime)
                collection = self._candidate_collection_seconds(
                    target, visibility_window, utime
                )
                if collection < snapshot_min_seconds(target):
                    if cut is not None:
                        blocked.append(cut)
                    continue
                if recorder_room_gb is not None:
                    # Collect no more than the recorder can hold from its
                    # fill now; a snapshot shorter than the minimum waits.
                    collection = min(
                        collection, self._collection_within(target, recorder_room_gb)
                    )
                    if collection < snapshot_min_seconds(target):
                        overfill += 1
                        continue
                target.end = min(
                    float(visibility_window[1]),
                    utime
                    + target.slewtime
                    + collection
                    + timing.total_seconds
                    + self._visit_overhead(target),
                )
                target.set_collection_window(timing)
                value = merit_model.value_terms(
                    target,
                    utime,
                    visibility_window=visibility_window,
                    delivered_shares=delivered_shares,
                )
                bonus = self._bonus(target)
                if bonus:
                    value = value.model_copy(update={"allocation": bonus * value.value})
                # If nothing beyond merit order matters, return the first
                # visible target in merit order (fast path)
                if not score_candidates:
                    self._log_blocked(utime, blocked)
                    self._select(target, value, cut)
                    return target
                # Otherwise, score all visible targets and pick best
                slewdist = getattr(target, "slewdist", 0.0)
                slew_minutes = float(target.slewtime) / 60.0
                slew_end = utime + float(target.slewtime)
                deadline = (
                    collection_deadline(target, slew_end)
                    if collection_deadline is not None
                    else None
                )
                collection_seconds = min(
                    collection,
                    self._candidate_collection_seconds(
                        target=target,
                        visibility_window=visibility_window,
                        utime=utime,
                        deadline=deadline,
                    ),
                )
                if collection_deadline is not None and collection_seconds < float(
                    snapshot_min_seconds(target)
                ):
                    continue
                collection_minutes = collection_seconds / 60.0
                data_gb = (
                    self._snapshot_science_gb(target, collection_seconds)
                    if self.data_volume_weight != 0.0
                    else 0.0
                )

                radiator = 0.0
                if (
                    self.radiator_sun_exposure_weight > 0.0
                    or self.radiator_earth_exposure_weight > 0.0
                ):
                    assert self.config is not None
                    radiators = self.config.spacecraft_bus.radiators
                    if radiators.num_radiators() > 0 and self.ephem is not None:
                        target_is_pointing = issubclass(type(target), Pointing)
                        spacecraft_attitude = (
                            target.spacecraft_attitude
                            if target_is_pointing
                            and target.spacecraft_attitude is not None
                            else (target.ra, target.dec, getattr(target, "roll", 0.0))
                        )
                        metrics = radiators.exposure_metrics(
                            ra_deg=spacecraft_attitude[0],
                            dec_deg=spacecraft_attitude[1],
                            utime=utime,
                            ephem=self.ephem,
                            roll_deg=spacecraft_attitude[2],
                        )
                        sun_exposure = cast(float, metrics.get("sun_exposure", 0.0))
                        earth_exposure = cast(float, metrics.get("earth_exposure", 0.0))
                        radiator = -(
                            self.radiator_sun_exposure_weight * sun_exposure
                            + self.radiator_earth_exposure_weight * earth_exposure
                        )

                breakdown = value.model_copy(
                    update={
                        "slew_distance": -(self.slew_distance_weight * slewdist),
                        "slew_time": -(self.slew_time_weight * slew_minutes),
                        "collection": self.collection_time_weight * collection_minutes,
                        "radiator": radiator,
                        "data": -(self.data_volume_weight * data_gb),
                    }
                )
                if breakdown.score > best_score:
                    best_score = breakdown.score
                    best_target = target
                    best_breakdown = breakdown

        if overfill and self.log is not None:
            self.log.log_event(
                utime=utime,
                event_type="QUEUE",
                description=(
                    f"{overfill} targets passed over: the recorder can take "
                    f"{recorder_room_gb:.2f} Gb of science before its next contact"
                ),
                obsid=None,
                acs_mode=None,
            )
        self._log_blocked(utime, blocked)
        if best_target is not None:
            assert best_breakdown is not None
            self._select(best_target, best_breakdown, cuts.get(int(best_target.obsid)))
        return best_target

    def _log_blocked(self, utime: float, blocked: list[tuple[float, int]]) -> None:
        """Log the candidates passed over because their minimum snapshot
        does not fit before a request's window opens."""
        if not blocked or self.log is None:
            return
        opening, request = min(blocked)
        self.log.log_event(
            utime=utime,
            event_type="QUEUE",
            description=(
                f"{len(blocked)} targets passed over: their minimum snapshot "
                f"does not fit before request {request} opens at "
                f"{unixtime2date(opening)}"
            ),
            obsid=None,
            acs_mode=None,
        )

    def _collection_within(self, target: Pointing, room_gb: float) -> float:
        """Seconds of collection whose science fits in ``room_gb``: without
        limit when the target generates nothing per second, and negative when
        not even its fixed per-observation data fits."""
        base = self._snapshot_science_gb(target, 0.0)
        per_second = self._snapshot_science_gb(target, 1.0) - base
        if per_second <= 0.0:
            return float("inf") if base <= room_gb + 1e-9 else -1.0
        return (room_gb - base) / per_second

    def _snapshot_science_gb(
        self, target: Pointing, collection_seconds: float
    ) -> float:
        """Gb of science a snapshot of ``target`` collecting for
        ``collection_seconds`` puts on the recorder, at the target's own
        rates: its rate for the seconds plus its fixed amount per observation."""
        assert self.config is not None
        return float(
            self.config.science_generated(
                collection_seconds, target.science_data, instrument_modes_of(target)
            )
        ) + extra_data_gb(target)

    def _select(
        self,
        target: Pointing,
        breakdown: MeritBreakdown,
        cut: tuple[float, int] | None = None,
    ) -> None:
        """Freeze the selected target's value and log its terms, and the
        request opening its snapshot ends at, if any."""
        target.merit_breakdown = breakdown
        if self.log is None:
            return
        self.log.log_event(
            utime=self.utime,
            event_type="QUEUE",
            description=f"Selected {target.obsid}: {breakdown.describe()}",
            obsid=target.obsid,
            acs_mode=None,
        )
        if cut is not None:
            self.log.log_event(
                utime=self.utime,
                event_type="QUEUE",
                description=(
                    f"Snapshot of {target.obsid} ends at "
                    f"{unixtime2date(cut[0])}, when request {cut[1]} opens"
                ),
                obsid=target.obsid,
                acs_mode=None,
            )

    def reset(self) -> None:
        """Reset queue by resetting target status.

        Resets done flags on remaining targets for reuse in subsequent
        scheduling cycles.
        """
        for target in self.targets:
            target.reset()


Queue = TargetQueue
