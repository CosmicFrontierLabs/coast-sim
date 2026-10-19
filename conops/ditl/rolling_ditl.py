"""Simulate a spacecraft that follows a plan rebuilt on a rolling horizon."""

import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from enum import Enum

import numpy as np
import rust_ephem
from pydantic import BaseModel, ConfigDict

from ..common import ObsType
from ..config import MissionConfig, ScienceData
from ..schedulers.context import Attitude, tracking_rest
from ..schedulers.priority_scheduler import PriorityScheduler, StartState
from ..simulation.slew import Slew
from ..targets import Plan, PlanEntry, Pointing
from ..targets.merit import MeritModel
from .ditl import DITL
from .ditl_log import DITLLog
from .plan_validator import PLAN_SCIENCE_OBSTYPES, entry_obstype
from .queue_ditl import TOORequest


class ReplanReason(str, Enum):
    """Why a plan was rebuilt."""

    INITIAL = "initial"
    SCHEDULED = "scheduled"
    RAPID = "rapid"


class ReplanRecord(BaseModel):
    """What one replan changed.

    Attributes:
        utime: When the replan ran.
        reason: Why it ran.
        cutoff: Activities starting before this time were already committed
            and kept.
        kept: Committed entries not yet finished when the replan ran.
        dropped: Planned entries that had not been committed, replaced.
        added: Entries in the new plan.
        unplaced: Targets the new plan could not fit.
        planning_seconds: Wall-clock time spent building the plan.
        trigger_obsid: The urgent ToO that triggered the replan or interrupt.
        interrupted_obsid: The observation cut short for it, if any.
    """

    model_config = ConfigDict(frozen=True)

    utime: float
    reason: ReplanReason
    cutoff: float
    kept: int
    dropped: int
    added: int
    unplaced: int
    planning_seconds: float
    trigger_obsid: int | None = None
    interrupted_obsid: int | None = None


class RollingHorizonDITL(DITL):
    """Follow a plan rebuilt from the spacecraft's actual state as time passes.

    This models a ground-planned mission. A plan for the next ``horizon`` is
    built with ``scheduler`` (by default
    :class:`~conops.schedulers.PriorityScheduler`) at the start and
    every ``replan_interval`` after that. Each replan keeps everything already
    commanded or due to start within ``commit_lead_time`` (the time a new plan
    takes to reach the spacecraft), and plans the rest again from the
    spacecraft's attitude and each target's remaining exposure. The first plan
    is built before the run starts, so it takes effect at once.

    Targets of Opportunity (see :meth:`submit_too`) join the target pool when
    they are submitted and go into the next scheduled plan. A ToO whose
    deadline falls before that plan could start collecting it (after the
    activities it keeps, a worst-case slew and the setup time) triggers a
    rapid replan at once, or an interrupt in a scheduled replan due then.
    The observation running at the commit cutoff is cut there if
    ``allow_interrupts`` is set, the ToO allows it (its ``interrupt`` flag),
    the observation's category is ``interruptible``, the ToO's tier and value,
    evaluated now, beat the tier and value frozen onto that observation, and
    the new plan can collect the ToO. When several ToOs need a rapid replan at once, they are
    tried highest-ranked first, and the first that qualifies and can be placed
    interrupts. A rejected attempt leaves no events in the log. Science
    already collected is kept, and the cut
    observation's cleanup follows it.

    Each replan is recorded in :attr:`replans`, and
    :meth:`too_response_times` reports how long each ToO waited for its first
    collection. :meth:`validate_plan_matches_execution` compares everything
    executed with the plans that were followed.

    The targets' remaining exposure is updated as science is collected, as in
    a :class:`~conops.targets.TargetQueue`.

    Args:
        config: Mission configuration.
        targets: Pool of targets to plan from. Obsids must be unique.
        ephem: Ephemeris; defaults to the one on the config's constraint.
        begin: Simulation start.
        end: Simulation end.
        horizon: How far ahead each plan looks.
        replan_interval: Time between scheduled replans.
        commit_lead_time: Time between building a plan and it taking effect.
            Activities starting before then are kept from the previous plan.
        rapid_replans: Whether a ToO with a close deadline triggers an
            immediate replan.
        allow_interrupts: Whether a rapid replan may cut short the
            observation in progress.
        include_passes: Whether plans reserve ground-station passes.
        scheduler: Scheduler class that builds each plan:
            :class:`~conops.schedulers.PriorityScheduler` or a subclass such as
            :class:`~conops.schedulers.LocalSearchScheduler` or
            :class:`~conops.schedulers.CpSatScheduler`.
        scheduler_options: Extra keyword arguments for the scheduler, such as
            ``{"time_limit": 5.0}`` for a local-search scheduler or
            ``{"solver_time_limit": 10.0}`` for the CP-SAT scheduler.
        calculate_field_of_regard: Whether to compute field-of-regard telemetry.
    """

    def __init__(
        self,
        config: MissionConfig,
        targets: Sequence[Pointing],
        ephem: rust_ephem.Ephemeris | None = None,
        begin: datetime | None = None,
        end: datetime | None = None,
        *,
        horizon: timedelta = timedelta(days=1),
        replan_interval: timedelta = timedelta(hours=12),
        commit_lead_time: timedelta = timedelta(0),
        rapid_replans: bool = True,
        allow_interrupts: bool = True,
        include_passes: bool = True,
        scheduler: type[PriorityScheduler] = PriorityScheduler,
        scheduler_options: Mapping[str, object] | None = None,
        calculate_field_of_regard: bool = False,
    ) -> None:
        if horizon <= timedelta(0) or replan_interval <= timedelta(0):
            raise ValueError("horizon and replan_interval must be positive")
        if commit_lead_time < timedelta(0):
            raise ValueError("commit_lead_time must not be negative")
        super().__init__(
            config=config,
            ephem=ephem,
            plan=Plan(),
            begin=begin,
            end=end,
            calculate_field_of_regard=calculate_field_of_regard,
        )
        self.targets = list(targets)
        self.horizon = horizon
        self.replan_interval = replan_interval
        self.commit_lead_time = commit_lead_time
        self.rapid_replans = rapid_replans
        self.allow_interrupts = allow_interrupts
        self.include_passes = include_passes
        self.scheduler = scheduler
        self.scheduler_options = dict(scheduler_options or {})
        self.merit_model = MeritModel(config)
        self.too_register: list[TOORequest] = []
        self.replans: list[ReplanRecord] = []
        self._too_targets: dict[int, Pointing] = {}
        # When each ToO counts as submitted: no earlier than the run it joined.
        self._too_submitted: dict[int, float] = {}
        self._first_collection: dict[int, float] = {}
        self._next_replan: float | None = None

    # ── Public API ────────────────────────────────────────────────────────

    def submit_too(
        self,
        obsid: int,
        ra: float,
        dec: float,
        merit: float,
        exptime: int,
        name: str,
        submit_time: float | datetime | None = None,
        deadline: float | datetime | None = None,
        science_data: Mapping[str, ScienceData] | None = None,
        interrupt: bool = True,
    ) -> TOORequest:
        """Submit a Target of Opportunity that joins the target pool when active.

        Args:
            obsid: Unique observation identifier
            ra: Right ascension in degrees
            dec: Declination in degrees
            merit: Base merit
            exptime: Requested exposure time in seconds, observed in one snapshot
            name: Human-readable name
            submit_time: When the ToO becomes active (Unix time or datetime);
                None means from the start of the simulation.
            deadline: Latest time science collection may begin (Unix time or
                datetime), or None for no deadline.
            science_data: Science data observing the ToO generates, keyed by
                instrument name; other instruments generate their own.
            interrupt: Whether it may cut short the observation in progress.
                If False, a rapid replan for it starts it once that
                observation ends.
        """
        too = TOORequest(
            obsid=obsid,
            ra=ra,
            dec=dec,
            merit=merit,
            exptime=exptime,
            name=name,
            submit_time=_unix_time(submit_time) if submit_time is not None else 0.0,
            deadline=_unix_time(deadline) if deadline is not None else None,
            science_data=dict(science_data or {}),
            interrupt=interrupt,
        )
        self.too_register.append(too)
        return too

    def too_response_times(self) -> dict[int, float | None]:
        """Seconds from each ToO's submission to its first science, by obsid.

        None means the ToO collected no science in the simulation.
        """
        responses: dict[int, float | None] = {}
        for too in self.too_register:
            first = self._first_collection.get(too.obsid)
            submitted = self._too_submitted.get(id(too))
            responses[too.obsid] = (
                None if first is None or submitted is None else first - submitted
            )
        return responses

    def calc(self) -> bool:
        """Run the simulation, building and rebuilding the plan as it goes.

        Calling it again for a later interval continues from where the last
        one left off: targets keep their collected science, and ToOs already
        in the target pool are neither added again nor re-run.
        """
        self.plan = Plan()
        self.replans = []
        self._next_replan = None
        return super().calc()

    # ── Execution hooks ──────────────────────────────────────────────────

    def _execute_plan(self, utime: float) -> bool:
        """Replan if one is due, then issue the commands the plan makes due."""
        changed = False
        if not self.acs.in_safe_mode:
            reason, triggers = self._replan_reason(utime)
            if reason is not None:
                had_active = self._active_entry is not None
                self._replan(utime, reason, triggers)
                self.ppt = self.plan.which_ppt(utime)
                # An interrupt may have cut the observation in progress to now.
                self._finish_expired_entry(utime)
                changed = had_active and self._active_entry is None
        return super()._execute_plan(utime) or changed

    def _credit_collection(self, entry: PlanEntry, start: float, stop: float) -> None:
        """Credit collected science to the target that requested it."""
        target = self._target_for(int(entry.obsid))
        if target is None:
            return
        target.record_collection(stop, stop - start)
        self._first_collection.setdefault(int(entry.obsid), start)
        for too in self.too_register:
            if too.obsid == entry.obsid:
                too.executed = True

    # ── Replanning ───────────────────────────────────────────────────────

    def _target_for(self, obsid: int) -> Pointing | None:
        for target in self.targets:
            if int(target.obsid) == obsid:
                return target
        return None

    def _activate_toos(self, utime: float) -> list[TOORequest]:
        """Add newly submitted ToOs to the target pool and return them."""
        activated = []
        for too in self.too_register:
            if id(too) in self._too_targets or too.submit_time > utime:
                continue
            target = Pointing(
                config=self.config,
                ra=too.ra,
                dec=too.dec,
                obsid=too.obsid,
                name=too.name,
                merit=too.merit,
                ss_min=min(300, too.exptime),
                ss_max=too.exptime,
                deadline=too.deadline,
                science_data=too.science_data,
            )
            target.exptime = too.exptime
            self._too_targets[id(too)] = target
            self._too_submitted[id(too)] = max(too.submit_time, self.ustart)
            self.targets.append(target)
            activated.append(too)
            self.log.log_event(
                utime=utime,
                event_type="TOO",
                description=f"ToO {too.name} (obsid={too.obsid}) joined the target pool",
                obsid=too.obsid,
            )
        return activated

    def _replan_reason(
        self, utime: float
    ) -> tuple[ReplanReason | None, list[TOORequest]]:
        """Return why a replan is due at ``utime``, and the urgent ToOs behind
        it, highest-ranked first."""
        activated = self._activate_toos(utime)
        if self._next_replan is None:
            return ReplanReason.INITIAL, []
        scheduled = utime >= self._next_replan
        triggers: list[TOORequest] = []
        if self.rapid_replans and activated:
            # The next plan could start collecting no sooner than the
            # activities it keeps allow, plus a slew and setup.
            next_effect = (
                self._plan_start(utime if scheduled else self._next_replan)
                + Slew.duration_upper_bound(self.config.spacecraft_bus.attitude_control)
                + self.config.payload.observation_timing.setup_seconds
                + self.step_size
            )
            urgent = [
                too
                for too in activated
                if too.deadline is not None and too.deadline < next_effect
            ]
            if urgent:
                # One replan serves them all. They are tried in rank order to
                # interrupt the observation in progress.
                shares = self.merit_model.delivered_shares(self.targets)
                triggers = sorted(
                    urgent,
                    key=lambda too: (
                        self.merit_model.value_terms(
                            self._too_targets[id(too)],
                            utime,
                            delivered_shares=shares,
                        ).value_rank
                    ),
                    reverse=True,
                )
        if scheduled:
            return ReplanReason.SCHEDULED, triggers
        if triggers:
            return ReplanReason.RAPID, triggers
        return None, []

    def _plan_start(self, plan_time: float) -> float:
        """Return when a plan built at ``plan_time`` could take over."""
        cutoff = self._cutoff(plan_time)
        committed = [e for e in self.plan.entries if float(e.begin) < cutoff]
        start_state = self._start_state(committed, plan_time, cutoff)
        return start_state.time if start_state is not None else cutoff

    def _ceil_step(self, utime: float) -> float:
        """Return the first simulation step at or after ``utime``."""
        steps = np.ceil((utime - self.ustart) / self.step_size - 1e-9)
        return self.ustart + float(steps) * self.step_size

    def _cutoff(self, utime: float) -> float:
        """Return the first step a new plan can change."""
        return self._ceil_step(utime + self.commit_lead_time.total_seconds())

    def _replan(
        self, utime: float, reason: ReplanReason, triggers: Sequence[TOORequest]
    ) -> None:
        """Keep committed activities and plan the rest of the horizon again."""
        # The first plan is built before the run, so nothing waits for it.
        cutoff = (
            self._ceil_step(utime)
            if reason is ReplanReason.INITIAL
            else self._cutoff(utime)
        )
        trigger = triggers[0] if triggers else None
        interrupted = None
        planned = None

        began = time.perf_counter()
        for too in triggers:
            # Each attempt logs privately, so a rejected one leaves no trace
            # in the live log or its store.
            attempt = DITLLog()
            cut = self._interrupt_for(too, utime, cutoff, attempt)
            if cut is None:
                continue
            planned = self._plan_after(utime, cutoff, attempt)
            if any(
                int(e.obsid) == too.obsid and entry_obstype(e) in PLAN_SCIENCE_OBSTYPES
                for e in planned[2]
            ):
                trigger, interrupted = too, int(cut[0].obsid)
                self.log.add_events(attempt.events)
                break
            # This ToO cannot be placed after all, so the observation stays
            # whole unless a lower-ranked one can replace it.
            entry, original = cut
            entry.end, entry.collection_begin, entry.collection_end = original
            planned = None
        if planned is None:
            planned = self._plan_after(utime, cutoff, self.log)
        committed, dropped, new_entries, unplaced = planned
        planning_seconds = time.perf_counter() - began

        self._splice(committed, dropped, new_entries, cutoff)
        record = ReplanRecord(
            utime=utime,
            reason=reason,
            cutoff=cutoff,
            kept=sum(1 for e in committed if float(e.end) > utime),
            dropped=len(dropped),
            added=len(new_entries),
            unplaced=unplaced,
            planning_seconds=planning_seconds,
            trigger_obsid=trigger.obsid if trigger is not None else None,
            interrupted_obsid=interrupted,
        )
        self.replans.append(record)
        self.log.log_event(
            utime=utime,
            event_type="SCHEDULER",
            description=(
                f"{reason.value.capitalize()} replan: kept {record.kept}, "
                f"dropped {record.dropped}, added {record.added} entries "
                f"in {planning_seconds:.2f}s"
            ),
            obsid=record.trigger_obsid,
        )
        if reason is not ReplanReason.RAPID:
            self._next_replan = utime + self.replan_interval.total_seconds()

    def _plan_after(
        self, utime: float, cutoff: float, log: DITLLog
    ) -> tuple[list[PlanEntry], list[PlanEntry], list[PlanEntry], int]:
        """Plan the horizon after the committed activities.

        Returns the committed and dropped entries, the new entries and how
        many targets could not be placed.
        """
        committed = [e for e in self.plan.entries if float(e.begin) < cutoff]
        dropped = [e for e in self.plan.entries if float(e.begin) >= cutoff]
        start_state = self._start_state(committed, utime, cutoff)
        start = start_state.time if start_state is not None else cutoff
        horizon_end = min(utime + self.horizon.total_seconds(), self.uend)
        if horizon_end <= start:
            return committed, dropped, [], 0
        scheduler = self.scheduler(
            self.config,
            # Finished targets too: the scheduler does not place them, but their
            # science counts towards their programs' shares.
            self.targets,
            _as_datetime(start),
            _as_datetime(horizon_end),
            step_size=self.step_size,
            include_passes=self.include_passes,
            log=log,
            simulation_start=self.begin,
            simulation_end=self.end,
            start_state=start_state,
            reserved_seconds=self._reserved_seconds(committed, utime),
            reserved_visits=self._reserved_visits(committed, utime),
            **self.scheduler_options,  # type: ignore[arg-type]
        )
        return (
            committed,
            dropped,
            list(scheduler.schedule().entries),
            len(scheduler.unplaced),
        )

    def _interrupt_for(
        self, too: TOORequest, utime: float, cutoff: float, log: DITLLog
    ) -> tuple[PlanEntry, tuple[float, float | None, float | None]] | None:
        """Cut the observation running at ``cutoff`` if the ToO outranks it.

        That observation is committed but may not have started yet, when the
        commit lead time reaches into it. Collection stops no earlier than
        ``utime``, as science before then is already collected, and the
        observation ends after its cleanup. Returns the cut entry with its
        original end and collection window.
        """
        entry = next(
            (
                e
                for e in self.plan.entries
                if entry_obstype(e) in PLAN_SCIENCE_OBSTYPES
                and float(e.begin) < cutoff < float(e.end)
            ),
            None,
        )
        if not self.allow_interrupts or not too.interrupt or entry is None:
            return None
        category = self.merit_model.categories.get_category(int(entry.obsid))
        if not category.interruptible:
            return None
        target = self._too_targets[id(too)]
        too_rank = self.merit_model.value_terms(
            target,
            utime,
            delivered_shares=self.merit_model.delivered_shares(self.targets),
        ).value_rank
        current_rank = (category.tier, float(entry.merit))
        if too_rank <= current_rank:
            return None
        # A slew under way finishes first, so the cut is at the step after the
        # observation's slew ends, if that is after the cutoff.
        arrival = float(entry.begin) + float(entry.slewtime)
        cut = max(cutoff, self._ceil_step(arrival))
        end = cut
        collection = None
        if entry.collection_begin is not None and entry.collection_end is not None:
            post = self.config.payload.observation_timing.post_collection_seconds
            begin = min(float(entry.collection_begin), cut)
            # Science before ``utime`` is already collected, so collection
            # cannot stop before then; cleanup follows wherever it stops.
            collected = min(float(entry.collection_end), max(begin, utime))
            stop = max(collected, min(float(entry.collection_end), cut - post))
            if stop > begin:
                end = max(cut, self._ceil_step(stop + post))
            collection = (begin, stop)
        if end >= float(entry.end):
            return None
        original = (float(entry.end), entry.collection_begin, entry.collection_end)
        entry.end = end
        if collection is not None:
            entry.collection_begin, entry.collection_end = collection
        log.log_event(
            utime=utime,
            event_type="TOO",
            description=(
                f"ToO {too.name} (obsid={too.obsid}) interrupts observation "
                f"{entry.obsid} at the commit cutoff"
            ),
            obsid=too.obsid,
        )
        return entry, original

    def _start_state(
        self, committed: Sequence[PlanEntry], utime: float, cutoff: float
    ) -> StartState | None:
        """Return where the new plan takes over from the committed activities."""
        pending = [e for e in committed if float(e.end) > utime]
        if pending:
            rest_time, attitude = max(
                (self._rest_after(e) for e in pending), key=lambda rest: rest[0]
            )
            return StartState(time=max(cutoff, rest_time), attitude=attitude)
        if not committed:
            # Never commanded: the scheduler replays the freshly started spacecraft.
            return None
        return StartState(
            time=cutoff,
            attitude=(float(self.acs.ra), float(self.acs.dec), float(self.acs.roll)),
        )

    def _rest_after(self, entry: PlanEntry) -> tuple[float, Attitude]:
        """Return when an entry leaves the spacecraft at rest, and where.

        A ground contact brakes past its tracking profile, so it comes to rest
        after the contact ends and away from the profile's last sample.
        """
        gspass = self._entry_passes.get(id(entry))
        if entry_obstype(entry) == ObsType.GSP and gspass is not None:
            # The profile the plan selected, which ACS flies.
            profile = list(
                zip(
                    gspass.ra,
                    gspass.dec,
                    gspass.roll or [gspass.gsstartroll] * len(gspass.ra),
                    strict=True,
                )
            )
            acquire = float(entry.contact_begin or entry.begin)
            rest = tracking_rest(
                gspass, profile, acquire, self.config.spacecraft_bus.attitude_control
            )
            if rest is not None:
                rest_time, attitude = rest
                return max(float(entry.end), rest_time), attitude
        return float(entry.end), _final_attitude(entry)

    @staticmethod
    def _reserved_seconds(
        committed: Sequence[PlanEntry], utime: float
    ) -> dict[int, float]:
        """Exposure committed entries will still collect after ``utime``, by obsid."""
        reserved: dict[int, float] = {}
        for entry in committed:
            if entry_obstype(entry) not in PLAN_SCIENCE_OBSTYPES:
                continue
            remaining = entry.collection_seconds_between(utime, float(entry.end))
            if remaining > 0:
                obsid = int(entry.obsid)
                reserved[obsid] = reserved.get(obsid, 0.0) + remaining
        return reserved

    @staticmethod
    def _reserved_visits(
        committed: Sequence[PlanEntry], utime: float
    ) -> dict[int, float]:
        """When committed entries still collecting after ``utime`` finish, by obsid."""
        visits: dict[int, float] = {}
        for entry in committed:
            if (
                entry_obstype(entry) not in PLAN_SCIENCE_OBSTYPES
                or entry.collection_end is None
                or float(entry.collection_end) <= utime
            ):
                continue
            obsid = int(entry.obsid)
            visits[obsid] = max(visits.get(obsid, 0.0), float(entry.collection_end))
        return visits

    def _splice(
        self,
        committed: Sequence[PlanEntry],
        dropped: Sequence[PlanEntry],
        new_entries: Sequence[PlanEntry],
        cutoff: float,
    ) -> None:
        """Replace the uncommitted part of the plan, in the plan and the executor."""
        dropped_ids = {id(e) for e in dropped}
        for entry_id in dropped_ids:
            self._entry_passes.pop(entry_id, None)
        self.plan.entries = [*committed, *new_entries]

        commanded = self._entries_to_command[: self._next_entry_index]
        pending = [
            e
            for e in self._entries_to_command[self._next_entry_index :]
            if id(e) not in dropped_ids and float(e.begin) < cutoff
        ]
        self._entries_to_command = [
            *commanded,
            *pending,
            *sorted(new_entries, key=lambda e: float(e.begin)),
        ]
        self._sync_planned_passes()


def _final_attitude(entry: PlanEntry) -> tuple[float, float, float]:
    """Return the attitude the spacecraft holds when an entry ends."""
    if entry_obstype(entry) == ObsType.GSP:
        assert entry.track_end_ra is not None and entry.track_end_dec is not None
        return (
            float(entry.track_end_ra),
            float(entry.track_end_dec),
            float(entry.track_end_roll or 0.0),
        )
    return entry.target_body_attitude()


def _unix_time(value: float | datetime) -> float:
    """Convert a Unix timestamp or datetime (naive means UTC) to Unix seconds."""
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.timestamp()
    return float(value)


def _as_datetime(utime: float) -> datetime:
    return datetime.fromtimestamp(utime, tz=timezone.utc)
