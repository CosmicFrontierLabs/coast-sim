"""Simulate a spacecraft that follows a plan rebuilt on a rolling horizon."""

import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta, timezone
from enum import Enum

import numpy as np
import rust_ephem
from pydantic import BaseModel, ConfigDict

from ..common import ACSMode, ObsType
from ..config import MissionConfig, ScienceData
from ..config.data_storage import DataStorage, RecorderView
from ..config.solar_panel import SolarArrayDriveState
from ..schedulers.allocator import Allocation, LongRangeAllocator
from ..schedulers.battery import Addition, BatteryLedger
from ..schedulers.context import Attitude, SchedulingContext, tracking_rest
from ..schedulers.priority_scheduler import PriorityScheduler, StartState
from ..schedulers.recorder import RecorderLedger
from ..simulation.passes import Pass
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
    the observation's category is ``interruptible``, and the value frozen
    onto the ToO's first snapshot in the replacement plan beats the value
    frozen onto that observation. When several ToOs need a rapid replan at once, they are
    tried highest-ranked first, and the first that qualifies and can be placed
    interrupts. A ToO that may not interrupt does not take the cut
    observation's place either: it is planned in after the interrupting ToO.
    A rejected attempt leaves no events in the log. Science
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
        include_passes: Whether plans reserve spacecraft tracking activities for passes.
        scheduler: Scheduler class that builds each plan:
            :class:`~conops.schedulers.PriorityScheduler` or a subclass such as
            :class:`~conops.schedulers.LocalSearchScheduler` or
            :class:`~conops.schedulers.CpSatScheduler`.
        scheduler_options: Extra keyword arguments for the scheduler, such as
            ``{"time_limit": 5.0}`` for a local-search scheduler or
            ``{"solver_time_limit": 10.0}`` for the CP-SAT scheduler.
        allocator: Long-range allocator over the whole run. If given, each
            replan allocates the remaining exposure again from when the new
            plan starts, and the scheduler plans each request's allocated
            seconds in their bin, worth ``allocation_bonus`` more (see
            :class:`~conops.schedulers.LongRangeAllocator`).
        allocation: A long-range allocation to follow instead, made
            elsewhere, such as a weekly long-range plan. Each replan plans each
            request's allocated seconds in their bin, less what has been
            collected or committed in the bin. Not with ``allocator``.
        allocation_bonus: Fraction of a request's value its allocated time
            gains; how strongly plans follow the allocation.
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
        allocator: LongRangeAllocator | None = None,
        allocation: Allocation | None = None,
        allocation_bonus: float = 0.5,
        calculate_field_of_regard: bool = False,
    ) -> None:
        if horizon <= timedelta(0) or replan_interval <= timedelta(0):
            raise ValueError("horizon and replan_interval must be positive")
        if allocator is not None and allocation is not None:
            raise ValueError("give allocator or allocation, not both")
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
        self._run_passes: list[Pass] | None = None
        """The interval's ground passes, predicted once for every plan in it."""
        self.allocator = allocator
        self.allocation_bonus = allocation_bonus
        self.allocation: Allocation | None = allocation
        """The allocation the plans follow: the one given, or the one made at
        the last replan if there is an allocator."""
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
        self.plan = Plan(ground_contacts=self._booked_ground_contacts())
        self.replans = []
        self._next_replan = None
        # The passes belong to the interval: a later one predicts its own.
        self._run_passes = None
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
                        ).value
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
            planned = self._plan_interrupt(too, utime, cutoff, attempt)
            if planned is not None:
                replacement = self._placed(planned[2], too.obsid)
                assert replacement is not None
                # The planned snapshot carries its actual allocation bonus,
                # after the horizon's proration and collected/committed
                # seconds have been accounted for by _plan_after().
                if float(replacement.merit) <= float(cut[0].merit):
                    planned = None
            if planned is not None:
                trigger, interrupted = too, int(cut[0].obsid)
                self.log.add_events(attempt.events)
                break
            # This ToO cannot be placed at a higher value after all, so the
            # observation stays whole unless another one can replace it.
            entry, original = cut
            entry.end, entry.collection_begin, entry.collection_end = original
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

    def _plan_interrupt(
        self, too: TOORequest, utime: float, cutoff: float, log: DITLLog
    ) -> tuple[list[PlanEntry], list[PlanEntry], list[PlanEntry], int] | None:
        """Plan the horizon after an interrupt for ``too``, or None if the
        plan cannot collect it.

        ToOs submitted with ``interrupt=False`` wait for the observation in
        progress, so they may not take its place either: a higher-ranked one
        would push the interrupting ToO past its deadline and undo the cut.
        They are held back while the interrupting ToO is placed, then the
        plan up to and including its entry is pinned and the rest of the
        horizon planned again with them; one whose deadline falls before the
        entry ends waits for the next plan.
        """
        waiting = [
            self._too_targets[id(request)]
            for request in self.too_register
            if not request.interrupt
            and id(request) in self._too_targets
            and not self._too_targets[id(request)].done
        ]
        first = DITLLog()
        planned = self._plan_after(utime, cutoff, first, held=waiting)
        placed = self._placed(planned[2], too.obsid)
        if placed is None:
            return None
        if not waiting:
            log.add_events(first.events)
            return planned
        follow = float(placed.end)
        held = [t for t in waiting if t.deadline is not None and t.deadline < follow]
        pinned = planned[2][: planned[2].index(placed) + 1]
        again = DITLLog()
        replanned = self._plan_after(utime, cutoff, again, held=held, pinned=pinned)
        log.add_events(again.events)
        return replanned

    @staticmethod
    def _placed(entries: Sequence[PlanEntry], obsid: int) -> PlanEntry | None:
        """The first science entry for ``obsid`` among ``entries``, if any."""
        return next(
            (
                e
                for e in entries
                if int(e.obsid) == obsid and entry_obstype(e) in PLAN_SCIENCE_OBSTYPES
            ),
            None,
        )

    def _plan_after(
        self,
        utime: float,
        cutoff: float,
        log: DITLLog,
        *,
        held: Sequence[Pointing] = (),
        pinned: Sequence[PlanEntry] = (),
    ) -> tuple[list[PlanEntry], list[PlanEntry], list[PlanEntry], int]:
        """Plan the horizon after the committed activities.

        ``held`` targets are not placed by this plan, though their science,
        collected and committed, still counts towards their programs' shares:
        one of them may be the observation being interrupted. ``pinned``
        entries, planned earlier for this replan, lead the new entries as
        they are: the plan takes over where the last of them leaves the
        spacecraft, and their science counts as collected. Returns the
        committed and dropped entries, the new entries and how many targets
        could not be placed.
        """
        committed = [e for e in self.plan.entries if float(e.begin) < cutoff]
        dropped = [e for e in self.plan.entries if float(e.begin) >= cutoff]
        leading = [*committed, *pinned]
        start_state = self._start_state(leading, utime, cutoff)
        start = start_state.time if start_state is not None else cutoff
        horizon_end = min(utime + self.horizon.total_seconds(), self.uend)
        if horizon_end <= start:
            return committed, dropped, list(pinned), 0
        start_deficit_wh, start_drive_state = self._battery_start(utime, start, leading)
        if self.include_passes and self._run_passes is None:
            # Predicted once, as every plan would predict them for itself:
            # the same seed, start and length give the same passes. Every
            # contact the plans hold comes from them, so they are also the
            # passes to command, and DITL need not predict them again.
            self._run_passes = list(
                SchedulingContext(self.config, self.begin, self.end).predict_passes()
            )
            self._predicted_passes = list(self._run_passes)
        reserved = self._reserved_seconds(leading, utime)
        options = dict(self.scheduler_options)
        used = None
        if self.allocator is not None:
            if self.allocator.passes is None and self._run_passes is not None:
                # The allocator's data budget predicts the run's passes for
                # itself otherwise; they are the ones just predicted.
                self.allocator.passes = list(self._run_passes)
            if self.coverage is not None:
                # The run's coverage, so the allocator does not build its own.
                self.allocator.ctx.coverage = self.coverage
            # Allocated from the exposure that remains, so nothing is used.
            self.allocation = self.allocator.allocate(
                self.targets,
                start,
                reserved,
                unplanned={int(t.obsid) for t in self._too_targets.values()},
                start_volume_gb=self._recorder_volume_at(utime, start, committed),
            )
        elif self.allocation is not None:
            used = self._used_by_bin(self.allocation, leading)
        if self.allocation is not None:
            options["allocated"] = self.allocation.allocated(
                [int(t.obsid) for t in self.targets if not t.done],
                start,
                horizon_end,
                used,
            )
            options["allocation_bonus"] = self.allocation_bonus
        scheduler = self.scheduler(
            self.config,
            # Finished and held targets too: the scheduler does not place
            # them, but their science counts towards their programs' shares.
            self.targets,
            _as_datetime(start),
            _as_datetime(horizon_end),
            step_size=self.step_size,
            include_passes=self.include_passes,
            log=log,
            simulation_start=self.begin,
            simulation_end=self.end,
            start_state=start_state,
            reserved_seconds=reserved,
            reserved_visits=self._reserved_visits(leading, utime),
            held=[int(t.obsid) for t in held],
            start_volume_gb=self._recorder_volume_at(utime, start, leading),
            start_deficit_wh=start_deficit_wh,
            start_drive_state=start_drive_state,
            passes=self._run_passes,
            coverage=self.coverage,
            **options,  # type: ignore[arg-type]
        )
        return (
            committed,
            dropped,
            [*pinned, *scheduler.schedule().entries],
            len(scheduler.unplaced),
        )

    def _interrupt_for(
        self, too: TOORequest, utime: float, cutoff: float, log: DITLLog
    ) -> tuple[PlanEntry, tuple[float, float | None, float | None]] | None:
        """Tentatively cut the observation if the ToO could outrank it.

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
        too_value = self.merit_model.value_terms(
            target,
            utime,
            delivered_shares=self.merit_model.delivered_shares(self.targets),
        ).value
        # An upper bound only: the candidate plan determines the bonus from
        # the snapshot it can place and the allocation it can still use.
        # _replan compares that entry's frozen value before keeping the cut.
        if self.allocation is not None or self.allocator is not None:
            too_value += max(too_value, 0.0) * max(self.allocation_bonus, 0.0)
        if too_value <= float(entry.merit):
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

    def _recorder_volume_at(
        self, utime: float, start: float, committed: Sequence[PlanEntry]
    ) -> float | dict[str, float]:
        """What is on the recorder when a new plan starts.

        The committed activities run on until then, generating data and
        downlinking during their contacts, so the plan starts from what they
        leave rather than from the volume now.
        """
        if isinstance(self.config.data_storage, DataStorage):
            return {
                name: self._project_store_volume(view, utime, start, committed)
                for name, view in self.config.recorder_views().items()
            }
        return self._project_store_volume(self.config, utime, start, committed)

    def _project_store_volume(
        self,
        config: MissionConfig | RecorderView,
        utime: float,
        start: float,
        committed: Sequence[PlanEntry],
    ) -> float:
        now = float(config.recorder.current_volume_gb)
        if start <= utime:
            return now
        contacts = [e for e in committed if entry_obstype(e) == ObsType.GSP]
        science: list[tuple[float, float]] = []
        for entry in committed:
            if entry.collection_begin is None or entry.collection_end is None:
                continue
            begin = max(float(entry.collection_begin), utime)
            end = min(float(entry.collection_end), start)
            if end <= begin:
                continue
            volume = config.science_generated(end - begin, entry.science_data)
            if float(entry.collection_begin) < utime:
                # Already collecting: its fixed amount is on the recorder.
                volume -= config.science_generated(0.0, entry.science_data)
            science.append((begin, volume))
        ledger = RecorderLedger(
            config,
            utime,
            start,
            contacts,
            start_volume_gb=now,
            step_size=self.step_size,
            origin=self.ustart,
            coverage=self.coverage,
        )
        volumes = [0.0] * ledger.segments
        for begin, volume in science:
            volumes[ledger.segment(begin)] += volume
        for entry in contacts:
            # The slew into a contact still to come, flown in PASS mode.
            volumes[ledger.ingress_segment(entry)] += ledger.ingress_volume(entry)
        return ledger.peaks(volumes)[-1]

    def _battery_start(
        self, utime: float, start: float, committed: Sequence[PlanEntry]
    ) -> tuple[float, SolarArrayDriveState]:
        """The battery's deficit and the solar-array drive's state when a new
        plan starts.

        The committed activities run on until then, drawing and charging as
        they go and turning the drive, so the plan starts from what they
        leave rather than from the battery now; the part of each already
        past ``utime`` is in the battery's charge already. The idle time
        between them, and after the last of them, is flown as ACS flies it:
        holding the attitude the activity before leaves (the attitude held
        now, before the first), in ``IDLE`` mode, with the drive tracking
        the Sun if its control lists that mode and the panels generating
        at the angles it reaches.
        """
        battery = self.config.battery
        now = max(0.0, float(battery.watthour) - float(battery.charge_level))
        drive = self.acs.solar_array_drive_state
        if start <= utime:
            return now, drive
        ledger = BatteryLedger(
            self.config,
            utime,
            start,
            start_deficit_wh=now,
            step_size=self.step_size,
            origin=self.ustart,
            drive_state=drive,
        )
        planned: list[list[Addition]] = [[] for _ in range(ledger.segments)]

        def add(addition: Addition | None) -> None:
            if addition is not None:
                planned[ledger.segment(addition.begin)].append(addition)

        # Each slew starts where the entry before leaves the spacecraft, and
        # the idle time before it is held there.
        previous: Attitude = (
            float(self.acs.ra),
            float(self.acs.dec),
            float(self.acs.roll),
        )
        cursor = utime
        for entry in sorted(committed, key=lambda e: float(e.begin)):
            # A contact flies its tracking profile and brakes to rest after
            # it, which is what turns the drive and generates meanwhile.
            gspass = self._entry_passes.get(id(entry))
            is_contact = entry_obstype(entry) == ObsType.GSP and gspass is not None
            slew = self._committed_slew(entry, previous, utime)
            addition = ledger.addition(
                entry,
                after=utime,
                rest=self._rest_after(entry)[0] if is_contact else None,
                track=gspass.attitude_at if is_contact and gspass is not None else None,
                path=slew.attitude if slew is not None else None,
                path_start=(
                    (float(slew.startra), float(slew.startdec), float(slew.startroll))
                    if slew is not None
                    else None
                ),
            )
            if addition is not None:
                if addition.begin > cursor:
                    add(
                        ledger.hold_addition(
                            previous, ACSMode.IDLE, cursor, addition.begin
                        )
                    )
                add(addition)
                cursor = max(cursor, addition.end)
            previous = self._rest_after(entry)[1]
        if start > cursor:
            add(ledger.hold_addition(previous, ACSMode.IDLE, cursor, start))
        energy = tuple(tuple(sorted(g)) for g in planned)
        return (
            ledger.level_at(energy, start),
            ledger.drive_state_at(energy, start) or drive,
        )

    def _committed_slew(
        self, entry: PlanEntry, start: Attitude, utime: float
    ) -> Slew | None:
        """The slew into a committed entry, as ACS flies it: the one under
        way now if it is this entry's, else as it will be flown from
        ``start``, where the entry before leaves the spacecraft. None for an
        entry without a slew, or whose slew is over by ``utime``: what it
        generated is in the battery already.

        The forecast flies it so that the drive turns along it as it does in
        the simulation, in ``PASS`` mode into a contact and ``SLEWING``
        otherwise, and the panels generate along the way.
        """
        begin = float(entry.begin)
        slewtime = float(entry.slewtime or 0)
        if slewtime <= 0.0 or begin + slewtime <= utime:
            return None
        flying = self.acs.last_slew
        if (
            flying is not None
            and flying.is_slewing(utime)
            and abs(float(flying.slewstart) - begin) <= self.step_size
        ):
            return flying
        slew = Slew(config=self.config)
        slew.startra, slew.startdec, slew.startroll = start
        slew.slewstart = begin
        if entry_obstype(entry) == ObsType.GSP:
            slew.endra, slew.enddec, slew.endroll = (
                float(entry.ra),
                float(entry.dec),
                float(entry.roll),
            )
            slew.obstype = ObsType.GSP
        else:
            slew.endra, slew.enddec, slew.endroll = entry.target_body_attitude()
        slew.obsid = entry.obsid
        slew.calc_slewtime()
        return slew

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
    def _used_by_bin(
        allocation: Allocation, committed: Sequence[PlanEntry]
    ) -> dict[int, dict[int, float]]:
        """Science collected or committed, by obsid and the bin it starts in."""
        used: dict[int, dict[int, float]] = {}
        for entry in committed:
            begin, end = entry.collection_begin, entry.collection_end
            if begin is None or end is None:
                continue
            for k, (b0, b1) in enumerate(allocation.bins):
                if b0 <= begin < b1:
                    per_bin = used.setdefault(int(entry.obsid), {})
                    per_bin[k] = per_bin.get(k, 0.0) + float(end) - float(begin)
                    break
        return used

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
