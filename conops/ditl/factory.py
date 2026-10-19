"""Build the simulation a mission configuration's scheduler section selects."""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import rust_ephem

from ..config import MissionConfig
from ..config.scheduler import AllocationSettings, SchedulerMode
from ..schedulers.allocator import LongRangeAllocator
from ..schedulers.registry import scheduler_class
from ..targets import Plan, Pointing, Queue
from .ditl import DITL
from .queue_ditl import QueueDITL
from .rolling_ditl import RollingHorizonDITL


def create_ditl(
    config: MissionConfig,
    targets: Sequence[Pointing],
    begin: datetime | None = None,
    end: datetime | None = None,
    ephem: rust_ephem.Ephemeris | None = None,
) -> QueueDITL | DITL | RollingHorizonDITL:
    """Return a simulation of ``targets`` scheduled as ``config.scheduler`` says.

    * ``dispatch``: a :class:`~conops.ditl.QueueDITL` with the targets queued.
    * ``planned``: a :class:`~conops.ditl.DITL` executing a plan built now by
      the configured scheduler.
    * ``rolling``: a :class:`~conops.ditl.RollingHorizonDITL` that builds and
      rebuilds its plan with the configured scheduler while it runs.

    With ``scheduler.allocation`` set, dispatch and rolling simulations get a
    :class:`~conops.schedulers.LongRangeAllocator` over the run, and follow
    its allocation as closely as its ``strictness`` says.

    The simulation is ready for :meth:`calc`. Targets of Opportunity can be
    submitted to the dispatch and rolling simulations before running them.

    Args:
        config: Mission configuration, including its ``scheduler`` section.
        targets: Targets to observe. Each is given the simulation's
            configuration, constraint, ephemeris and ACS settings. Dispatch
            and planned modes use copies of them; rolling replanning uses
            them, and updates their remaining exposure.
        begin: Simulation start; defaults to the start of the ephemeris.
        end: Simulation end; defaults to the end of the ephemeris.
        ephem: Ephemeris; defaults to the one on the config's constraint.
    """
    if ephem is not None:
        config.constraint.ephem = ephem
    ephemeris = config.constraint.ephem
    if ephemeris is None:
        raise ValueError("an ephemeris is required: pass ephem or set it on the config")
    begin = begin or _as_utc(ephemeris.timestamp[0])
    end = end or _as_utc(ephemeris.timestamp[-1])
    settings = config.scheduler
    allocation = settings.allocation
    allocator = None
    if allocation is not None:
        allocator = LongRangeAllocator(
            config,
            begin,
            end,
            bin_length=timedelta(seconds=allocation.bin_seconds),
            efficiency=allocation.efficiency,
            reserve=allocation.reserve,
            solver=allocation.solver,
            time_limit=allocation.time_limit_seconds,
        )
    # How closely scheduling follows the allocation, if there is one.
    steering = allocation or AllocationSettings()

    if settings.mode is SchedulerMode.DISPATCH:
        queue_ditl = QueueDITL(
            config=config,
            begin=begin,
            end=end,
            allocator=allocator,
            allocation_strictness=steering.strictness,
            allocation_bonus=steering.bonus,
        )
        queue_targets(queue_ditl, targets)
        return queue_ditl

    scheduler = scheduler_class(settings.kind)
    options = settings.scheduler_options()
    if settings.mode is SchedulerMode.PLANNED:
        copies = [target.model_copy() for target in targets]
        _bind_targets(config, copies)
        builder = scheduler(
            config,
            copies,
            begin,
            end,
            include_passes=settings.include_passes,
            **options,  # type: ignore[arg-type]
        )
        ditl = DITL(config=config, plan=builder.schedule(), begin=begin, end=end)
        ditl.step_size = builder.ctx.step_size
        return ditl

    # Rolling replanning records collection on the targets themselves.
    _bind_targets(config, targets)
    replanning = settings.replanning
    rolling = RollingHorizonDITL(
        config,
        targets,
        begin=begin,
        end=end,
        horizon=timedelta(seconds=replanning.horizon_seconds),
        replan_interval=timedelta(seconds=replanning.replan_interval_seconds),
        commit_lead_time=timedelta(seconds=replanning.commit_lead_time_seconds),
        rapid_replans=replanning.rapid_replans,
        allow_interrupts=replanning.allow_interrupts,
        include_passes=settings.include_passes,
        scheduler=scheduler,
        scheduler_options=options,
        allocator=allocator,
        allocation_strictness=steering.strictness,
        allocation_bonus=steering.bonus,
    )
    rolling.step_size = int(ephemeris.step_size)
    return rolling


def queue_targets(ditl: QueueDITL, targets: Sequence[Pointing]) -> None:
    """Add a copy of each target to a QueueDITL's queue.

    The copy keeps the target's settings and its observation history: the
    science it has collected, when it last collected it, and its original
    exposure. Cadence and program-share merit follow that history, so a
    target observed before the run is ranked as one. The targets passed in
    are not changed by the run.
    """
    queue = ditl.queue
    if not isinstance(queue, Queue):
        raise TypeError("targets can only be added to a QueueDITL's own Queue")
    config = ditl.config
    copies = [target.model_copy() for target in targets]
    _bind_targets(config, copies)
    for queued in copies:
        # The queue ranks targets from fom, a legacy alias of merit that is
        # not kept in step with it.
        queued.fom = queued.merit
        if queued.exptime is None:
            queued.exptime = queued.ss_max
        # As Queue.add does: observe with the telescope the payload assigns.
        telescope = config.payload.telescope_for_target(queued.instrument_name)
        queued.instrument_name = telescope.name if telescope is not None else None
        queued.visibility()
        queue.append(queued)


def _bind_targets(config: MissionConfig, targets: Sequence[Pointing]) -> None:
    """Give targets the simulation's configuration, constraint, ephemeris and
    ACS settings.

    A target caches these from the configuration it was made with, and one
    loaded from JSON has none. Visibility, and the telescope a target asks
    for, are resolved from them.
    """
    Plan(entries=list(targets)).bind_runtime(config)


def _as_utc(timestamp: datetime) -> datetime:
    return (
        timestamp
        if timestamp.tzinfo is not None
        else timestamp.replace(tzinfo=timezone.utc)
    )
