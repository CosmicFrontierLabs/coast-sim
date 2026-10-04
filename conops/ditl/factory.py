"""Build the simulation a mission configuration's scheduler section selects."""

from collections.abc import Sequence
from datetime import datetime, timedelta, timezone

import rust_ephem

from ..config import MissionConfig, bind_ephemeris
from ..config.scheduler import SchedulerMode
from ..schedulers.registry import planner_class
from ..targets import Pointing
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
      the configured planner.
    * ``rolling``: a :class:`~conops.ditl.RollingHorizonDITL` that builds and
      rebuilds its plan with the configured planner while it runs.

    The simulation is ready for :meth:`calc`. Targets of Opportunity can be
    submitted to the dispatch and rolling simulations before running them.

    Args:
        config: Mission configuration, including its ``scheduler`` section.
        targets: Targets to observe. Dispatch adds copies of them to its
            queue; rolling replanning updates their remaining exposure.
        begin: Simulation start; defaults to the start of the ephemeris.
        end: Simulation end; defaults to the end of the ephemeris.
        ephem: Ephemeris; defaults to the one on the config's constraint.
    """
    if ephem is not None:
        bind_ephemeris(config, ephem)
    ephemeris = config.constraint.ephem
    if ephemeris is None:
        raise ValueError("an ephemeris is required: pass ephem or set it on the config")
    begin = begin or _as_utc(ephemeris.timestamp[0])
    end = end or _as_utc(ephemeris.timestamp[-1])
    scheduler = config.scheduler
    step_size = int(ephemeris.step_size)

    if scheduler.mode is SchedulerMode.DISPATCH:
        queue_ditl = QueueDITL(config=config, begin=begin, end=end)
        queue_targets(queue_ditl, targets)
        return queue_ditl

    planner = planner_class(scheduler.planner.kind)
    options = scheduler.planner.options()
    if scheduler.mode is SchedulerMode.PLANNED:
        builder = planner(config, targets, begin, end, **options)  # type: ignore[arg-type]
        ditl = DITL(config=config, plan=builder.schedule(), begin=begin, end=end)
        ditl.step_size = builder.ctx.step_size
        return ditl

    replanning = scheduler.replanning
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
        include_passes=scheduler.planner.include_passes,
        planner=planner,
        planner_options={k: v for k, v in options.items() if k != "include_passes"},
    )
    rolling.step_size = step_size
    return rolling


def queue_targets(ditl: QueueDITL, targets: Sequence[Pointing]) -> None:
    """Add each target to a QueueDITL's queue, with its settings."""
    for target in targets:
        ditl.queue.add(
            ra=target.ra,
            dec=target.dec,
            obsid=target.obsid,
            name=target.name,
            merit=float(target.fom),
            exptime=int(
                target.exptime if target.exptime is not None else target.ss_max
            ),
            ss_min=int(target.ss_min),
            ss_max=int(target.ss_max),
            instrument_name=target.instrument_name,
            deadline=target.deadline,
        )


def _as_utc(timestamp: datetime) -> datetime:
    return (
        timestamp
        if timestamp.tzinfo is not None
        else timestamp.replace(tzinfo=timezone.utc)
    )
