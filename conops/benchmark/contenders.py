"""Ways of scheduling a scenario: dispatch, planned, rolling and configured."""

import time
from collections.abc import Mapping, Sequence
from datetime import timedelta

from ..config import MissionConfig
from ..ditl import DITL, QueueDITL, RollingHorizonDITL, create_ditl
from ..ditl.factory import queue_targets
from ..schedulers import LocalSearchScheduler, LongRangeAllocator, PriorityScheduler
from ..targets import Pointing
from .harness import BenchmarkScenario, Contender, Run, TOOSpec


def _calculate(ditl: DITL | QueueDITL) -> None:
    """Run a simulation, reporting diagnosed failures to the harness."""
    if not ditl.calc():
        causes = [
            f"{event.name}: {event.cause}"
            for event in ditl.fault_management.events
            if event.event_type == "operational_fault"
        ]
        detail = "; ".join(causes) or "no operational fault recorded"
        raise RuntimeError(f"calc() returned False: {detail}")


def _allocator(
    config: MissionConfig,
    scenario: BenchmarkScenario,
    bin_length: timedelta | None,
    reserve: float | None,
) -> LongRangeAllocator | None:
    if bin_length is None:
        return None
    if reserve is None:
        return LongRangeAllocator(
            config, scenario.begin, scenario.end, bin_length=bin_length
        )
    return LongRangeAllocator(
        config, scenario.begin, scenario.end, bin_length=bin_length, reserve=reserve
    )


def _allocation_suffix(allocation: timedelta | None, reserve: float | None) -> str:
    if allocation is None:
        return ""
    return "+alloc" if reserve is None else f"+alloc(reserve {reserve:g})"


def dispatch(
    name: str | None = None,
    *,
    allocation: timedelta | None = None,
    allocation_reserve: float | None = None,
) -> Contender:
    """Queue dispatch: QueueDITL selects each next target as the run goes.

    Args:
        name: Contender name; defaults to "dispatch", with "+alloc" when
            allocating.
        allocation: Bin length of a long-range allocator steering the queue
            (see :class:`~conops.schedulers.LongRangeAllocator`); None for none.
        allocation_reserve: The allocator's reserve; its default if None.
    """

    def simulate(scenario: BenchmarkScenario) -> Run:
        return Run(simulation=_dispatch(scenario, allocation, allocation_reserve))

    suffix = _allocation_suffix(allocation, allocation_reserve)
    return Contender(name=name or f"dispatch{suffix}", simulate=simulate)


def _dispatch(
    scenario: BenchmarkScenario,
    allocation: timedelta | None = None,
    allocation_reserve: float | None = None,
) -> QueueDITL:
    """Run the scenario dispatched by the queue, with its ToOs and activities."""
    config = scenario.make_config()
    ditl = QueueDITL(
        config=config,
        begin=scenario.begin,
        end=scenario.end,
        activities=scenario.activities,
        allocator=_allocator(config, scenario, allocation, allocation_reserve),
    )
    queue_targets(ditl, scenario.make_targets(config))
    for too in scenario.toos:
        ditl.submit_too(**too.model_dump())
    _calculate(ditl)
    return ditl


def too_targets(config: MissionConfig, toos: Sequence[TOOSpec]) -> list[Pointing]:
    """The scenario's ToOs as requests a plan built in advance may take as
    known: each may begin collecting from its submission until its deadline,
    as the rolling simulation requests them when they arrive."""
    targets = []
    for too in toos:
        target = Pointing(
            config=config,
            ra=too.ra,
            dec=too.dec,
            obsid=too.obsid,
            name=too.name,
            merit=too.merit,
            ss_min=min(300, too.exptime),
            ss_max=too.exptime,
            earliest_start=too.submit_time,
            deadline=too.deadline,
        )
        target.exptime = too.exptime
        targets.append(target)
    return targets


def planned(
    scheduler: type[PriorityScheduler] = PriorityScheduler,
    name: str | None = None,
    *,
    seeded_by_dispatch: bool = False,
    **options: object,
) -> Contender:
    """One plan for the whole run, built up front and executed by DITL.

    Args:
        scheduler: Scheduler class to build the plan with.
        name: Contender name; defaults to the scheduler's, with
            "seeded by dispatch" when seeded.
        seeded_by_dispatch: Run the scenario dispatched by the queue first and
            give the scheduler, a :class:`~conops.schedulers.LocalSearchScheduler`
            or one built on it, the plan that run executed as its
            ``start_plan``, with the scenario's ToOs as requests known in
            advance, so the plan it returns scores at least as well as the
            dispatch run in its own terms.
        **options: Extra keyword arguments for the scheduler.
    """
    if seeded_by_dispatch and not issubclass(scheduler, LocalSearchScheduler):
        raise TypeError("only the local-search schedulers take a start plan")

    def simulate(scenario: BenchmarkScenario) -> Run:
        settings = dict(options)
        config = scenario.make_config()
        targets = scenario.make_targets(config)
        if seeded_by_dispatch:
            settings["start_plan"] = _dispatch(scenario).plan
            targets = [*targets, *too_targets(config, scenario.toos)]
        began = time.perf_counter()
        builder = scheduler(
            config,
            targets,
            scenario.begin,
            scenario.end,
            activities=scenario.activities,
            **settings,  # type: ignore[arg-type]
        )
        plan = builder.schedule()
        planning_seconds = time.perf_counter() - began
        ditl = DITL(config=config, plan=plan, begin=scenario.begin, end=scenario.end)
        ditl.step_size = builder.ctx.step_size
        _calculate(ditl)
        return Run(simulation=ditl, planning_seconds=planning_seconds)

    suffix = " seeded by dispatch" if seeded_by_dispatch else ""
    return Contender(
        name=name or f"planned:{scheduler.scheduler_name}{suffix}", simulate=simulate
    )


def rolling(
    scheduler: type[PriorityScheduler] = PriorityScheduler,
    name: str | None = None,
    *,
    horizon: timedelta = timedelta(days=1),
    replan_interval: timedelta = timedelta(hours=12),
    commit_lead_time: timedelta = timedelta(0),
    scheduler_options: Mapping[str, object] | None = None,
    allocation: timedelta | None = None,
    allocation_reserve: float | None = None,
) -> Contender:
    """Rolling-horizon replanning with ``scheduler``, reacting to ToOs.

    Args:
        scheduler: Scheduler class to build each plan with.
        name: Contender name; defaults to one naming the scheduler.
        horizon: How far ahead each plan looks.
        replan_interval: Time between scheduled replans.
        commit_lead_time: Time between building a plan and it taking effect.
        scheduler_options: Extra keyword arguments for the scheduler.
        allocation: Bin length of a long-range allocator steering each replan
            (see :class:`~conops.schedulers.LongRangeAllocator`); None for none.
        allocation_reserve: The allocator's reserve; its default if None.
    """

    def simulate(scenario: BenchmarkScenario) -> Run:
        config = scenario.make_config()
        ditl = RollingHorizonDITL(
            config,
            scenario.make_targets(config),
            begin=scenario.begin,
            end=scenario.end,
            horizon=horizon,
            replan_interval=replan_interval,
            commit_lead_time=commit_lead_time,
            scheduler=scheduler,
            scheduler_options=scheduler_options,
            activities=scenario.activities,
            allocator=_allocator(config, scenario, allocation, allocation_reserve),
        )
        ditl.step_size = int(ditl.ephem.step_size)
        for too in scenario.toos:
            ditl.submit_too(**too.model_dump())
        _calculate(ditl)
        return Run(
            simulation=ditl,
            planning_seconds=sum(r.planning_seconds for r in ditl.replans),
        )

    suffix = _allocation_suffix(allocation, allocation_reserve)
    return Contender(
        name=name or f"rolling:{scheduler.scheduler_name}{suffix}", simulate=simulate
    )


def configured(name: str = "configured") -> Contender:
    """Whatever the scenario's configuration selects in its ``scheduler`` section.

    The simulation is built with :func:`~conops.ditl.create_ditl`, so the
    mode, the scheduler, its settings, replanning and allocation all come
    from the configuration, as they would in a configured run.
    """

    def simulate(scenario: BenchmarkScenario) -> Run:
        config = scenario.make_config()
        began = time.perf_counter()
        ditl = create_ditl(
            config,
            scenario.make_targets(config),
            scenario.begin,
            scenario.end,
            activities=scenario.activities,
        )
        planning: float | None = time.perf_counter() - began
        if isinstance(ditl, (QueueDITL, RollingHorizonDITL)):
            for too in scenario.toos:
                ditl.submit_too(**too.model_dump())
        _calculate(ditl)
        if isinstance(ditl, RollingHorizonDITL):
            planning = sum(r.planning_seconds for r in ditl.replans)
        elif isinstance(ditl, QueueDITL):
            planning = None
        return Run(simulation=ditl, planning_seconds=planning)

    return Contender(name=name, simulate=simulate)
