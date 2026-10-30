"""Ways of scheduling a scenario: dispatch, planned, rolling and configured."""

import time
from collections.abc import Mapping
from datetime import timedelta

from ..config import MissionConfig
from ..ditl import DITL, QueueDITL, RollingHorizonDITL, create_ditl
from ..ditl.factory import queue_targets
from ..schedulers import LongRangeAllocator, PriorityScheduler
from .harness import BenchmarkScenario, Contender, Run


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
        config = scenario.make_config()
        ditl = QueueDITL(
            config=config,
            begin=scenario.begin,
            end=scenario.end,
            allocator=_allocator(config, scenario, allocation, allocation_reserve),
        )
        queue_targets(ditl, scenario.make_targets(config))
        for too in scenario.toos:
            ditl.submit_too(**too.model_dump())
        _calculate(ditl)
        return Run(simulation=ditl)

    suffix = _allocation_suffix(allocation, allocation_reserve)
    return Contender(name=name or f"dispatch{suffix}", simulate=simulate)


def planned(
    scheduler: type[PriorityScheduler] = PriorityScheduler,
    name: str | None = None,
    **options: object,
) -> Contender:
    """One plan for the whole run, built up front and executed by DITL.

    Args:
        scheduler: Scheduler class to build the plan with.
        name: Contender name; defaults to the scheduler's.
        **options: Extra keyword arguments for the scheduler.
    """

    def simulate(scenario: BenchmarkScenario) -> Run:
        config = scenario.make_config()
        targets = scenario.make_targets(config)
        began = time.perf_counter()
        builder = scheduler(
            config,
            targets,
            scenario.begin,
            scenario.end,
            **options,  # type: ignore[arg-type]
        )
        plan = builder.schedule()
        planning_seconds = time.perf_counter() - began
        ditl = DITL(config=config, plan=plan, begin=scenario.begin, end=scenario.end)
        ditl.step_size = builder.ctx.step_size
        _calculate(ditl)
        return Run(simulation=ditl, planning_seconds=planning_seconds)

    return Contender(
        name=name or f"planned:{scheduler.scheduler_name}", simulate=simulate
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
            config, scenario.make_targets(config), scenario.begin, scenario.end
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
