"""Compare scheduling modes on the same scenario.

A :class:`BenchmarkScenario` describes a mission configuration, a pool of
targets and the Targets of Opportunity that arrive during the run. Each
:class:`Contender` simulates the scenario in one scheduling mode:

* :func:`dispatch`: :class:`~conops.ditl.QueueDITL` picks each next target.
* :func:`planned`: a planner builds one plan for the whole run up front, and
  :class:`~conops.ditl.DITL` executes it. A plan built in advance cannot react
  to ToOs, so they are never observed.
* :func:`rolling`: :class:`~conops.ditl.RollingHorizonDITL` replans as the run
  goes, including rapid replans for ToOs.
* :func:`configured`: whatever the scenario configuration's ``scheduler``
  section selects, built with :func:`~conops.ditl.create_ditl`.

:func:`run_benchmark` runs every contender on fresh copies of the scenario and
returns a :class:`BenchmarkResult` for each, measured the same way from
telemetry. :func:`format_results` renders them as a table.
"""

import time
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime, timedelta

from pydantic import BaseModel, ConfigDict, Field

from ..common import ACSMode
from ..config import MissionConfig
from ..ditl import DITL, QueueDITL, RollingHorizonDITL, create_ditl
from ..ditl.ditl_mixin import DITLMixin
from ..ditl.factory import queue_targets
from ..ditl.plan_validator import PLAN_SCIENCE_OBSTYPES, entry_obstype
from ..schedulers import LongRangeAllocator, PriorityPlanner
from ..targets import Pointing
from .metrics import Collection, cadence_error, program_shares, visit_starts


class TOOSpec(BaseModel):
    """A Target of Opportunity that arrives during a benchmark run."""

    model_config = ConfigDict(frozen=True)

    obsid: int
    ra: float
    dec: float
    merit: float
    exptime: int
    name: str
    submit_time: float
    """Unix time the ToO is submitted."""
    deadline: float | None = None
    """Latest Unix time its science may begin."""


class BenchmarkScenario(BaseModel):
    """A mission, a target pool and the ToOs that arrive while it runs.

    ``make_config`` and ``make_targets`` are called afresh for every
    contender, because simulations change their configuration and targets.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    begin: datetime
    end: datetime
    make_config: Callable[[], MissionConfig]
    """Return a new configuration with its ephemeris set."""
    make_targets: Callable[[MissionConfig], list[Pointing]]
    """Return new targets bound to the given configuration."""
    toos: list[TOOSpec] = Field(default_factory=list)


class _Run(BaseModel):
    """What a contender's simulation produced."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    simulation: DITLMixin
    planning_seconds: float | None = None


class Contender(BaseModel):
    """A named way of scheduling a scenario."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    simulate: Callable[[BenchmarkScenario], _Run]


class BenchmarkResult(BaseModel):
    """How one contender did on one scenario, measured from telemetry."""

    contender: str
    scenario: str
    science_hours: float = 0.0
    """Science collected, to the second."""
    weighted_science: float = 0.0
    """Science seconds times the merit of the target observed."""
    slewing_hours: float = 0.0
    """Time in each ACS mode, sampled at simulation steps. A step in which a
    slew ends and collection starts counts as slewing, so mode hours and
    ``science_hours`` can overlap by up to a step per observation."""
    idle_hours: float = 0.0
    pass_hours: float = 0.0
    observations: int = 0
    """Science entries in the plan that was executed."""
    too_response_seconds: dict[int, float | None] = Field(default_factory=dict)
    """Seconds from each ToO's submission to its first science, by obsid."""
    too_on_time: int = 0
    """ToOs whose science began by their deadline (or at all, without one)."""
    deadline_requests: int = 0
    """Requests, other than ToOs, with a deadline."""
    deadlines_met: int = 0
    """Of those, the ones whose whole exposure was collected."""
    program_share: dict[str, float] = Field(default_factory=dict)
    """Each program's fraction of the science collected."""
    cadence_error: float | None = None
    """Mean relative miss of the requested revisit interval, over revisited
    targets with a cadence; None if none was revisited."""
    cadence_revisited: int = 0
    """Targets with a cadence that were visited at least twice."""
    cadence_targets: int = 0
    """Targets with a cadence."""
    planning_seconds: float | None = None
    """Wall-clock time spent building plans, for planning contenders."""
    run_seconds: float = 0.0
    """Wall-clock time for the whole contender, planning included."""
    mismatches: int | None = None
    """Plan/execution mismatches found by the plan validator."""
    error: str | None = None
    """Why the contender failed, if it did."""


# ── Contenders ───────────────────────────────────────────────────────────


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

    def simulate(scenario: BenchmarkScenario) -> _Run:
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
        ditl.calc()
        return _Run(simulation=ditl)

    suffix = _allocation_suffix(allocation, allocation_reserve)
    return Contender(name=name or f"dispatch{suffix}", simulate=simulate)


def planned(
    planner: type[PriorityPlanner] = PriorityPlanner,
    name: str | None = None,
    **options: object,
) -> Contender:
    """One plan for the whole run, built up front and executed by DITL.

    Args:
        planner: Planner class to build the plan with.
        name: Contender name; defaults to the planner's.
        **options: Extra keyword arguments for the planner.
    """

    def simulate(scenario: BenchmarkScenario) -> _Run:
        config = scenario.make_config()
        targets = scenario.make_targets(config)
        began = time.perf_counter()
        builder = planner(
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
        ditl.calc()
        return _Run(simulation=ditl, planning_seconds=planning_seconds)

    return Contender(name=name or f"planned:{planner.planner_name}", simulate=simulate)


def rolling(
    planner: type[PriorityPlanner] = PriorityPlanner,
    name: str | None = None,
    *,
    horizon: timedelta = timedelta(days=1),
    replan_interval: timedelta = timedelta(hours=12),
    commit_lead_time: timedelta = timedelta(0),
    planner_options: Mapping[str, object] | None = None,
    allocation: timedelta | None = None,
    allocation_reserve: float | None = None,
) -> Contender:
    """Rolling-horizon replanning with ``planner``, reacting to ToOs.

    Args:
        planner: Planner class to build each plan with.
        name: Contender name; defaults to one naming the planner.
        horizon: How far ahead each plan looks.
        replan_interval: Time between scheduled replans.
        commit_lead_time: Time between building a plan and it taking effect.
        planner_options: Extra keyword arguments for the planner.
        allocation: Bin length of a long-range allocator steering each replan
            (see :class:`~conops.schedulers.LongRangeAllocator`); None for none.
        allocation_reserve: The allocator's reserve; its default if None.
    """

    def simulate(scenario: BenchmarkScenario) -> _Run:
        config = scenario.make_config()
        ditl = RollingHorizonDITL(
            config,
            scenario.make_targets(config),
            begin=scenario.begin,
            end=scenario.end,
            horizon=horizon,
            replan_interval=replan_interval,
            commit_lead_time=commit_lead_time,
            planner=planner,
            planner_options=planner_options,
            allocator=_allocator(config, scenario, allocation, allocation_reserve),
        )
        ditl.step_size = int(ditl.ephem.step_size)
        for too in scenario.toos:
            ditl.submit_too(**too.model_dump())
        ditl.calc()
        return _Run(
            simulation=ditl,
            planning_seconds=sum(r.planning_seconds for r in ditl.replans),
        )

    suffix = _allocation_suffix(allocation, allocation_reserve)
    return Contender(
        name=name or f"rolling:{planner.planner_name}{suffix}", simulate=simulate
    )


def configured(name: str = "configured") -> Contender:
    """Whatever the scenario's configuration selects in its ``scheduler`` section."""

    def simulate(scenario: BenchmarkScenario) -> _Run:
        config = scenario.make_config()
        began = time.perf_counter()
        ditl = create_ditl(
            config, scenario.make_targets(config), scenario.begin, scenario.end
        )
        planning: float | None = time.perf_counter() - began
        if isinstance(ditl, (QueueDITL, RollingHorizonDITL)):
            for too in scenario.toos:
                ditl.submit_too(**too.model_dump())
        ditl.calc()
        if isinstance(ditl, RollingHorizonDITL):
            planning = sum(r.planning_seconds for r in ditl.replans)
        elif isinstance(ditl, QueueDITL):
            planning = None
        return _Run(simulation=ditl, planning_seconds=planning)

    return Contender(name=name, simulate=simulate)


# ── Running and reporting ────────────────────────────────────────────────


def run_benchmark(
    scenario: BenchmarkScenario, contenders: Sequence[Contender]
) -> list[BenchmarkResult]:
    """Run every contender on the scenario and measure each the same way.

    A contender that raises is reported with its error rather than stopping
    the benchmark.
    """
    results = []
    for contender in contenders:
        began = time.perf_counter()
        try:
            run = contender.simulate(scenario)
        except Exception as exc:  # report the failure and carry on
            results.append(
                BenchmarkResult(
                    contender=contender.name,
                    scenario=scenario.name,
                    run_seconds=time.perf_counter() - began,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        results.append(
            _measure(contender.name, scenario, run, time.perf_counter() - began)
        )
    return results


def _measure(
    name: str, scenario: BenchmarkScenario, run: _Run, run_seconds: float
) -> BenchmarkResult:
    """Measure a finished simulation from its telemetry and plan."""
    simulation = run.simulation
    targets = scenario.make_targets(simulation.config)
    merit = {int(t.obsid): float(t.fom) for t in targets}
    merit.update({too.obsid: too.merit for too in scenario.toos})
    step_hours = float(simulation.step_size) / 3600.0
    modes = [ACSMode(int(m)) for m in simulation.mode]

    records: list[Collection] = []
    collected: dict[int, float] = {}
    science = weighted = 0.0
    for record in simulation.telemetry.housekeeping:
        seconds = record.collection_seconds or 0.0
        if seconds <= 0 or record.obsid is None:
            continue
        obsid = int(record.obsid)
        science += seconds
        weighted += seconds * merit.get(obsid, 0.0)
        collected[obsid] = collected.get(obsid, 0.0) + seconds
        records.append((obsid, record.timestamp.timestamp(), seconds))
    starts = visit_starts(records, float(simulation.step_size))

    responses: dict[int, float | None] = {}
    on_time = 0
    for too in scenario.toos:
        first = starts[too.obsid][0] if too.obsid in starts else None
        submitted = max(too.submit_time, scenario.begin.timestamp())
        responses[too.obsid] = None if first is None else first - submitted
        if first is not None and (too.deadline is None or first <= too.deadline):
            on_time += 1

    # Snapshots only start by a deadline, so all the exposure collected was
    # started in time.
    due = [t for t in targets if t.deadline is not None]
    met = sum(
        1
        for t in due
        if collected.get(int(t.obsid), 0.0) >= float(t.exptime or t.ss_max) - 1.0
    )

    categories = simulation.config.observation_categories
    program = {obsid: categories.get_category(obsid).program_name for obsid in merit}
    cadence: dict[int, float] = {}
    for obsid in merit:
        interval = categories.get_category(obsid).cadence_seconds
        if interval is not None:
            cadence[obsid] = interval
    error, revisited = cadence_error(starts, cadence)

    validate = getattr(simulation, "validate_plan_matches_execution", None)
    return BenchmarkResult(
        contender=name,
        scenario=scenario.name,
        science_hours=science / 3600.0,
        weighted_science=weighted,
        slewing_hours=modes.count(ACSMode.SLEWING) * step_hours,
        idle_hours=modes.count(ACSMode.IDLE) * step_hours,
        pass_hours=modes.count(ACSMode.PASS) * step_hours,
        observations=sum(
            1
            for entry in simulation.plan
            if entry_obstype(entry) in PLAN_SCIENCE_OBSTYPES
        ),
        too_response_seconds=responses,
        too_on_time=on_time,
        deadline_requests=len(due),
        deadlines_met=met,
        program_share=program_shares(collected, program),
        cadence_error=error,
        cadence_revisited=revisited,
        cadence_targets=len(cadence),
        planning_seconds=run.planning_seconds,
        run_seconds=run_seconds,
        mismatches=len(validate()) if validate is not None else None,
    )


def format_results(results: Sequence[BenchmarkResult]) -> str:
    """Render benchmark results as a plain-text table.

    Deadline, program share and cadence columns appear only when a result has
    them. A
    contender that failed shows its error's type in the table, and the full
    message below it.
    """
    show_programs = any(len(r.program_share) > 1 for r in results)
    show_deadlines = any(r.deadline_requests for r in results)
    show_cadence = any(r.cadence_targets for r in results)
    headers = [
        "contender",
        "science h",
        "weighted",
        "slew h",
        "idle h",
        "obs",
        "ToO on time",
        "ToO median",
    ]
    if show_deadlines:
        headers.append("deadlines met")
    if show_programs:
        headers.append("programs")
    if show_cadence:
        headers.append("cadence")
    headers += ["plan s", "run s", "mismatches"]

    rows: list[list[str]] = [headers]
    for r in results:
        if r.error is not None:
            kind = r.error.split(":", 1)[0]
            rows.append([r.contender, f"error: {kind}", *[""] * (len(headers) - 2)])
            continue
        responses = sorted(
            seconds
            for seconds in r.too_response_seconds.values()
            if seconds is not None
        )
        row = [
            r.contender,
            f"{r.science_hours:.2f}",
            f"{r.weighted_science:.3g}",
            f"{r.slewing_hours:.2f}",
            f"{r.idle_hours:.2f}",
            str(r.observations),
            f"{r.too_on_time}/{len(r.too_response_seconds)}"
            if r.too_response_seconds
            else "n/a",
            f"{responses[len(responses) // 2] / 60:.0f} min" if responses else "-",
        ]
        if show_deadlines:
            row.append(f"{r.deadlines_met}/{r.deadline_requests}")
        if show_programs:
            row.append(
                " ".join(
                    f"{name} {share:.0%}" for name, share in r.program_share.items()
                )
            )
        if show_cadence:
            row.append(
                "-"
                if r.cadence_error is None
                else f"{r.cadence_error:.2f} ({r.cadence_revisited}/{r.cadence_targets})"
            )
        row += [
            "" if r.planning_seconds is None else f"{r.planning_seconds:.1f}",
            f"{r.run_seconds:.1f}",
            "" if r.mismatches is None else str(r.mismatches),
        ]
        rows.append(row)
    widths = [max(len(row[i]) for row in rows) for i in range(len(headers))]
    lines = [
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in rows
    ]
    lines.insert(1, "  ".join("-" * width for width in widths))
    errors = [f"{r.contender}: {r.error}" for r in results if r.error is not None]
    if errors:
        lines += ["", "Errors:", *errors]
    return "\n".join(lines)
