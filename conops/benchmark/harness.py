"""The benchmark harness: scenarios, contenders, results and their measurement."""

import time
from collections.abc import Callable, Sequence
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from ..common import ACSMode
from ..config import MissionConfig
from ..ditl.ditl_mixin import DITLMixin
from ..ditl.plan_validator import PLAN_SCIENCE_OBSTYPES, entry_obstype
from ..targets import Pointing, ScheduledActivity
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
    activities: list[ScheduledActivity] = Field(default_factory=list)
    """Scheduled activities every contender executes, such as calibration
    blocks and a maintenance window."""


class Run(BaseModel):
    """What a contender's simulation produced."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    simulation: DITLMixin
    planning_seconds: float | None = None


class Contender(BaseModel):
    """A named way of scheduling a scenario."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str
    simulate: Callable[[BenchmarkScenario], Run]


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
    name: str, scenario: BenchmarkScenario, run: Run, run_seconds: float
) -> BenchmarkResult:
    """Measure a finished simulation from its telemetry and plan."""
    simulation = run.simulation
    targets = scenario.make_targets(simulation.config)
    merit = {int(t.obsid): float(t.merit) for t in targets}
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
    them. A contender that failed shows its error's type in the table, and
    the full message below it.
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
