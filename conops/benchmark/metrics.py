"""Measures of a run's science, computed from per-step collection records."""

from collections.abc import Iterable, Mapping

Collection = tuple[int, float, float]
"""One step's science: obsid, step start (Unix seconds) and seconds collected."""


def visit_starts(
    records: Iterable[Collection], step_size: float
) -> dict[int, list[float]]:
    """Return when each visit to each target began.

    A visit is a run of consecutive steps collecting science for the same
    obsid; a step without collection for it ends the visit.
    """
    starts: dict[int, list[float]] = {}
    last_step: dict[int, float] = {}
    for obsid, utime, seconds in sorted(records, key=lambda r: r[1]):
        if seconds <= 0:
            continue
        previous = last_step.get(obsid)
        if previous is None or utime - previous > step_size * 1.5:
            starts.setdefault(obsid, []).append(utime)
        last_step[obsid] = utime
    return starts


def cadence_error(
    starts: Mapping[int, list[float]], cadence: Mapping[int, float]
) -> tuple[float | None, int]:
    """Return how far revisits missed their cadence, and how many targets were revisited.

    For each target with a cadence and at least two visits, the error is the
    mean gap between visits minus the cadence, as a fraction of the cadence,
    taken absolutely. The result is the mean over those targets, or None if
    none was revisited.
    """
    errors = []
    for obsid, interval in cadence.items():
        visits = sorted(starts.get(obsid, []))
        if len(visits) < 2 or interval <= 0:
            continue
        mean_gap = (visits[-1] - visits[0]) / (len(visits) - 1)
        errors.append(abs(mean_gap - interval) / interval)
    if not errors:
        return None, 0
    return sum(errors) / len(errors), len(errors)


def program_shares(
    collected: Mapping[int, float], program: Mapping[int, str]
) -> dict[str, float]:
    """Return each program's fraction of the science collected."""
    totals: dict[str, float] = {}
    for obsid, seconds in collected.items():
        if seconds <= 0:
            continue
        name = program.get(obsid)
        if name is not None:
            totals[name] = totals.get(name, 0.0) + seconds
    total = sum(totals.values())
    if total <= 0:
        return {}
    return {name: seconds / total for name, seconds in sorted(totals.items())}
