"""Compare scheduling modes on the same scenario.

A :class:`BenchmarkScenario` describes a mission configuration, a pool of
targets and the Targets of Opportunity that arrive during the run. Each
:class:`Contender` simulates the scenario in one scheduling mode:

* :func:`dispatch`: :class:`~conops.ditl.QueueDITL` picks each next target.
* :func:`planned`: a scheduler builds one plan for the whole run up front, and
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

from .contenders import configured, dispatch, planned, rolling
from .harness import (
    BenchmarkResult,
    BenchmarkScenario,
    Contender,
    Run,
    TOOSpec,
    format_results,
    run_benchmark,
)

__all__ = [
    "BenchmarkResult",
    "BenchmarkScenario",
    "Contender",
    "Run",
    "TOOSpec",
    "configured",
    "dispatch",
    "format_results",
    "planned",
    "rolling",
    "run_benchmark",
]
