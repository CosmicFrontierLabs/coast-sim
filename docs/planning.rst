Offline Planning
================

Overview
--------

COASTSim runs a scheduler in one of three ways:

* **Dispatch** (closed loop): :class:`~conops.ditl.queue_ditl.QueueDITL` asks a
  target queue for the next target each time the spacecraft is free. This is the
  default way to simulate a queue-scheduled mission.
* **Planning** (open loop): a planner builds a whole
  :class:`~conops.targets.Plan` up front, and :class:`~conops.ditl.ditl.DITL`
  executes it.
* **Rolling-horizon replanning** (closed loop):
  :class:`~conops.ditl.rolling_ditl.RollingHorizonDITL` follows a plan and rebuilds
  it from the spacecraft's actual state as time passes, as a ground-planned mission
  does. See `Rolling-horizon replanning`_.

A mission configuration's ``scheduler`` section selects the mode and planner, and
:func:`~conops.ditl.create_ditl` builds the matching simulation (see
:doc:`configuration`). The classes below can also be used directly.

:class:`~conops.schedulers.PriorityPlanner` is the planning engine. It reserves
ground contacts first, then takes requests in priority order and places each
snapshot in the earliest slot that still fits, without moving anything already
placed. It is fast and deterministic, and its plans are traceable to the priority
order. Because placements are never revisited, an early placement can block a
better packing.

Building and executing a plan
-----------------------------

.. code-block:: python

   from datetime import datetime, timedelta, timezone

   from rust_ephem import TLEEphemeris

   from conops import DITL, MissionConfig, PriorityPlanner
   from conops.targets import Pointing

   begin = datetime(2025, 11, 1, tzinfo=timezone.utc)
   end = begin + timedelta(days=1)
   config = MissionConfig.from_json_file("examples/example_config.json")
   config.constraint.ephem = TLEEphemeris(
       tle="examples/example.tle", begin=begin, end=end, step_size=60
   )

   targets = []
   for obsid, (ra, dec, merit) in enumerate([(105, 10, 90), (30, 40, 60)], 10000):
       target = Pointing(
           config=config, obsid=obsid, ra=ra, dec=dec, merit=merit, fom=merit,
           ss_min=300, ss_max=1200,
       )
       target.exptime = 2400  # split into snapshots of up to ss_max
       targets.append(target)

   planner = PriorityPlanner(config, targets, begin, end)
   plan = planner.schedule()
   print(f"{len(plan)} entries; unplaced: {[t.obsid for t in planner.unplaced]}")

   ditl = DITL(config=config, plan=plan, begin=begin, end=end)
   ditl.step_size = planner.ctx.step_size
   ditl.calc()
   assert ditl.validate_plan_matches_execution() == []

Execute the plan with the same configuration, ephemeris, horizon and step size it
was built with. The plan can also be saved and loaded (see :doc:`plan_serialization`)
before execution.

How a plan is built
-------------------

1. **Locked entries** (the ``locked`` argument) keep their collection windows and are
   placed first. Only the slews into them are recomputed. Use this to pin
   placements and re-plan around them.
2. **Ground passes** are predicted from the configured ground stations, with the
   configuration's random seed, so ``DITL`` predicts the same passes when it executes
   the plan. Each pass is reserved on the first tracking profile the spacecraft can
   reach safely. Pass ``include_passes=False`` to plan science only.
3. **Requests** are sorted by tier, then by merit value at the start of the horizon
   (base merit plus the urgency, cadence and completion-deficit terms; see
   :doc:`configuration`). Each request is split into snapshots of up to ``ss_max``
   seconds and never shorter than ``ss_min``, until its ``exptime`` is used or it no
   longer fits. A request whose deadline is close gains urgency and so is placed
   ahead of flexible requests with higher base merit. Cadence and completion deficit
   follow the plan as it is built (see below).
4. **Each snapshot** goes in the earliest slot where every check ``DITL`` and the plan
   validator will apply passes:

   * the slew starts on a simulation step, after the previous activity ends;
   * the slew path clears the ``SLEWING`` attitude-constraint scopes;
   * the target is visible when the slew starts (ACS will not start it otherwise);
   * the held attitude clears the ``SCIENCE`` scopes for the whole observation,
     including setup, cleanup and handoff time; an observation that would run into
     a constraint is shortened, but never below ``ss_min``;
   * any wait at the previous attitude clears the ``IDLE`` scopes;
   * collection starts by the target's ``deadline``;
   * the next activity can still be reached on time, with its own slew and wait
     checked the same way.

Requests left with at least ``ss_min`` of exposure unplanned are listed in
``planner.unplaced``. Every placement and rejection is logged to ``planner.log``.
The input targets are not modified.

**Planning faster when many requests cannot fit.** Most planning time in a crowded
horizon goes to requests that end up unplaced: a snapshot fits a gap, but the activity
after it cannot then be reached, so each later start in the gap is tried in turn, each
with a full search for the slew onward. ``successor_retries`` limits how many later
starts are tried after that happens before the gap is given up. Unset (the default),
every start is tried and every fit is found; a small number such as 0 or 3 plans much
faster, but can miss a fit late in a gap. It applies to all three planners and to
rolling replans, and can be set in the configuration as
``scheduler.planner.successor_retries``.

Cadence and program shares in plans
-----------------------------------

The cadence and completion-deficit merit terms depend on what has already been
observed, so the planners evaluate them against the plan as well as the science
already collected, as :class:`~conops.ditl.QueueDITL` does against what it has
executed:

* **Completion deficit.** With ``completion_deficit_weight`` set and programs with a
  ``time_share``, each program's share counts the science its targets have collected
  (``collected_seconds``) plus what the plan has given them so far. The priority-first
  planner re-ranks the requests after every snapshot, so a program falling behind its
  share moves up; local search values each snapshot by the shares in the plan before
  it; CP-SAT uses the shares at the start of each chunk.
* **Cadence.** With ``cadence_weight`` set, a target in a category with
  ``cadence_seconds`` is visited no sooner than that long after its last visit,
  planned or collected (``last_collection_time``), so every visit has the full
  cadence value. The priority-first planner and local search place each visit at the
  earliest fit once it is due; CP-SAT keeps the same spacing between a target's
  candidates and discounts a visit by ``earliness_weight`` for each cadence it waits
  after falling due.

When :class:`~conops.ditl.RollingHorizonDITL` replans, observations it has already
committed but not finished collecting count too: their exposure towards their
program's share (``reserved_seconds``) and their collection end as the target's last
visit (``reserved_visits``), so a new plan does not revisit a target straight after a
committed visit.

Without these weights, plans are built exactly as before.

Limitations
-----------

* **No battery model.** The planner does not schedule charging, and ``DITL`` does not
  charge on its own while executing a plan.
* **Urgency at the start of the horizon.** Urgency is evaluated once, when the plan is
  built; deadlines are kept as constraints and, in local search and CP-SAT, by the
  earliness discount.
* **Cadence as a minimum spacing.** A visit before its cadence is due is never
  planned, though dispatch may take one when nothing else is worth more. Spreading
  many visits through the day also leaves the priority-first planner more, smaller
  gaps to search, which lengthens planning.
* **Greedy.** Placements are never revisited. If a flexible, high-merit request is
  placed first, it can take the only slot a lower-merit request with a short window
  could use, and that request goes unplaced. Urgency avoids this when it raises the
  short-window request above the flexible one, so it is placed first.

Improving plans by local search
-------------------------------

Placing requests in priority order leaves gaps: a high-priority snapshot placed
early can strand time around it that nothing else fits into.
:class:`~conops.schedulers.LocalSearchPlanner` builds the priority-first plan and
then improves it.

The search works on the *order* of the plan's science snapshots. A decoder turns an
order into a timeline by placing each snapshot at the earliest time it fits after the
one before, with every check described above, so every plan it considers executes as
planned. Each snapshot starts out tied to the time and length the priority-first plan
gave it, so the starting point is exactly that plan. The search then tries:

* **inserting** a snapshot of a request that still has exposure to place;
* **releasing** a snapshot from its earlier time and length, letting it move earlier
  and run to its full ``ss_max``;
* **removing** a snapshot;
* **swapping** two nearby snapshots, or **moving** one a few positions.

A change is kept when it is no worse than the current plan, or than the plan of
``history_length`` steps ago (late-acceptance hill climbing), which lets the search
cross plateaus. The objective is merit-weighted science time, compared tier by tier
from the highest, with each snapshot's completion deficit taken from the program
shares in the plan before it. A snapshot of a request with a deadline is worth less
the later it starts, so ToOs and other time-critical requests are not pushed later than they need
to be. The best plan found is returned, and it is never worse than the priority-first
plan.

.. code-block:: python

   from conops import LocalSearchPlanner

   planner = LocalSearchPlanner(config, targets, begin, end, time_limit=30.0)
   plan = planner.schedule()
   print(planner.initial_score, "->", planner.score, f"after {planner.iterations} changes")

``LocalSearchPlanner`` takes the same arguments as ``PriorityPlanner``, plus:

* ``time_limit``: seconds of search after the priority-first plan is built
  (default 10);
* ``max_iterations``: changes to try at most. A run limited only by time depends on
  machine speed; set ``max_iterations`` (and ``seed``) for reproducible plans;
* ``seed``: random seed, defaulting to the configuration's;
* ``neighborhood``: how many positions apart a swap or move can be;
* ``history_length``: length of the late-acceptance history;
* ``earliness_weight``: fraction of a deadline request's value lost if it starts at
  its deadline rather than at the start of the horizon (default 0.5). It must be at
  most 1, so a late snapshot is always worth more than none.

Ground passes and locked entries stay where the priority-first plan put them.

**Limitations.** Only requests with a deadline are rewarded for starting early. Each
change is decoded from the point it touches, so a search of a few
thousand changes takes tens of seconds for a day with a few hundred targets.

Optimizing plans with CP-SAT
----------------------------

:class:`~conops.schedulers.CpSatPlanner` builds plans with the CP-SAT constraint solver
from Google's OR-Tools (an optional dependency; install ``coast-sim[cpsat]``). It
optimizes the same objective as local search, but chooses which snapshots to observe
and in what order by solving a constraint model rather than by trying changes one at
a time.

The horizon is solved in consecutive ``chunk`` lengths (three hours by default). For
each chunk, a candidate is one snapshot of a request in one of its visibility windows
(as many per window as the exposure needs and the window holds), with an optional
arrival time and a collection length between its ``ss_min`` and ``ss_max``. CP-SAT
chooses candidates and their order:

* a circuit through the chosen candidates fixes the order;
* each slew starts on the first simulation step after the task before it ends, and
  once its target is visible, as ACS requires; each snapshot finishes within its
  window;
* ground passes are fixed tasks that snapshots must leave room to slew to;
* a request's snapshots never add up to more than its remaining exposure;
* a cadence target's visits are at least its cadence apart, and after its last one;
* the objective is merit-weighted science, with the completion deficit at the chunk's
  start and the earliness discount for requests with a deadline and for cadence
  visits.

The priority-first plan's snapshots in each chunk are the solver's starting hint,
adjusted where the model's approximate slews need it, so the solver starts from a
feasible solution. Each chunk finishes before the priority-first plan's next slew and
leaves the exposure that plan collects later to it, so the rest of that plan can
always follow. Because the model approximates slews (each target's roll is chosen
before solving), each chunk's order is decoded with the planner's exact checks. It is
kept if, followed by the rest of the priority-first plan, it scores at least as well
as that plan does from the same point; otherwise, or if the solver found no solution
in its time, the priority-first plan's snapshots for the chunk are kept. The next
chunk is solved from where the kept snapshots leave the spacecraft. Every plan
therefore executes as planned and is never worse than the priority-first plan;
``solver_chunks_used`` shows which chunks the solver improved.

.. code-block:: python

   from conops import CpSatPlanner

   planner = CpSatPlanner(config, targets, begin, end, solver_time_limit=20.0)
   plan = planner.schedule()
   print(planner.solver_statuses, planner.initial_score, "->", planner.score)

``CpSatPlanner`` takes the arguments of ``LocalSearchPlanner``, plus:

* ``solver_time_limit``: seconds CP-SAT may search, shared across the chunks
  (default 20);
* ``chunk``: length of each chunk (default three hours);
* ``workers``: CP-SAT search workers (default 8). Use 1, with ``seed``, for runs that
  repeat exactly;
* ``max_candidates``: candidates per chunk at most (default 120), those of the
  priority-first plan first, then in priority order.

``time_limit`` defaults to 0 here; set it to spend that many seconds improving the
solver's plan by local search afterwards.

**Limitations.** Chunks are solved one after another. A chunk cannot borrow time or
exposure the priority-first plan uses later, so improvements that would need to
rearrange several chunks at once are left to local search. The model's slews are
approximate; the exact decoding corrects them but can drop or shorten a snapshot the
solver chose. With very short limits (well under a second per chunk) the solver may
find nothing, and the plan is the priority-first one.

The scheduling context
----------------------

:class:`~conops.schedulers.SchedulingContext` answers the questions a planner asks,
using the models ``DITL`` executes with: slew durations and paths from the real
:class:`~conops.simulation.Slew`, scoped attitude-constraint checks as ``DITL``
records them, the attitude a fresh simulation idles at before its first command,
and the ground passes in the horizon. Future planners are built on the same
context so that their plans also execute as planned.

Rolling-horizon replanning
--------------------------

:class:`~conops.ditl.rolling_ditl.RollingHorizonDITL` simulates a mission that is
planned on the ground and replanned periodically. It runs like
:class:`~conops.ditl.ditl.DITL`, but builds its own plan with ``PriorityPlanner`` and
rebuilds it as the simulation runs:

1. **Initial plan.** At the start, a plan is built for the next ``horizon``.
2. **Scheduled replans.** Every ``replan_interval``, the plan is rebuilt. Everything
   already commanded, or due to start within ``commit_lead_time`` (the time a new
   plan takes to reach the spacecraft), is kept. The rest is planned again from
   where the committed activities leave the spacecraft, using each target's
   remaining exposure.
3. **Targets of Opportunity.** A ToO submitted with
   :meth:`~conops.ditl.rolling_ditl.RollingHorizonDITL.submit_too` joins the target
   pool when it becomes active and goes into the next scheduled plan. If its
   ``deadline`` falls before that plan could start collecting it (its lead time plus a
   worst-case slew and the setup time), a **rapid replan** runs at once. In a rapid
   replan the observation running at the commit cutoff, started or not, is cut short
   there when ``allow_interrupts`` is set and the ToO's tier and value, evaluated now,
   beat the tier and value frozen onto that observation.

.. code-block:: python

   from datetime import timedelta

   from conops import RollingHorizonDITL

   ditl = RollingHorizonDITL(
       config,
       targets,
       begin=begin,
       end=end,
       horizon=timedelta(days=1),
       replan_interval=timedelta(hours=12),
       commit_lead_time=timedelta(hours=1),
   )
   ditl.submit_too(
       obsid=1000001, ra=105.0, dec=10.0, merit=500, exptime=900, name="GRB",
       submit_time=begin + timedelta(hours=5),
       deadline=begin + timedelta(hours=6),
   )
   ditl.calc()

   for replan in ditl.replans:
       print(replan.reason.value, replan.kept, replan.dropped, replan.added)
   print(ditl.too_response_times())       # seconds to first science, by obsid
   assert ditl.validate_plan_matches_execution() == []

Pass ``planner=LocalSearchPlanner`` or ``planner=CpSatPlanner`` (and, for example,
``planner_options={"time_limit": 10.0}`` or ``{"solver_time_limit": 10.0}``) to build
each plan by local search or with CP-SAT.

**Results**

* ``ditl.replans`` lists each replan as a
  :class:`~conops.ditl.rolling_ditl.ReplanRecord`: when and why it ran, the commit
  cutoff, how many entries were kept, dropped and added, how many targets did not
  fit, the planning time, and any ToO that triggered it or observation it
  interrupted.
* ``ditl.too_response_times()`` gives the seconds from each ToO's submission to its
  first science, or None if it was never observed.
* ``ditl.plan`` is the plan that was actually followed: the committed parts of
  every plan. ``validate_plan_matches_execution()`` checks the execution against it.
* Collected science is credited to the targets, so their remaining exposure
  (``exptime``) and ``collected_seconds`` reflect the run, as with a target queue.

Comparing a run with :class:`~conops.ditl.queue_ditl.QueueDITL` dispatch on the same
targets shows what planning ahead gains or costs in science time and ToO response.
A ``RollingHorizonDITL`` instance runs once; create a new one for each simulation.

Comparing scheduling modes
--------------------------

:mod:`conops.benchmark` runs the same scenario through each scheduling mode and
measures them the same way, from telemetry. A
:class:`~conops.benchmark.BenchmarkScenario` provides factories for a fresh
configuration and target pool, since simulations change both, and lists the ToOs
that arrive during the run. The contenders are:

* :func:`~conops.benchmark.dispatch`: ``QueueDITL`` picks each next target;
* :func:`~conops.benchmark.planned`: one plan built up front by a planner and executed
  by ``DITL``. A plan built in advance cannot react to ToOs;
* :func:`~conops.benchmark.rolling`: ``RollingHorizonDITL`` with a planner and
  replanning settings.

.. code-block:: python

   from conops.benchmark import (
       dispatch, format_results, planned, rolling, run_benchmark,
   )
   from conops.schedulers import LocalSearchPlanner, PriorityPlanner

   results = run_benchmark(scenario, [
       dispatch(),
       planned(PriorityPlanner),
       planned(LocalSearchPlanner, time_limit=10.0),
       rolling(LocalSearchPlanner, planner_options={"time_limit": 10.0}),
   ])
   print(format_results(results))

Each :class:`~conops.benchmark.BenchmarkResult` reports, measured the same way for every
contender:

* science time and merit-weighted science;
* time slewing, idle and in contact, and the number of observations;
* each ToO's response time, and how many started by their deadline;
* each program's share of the science collected;
* for targets with a cadence, how far the mean gap between visits missed the requested
  interval, and how many targets were revisited at all;
* planning and run time, and plan/execution mismatches.

A contender that fails is reported with its error instead of stopping the benchmark.

Standard scenarios
^^^^^^^^^^^^^^^^^^

:mod:`conops.benchmark.scenarios` provides five scenarios, each stressing a different
part of scheduling. All use one low-Earth orbit from a TLE, Sun and Earth-limb
avoidance, the default ground stations, a battery that never limits operations, and
seeded random targets:

* ``baseline``: a day of 200 targets and one ToO with a one-hour deadline;
* ``too-heavy``: eight ToOs across the day, half urgent (tier 1, deadlines of 30
  minutes to 2 hours) and half routine (tier 0, 4 to 12 hours);
* ``oversubscribed``: 600 targets requesting far more time than the day holds, a third
  of them with deadlines;
* ``cadence``: twelve monitoring targets wanting a visit every four hours, and two
  survey programs with equal time shares but different merit, with the cadence and
  completion-deficit merit terms switched on;
* ``multi-day``: three days, 400 targets and a ToO a day, for replanning over a long
  run.

``scripts/benchmark_schedulers.py`` runs the standard contenders (adding the CP-SAT
planner, planned and rolling, when OR-Tools is installed) on one scenario or all of
them, with rolling replans every six hours (or four times in a shorter run) and a
30-minute commit lead time::

   uv run python scripts/benchmark_schedulers.py --scenario all --time-limit 10
