Offline Planning
================

Overview
--------

COASTSim runs a scheduler in one of three ways:

* **Dispatch** (closed loop): :class:`~conops.ditl.queue_ditl.QueueDITL` asks a
  target queue for the next target each time the spacecraft is free. This is the
  default way to simulate a queue-scheduled mission.
* **Planning** (open loop): a scheduler builds a whole
  :class:`~conops.targets.Plan` up front, and :class:`~conops.ditl.ditl.DITL`
  executes it.
* **Rolling-horizon replanning** (closed loop):
  :class:`~conops.ditl.rolling_ditl.RollingHorizonDITL` follows a plan and rebuilds
  it from the spacecraft's actual state as time passes, as a ground-planned mission
  does. See `Rolling-horizon replanning`_.

Over runs of days to months, a long-range allocator can decide on which day each
request is observed, and dispatch or rolling replanning follow it (see
`Long-range allocation`_).

A mission configuration's ``scheduler`` section selects the mode and scheduler, and
:func:`~conops.ditl.create_ditl` builds the matching simulation (see
:doc:`configuration`). The classes below can also be used directly.

:class:`~conops.schedulers.PriorityScheduler` is the planning engine. It reserves
ground contacts first, then takes requests in priority order and places each
snapshot in the earliest slot that still fits, without moving anything already
placed. It is fast and deterministic, and its plans are traceable to the priority
order. Because placements are never revisited, an early placement can block a
better packing. :class:`~conops.schedulers.LocalSearchScheduler` starts from its plan
and improves it (see `Improving plans by local search`_), and
:class:`~conops.schedulers.CpSatScheduler` optimizes it with a constraint solver (see
`Optimizing plans with CP-SAT`_).

Building and executing a plan
-----------------------------

.. code-block:: python

   from datetime import datetime, timedelta, timezone

   from rust_ephem import TLEEphemeris

   from conops import DITL, MissionConfig, PriorityScheduler
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
           config=config, obsid=obsid, ra=ra, dec=dec, merit=merit,
           ss_min=300, ss_max=1200,
       )
       target.exptime = 2400  # split into snapshots of up to ss_max
       targets.append(target)

   scheduler = PriorityScheduler(config, targets, begin, end)
   plan = scheduler.schedule()
   print(f"{len(plan)} entries; unplaced: {[t.obsid for t in scheduler.unplaced]}")

   ditl = DITL(config=config, plan=plan, begin=begin, end=end)
   ditl.step_size = scheduler.ctx.step_size
   ditl.calc()
   assert ditl.validate_plan_matches_execution() == []

A target's ``earliest_start`` and ``deadline`` (Unix seconds, a datetime or an
ISO-8601 string) limit when its science collection may begin.

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
   * the target is visible when the slew starts (ACS will not start it otherwise),
     or, with ``slew_before_visible`` set in the ACS configuration, when the slew
     arrives;
   * the held attitude clears the ``SCIENCE`` scopes for the whole observation,
     including setup, cleanup and handoff time; an observation that would run into
     a constraint is shortened, but never below ``ss_min``;
   * any wait at the previous attitude clears the ``IDLE`` scopes;
   * collection starts no earlier than the target's ``earliest_start`` and by its
     ``deadline``;
   * the next activity can still be reached on time, with its own slew and wait
     checked the same way.

   **Moving snapshots later to make room.** A request whose snapshots straddle a time
   its target is hidden gets its next snapshot the moment the target rises. By
   default ACS starts a slew only once its target is visible, so the only way to be
   on the target then is to stay on it while it is hidden: anything else placed in
   that time would get there late. So if a snapshot would fit a gap but the science
   snapshot after it could then not be reached in time, that snapshot is moved later
   to make room, along with up to three more packed behind it, as long as the
   activity after them can still be reached. Moving a snapshot can cut it short
   where its target's visibility ends; that is allowed only if no snapshot cut short
   is in a higher tier than the new one and the new snapshot's value-weighted seconds
   exceed those lost. Exposure cut short is left for the request to plan elsewhere.
   Ground passes, locked entries and snapshots of cadence targets are never moved.

   **Slewing before a target is visible.** Setting ``slew_before_visible`` in the
   ACS configuration lets a slew start early enough to arrive as its target rises,
   rather than waiting for it to rise. That saves a slew's time each time a target
   rises, and the snapshot after a gap no longer needs to move to make room.

Requests left with at least ``ss_min`` of exposure unplanned are listed in
``scheduler.unplaced``. Every placement and rejection is logged to ``scheduler.log``.
The input targets are not modified.

**Planning faster when many requests cannot fit.** Most planning time in a crowded
horizon goes to requests that end up unplaced: a snapshot fits a gap, but the activity
after it cannot then be reached, so each later start in the gap is tried in turn, each
with a full search for the slew onward. ``successor_retries`` limits how many later
starts are tried after that happens before the gap is given up (3 by default). This
plans much faster in a crowded horizon, but can miss a fit late in a gap; set it to
None to try every start. Rolling replans use the default.

Cadence and program shares in plans
-----------------------------------

The cadence and completion-deficit merit terms depend on what has already been
observed, so the scheduler evaluates them against the plan as well as the science
already collected, as :class:`~conops.ditl.QueueDITL` does against what it has
executed:

* **Completion deficit.** With ``completion_deficit_weight`` set and programs with a
  ``time_share``, each program's share counts the science its targets have collected
  (``collected_seconds``) plus what the plan has given them so far. The scheduler
  re-ranks the requests after every snapshot, so a program falling behind its share
  moves up; local search values each snapshot by the shares in the plan before it;
  CP-SAT uses the shares at the start of each chunk.
* **Cadence.** With ``cadence_weight`` set, a target in a category with
  ``cadence_seconds`` is visited no sooner than that long after its last visit,
  planned or collected (``last_collection_time``), so every visit has the full
  cadence value. The priority-first scheduler and local search place each visit at the
  earliest fit once it is due; CP-SAT keeps the same spacing between a target's
  candidates and discounts a visit by ``earliness_weight`` for each cadence it waits
  after falling due, a cadence after the visit before it, up to ``earliness_weight``
  in all.

When :class:`~conops.ditl.RollingHorizonDITL` replans, observations it has already
committed but not finished collecting count too: their exposure towards their
program's share (``reserved_seconds``) and their collection end as the target's last
visit (``reserved_visits``), so a new plan does not revisit a target straight after a
committed visit.

Without these weights, plans are built exactly as before.

Limitations
-----------

* **No battery model.** The scheduler does not schedule charging, and ``DITL`` does not
  charge on its own while executing a plan.
* **Urgency at the start of the horizon.** Urgency is evaluated once, when the plan is
  built; deadlines are kept as constraints and, in local search and CP-SAT, by the
  earliness discount.
* **Cadence as a minimum spacing.** A visit before its cadence is due is never
  planned, though dispatch may take one when nothing else is worth more. Spreading
  many visits through the day also leaves the scheduler more, smaller gaps to search,
  which lengthens planning.
* **Greedy.** Apart from moving a snapshot later to make room before it, placements
  are never revisited. If a flexible, high-merit request is placed first, it can take
  the only slot a lower-merit request with a short window could use, and that request
  goes unplaced. Urgency avoids this when it raises the short-window request above
  the flexible one, so it is placed first; local search can also recover it (see
  below).

Improving plans by local search
-------------------------------

Placing requests in priority order leaves gaps: a high-priority snapshot placed
early can strand time around it that nothing else fits into.
:class:`~conops.schedulers.LocalSearchScheduler` builds the priority-first plan and
then improves it.

The search works on the *order* of the plan's science snapshots. A decoder turns an
order into a timeline by placing each snapshot at the earliest time it fits after the
one before, with every check described above, so every plan it considers executes as
planned. Each snapshot starts out tied to the time and length the priority-first plan
gave it, so the starting point is that plan. The search then tries:

* **inserting** a snapshot of a request that still has exposure to place;
* **replacing** a snapshot with one of another request that still has exposure to
  place, which frees time that only a different target can use, such as the time the
  spacecraft would otherwise spend waiting on a hidden target until it rises;
* **releasing** a snapshot from its earlier time and length, letting it move earlier
  and run to its full ``ss_max``;
* **removing** a snapshot;
* **swapping** two nearby snapshots, or **moving** one a few positions.

A change is kept when it is no worse than the current plan, or than the plan of
``history_length`` steps ago (late-acceptance hill climbing), which lets the search
cross plateaus. The objective is merit-weighted science time, compared tier by tier
from the highest, with each snapshot's completion deficit taken from the program
shares in the plan before it. A snapshot of a request with a deadline is worth less
the later it starts, so ToOs and other time-critical requests are not pushed later
than they need to be. The best plan found is returned, and it is never worse than the
priority-first plan.

Cadence visits keep their spacing, from each other and from locked visits, and each
plan entry's ``merit`` is the value its snapshot has where it is placed, as in the
priority-first scheduler.

.. code-block:: python

   from conops import LocalSearchScheduler

   scheduler = LocalSearchScheduler(config, targets, begin, end, time_limit=30.0)
   plan = scheduler.schedule()
   print(scheduler.initial_score, "->", scheduler.score, f"after {scheduler.iterations} changes")

``LocalSearchScheduler`` takes the same arguments as ``PriorityScheduler``, plus:

* ``time_limit``: seconds of search after the priority-first plan is built
  (default 10);
* ``max_iterations``: changes to try at most. A run limited only by time depends on
  machine speed; set ``max_iterations`` (and ``seed``) for reproducible plans;
* ``seed``: random seed, defaulting to the configuration's;
* ``neighborhood``: how many positions apart a swap or move can be (default 3);
* ``history_length``: length of the late-acceptance history (default 50);
* ``earliness_weight``: fraction of a deadline request's value lost if it starts at
  its deadline rather than at the start of the horizon, or at its earliest start if
  that is later (default 0.5). It must be at most 1, so a late snapshot is always
  worth more than none.

Ground passes and locked entries stay where the priority-first plan put them.

**Limitations.** Only requests with a deadline are rewarded for starting early. Each
change is decoded from the point it touches, so a search of a few thousand changes
takes tens of seconds for a day with a few hundred targets.

The notebook ``examples/Local_Search_Scheduler_Example.ipynb`` compares the two schedulers
on a one-day plan, as the number of changes grows, and in rolling-horizon runs.

Optimizing plans with CP-SAT
----------------------------

:class:`~conops.schedulers.CpSatScheduler` builds plans with the CP-SAT constraint solver
from Google's OR-Tools, which is an optional dependency (see :doc:`installation`). It
optimizes the same objective as local search, but chooses which snapshots to observe
and in what order by solving a constraint model rather than by trying changes one at
a time.

The horizon is solved in consecutive ``chunk`` lengths (three hours by default). A
chunk holds the snapshots whose slews start in it; a snapshot may finish in the next.
For each chunk, a candidate is one snapshot of a request in one of its visibility
windows (as many per window as the exposure needs and the window holds), with an
optional arrival time and a collection length between its ``ss_min`` and ``ss_max``.
CP-SAT chooses candidates and their order:

* a circuit through the chosen candidates fixes the order;
* each slew starts on the first simulation step after the task before it ends, and
  once its target is visible; each snapshot finishes within its window;
* collection starts no earlier than the request's ``earliest_start`` and by its
  ``deadline``, though its slew may start before the earliest start;
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
before solving), each chunk's order is decoded with the scheduler's exact checks, as in
local search. It is kept if, followed by the rest of the priority-first plan, it
scores at least as well as that plan does from the same point; otherwise, or if the
solver found no solution in its time, the priority-first plan's snapshots for the
chunk are kept. The next chunk is solved from where the kept snapshots leave the
spacecraft. Every plan therefore executes as planned and is never worse than the
priority-first plan; ``solver_chunks_used`` shows which chunks the solver improved.

The exact decoding also applies the rules the model leaves out: visits stay a cadence
clear of locked visits of the same target, ``slew_before_visible`` lets a slew start
before its target rises, and each plan entry's ``merit`` is the value its snapshot has
where it is placed.

.. code-block:: python

   from conops import CpSatScheduler

   scheduler = CpSatScheduler(config, targets, begin, end, solver_time_limit=20.0)
   plan = scheduler.schedule()
   print(scheduler.solver_statuses, scheduler.initial_score, "->", scheduler.score)

``CpSatScheduler`` takes the arguments of ``LocalSearchScheduler``, plus:

* ``solver_time_limit``: seconds CP-SAT may search, shared across the chunks
  (default 20);
* ``chunk``: length of each chunk (default three hours);
* ``workers``: CP-SAT search workers (default 8). Use 1, with ``seed``, for runs that
  repeat exactly;
* ``max_candidates``: candidates per chunk at most (default 120), those of the
  priority-first plan first, then in priority order.

``time_limit`` defaults to 0 here; set it to spend that many seconds improving the
solver's plan by local search afterwards. The plan's metadata records
``"scheduler": "cp_sat"``, and ``solver_score`` and ``solver_seconds`` give the score
and time of the solver's plan before any local search.

**Limitations.** Chunks are solved one after another. A chunk cannot borrow time or
exposure the priority-first plan uses later, so improvements that would need to
rearrange several chunks at once are left to local search. The model's slews are
approximate; the exact decoding corrects them but can drop or shorten a snapshot the
solver chose. With very short limits (well under a second per chunk) the solver may
find nothing, and the plan is the priority-first one.

The notebook ``examples/CP_SAT_Scheduler_Example.ipynb`` compares CP-SAT with local
search on a one-day plan, chunk by chunk, as the planning time grows, and in
rolling-horizon runs.

The scheduling context
----------------------

:class:`~conops.schedulers.SchedulingContext` answers the questions a scheduler asks,
using the models ``DITL`` executes with: slew durations and paths from the real
:class:`~conops.simulation.Slew`, scoped attitude-constraint checks as ``DITL``
records them, the attitude a fresh simulation idles at before its first command,
and the ground passes in the horizon. Future schedulers are built on the same
context so that their plans also execute as planned.

Rolling-horizon replanning
--------------------------

:class:`~conops.ditl.rolling_ditl.RollingHorizonDITL` simulates a mission that is
planned on the ground and replanned periodically. It runs like
:class:`~conops.ditl.ditl.DITL`, but builds its own plan with ``PriorityScheduler`` and
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
   there when ``allow_interrupts`` is set, the ToO allows it (``interrupt``, an
   argument of ``submit_too``), the observation's category is ``interruptible``, and
   the ToO's tier and value, evaluated now, beat the tier and value frozen onto that
   observation. Otherwise the ToO can start only after the observation ends. Science it has already
   collected is kept, and its cleanup follows the cut. The cut is kept only if the new
   plan can collect the ToO. If several ToOs need a rapid replan at the same time, they
   are tried highest-ranked first, and the first that outranks the observation and can
   be placed interrupts it. A ToO that may not interrupt does not take the cut
   observation's place either, however it ranks: it is planned in after the
   interrupting ToO, or, if its deadline falls before that, in the next plan.

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

Pass ``scheduler=LocalSearchScheduler`` or ``scheduler=CpSatScheduler`` (and, for example,
``scheduler_options={"time_limit": 10.0}`` or ``{"solver_time_limit": 10.0}``) to build
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

Long-range allocation
---------------------

:class:`~conops.schedulers.LongRangeAllocator` is a long-range plan: it decides on
which day (or other period) each request should be observed, over a run of days to
months. The short-term scheduling, rolling replans or dispatch, then decides the order
within each day. This is the split between long-range and short-term planning that
observatories use.

It matters because the short-term scheduling looks only hours ahead. On its own, it
spends each day on the best targets available that day, so a target observable for
only part of the run, such as one the Sun will soon cover, or a request due beyond the
horizon, can miss its chance while time goes to targets that could have waited.

The allocator splits the run into bins (a day by default) and decides how much of
each request's remaining exposure to collect in which bin:

* each request can be allocated only the seconds its target is visible in a bin,
  between its earliest start and its deadline, from the same visibility windows the
  schedulers use;
* each bin is expected to hold ``efficiency`` (default 0.75) of its length in science,
  the rest going to slews, setup and ground passes;
* ``reserve`` (default 0.1) of that capacity is kept free for work that isn't known
  yet, such as ToOs. Planned requests are never allocated it. ToOs the simulation has
  received are allocated first and use the reserve before anything else, so a ToO
  that fits in the reserve doesn't push planned requests to later days;
* the allocation is solved as a mixed-integer program with HiGHS, which comes with
  OR-Tools (the optional ``cpsat`` extra; see :doc:`installation`), maximizing
  merit-weighted time. ToOs come first, then each tier in turn, each solved with the
  allocation of the ones above it fixed, so a tier never gives anything up for a lower
  one. Each tier starts from a greedy allocation of its requests into the room left,
  and if the solver finds no solution in time, that tier and the ones below it keep
  the greedy allocation. A request gets either nothing in a bin or at least its
  ``ss_min``, never more than its remaining exposure, and a request with a cadence no
  more than an even share per bin. Requests with a deadline are worth slightly more
  in earlier bins;
* the planned load is spread evenly over the bins as far as visibility allows, so
  every week keeps some slack. Each tier evens out the load it and the tiers above
  it place, so a lower tier, such as filler, fills what the tiers above leave
  light. This is a tiebreak: it never costs a second of science for less than a
  thousand seconds of better balance;
* re-allocating keeps the previous allocation unless changing it gains something:
  keeping a request's seconds where they were is worth an extra 1%. A ToO that fits
  in the reserve leaves every planned request where it was;
* ``solver="greedy"`` instead takes requests in order of tier and merit, each into
  the bins with the least competing demand. It is quicker but not optimal, is the
  MILP's starting point, and does not need OR-Tools.

The allocation says when each request should be observed, and short-term
scheduling follows it:

* a plan covering several bins plans each request's allocated seconds in the bin
  they are allocated to. Collection starting in the bin counts, up to the seconds
  allocated there; a bin the plan's end cuts short counts pro rata;
* how far ahead allocated time comes is the strictness (``allocation_strictness``,
  or ``strictness`` in the configuration). ``strict``, the default, puts it ahead of
  every unallocated request, with tiers in order among allocated time: the
  allocation has already weighed the tiers, so even a filler tier's allocated time
  is kept. ``tier`` puts it ahead of the unallocated requests in its own tier only.
  ``weighted`` keeps it in its tier, worth ``allocation_bonus`` more, so scheduling
  trades keeping to the allocation against merit;
* the rest of a request's exposure, and allocated seconds that don't fit their bin,
  are planned like unallocated requests, by tier, in the time the allocated ones
  leave. So with ``strict`` strictness a plan can work ahead, but not at the expense
  of what the allocation put in a bin;
* local search and CP-SAT score allocated time the same way, so they improve a plan
  without trading away what the allocation put in a bin;
* dispatch takes the requests allocated to the current bin first, with the same
  strictness.

A request the allocator hasn't seen, such as a ToO submitted after the last
allocation, is treated as allocated throughout. Rolling runs re-allocate at every
replan and dispatch at the start of every bin, using the exposure still remaining.
A rolling run can instead follow a fixed allocation made elsewhere, such as a weekly
long-range plan (``allocation=``); each replan then counts what has been collected or
committed in each bin against its seconds.

.. code-block:: python

   from datetime import timedelta
   from conops import LongRangeAllocator, RollingHorizonDITL

   allocator = LongRangeAllocator(config, begin, end, bin_length=timedelta(days=1))
   ditl = RollingHorizonDITL(config, targets, begin=begin, end=end, allocator=allocator)
   ditl.calc()
   print(ditl.allocation.seconds[10042])   # seconds allocated to each bin

Pass the same ``allocator`` to :class:`~conops.ditl.QueueDITL` for dispatch, or set
``scheduler.allocation`` in the configuration (see :doc:`configuration`). Schedulers
take the steering as ``allocated``, the seconds each request has in each span of
time, from :meth:`~conops.schedulers.Allocation.allocated`:

.. code-block:: python

   allocation = allocator.allocate(targets, begin.timestamp())
   scheduler = PriorityScheduler(
       config, targets, begin, end,
       allocated=allocation.allocated(
           [t.obsid for t in targets], begin.timestamp(), end.timestamp()
       ),
   )
