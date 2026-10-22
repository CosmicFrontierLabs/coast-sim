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

A target is observed at the power-optimal instrument roll unless it asks for
another: ``roll_offset_deg`` is added to the power-optimal roll, and ``roll_deg`` is
the roll itself. Dispatch and the planners observe the target at that roll where the
constraints allow it, and the plan entry's ``roll`` records it. The ACS's
``max_roll_from_optimum_deg`` bounds either: an offset beyond it is a configuration
error, and a target whose absolute roll is beyond it at a time is not selected or
placed then, but may be later as the optimal roll moves.

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
   **Scheduled activities** (the ``activities`` argument, a list of
   :class:`~conops.targets.ScheduledActivity`) are reserved the same way: each is a
   fixed block from its start for its duration, in ``IDLE`` mode, holding the attitude
   the timeline has when it starts as reserved, so no science is placed across it and
   the slews around it are planned. A maintenance window, a calibration that needs the
   spacecraft still or an engineering block is an activity; ``DITL`` commands the
   plan's activity entries, and the plan validator checks that no science, contact or
   charge ran during one. Dispatch takes the same list: it ends the observation under
   way when an activity begins, marking it interrupted so it is revisited, holds the
   attitude it has in ``IDLE`` until the activity ends, and starts no observation that
   would run into the next activity.
3. **Requests** are sorted by merit value at the start of the horizon
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
   where its target's visibility ends; that is allowed only if the new snapshot's
   value-weighted seconds exceed those lost. Exposure cut short is left for the request to plan elsewhere.
   Ground passes and locked entries are never moved. A cadence target's visit moves
   only if it is the target's latest, and never earlier than it was, so its visits
   stay a cadence apart.

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

Planning within the recorder
----------------------------

Every scheduler keeps the onboard recorder below its planning limit,
``recorder.planning_limit`` as a fraction of ``capacity_gb`` (see
:doc:`data_management`). By default it is the recorder's ``red_threshold``, which
is where the red alert fires, so a plan stops just short of the red; set it lower
to keep margin, or to ``None`` to plan without regard to the recorder.

The recorder fills between ground contacts and empties during them, so its volume
is highest just before each contact and at the end of the horizon. The reserved
contacts cut the horizon into segments, and the plan is checked at every segment's
peak:

* each segment's peak is what was left after the contact before it, plus the
  housekeeping data generated in it, plus the science data of the snapshots placed
  in it: for each snapshot, each instrument's rate for its collection time plus its
  fixed per-observation amount, at the target's own rates where it sets them;
* each contact removes what its rate and length allow, and no more than is there,
  so what it cannot remove carries over to the segments after it;
* housekeeping is counted at the nominal rate throughout, and at the ``PASS`` rate
  during spacecraft tracking, since the plan does not know when the spacecraft will be in
  other modes. It is counted on the simulation's step grid, as the simulation
  generates it: each step counts whole at the rate of the mode it starts in;
* when a pass requires a spacecraft slew, ACS flies it in ``PASS`` mode too, adding the
  difference between the two rates for the steps it is under way. How long it is
  depends on where the spacecraft is before it, so the schedulers add it with the
  contact, as they add science, and recompute it whenever a snapshot placed before
  the contact changes the slew. CP-SAT counts each contact's slew as the plan it
  starts from has it, and a solution that lengthens one is checked when it is
  decoded;
* a contact downlinks for the part of its link inside the horizon, and what the
  step its link ends in generates after the link stays on the recorder, as the
  simulation stores it after the downlink.

All ground-station passes are booked by :class:`~conops.simulation.PassTimes`
(see :doc:`communications`). A pass that requires no spacecraft tracking does
not block science: its usable link spans downlink at the station's rate while
housekeeping stays at the nominal rate. A snapshot that does not fit before
such a pass is tried after it within the same timeline gap.
The schedulers and :class:`~conops.ditl.RollingHorizonDITL`
take the coverage from the configuration, or from ``coverage=`` where the caller
has it: a rolling run builds it once and gives it to every plan. Where the antenna is
re-pointed within contacts (see :doc:`communications`), a target whose category sets
``antenna_repoints: forbidden`` has its visibility windows cut at each repoint, so every
scheduler places its snapshots between repoints as it places them between constraint
violations, and dispatch starts such an observation only when its minimum snapshot
fits before the next repoint and ends it there.

The arithmetic is not particular to the recorder. Anything a spacecraft
accumulates between the events that remove it, and must keep within a limit
meanwhile, has the same shape: stored momentum builds up until a dump unloads it,
for instance. :class:`~conops.schedulers.ConsumableLedger` holds that arithmetic
(segments between removals, accrual, removals that take no more than is there,
and the check at every later peak), :class:`~conops.schedulers.RecorderLedger`
maps the recorder onto it, and :class:`~conops.schedulers.BatteryLedger` the
battery (see `Planning within the battery`_).

A snapshot is placed only where every peak from its segment onwards stays within
the limit. It is never shortened to fit: a snapshot that does not fit before a
contact is placed after one, or not at all, and the log says so. The priority-first
scheduler places higher-merit requests first, so when data runs short it is the
lowest-merit science that waits, whatever its data rate; local search then trades
snapshots in and out on merit within the same limit; and CP-SAT has the recorder in
its model, so the solver trades high-data science against low-data science directly
(see `Optimizing plans with CP-SAT`_).

:class:`~conops.ditl.RollingHorizonDITL` starts each plan from the recorder's
actual volume, so a contact that downlinked less than planned is made up for at
the next replan. A plan takes effect after the commit lead time, so it starts from
what the committed activities leave on the recorder by then: what they go on to
generate, less what a committed contact downlinks.

Observation sequences
^^^^^^^^^^^^^^^^^^^^^

Some observations are a fixed series of steps rather than one pointing held for a
time: a reference star, then the target, then the reference again, each for its own
duration and perhaps with its own instrument, with calibration steps that take time
but collect no science. A :class:`~conops.targets.ObservationSequence` on a
:class:`~conops.targets.Pointing` lists the steps (``SequenceStep``: a name, a
duration, an optional pointing and instrument, whether it collects science, and
whether it pays the setup budget), and ``visits`` says how many times the sequence is
wanted. A sequence target's snapshot is one visit's science, no more and no less, and
its exposure is its visits' science unless given.

Dispatch observes a visit step by step. The target is selected as any other, with the
visit's calibration, the slews between steps that point elsewhere and any step setup
counted in the time it needs; its first step is the observation admitted, paying the
payload's setup, and when a step's time is up the next starts at once, with a real
slew where it points elsewhere, no setup unless the step asks for it, and cleanup and
handoff after the last step only. Nothing is inserted inside a visit: a Target of
Opportunity waits for it to end. A step that cannot be observed, because its pointing
is not visible throughout, its attitude or slew is constrained, or a ground pass
falls in it, cuts the visit short, as does any other termination of a step; the cut
is recorded on the target, which is revisited for the rest from its first step.
Each step is its own plan entry with the target's obsid, its ``step`` name and, for
a calibration step, ``calibration`` set; the target is credited the science steps'
collection and, separately, the calibration time.

The planners place a visit as one block of steps. The priority-first scheduler
builds the chain from a candidate start: the first step's collection begins its
setup after the slew in, each later step follows a slew where it points elsewhere
and the setup budget where it asks, and cleanup and handoff come after the last;
every step must be visible and allowed throughout and every slew between steps
allowed, else the start is tried later. The visit fits before the next activity or
not at all, the slew to that activity leaves from the last step's attitude, and the
plan lists the steps as entries. The recorder ledger counts the science steps'
data and the battery ledger the slews and holds of every step. Local search moves a
visit as one block, and CP-SAT models it as one task whose length includes the
calibration, the slews between steps and any step setup, arriving at the first
step's attitude and leaving from the last's. A rolling run credits each executed
step to its target, science or calibration, and ``DITL`` commands the steps as it
commands any science entries.

Dispatch (:class:`~conops.ditl.queue_ditl.QueueDITL`) keeps within the recorder
too. When it selects the next target it works out how much the recorder can take
now: the most that keeps the volume under the planning limit at every contact's
start and at the run's end, carrying the recorder through the rest of the run, with
each contact removing what its station's rate downlinks for its length and never
below empty, so that a contact too short or too slow to empty the recorder leaves
the rest to the next and the housekeeping between has to fit as well. Housekeeping
is counted as the simulation generates it, for each whole
step at the rate of the mode the step starts in: at the waiting rate, the highest of
the nominal, ``IDLE``, ``CHARGING`` and ``SAA`` rates, since the dispatcher can spend
the time in any of those modes, and at the ``PASS`` rate for the slew into the
contact, which ACS flies in ``PASS`` mode, bounded as admission bounds it, where that
rate is higher. A candidate collects no more than that room holds of its science, at
its own rates, and of the housekeeping its own slew, the attitude-aware slew it will
fly, and science mode generate beyond the waiting rate; a mode that generates less
is not credited with the saving, since the spans are upper bounds. One that cannot
collect its minimum snapshot within it is passed over, and a snapshot shortened to
fit stays shortened through admission, which asks the room again with the slew it
flies. A Target of Opportunity admitted ahead of the observation under way, to start
once that observation's cleanup is over, is admitted against the volume projected to
then: the science the observation still collects and the housekeeping of the steps
between. Passes that require no spacecraft tracking have no ingress slew;
housekeeping stays at the waiting rate, and each usable link span downlinks at
its mean rate for its length.
The log says how many were passed over and how much room there was. A contact during the snapshot is not counted on. ``config.targets.data_volume_weight`` adds a price on top, in merit
points per gigabit of science data, so that high-data science gives way to low-data
science before the recorder is full rather than only once it is (see
:doc:`configuration`).

Planning within the battery
---------------------------

Every scheduler keeps the battery above its planning level, ``battery.planning_level``
as a fraction of full (see :doc:`configuration`). By default it is the
depth-of-discharge floor, ``1 - max_depth_of_discharge``, so no plan takes the
battery past the discharge it is allowed; set it higher to keep margin, or to
``None`` to plan without regard to the battery. ``DITL`` does not charge on its own
while executing a plan, so the plan carries its own charging.

The battery is the second consumable on :class:`~conops.schedulers.ConsumableLedger`,
as the *deficit*: the energy below full, in watt-hours. Unlike data on the recorder,
which only comes off during a contact, energy comes back whenever the panels are
lit, whatever the spacecraft is doing, so :class:`~conops.schedulers.BatteryLedger`
orders a segment's additions in time and walks them, step by step as the simulation
drains and charges the battery, every span at the attitude the spacecraft holds
through it:

* a slew draws ``SLEWING`` power and generates along its path; a science snapshot
  draws ``SCIENCE`` power and generates at its attitude and the power-optimal roll
  the scheduler gives it, which the simulation flies; a contact draws ``PASS`` power
  from the start of its slew, which ACS runs in ``PASS`` mode, and generates along
  its tracking profile, then ``SLEWING`` power until the spacecraft is at rest after
  it; a charge draws ``CHARGING`` power and generates at the charging attitude;
* idle time draws the idle power and generates at the attitude the activity before
  it leaves the spacecraft in;
* a panel on a finite solar-array drive is carried along as the simulation carries
  it, tracking the Sun in the modes its control lists and holding its angle in the
  rest, so each activity is counted at the drive angles the plan reaches it with,
  and a plan's charges take the charging attitude for those angles; a rolling
  replan starts from where the drive is, the committed activities flown as ACS
  flies them, slews and the idle time between and after them included, until
  the plan takes over;
* the deficit never falls below zero: a surplus before a draw is lost, as a full
  battery wastes what its panels make; and generation is not counted past
  ``planning_recharge_level``, so a plan never assumes the battery above it (a
  deficit already below that floor is left where it is until something draws);
* the segments start at each sunrise, and the plan is checked at every segment's
  peak.

The scheduler charges where it can. As it evaluates each placement, it puts a charge
into every idle sunlit stretch of the timeline long enough for one and its two
slews: slewing from the activity before to the charging attitude, found as emergency
charging finds it, holding it while the charging mode's constraints allow until the
stretch ends, and leaving the activity after it reachable from the charging attitude,
shortened or dropped where it is not. A snapshot is placed only where the deficit,
with those charges, stays within the limit from there to the end of the horizon.
Where it does not, the scheduler tries the snapshot later in the same gap, after the
charging time its idle sunlit start then leaves, before moving to a later gap; it is
never shortened to fit, and the log says when a request waits for charge. The
finished plan carries its charges as ``CHARGE`` entries, which ``DITL`` executes as it
does contacts and :meth:`~conops.ditl.DITL.validate_plan_matches_execution` checks.

The ledger is the simulation's own arithmetic at the attitudes the simulation flies,
so a plan it accepts keeps the battery above the level when executed. A spacecraft
whose panels generate more than observing draws, such as a side-mounted array rolled
towards the Sun, charges only in the sunlit idle time it has anyway and fills the
day with science. One whose panels only generate pointing at the Sun charges in
every idle sunlit stretch and refuses the science the battery cannot carry through
the eclipse after it; it is then the lowest-merit science that waits, as with the
recorder. Local search decodes its sequences through the same check;
:class:`~conops.ditl.RollingHorizonDITL` starts each plan from the battery the
committed activities, charges included, leave by the time the plan takes over; and
CP-SAT has the battery in its model (see `Optimizing plans with CP-SAT`_).

Limitations
-----------

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
cross plateaus. The objective is merit-weighted science time, with each snapshot's completion deficit taken from the program
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
* the recorder stays below its planning limit: each candidate's data (its rate for
  its collection, plus its fixed amount, at the observed target's rates, rounded up
  to whole kilobits so the model never counts less than the ledger) goes on the
  recorder in the segment, between contacts, where its collection starts; each
  segment's volume is what entered it after its contact, plus housekeeping and the
  science already planned there, plus the chunk's candidates; what a contact cannot
  remove carries over; and every segment from the chunk's to the end of the horizon
  is kept within the limit, so a candidate is traded against the room it takes from
  everything after it (see `Planning within the recorder`_);
* the objective is merit-weighted science, with the completion deficit at the chunk's
  start and the earliness discount for requests with a deadline and for cadence
  visits.

The battery is in the model too, where the battery limits the plan. The horizon
from the chunk's run to its end is cut into sunlit and eclipse runs. Each candidate
is in the run its arrival falls in, placed directly where its arrival domain spans
one run and by a boolean per run where it spans several, and adds to that run's
energy its net power, its mode's draw less what its panels generate at its
power-optimal roll, less the run's idle power, for its collection and overhead, plus
a typical slew in at the slewing power; what spills past the run's end is counted in
each run after it for the seconds it holds. A run's energy is then its idle power for
its whole length, charging in sunlight at the charging attitude and drawing in
eclipse, plus what the candidates and the already planned activities change; the
deficit carried into each run is bounded below by the previous run's, by zero, and
by the smaller of the recharge floor and the deficit entering the run, and every
run's end must stay within the limit. In an eclipse run everything draws,
so the run's end is its peak; in a sunlit run the deficit can rise and fall within
the run, and the model is linear, so the decoder, which walks the plan exactly, has
the last word: a chunk whose solution does not hold up keeps the priority-first
plan. The priority-first plan is hinted into the run and carry-over variables, so
the solver starts from a feasible solution.

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
   the value frozen onto the ToO's first snapshot in the replacement plan beats
   the value frozen onto that observation. Both include any `long-range allocation`_'s
   bonus. The replacement plan accounts for the horizon's share of each bin and
   the allocation already collected or committed before valuing the snapshot.
   Otherwise the ToO can start
   only after the observation ends. Science it has already
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
  merit-weighted time. ToOs come first, and their allocation is fixed while the rest
  are solved, so they give up nothing for planned requests. Each level starts from a
  greedy allocation of its requests into the room left, and if the solver finds no
  solution in time, that level and any after it keep the greedy allocation. A request gets either nothing in a bin or at least its
  ``ss_min``, never more than its remaining exposure, and a request with a cadence no
  more than an even share per bin. Requests with a deadline are worth slightly more
  in earlier bins;
* the planned load is spread evenly over the bins as far as visibility allows, so
  every week keeps some slack. This is a tiebreak: it never costs a second of science for less than a
  thousand seconds of better balance;
* re-allocating keeps the previous allocation unless changing it gains something:
  keeping a request's seconds where they were is worth an extra 1%. A ToO that fits
  in the reserve leaves every planned request where it was;
* the data the run generates fits what its ground contacts can downlink. Each bin
  generates housekeeping, at the ``PASS`` rate during its contacts, plus the science
  data of the requests allocated to it: each target's rate for its allocated seconds
  and its fixed amount per observation, once per bin. Its contacts, from the same
  booked ground-station passes the schedulers use, take off what they can, and what they cannot
  is carried on the recorder into the next bin, where it must stay within the
  recorder's planning limit (see `Planning within the recorder`_). The allocation
  starts from what is on the recorder, so a run that begins with a full recorder
  allocates its first day no more data than that day downlinks. High-data science
  then gives way to low-data science where the contacts cannot keep up, rather than
  being allocated and refused day by day. There is no budget without a planning
  limit, or when no request generates data, and nothing is predicted for it then;
  :attr:`~conops.schedulers.Allocation.downlink`, ``housekeeping`` and ``science``
  by bin and :meth:`~conops.schedulers.Allocation.recorder_volumes` report it. A
  mission with named stores (see :doc:`data_management`) is budgeted store by
  store: each bin generates on each store what is routed to it, each store's
  contacts take off only its reserved share of the link, and each store stays
  within its own planning limit, so a request is allocated no more than every
  store it generates data on can take.
  :attr:`~conops.schedulers.Allocation.recorders` reports each store's budget by
  name, and the totals above sum them;
* ``solver="greedy"`` instead takes requests in order of merit, each into
  the bins with the least competing demand, under the same bounds: nothing in a
  bin or at least a snapshot, and no more than an even share of a cadence
  request in any bin. It is quicker but not optimal, is the MILP's starting
  point, and does not need OR-Tools.

The allocation says when each request should be observed, and short-term
scheduling follows it:

* a plan covering several bins plans each request's allocated seconds in the bin
  they are allocated to. Collection starting in the bin counts, up to the seconds
  allocated there; a bin the plan's end cuts short counts pro rata;
* allocated time is worth ``allocation_bonus`` (``bonus`` in the configuration,
  default 0.5) more of its request's value, shown as the ``allocation`` merit term.
  That is the only way the allocation steers scheduling: allocated time competes
  with everything else on merit, so a larger bonus follows the allocation more
  closely, and ``0`` ignores the allocation altogether, so no request is held to
  its bin;
* the rest of a request's exposure, and allocated seconds that don't fit their bin,
  are planned like unallocated requests, by merit, so a plan can work ahead where
  that is worth more than the allocated time it displaces;
* local search and CP-SAT score allocated time the same way, so they improve a plan
  without trading away what the allocation put in a bin;
* every scheduler freezes the value with the bonus onto the entry. Rolling
  replanning compares the ToO's first planned snapshot with the observation it
  would interrupt, using those frozen values (see `Targets of Opportunity`_ above);
* dispatch adds the same bonus to the requests allocated to the current bin, for
  the seconds allocated there that the bin has not yet collected: a snapshot
  only partly covered gets that part of the bonus, and once a request has
  collected its allocation the rest competes at its merit.

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

Comparing scheduling modes
--------------------------

:mod:`conops.benchmark` runs the same scenario through each scheduling mode and
measures them the same way, from telemetry. A
:class:`~conops.benchmark.BenchmarkScenario` provides factories for a fresh
configuration and target pool, since simulations change both, and lists the ToOs
that arrive during the run. The contenders are:

* :func:`~conops.benchmark.dispatch`: ``QueueDITL`` picks each next target;
* :func:`~conops.benchmark.planned`: one plan built up front by a scheduler and
  executed by ``DITL``. A plan built in advance cannot react to ToOs;
* :func:`~conops.benchmark.rolling`: ``RollingHorizonDITL`` with a scheduler and
  replanning settings;
* :func:`~conops.benchmark.configured`: whatever the scenario configuration's
  ``scheduler`` section selects (see :doc:`configuration`), built with
  :func:`~conops.ditl.create_ditl`, so a configured run can be measured against
  the explicit modes.

Dispatch and rolling contenders can be steered by a long-range allocator
(``allocation``, the bin length, and ``allocation_reserve``), which their names show
as ``+alloc``.

.. code-block:: python

   from conops.benchmark import (
       configured, dispatch, format_results, planned, rolling, run_benchmark,
   )
   from conops.schedulers import LocalSearchScheduler, PriorityScheduler

   results = run_benchmark(scenario, [
       dispatch(),
       planned(PriorityScheduler),
       planned(LocalSearchScheduler, time_limit=10.0),
       rolling(LocalSearchScheduler, scheduler_options={"time_limit": 10.0}),
       configured(),
   ])
   print(format_results(results))

Each :class:`~conops.benchmark.BenchmarkResult` reports, measured the same way for every
contender:

* science time and merit-weighted science;
* time slewing, idle and in contact, and the number of observations;
* each ToO's response time, and how many started by their deadline;
* how many other requests with a deadline got their whole exposure;
* each program's share of the science collected;
* for targets with a cadence, how far the mean gap between visits missed the requested
  interval, and how many targets were revisited at all;
* planning and run time, and plan/execution mismatches.

A contender that fails is reported with its error instead of stopping the benchmark:
the table shows the error's type, and the full message is listed below it.

Standard scenarios
^^^^^^^^^^^^^^^^^^

:mod:`conops.benchmark.scenarios` provides six scenarios, each stressing a different
part of scheduling. All use one low-Earth orbit from a TLE, Sun and Earth-limb
avoidance, the default ground stations, a battery that never limits operations (the
spacecraft has no solar panels, so the battery's recharge alerts are turned off rather
than letting it drain into emergency charging), and seeded random targets:

* ``baseline``: a day of 200 targets and one ToO with a one-hour deadline;
* ``too-heavy``: eight ToOs across the day, half urgent (merit 1000, deadlines of 30
  minutes to 2 hours) and half routine (merit 500, 4 to 12 hours);
* ``oversubscribed``: 600 targets requesting far more time than the day holds, a third
  of them with deadlines;
* ``cadence``: twelve monitoring targets wanting a visit every four hours, and two
  survey programs with equal time shares but different merit, with the cadence and
  completion-deficit merit terms switched on;
* ``multi-day``: three days, 400 targets and a ToO a day, for replanning over a long
  run;
* ``long-range``: a week of 300 targets, half with deadlines spread across it, two
  programs by the Sun's avoidance zone ("Early" targets that the Sun covers as the
  week goes on and "Late" ones it uncovers), and 10 ToOs arriving through the week,
  alternately due within hours and within days, for long-range allocation.

``scripts/benchmark_schedulers.py`` runs the standard contenders on one scenario or all of
them, with rolling replans every six hours (or four times in a shorter run) and a
30-minute commit lead time. Runs of two days or more also compare dispatch and rolling
steered by an allocator with one-day bins::

   uv run python scripts/benchmark_schedulers.py --scenario all --time-limit 10
