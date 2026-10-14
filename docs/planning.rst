Offline Planning
================

Overview
--------

COASTSim runs a scheduler in one of two ways:

* **Dispatch** (closed loop): :class:`~conops.ditl.queue_ditl.QueueDITL` asks a
  target queue for the next target each time the spacecraft is free.
* **Planning** (open loop): a planner builds a whole
  :class:`~conops.targets.Plan` up front, and :class:`~conops.ditl.ditl.DITL`
  executes it.

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
   ahead of flexible requests with higher base merit.
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

Limitations
-----------

* **No battery model.** The planner does not schedule charging, and ``DITL`` does not
  charge on its own while executing a plan.
* **Static ordering.** Merit is evaluated once, at the start of the horizon. Cadence
  and completion deficit therefore set the order of requests but do not space or
  balance their snapshots through the plan.
* **Greedy.** Placements are never revisited. If a flexible, high-merit request is
  placed first, it can take the only slot a lower-merit request with a short window
  could use, and that request goes unplaced. Urgency avoids this when it raises the
  short-window request above the flexible one, so it is placed first.

The scheduling context
----------------------

:class:`~conops.schedulers.SchedulingContext` answers the questions a planner asks,
using the models ``DITL`` executes with: slew durations and paths from the real
:class:`~conops.simulation.Slew`, scoped attitude-constraint checks as ``DITL``
records them, the attitude a fresh simulation idles at before its first command,
and the ground passes in the horizon. Future planners are built on the same
context so that their plans also execute as planned.
