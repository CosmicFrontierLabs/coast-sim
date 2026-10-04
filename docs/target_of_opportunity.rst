Target of Opportunity (TOO)
===========================

Overview
--------

The Target of Opportunity (TOO) system allows simulating time-critical observations that interrupt normal queue-scheduled operations. This is essential for modeling responses to transient astronomical events such as gamma-ray bursts (GRBs), gravitational wave counterparts, supernovae, or other phenomena requiring immediate observation.

A TOO is an ordinary queue request with a short window. When its ``submit_time`` is reached it joins the target queue with its own merit and optional deadline, and is selected like any other target. If it is visible and outranks the observation in progress, that observation is preempted and the queue selects again.

Key Features
------------

* **Ordinary requests**: An active TOO is a queue target; nothing boosts its merit
* **Tier- and merit-based preemption**: A TOO interrupts only when its tier and value, evaluated now, beat the tier and value frozen onto the current observation when it was selected
* **Deadlines**: An optional deadline drives the urgency merit term and stops the TOO being selected once it passes
* **Visibility checking**: TOOs only interrupt when the target is actually observable
* **Scheduled submission**: TOOs can be scheduled to become active at a future time
* **Full event logging**: Queuing, interrupts and the merit breakdown of each selection are logged

TOORequest Model
----------------

The ``TOORequest`` class is a Pydantic model that represents a pending TOO. It contains all the information needed to observe the target and track its status.

.. code-block:: python

   from conops.ditl import TOORequest

   too = TOORequest(
       obsid=1000001,        # Unique observation ID
       ra=180.0,             # Right ascension (degrees)
       dec=45.0,             # Declination (degrees)
       merit=10000.0,        # Priority (higher = more urgent)
       exptime=3600,         # Exposure time (seconds)
       name="GRB 250101A",   # Human-readable name
       submit_time=0.0,      # When TOO becomes active (Unix timestamp)
       deadline=None,        # Latest start of science (Unix timestamp), or None
       executed=False,       # Whether TOO has been dispatched
   )

Attributes
^^^^^^^^^^

* ``obsid`` (int): Unique observation identifier for this TOO
* ``ra`` (float): Right ascension in degrees
* ``dec`` (float): Declination in degrees
* ``merit`` (float): Base merit. Within a tier, the TOO interrupts only if its value exceeds the current observation's
* ``exptime`` (int): Requested exposure time in seconds
* ``name`` (str): Human-readable name for the TOO target (e.g., "GRB 250101A")
* ``submit_time`` (float): Unix timestamp when the TOO becomes active. Default is 0.0, meaning active from simulation start
* ``deadline`` (float or None): Latest Unix time science collection may begin. Drives the urgency term (``config.targets.urgency_weight``); the TOO cannot be selected after it
* ``executed`` (bool): Whether this TOO has been dispatched. Set to True once it becomes the current observation

Submitting TOOs
---------------

Use the ``submit_too()`` method on ``QueueDITL`` to register a TOO:

.. code-block:: python

   from conops import MissionConfig, QueueDITL
   from rust_ephem import TLEEphemeris
   from datetime import datetime, timedelta

   # Set up simulation
   cfg = MissionConfig.from_json_file("examples/example_config.json")
   begin = datetime(2025, 1, 1, 0, 0, 0)
   end = begin + timedelta(days=1)
   ephem = TLEEphemeris(tle="examples/example.tle", begin=begin, end=end)

   ditl = QueueDITL(config=cfg, ephem=ephem, begin=begin, end=end)

   # Submit a TOO that is active immediately
   ditl.submit_too(
       obsid=1000001,
       ra=180.0,
       dec=45.0,
       merit=10000.0,
       exptime=3600,
       name="GRB 250101A",
   )

   # Run the simulation
   ditl.calc()

Scheduled TOOs
^^^^^^^^^^^^^^

TOOs can be scheduled to become active at a future time. This is useful for simulating scenarios where a TOO alert arrives mid-observation:

.. code-block:: python

   # TOO becomes active 2 hours into the simulation (using Unix timestamp)
   ditl.submit_too(
       obsid=1000002,
       ra=90.0,
       dec=-30.0,
       merit=10000.0,
       exptime=1800,
       name="GRB 250101B",
       submit_time=ditl.ustart + 7200,  # 2 hours after start
   )

   # Or use a datetime object
   from datetime import datetime
   ditl.submit_too(
       obsid=1000003,
       ra=270.0,
       dec=60.0,
       merit=10000.0,
       exptime=2400,
       name="GW Event",
       submit_time=datetime(2025, 1, 1, 6, 0, 0),
   )

How TOO Interrupts Work
-----------------------

During each simulation step in science or idle modes:

1. Every TOO whose ``submit_time`` has passed is **added to the queue** once, with its own merit and deadline.
2. A TOO that is already the current observation, or has collected science, is marked **executed**.
3. If an observation is in progress, a pending TOO **interrupts** it when:

   * its deadline, if any, has not passed;
   * its target is visible now; and
   * its tier and value, evaluated now, beat the current observation's frozen tier and value.

When a TOO interrupts:

1. The current observation is **terminated** (preempted, not completed).
2. The queue **selects again**, with the interrupted target held out of this one selection so it cannot simply resume. The highest-ranked target is chosen, normally the TOO.
3. The TOO is marked **executed** if it was selected.
4. The events and the TOO's merit breakdown are **logged**.

With no observation in progress nothing is interrupted: the TOO is already in the queue and is considered at the next selection.

Making TOOs Win
---------------

There is no automatic merit boost. A TOO outranks other work through:

* **Tier**: Put TOO obsids in an observation category with a higher ``tier`` than normal science. A higher tier always wins, whatever the merits.
* **Merit**: Within a tier, give TOOs a base merit above normal targets (normal targets typically use 1-1000).
* **Deadline and urgency**: Give the TOO a ``deadline`` and set ``config.targets.urgency_weight``; its value rises as the deadline approaches.

.. code-block:: python

   from conops.config import ObservationCategories, ObservationCategory

   cfg.observation_categories = ObservationCategories(
       categories=[
           ObservationCategory(name="TOO", obsid_min=1000000, obsid_max=2000000, tier=1),
       ]
   )
   cfg.targets.urgency_weight = 50.0

   ditl.submit_too(
       obsid=1000001,
       ra=180.0,
       dec=45.0,
       merit=1000.0,
       exptime=3600,
       name="GRB 250101A",
       submit_time=ditl.ustart + 3600,
       deadline=ditl.ustart + 3600 + 4 * 3600,  # photons on target within 4 hours
   )

Accessing TOO Status
--------------------

The TOO register is accessible as ``ditl.too_register``:

.. code-block:: python

   # After running the simulation
   ditl.calc()

   # Check all TOOs
   for too in ditl.too_register:
       status = "Executed" if too.executed else "Pending"
       print(f"{too.name}: {status}")

   # Find executed TOOs
   executed_toos = [t for t in ditl.too_register if t.executed]

   # Find TOOs that were never dispatched (not visible, outranked, deadline passed, etc.)
   missed_toos = [t for t in ditl.too_register if not t.executed]

Event Logging
-------------

TOO events are logged to ``ditl.log`` with event type ``"TOO"``. You can filter for these events:

.. code-block:: python

   # Get all TOO-related events
   too_events = [e for e in ditl.log.events if e.event_type == "TOO"]

   for event in too_events:
       print(f"{event.time_formatted}: {event.description}")

Example output::

   2025-01-01T02:30:00Z: Queued TOO GRB 250101A (obsid=1000001, merit=10000.0)
   2025-01-01T02:34:56Z: TOO interrupt: GRB 250101A (obsid=1000001, tier=0 base=10000 value=10000.000 score=10000.000) preempting current observation (tier=0, value=500)

Complete Example
----------------

.. code-block:: python

   from conops import MissionConfig, QueueDITL
   from conops.targets import Queue
   from rust_ephem import TLEEphemeris
   from datetime import datetime, timedelta

   # Configuration
   cfg = MissionConfig.from_json_file("examples/example_config.json")
   begin = datetime(2025, 6, 1, 0, 0, 0)
   end = begin + timedelta(days=1)
   ephem = TLEEphemeris(tle="examples/example.tle", begin=begin, end=end)

   # Create queue with normal targets
   queue = Queue(config=cfg, ephem=ephem)
   queue.add(ra=0.0, dec=0.0, obsid=1, name="Target 1", merit=100.0, exptime=3600)
   queue.add(ra=45.0, dec=30.0, obsid=2, name="Target 2", merit=200.0, exptime=3600)
   queue.add(ra=90.0, dec=60.0, obsid=3, name="Target 3", merit=150.0, exptime=3600)

   # Create DITL
   ditl = QueueDITL(config=cfg, ephem=ephem, begin=begin, end=end, queue=queue)

   # Submit a TOO that arrives 3 hours into the simulation
   ditl.submit_too(
       obsid=9999,
       ra=120.0,
       dec=20.0,
       merit=10000.0,
       exptime=7200,
       name="GRB 250601A",
       submit_time=ditl.ustart + 10800,  # 3 hours
   )

   # Run simulation
   ditl.calc()

   # Analyze results
   print(f"TOO executed: {ditl.too_register[0].executed}")

   # Find the TOO observation in the plan
   too_obs = [p for p in ditl.plan if p.obsid == 9999]
   if too_obs:
       print(f"TOO observed for {too_obs[0].exposure_time} seconds")

   # Check which observation was preempted
   too_events = [e for e in ditl.log.events if e.event_type == "TOO"]
   for event in too_events:
       print(event.description)

API Reference
-------------

.. autoclass:: conops.ditl.TOORequest
   :members:
   :undoc-members:
   :show-inheritance:
   :no-index:

.. automethod:: conops.ditl.QueueDITL.submit_too
   :no-index:
