Target of Opportunity (TOO)
===========================

Overview
--------

The Target of Opportunity (TOO) system allows simulating time-critical observations that interrupt normal queue-scheduled operations. This is essential for modeling responses to transient astronomical events such as gamma-ray bursts (GRBs), gravitational wave counterparts, supernovae, or other phenomena requiring immediate observation.

When a TOO is submitted, it is held in a register and continuously checked during the simulation. If the TOO target becomes visible and its merit exceeds that of the current observation, the current observation is immediately preempted and the TOO is observed.

Key Features
------------

* **Merit-based preemption**: TOOs only interrupt when their merit exceeds the current observation
* **Visibility checking**: TOOs are only triggered when the target is actually observable
* **Scheduled submission**: TOOs can be scheduled to become active at a future time
* **Full event logging**: All TOO events (submission, interrupt, observation) are logged
* **Queue integration**: TOOs are inserted into the queue with boosted merit for immediate selection

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
       executed=False,       # Whether TOO has been observed
   )

Attributes
^^^^^^^^^^

* ``obsid`` (int): Unique observation identifier for this TOO
* ``ra`` (float): Right ascension in degrees
* ``dec`` (float): Declination in degrees
* ``merit`` (float): Priority merit value. Should be significantly higher than normal queue targets (e.g., 10000+) to ensure immediate observation
* ``exptime`` (int): Requested exposure time in seconds
* ``name`` (str): Human-readable name for the TOO target (e.g., "GRB 250101A")
* ``submit_time`` (float): Unix timestamp when the TOO becomes active. Default is 0.0, meaning active from simulation start
* ``executed`` (bool): Whether this TOO has been executed. Automatically set to True when the TOO triggers

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

During each simulation step, the TOO system performs the following checks:

1. **Submission time check**: Is ``submit_time <= current_time``? If not, the TOO is not yet active.

2. **Execution check**: Has the TOO already been executed? If so, skip it.

3. **Merit comparison**: Is the TOO's merit higher than the current observation's merit? If not, no interrupt occurs.

4. **Visibility check**: Is the TOO target currently visible (not occulted, not in constraint violation)? If not, the interrupt is deferred until visibility is achieved.

If all conditions are met:

1. The current observation is **terminated** (marked as preempted, not completed)
2. The TOO is **added to the queue** with a boosted merit (+100,000) to guarantee immediate selection
3. The TOO is **marked as executed** in the register
4. The spacecraft begins **slewing** to the TOO target
5. All events are **logged** for later analysis

Merit Guidelines
----------------

Normal queue targets typically have merit values in the range of 1-1000. To ensure TOOs take priority:

* **Standard TOO**: Merit 10,000+
* **High-priority TOO**: Merit 50,000+
* **Emergency TOO**: Merit 100,000+

The boosted merit added when inserting into the queue (+100,000) ensures the TOO is selected next, regardless of other queue contents.

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

   # Find TOOs that never triggered (target not visible, merit too low, etc.)
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

   2025-01-01T02:34:56Z: TOO interrupt: GRB 250101A (obsid=1000001, merit=10000.0) preempting current observation (merit=500.0)
   2025-01-01T02:34:56Z: Added TOO GRB 250101A to queue with boosted merit 110000.0

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
